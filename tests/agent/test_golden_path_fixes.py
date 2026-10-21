"""Golden-path fixes: dashboard IO, open-book evidence, and the "parrot" loophole.

Three defects, each observed in a real run on 2026-10-19/20:

1. **Empty root traces.** The trace table showed dashes for every row because the
   wrapper that opens the root span never set the trace's `input`/`output`. Reading a
   run meant opening the graph and hunting for the question.

2. **Evidence the judge was not shown.** `get_lab_trends` returns a `synthetic` flag
   and a `note`; both were dropped before the prompt was built, so an answer quoting
   the tool's own "these values are synthetic" disclaimer read as a fabrication. The
   patient manifest was worse: it arrives as pre-flight context, never as a tool call,
   so it never reached the judge at all.

3. **The parrot loophole.** A turn could repeat the previous answer verbatim, ignore
   the question, and still pass, because the pass rule did not require responsiveness.

Run: `./.venv/bin/python -m pytest tests/agent/test_golden_path_fixes.py -q`
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "evaluation" / "agent"))

import judge  # noqa: E402
from services.agent import observability  # noqa: E402


# --- 1. trace input/output ---------------------------------------------------


class _RecordingClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def set_current_trace_io(self, *, input=None, output=None) -> None:  # noqa: A002
        self.calls.append({"input": input, "output": output})


def _install_fake_langfuse(monkeypatch, client=None):
    client = client or _RecordingClient()
    monkeypatch.setitem(sys.modules, "langfuse",
                        types.SimpleNamespace(get_client=lambda: client))
    monkeypatch.setattr(observability, "configured", lambda: True)
    return client


def test_trace_io_is_written_to_the_current_trace(monkeypatch):
    client = _install_fake_langfuse(monkeypatch)
    observability.set_trace_io("Are any high-risk medications listed?", "Yes: ...")
    assert client.calls == [{"input": "Are any high-risk medications listed?",
                             "output": "Yes: ..."}]


def test_empty_strings_stay_empty_rather_than_becoming_sentinels(monkeypatch):
    """An empty answer is an empty answer; the column should be blank, not a word."""
    client = _install_fake_langfuse(monkeypatch)
    observability.set_trace_io("q", "")
    assert client.calls == [{"input": "q", "output": None}]


def test_trace_io_never_raises(monkeypatch):
    """Rule 1: a dashboard defect must never fail an answer."""

    class _Broken:
        def set_current_trace_io(self, **_kwargs):
            raise RuntimeError("langfuse is down")

    _install_fake_langfuse(monkeypatch, _Broken())
    observability.set_trace_io("q", "a")  # must not raise


def test_trace_io_is_a_no_op_when_tracing_is_off(monkeypatch):
    called = []
    monkeypatch.setitem(sys.modules, "langfuse", types.SimpleNamespace(
        get_client=lambda: called.append(1)))
    monkeypatch.setattr(observability, "configured", lambda: False)
    observability.set_trace_io("q", "a")
    assert called == []


def test_a_command_turn_reports_the_prose_it_actually_produced():
    """The root span of a command turn used to carry no output at all.

    It looked for `narrative_text`, `answer` and `text` on the command's state. None of
    those keys exist there — the first is composed later, in the HTTP envelope — so
    2026-10-21 finished a run with 25 command traces and not one output in the table,
    while every free-text trace had one.
    """
    from langchain_core.messages import AIMessage

    from services.agent.graph import command_trace_output

    prose = {"messages": [AIMessage(content="**Clinical course.** The patient …")],
             "command": "summarize"}
    assert command_trace_output(prose).startswith("**Clinical course.**")

    # `/meds`: the model reply is extraction JSON; the sentence a reader sees is the
    # code-written narrative, and that is what belongs on the trace.
    meds = {"messages": [AIMessage(content='{"medications": []}')],
            "command": "meds",
            "deterministic_answer": "13 discharge medications, as recorded …"}
    assert command_trace_output(meds).startswith("13 discharge medications")

    # `/risk` has no prose by design: the widget carries the answer, so the payload is
    # the honest record of what the turn returned.
    risk = {"messages": [], "command": "risk",
            "command_payload": {"medications": [], "threshold": 0.11}}
    assert '"threshold": 0.11' in command_trace_output(risk)

    # A turn that produced nothing says so with an empty string, never a sentinel.
    assert command_trace_output({"messages": [], "command": "risk"}) == ""


# --- 2. open-book evidence ---------------------------------------------------


def test_lab_disclaimers_reach_the_judge():
    ev = judge._evidence({
        "name": "get_lab_trends",
        "response": {"trends": [{"lab": "creatinine"}], "synthetic": True,
                     "note": "values are synthetic and simplified for this demo"},
    })
    assert ev["synthetic"] is True
    assert "synthetic" in (ev["note"] or "")
    assert ev["labs"], "the series must still be present"


def test_manifest_fields_are_passed_untruncated():
    """LOS, admission type and prior inpatient days are the fields answers get failed on."""
    resp = {"admission_type": "EW EMER.", "prior_inpatient_days": 3,
            "length_of_stay_days": 6, "age": 71}
    ev = judge._evidence({"name": "get_admission_profile", "response": resp})
    assert ev["patient_manifest"] == resp


def test_record_level_manifest_is_evidence_and_goes_first():
    rec = {
        "prompt": "labs",
        "tool_calls": [{"name": "get_lab_trends", "response": {"trends": []}}],
        "admission_profile": {"admission_type": "EW EMER.", "prior_inpatient_days": 3},
    }
    ev = judge._evidence_for(rec)
    assert ev[0]["tool"] == "admission_manifest"
    assert ev[0]["patient_manifest"]["prior_inpatient_days"] == 3


def test_absent_manifest_adds_no_phantom_evidence():
    ev = judge._evidence_for({"prompt": "labs", "tool_calls": []})
    assert ev == []


# --- 2b. manifest capture at collection time (Option A) ----------------------


def test_census_case_carries_the_decoded_manifest():
    """`build_census` puts the manifest on the case, decoded by the server's decoder."""
    import build_census

    manifest = {"index_los_days": "6", "admission_type": "Urgent (URGENT)",
                "prior_inpatient_days": "3"}
    case = build_census.case_for(90000015, {}, {90000015: manifest})
    assert case["manifest"] == manifest


def test_a_case_without_a_decoded_manifest_still_builds():
    """An admission the decoder could not read is a missing manifest, not a crash."""
    import build_census

    assert build_census.case_for(90000015, {})["manifest"] == {}


def test_collected_census_record_carries_the_manifest():
    """The record the judge reads must carry it; the case file alone reaches nobody."""
    import collect_http

    case = {"hadm_id": 90000015, "prompt": "labs", "question": "q",
            "manifest": {"index_los_days": "6", "prior_inpatient_days": "3"}}
    rec = collect_http._record_from_response(case, 200, {"answer": "a"})
    assert rec["admission_profile"]["index_los_days"] == "6"
    # and it must actually reach the judge's evidence, first, as the sheet to check on
    ev = judge._evidence_for(rec)
    assert ev[0]["tool"] == "admission_manifest"
    assert ev[0]["patient_manifest"]["prior_inpatient_days"] == "3"


def test_a_record_without_a_manifest_gains_no_empty_field():
    import collect_http

    rec = collect_http._record_from_response(
        {"hadm_id": 90000015, "prompt": "labs", "question": "q"}, 200, {"answer": "a"})
    assert "admission_profile" not in rec


def test_the_census_is_never_read_from_the_mounted_fixture():
    """A mounted cohort would shadow the one just built from the served table."""
    import collect_http

    assert collect_http.CASES == collect_http.RESULTS / "cases_82.json"
    assert collect_http.SUITE2.name == "suite2_cases.json"


# --- 3. the parrot loophole --------------------------------------------------


def _sample(verdict="PASS", **dims):
    dimensions = {"fidelity": 3, "grounding": 3, "safety": 3, "citation": 3,
                  "responsiveness": 3, "context": 3}
    dimensions.update(dims)
    return {"verdict": verdict, "dimensions": dimensions, "flags": []}


def test_a_grounded_answer_that_ignores_the_question_now_fails():
    """The pilot's verbatim Turn-2 repeats: every fact right, the question unanswered."""
    j = judge._apply_pass_rule(_sample(responsiveness=1))
    assert j["verdict"] == "FAIL"
    assert any("responsiveness" in f for f in j["flags"])


def test_responsiveness_at_two_still_passes():
    assert judge._apply_pass_rule(_sample(responsiveness=2))["verdict"] == "PASS"


def test_missing_responsiveness_fails_closed():
    """A PASS nobody can verify from the scores is not a PASS."""
    sample = _sample()
    del sample["dimensions"]["responsiveness"]
    assert judge._apply_pass_rule(sample)["verdict"] == "FAIL"


def test_the_rule_never_upgrades_a_fail():
    """The rule closes a loophole; it must not invent passes the judge refused."""
    assert judge._apply_pass_rule(_sample(verdict="FAIL"))["verdict"] == "FAIL"


def test_an_error_sample_is_left_alone():
    broken = {"error": "429 RESOURCE_EXHAUSTED"}
    assert judge._apply_pass_rule(broken) == {"error": "429 RESOURCE_EXHAUSTED"}


def test_a_split_median_downgrades_a_majority_pass():
    """Two samples at 1 and one at 2 give a median of 1: the row cannot claim PASS."""
    samples = [_sample(responsiveness=1), _sample(responsiveness=1),
               _sample(responsiveness=2)]
    out = judge._consensus(samples)
    assert out["dimensions"]["responsiveness"] == 1.0
    assert out["verdict"] == "FAIL"


def test_rubric_states_the_new_pass_rule():
    """A prompt invariant: the judge is told the rule the harness enforces."""
    rubric = (Path(__file__).resolve().parents[2] / "evaluation" / "agent" / "rubric.md")
    text = rubric.read_text()
    assert "responsiveness >= 2" in text
    assert "Version 4" in text
