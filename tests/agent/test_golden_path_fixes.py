"""Golden-path fixes: dashboard IO, open-book evidence, and the "parrot" loophole.

Three defects, each observed in a real run on 2026-10-19/20:

1. **Empty root traces.** The trace table showed dashes for every row because the
   wrapper that opens the root span never set the trace's `input`/`output`. Reading a
   run meant opening the graph and hunting for the question.

2. **Evidence the judge was not shown.** `get_lab_trends` returns a `synthetic` flag
   and a `note`; both were dropped before the prompt was built, so an answer quoting
   the tool's own "these values are synthetic" disclaimer read as a fabrication. The
   patient manifest was worse: it arrives as pre-flight context, never as a tool call,
   so it never reached the judge at all. That plumbing now lives in
   `evaluation/platform/judge_call.py` and is asserted by `dry_run.py`; the census-side
   half of it is what the tests below still cover.

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

from evaluation.platform import judge_client  # noqa: E402
from evaluation.platform.schemas import PASS_DIMENSIONS  # noqa: E402
from services.agent.telemetry import observability  # noqa: E402


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


def test_the_trace_id_is_read_from_the_span_not_from_the_context():
    """Four command turns on 2026-10-21 answered with no trace id at all.

    `current_trace_id()` asks the ambient context, and the context is the part that can
    go missing — copied into a worker thread, or not yet populated. The span we opened
    already knows its trace (`LangfuseSpan.trace_id`, which is what the SDK itself
    reads), so that is the source the response uses.
    """
    span = types.SimpleNamespace(trace_id="2f9c1e4a5b6d")
    assert observability.span_trace_id(span) == "2f9c1e4a5b6d"

    # A span without an id is not an error: the caller falls back to the context,
    # and an empty string is how "no id" is spelled everywhere in this module.
    assert observability.span_trace_id(types.SimpleNamespace()) == ""
    assert observability.span_trace_id(None) == ""
    assert observability.span_trace_id(object()) == ""


def test_both_routes_take_the_id_from_the_span():
    """The wiring, asserted where it lives: both root spans hand their id over."""
    from pathlib import Path

    from services.agent.engine import graph

    source = Path(graph.__file__).read_text(encoding="utf-8")
    assert source.count("observability.span_trace_id(root_span)") == 2, (
        "the command path and the free-text path must both read the span's own id"
    )


def test_a_command_turn_reports_the_prose_it_actually_produced():
    """The root span of a command turn used to carry no output at all.

    It looked for `narrative_text`, `answer` and `text` on the command's state. None of
    those keys exist there — the first is composed later, in the HTTP envelope — so
    2026-10-21 finished a run with 25 command traces and not one output in the table,
    while every free-text trace had one.
    """
    from langchain_core.messages import AIMessage

    from services.agent.engine.graph import command_trace_output

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


# --- 2. the manifest reaches the judge --------------------------------------
#
# The evidence list this section used to build (`judge._evidence_for`) is gone: the
# platform builds one prompt per item in `evaluation/platform/judge_call.py`, with the
# manifest first, and `dry_run.py` asserts that end to end. What stays checkable here
# without a live judge is that the census still decodes the manifest onto the case.


def test_census_case_carries_the_decoded_manifest():
    """`build_census` puts the manifest on the case, decoded by the server's decoder."""
    from evaluation.agent import build_census

    manifest = {"index_los_days": "6", "admission_type": "Urgent (URGENT)",
                "prior_inpatient_days": "3"}
    case = build_census.case_for(90000015, {}, {90000015: manifest})
    assert case["manifest"] == manifest


def test_a_case_without_a_decoded_manifest_still_builds():
    """An admission the decoder could not read is a missing manifest, not a crash."""
    from evaluation.agent import build_census

    assert build_census.case_for(90000015, {})["manifest"] == {}


# --- 3. the parrot loophole --------------------------------------------------

def _sample(verdict="PASS", **dims):
    dimensions = {"fidelity": 3, "grounding": 3, "safety": 3, "citation": 3,
                  "responsiveness": 3, "context": 3}
    dimensions.update(dims)
    return {"verdict": verdict, "dimensions": dimensions, "flags": []}


def test_a_grounded_answer_that_ignores_the_question_now_fails():
    """The pilot's verbatim Turn-2 repeats: every fact right, the question unanswered."""
    j = judge_client.apply_pass_rule(_sample(responsiveness=1))
    assert j["verdict"] == "FAIL"
    assert any("responsiveness" in f for f in j["flags"])


def test_responsiveness_at_two_still_passes():
    assert judge_client.apply_pass_rule(_sample(responsiveness=2))["verdict"] == "PASS"


def test_missing_responsiveness_fails_closed():
    """A PASS nobody can verify from the scores is not a PASS."""
    sample = _sample()
    del sample["dimensions"]["responsiveness"]
    assert judge_client.apply_pass_rule(sample)["verdict"] == "FAIL"


def test_the_rule_never_upgrades_a_fail():
    """The rule closes a loophole; it must not invent passes the judge refused."""
    assert judge_client.apply_pass_rule(_sample(verdict="FAIL"))["verdict"] == "FAIL"


def test_a_citation_failure_downgrades_a_pass():
    """`00e`: citation was scored and excluded, and the gate read True at 4/15 zeroes.

    A marker that opens a passage which does not support its sentence is a false
    statement about where the claim came from, so `PASS` beside `citation: 1` — or 0 —
    cannot stand.
    """
    for citation in (0, 1):
        graded = judge_client.apply_pass_rule(_sample(citation=citation))
        assert graded["verdict"] == "FAIL", citation
        assert any("citation" in flag for flag in graded["flags"]), citation


def test_a_citation_of_two_still_passes():
    """Some citations misplaced is a 2 on the rubric, and 2 is a pass."""
    assert judge_client.apply_pass_rule(_sample(citation=2))["verdict"] == "PASS"


def test_citation_is_one_of_the_gated_dimensions():
    assert "citation" in PASS_DIMENSIONS


def test_an_error_sample_is_left_alone():
    broken = {"error": "429 RESOURCE_EXHAUSTED"}
    assert judge_client.apply_pass_rule(broken) == {"error": "429 RESOURCE_EXHAUSTED"}


def test_rubric_states_the_new_pass_rule():
    """A prompt invariant: the judge is told the rule the harness enforces."""
    rubric = Path(__file__).resolve().parents[2] / "evaluation" / "rubric.md"
    text = rubric.read_text()
    assert "responsiveness >= 2" in text
    assert "Version 4" in text

