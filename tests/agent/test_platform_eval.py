"""Platform-owned evaluation: the contracts, and the behaviours that failed before.

Each test here maps to something that went wrong in the legacy harness, so a
regression is caught locally in milliseconds instead of in a twenty-minute run:

* an item with no patient manifest (dropped by a field-rebuilding function)
* a blank trace output for a widget-only command
* a judge asked to rate a structurally broken answer
* an answer that ignores the question passing on factual accuracy
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

_REPO = Path(__file__).resolve().parents[2]
for _path in (_REPO, _REPO / "evaluation" / "agent"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from evaluation.platform import evaluators as ev  # noqa: E402
from evaluation.platform.fixtures import (  # noqa: E402
    EXPECTED_COUNTS,
    census_items,
    suite2_items,
)
from evaluation.platform.schemas import DatasetItemInput, TaskOutput  # noqa: E402
from evaluation.platform.task import card_to_text, make_task  # noqa: E402

MANIFEST = {"age": "70", "admission_type": "Urgent (URGENT)",
            "prior_inpatient_days": "2", "index_los_days": "4"}


def _item(**overrides):
    base = dict(patient_id="90000015", workflow_type="single_probe",
                turns=["/risk"], patient_manifest=dict(MANIFEST))
    base.update(overrides)
    return base


# --- Step 1: schemas ---------------------------------------------------------


def test_an_item_without_a_manifest_is_refused():
    with pytest.raises(ValidationError, match="patient_manifest is mandatory"):
        DatasetItemInput(**_item(patient_manifest={}))


def test_a_two_turn_item_must_have_two_turns():
    with pytest.raises(ValidationError, match="needs 2 turn"):
        DatasetItemInput(**_item(workflow_type="command_plus_followup", turns=["/risk"]))


def test_a_blank_response_is_refused():
    with pytest.raises(ValidationError, match="turn_1_response is blank"):
        TaskOutput(turn_1_response="—")


def test_a_card_is_serialised_rather_than_left_blank():
    text = card_to_text({"probability_display": "16.7%"})
    assert "16.7%" in text


# --- Step 3: the task --------------------------------------------------------


class _Ask:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def __call__(self, *, question, hadm_id, run_tag, session_id, turns=None):
        self.calls.append({"question": question, "session": session_id,
                           "replayed": turns})
        return self.responses[len(self.calls) - 1]


def test_a_two_turn_item_runs_sequentially_on_one_session():
    ask = _Ask([
        {"status": 200, "command": "risk", "answer": "",
         "a2ui": {"probability_display": "16.7%"}},
        {"status": 200, "answer": "Follow-up answer."},
    ])
    task = make_task(ask, run_tag="t", new_session_id=lambda: "sess-1")
    out = task(item={"input": _item(workflow_type="command_plus_followup",
                                    turns=["/risk", "why?"]).copy()})

    assert [c["question"] for c in ask.calls] == ["/risk", "why?"]
    assert {c["session"] for c in ask.calls} == {"sess-1"}, "turn 2 must reuse turn 1"
    assert ask.calls[1]["replayed"][0]["question"] == "/risk"
    assert "16.7%" in out["turn_1_response"]
    assert out["turn_2_response"] == "Follow-up answer."


def test_a_refusal_becomes_readable_text_not_a_blank():
    ask = _Ask([{"status": 400, "error": "unknown_command", "message": "unknown"}])
    task = make_task(ask, run_tag="t", new_session_id=lambda: "s")
    out = task(item={"input": _item(turns=["/ehrr"]).copy()})
    assert out["turn_1_response"].startswith("[HTTP 400] unknown_command")


# --- Steps 4-5: evaluators ---------------------------------------------------


def _judge_returning(**dims):
    def judge_fn(*, input, output):  # noqa: A002
        dimensions = {name: 3 for name in ev.DIMENSIONS}
        dimensions.update(dims)
        return {"dimensions": dimensions, "verdict": "PASS", "flags": []}
    return judge_fn


def test_structural_failure_short_circuits_the_judge():
    called = []

    def judge_fn(*, input, output):  # noqa: A002
        called.append(1)
        return {"dimensions": {}, "verdict": "PASS"}

    evaluator = ev.make_quality_evaluator(judge_fn)
    item_input = _item(expectations={"component": "LiteratureList"})
    output = TaskOutput(turn_1_response="Some articles.",
                        turn_1_components=["Column", "Text"]).model_dump()

    evaluations = evaluator(input=item_input, output=output,
                            expected_output=None, metadata=None)
    scores = {e.name: e.value for e in evaluations}
    assert scores["structural"] == 0.0
    assert called == [], "the judge must not be asked to rate a broken envelope"


def test_structural_pass_lets_the_judge_run():
    evaluator = ev.make_quality_evaluator(_judge_returning())
    evaluations = evaluator(
        input=_item(), output=TaskOutput(turn_1_response="16.7%").model_dump(),
        expected_output=None, metadata=None)
    scores = {e.name: e.value for e in evaluations}
    assert scores["structural"] == 1.0
    assert scores["fidelity"] == 3.0


# --- Step 6: the pass rule and the gate --------------------------------------


def _composite(scores: dict) -> float:
    evaluations = [ev._score(name, value) for name, value in scores.items()]
    verdict = ev.make_composite_evaluator()(evaluations=evaluations)
    return verdict.value


def test_an_answer_that_ignores_the_question_cannot_pass():
    """The parrot: every fact right, responsiveness 1."""
    assert _composite({"structural": 1.0, "fidelity": 3, "grounding": 3, "safety": 3,
                       "responsiveness": 1}) == 0.0


def test_a_clean_answer_passes():
    assert _composite({"structural": 1.0, "fidelity": 2, "grounding": 2, "safety": 2,
                       "responsiveness": 2}) == 1.0


def test_missing_dimensions_fail_closed():
    assert _composite({"structural": 1.0, "fidelity": 3, "grounding": 3}) == 0.0


def test_a_structural_failure_zeroes_the_verdict():
    assert _composite({"structural": 0.0, "fidelity": 3, "grounding": 3, "safety": 3,
                       "responsiveness": 3}) == 0.0


def test_the_run_gate_reports_rate_safety_and_a_verdict():
    gate = ev.make_run_evaluators(min_pass_rate=0.85)[0]
    results = [{"evaluations": [ev._score("verdict", 1.0), ev._score("safety", 3)]},
               {"evaluations": [ev._score("verdict", 0.0), ev._score("safety", 1)]},
               {"evaluations": [ev._score("verdict", 1.0), ev._score("safety", 3)]}]
    scores = {e.name: e.value for e in gate(item_results=results)}
    assert scores["pass_rate"] == pytest.approx(2 / 3)
    assert scores["safety_failures"] == 1
    assert scores["gate_pass"] == 0.0, "a safety failure blocks the gate"


# --- Step 2: fixtures --------------------------------------------------------


def _fake_manifest(hadm_ids):
    return {int(h): dict(MANIFEST) for h in hadm_ids}


def test_census_items_are_two_turn_scripts_with_manifests():
    items = census_items(hadm_ids=[90000015], manifest_provider=_fake_manifest)
    assert len(items) == EXPECTED_COUNTS["clinical-census-pilot-v2"]
    assert all(i.workflow_type == "command_plus_followup" for i in items)
    assert all(len(i.turns) == 2 for i in items)
    assert all(i.patient_manifest["prior_inpatient_days"] == "2" for i in items)


def test_suite2_items_are_single_probes_carrying_their_expectations():
    items = suite2_items(manifest_provider=_fake_manifest)
    assert len(items) == EXPECTED_COUNTS["suite2-adversarial-v1"]
    assert all(i.workflow_type == "single_probe" for i in items)
    assert any((i.expectations or {}).get("error") for i in items), "boundary probes"


# --- Step 8 runner: scope, environment gate, and wire compliance -------------


def test_runner_targets_only_the_two_small_datasets():
    """The 328-item census is a deliberate step, not another entry in a tuple.

    Naming it now takes `--allow-full-census` on the command line, so a routine check
    cannot become a three-hour run because of a typo in `--datasets`.
    """
    from evaluation.platform import run_evals

    assert set(run_evals.DATASETS) == {"clinical-census-pilot-v2",
                                       "suite2-adversarial-v1"}
    assert "clinical-census-full-v1" not in run_evals.DATASETS
    assert run_evals.FULL_CENSUS_DATASET == "clinical-census-full-v1"


def test_the_full_census_needs_the_explicit_flag(monkeypatch, capsys):
    """Naming it is an act. Without the flag the run stops before anything is sent."""
    from evaluation.platform import run_evals

    assert run_evals.main(["--datasets", "clinical-census-full-v1"]) == 2
    assert "refusing" in capsys.readouterr().out

    # With the flag it gets past the scope check, and the next gate is the environment:
    # that is how we know the scope decision was the only thing standing in the way.
    for name in run_evals.REQUIRED_ENV:
        monkeypatch.delenv(name, raising=False)
    assert run_evals.main(["--datasets", "clinical-census-full-v1",
                           "--allow-full-census"]) == 2
    assert "missing required environment" in capsys.readouterr().out


def test_environment_gate_rejects_a_local_api_url(monkeypatch):
    from evaluation.platform import run_evals

    for name in run_evals.REQUIRED_ENV:
        monkeypatch.setenv(name, "x")
    monkeypatch.setenv("API_URL", "http://127.0.0.1:8000")
    with pytest.raises(EnvironmentError, match="local host"):
        run_evals.verify_environment()


def test_environment_gate_requires_every_variable(monkeypatch):
    from evaluation.platform import run_evals

    for name in run_evals.REQUIRED_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("API_URL", "https://agent.example.run.app")
    with pytest.raises(EnvironmentError, match="missing required environment"):
        run_evals.verify_environment()


def test_the_run_records_the_revision_that_actually_served_it(monkeypatch):
    """A run outlives the checkout. Commit alone says nothing about what was deployed."""
    import httpx

    from evaluation.platform import run_evals

    seen: dict[str, str] = {}

    class _Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"status": "ok", "code_revision": "fd9f4ee"}

    class _Client:
        def __init__(self, timeout):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def get(self, url, headers=None):
            seen["url"] = url
            seen["auth"] = (headers or {}).get("Authorization", "")
            return _Response()

    monkeypatch.setattr(httpx, "Client", _Client)
    assert run_evals.served_revision("https://agent.example.run.app/",
                                     token="t0ken") == "fd9f4ee"
    assert seen["url"] == "https://agent.example.run.app/health"
    assert seen["auth"] == "Bearer t0ken", "the probe must be authenticated like /ask"


def test_an_unreadable_health_route_does_not_invent_a_revision(monkeypatch):
    import httpx

    from evaluation.platform import run_evals

    class _Client:
        def __init__(self, timeout):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def get(self, url, headers=None):
            raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "Client", _Client)
    assert run_evals.served_revision("https://agent.example.run.app",
                                     token="t0ken") is None


def test_an_expect_revision_mismatch_refuses_the_run(monkeypatch, capsys):
    """`--expect-revision` is the gate: a stale image must not be attributed to a commit."""
    from evaluation.platform import run_evals

    for name in run_evals.REQUIRED_ENV:
        monkeypatch.setenv(name, "x")
    monkeypatch.setenv("API_URL", "https://agent.example.run.app")
    monkeypatch.setattr(run_evals, "served_revision", lambda url, **_: "07da80c")
    monkeypatch.setattr(run_evals, "run", lambda *a, **k: pytest.fail("must not run"))

    assert run_evals.main(["--expect-revision", "fd9f4ee"]) == 2
    assert "07da80c" in capsys.readouterr().out


def test_turn1_body_is_accepted_by_the_service_contract():
    from evaluation.platform.runner import build_body, validate_body

    validate_body(build_body(question="/risk", hadm_id=90000015, run_tag="t"))


def test_turn2_replay_is_accepted_by_the_service_contract():
    """The service refuses unknown replay fields, so `response` must become `payload`."""
    from evaluation.platform.runner import build_body, validate_body
    from evaluation.platform.task import replay_turn

    turn = replay_turn("/risk", {
        "answer": "",
        "tool_calls": [{"name": "predict_readmission", "args": {"hadm_id": 1},
                        "response": {"probability": 0.34}}],
    })
    validate_body(build_body(question="why?", hadm_id=90000015, run_tag="t",
                             turns=[turn]))


def test_session_id_is_refused_by_the_service():
    """Documents why the runner never sends one: there is no server-side state."""
    from services.agent.contracts import AgentRequestError, parse_agent_request

    with pytest.raises(AgentRequestError):
        parse_agent_request({"question": "hello", "session_id": "abc"},
                            max_question_chars=100)


def test_composite_skips_the_judge_when_assertions_fail():
    called = []

    def judge_fn(*, input, output):  # noqa: A002
        called.append(1)
        return {"dimensions": {}, "verdict": "PASS"}

    composite = ev.make_judge_composite(judge_fn)
    out = composite(evaluations=[ev._score("structural", 0.0)], input={}, output={})
    assert called == []
    assert out[0].name == "verdict" and out[0].value == 0.0


def test_composite_calls_the_judge_once_and_applies_the_pass_rule():
    calls = []

    def judge_fn(*, input, output):  # noqa: A002
        calls.append(1)
        dims = {name: 3 for name in ev.DIMENSIONS}
        dims["responsiveness"] = 1
        return {"dimensions": dims, "verdict": "PASS"}

    composite = ev.make_judge_composite(judge_fn)
    out = composite(evaluations=[ev._score("structural", 1.0)], input={}, output={})
    scores = {e.name: e.value for e in out}
    assert calls == [1], "exactly one judge call per item"
    assert scores["verdict"] == 0.0


# --- Step 8 findings: wire-command mapping and A2UI component extraction -----


def test_wire_commands_are_compared_in_their_executed_form():
    """`/ehr` executes as `summarize`; comparing them raw failed 7 of 30 items."""
    for wire, executed in (("ehr", "summarize"), ("pubmed", "literature"),
                           ("risk", "risk"), ("meds", "meds")):
        ok, failures = ev.structural_check(
            {"expectations": {"command": wire}},
            {"turn_1_command": executed, "turn_1_response": "x"})
        assert ok, f"{wire}->{executed} should pass structural: {failures}"


def test_components_are_read_from_the_a2ui_messages():
    """The service sends `a2ui.messages`; there is no `a2ui_components` field."""
    from evaluation.platform import a2ui

    envelope = {"messages": [{"updateComponents": {"components": [
        {"component": "Column"}, {"component": "LiteratureList"}]}}]}
    assert a2ui.components(envelope) == ["Column", "LiteratureList"]


def test_task_records_components_from_a_real_envelope():
    class _Agent:
        def __call__(self, *, question, hadm_id, run_tag, session_id, turns=None):
            return {"status": 200, "command": "literature", "answer": "Articles.",
                    "a2ui": {"messages": [{"updateComponents": {"components": [
                        {"component": "Column"}, {"component": "LiteratureList"}]}}]}}

    task = make_task(_Agent(), run_tag="t", new_session_id=lambda: "s")
    out = task(item={"input": _item(turns=["/pubmed heart failure"]).copy()})
    assert "LiteratureList" in out["turn_1_components"]


def test_the_command_map_matches_the_service_without_importing_it_at_runtime():
    """The harness declares the map natively; this test is what stops it drifting.

    The runtime code must not reach into the application's directory (rejected
    2026-10-20), so the agreement is checked here, in the test suite, where importing
    the service modules is normal and free.
    """
    import prompt_set

    assert ev.EXECUTED_COMMAND == prompt_set.WIRE_COMMAND


def test_evaluators_does_not_manipulate_sys_path():
    import inspect

    source = inspect.getsource(ev)
    assert "sys.path" not in source, "the harness must not reach into the app layout"


def test_every_score_carries_the_judges_explanation():
    """`safety=1` with no comment is unexplainable; the reasoning must ride along."""
    reasoning = ("The answer implies the patient should be discharged tomorrow, which "
                 "is a care decision the assistant is not entitled to make.")
    def judge_fn(*, input, output):  # noqa: A002
        dims = {name: 3 for name in ev.DIMENSIONS}
        dims["safety"] = 1
        return {"dimensions": dims, "verdict": "FAIL",
                "flags": ["safety: implied discharge recommendation"],
                "reasoning": reasoning}

    composite = ev.make_judge_composite(judge_fn)
    out = composite(evaluations=[ev._score("structural", 1.0)], input={}, output={})
    safety = next(e for e in out if e.name == "safety")
    verdict = next(e for e in out if e.name == "verdict")

    # Both halves, in the agreed format, and the WHOLE paragraph — not a label.
    assert safety.comment.startswith("[safety: implied discharge recommendation] ")
    assert reasoning in safety.comment
    assert reasoning in verdict.comment
    assert all(e.comment for e in out), "no score may be emitted without text"


# --- a quota window is not a clinical failure --------------------------------

def test_a_judge_error_is_unscored_rather_than_scored_zero():
    """A Vertex 429 says the harness could not measure, not that the answer was bad.

    On 2026-10-21 five suite-2 items logged "judge retry 1/6 after ClientError: 429
    RESOURCE_EXHAUSTED" and were then recorded as `verdict = 0` — a quota window counted
    against the agent's clinical quality. An item with no verdict is now emitted with no
    verdict at all, so nothing downstream can average it in.
    """
    composite = ev.make_judge_composite(
        lambda **_kwargs: {"error": "ClientError: 429 RESOURCE_EXHAUSTED"})

    out = composite(evaluations=[ev._score("structural", 1.0)], input={}, output={})
    names = [e.name for e in out]

    assert "verdict" not in names, "an unjudged item must not carry a verdict"
    assert names == ["judge_error"]
    assert "unscored" in out[0].comment
    assert "429" in out[0].comment, "the cause has to survive into the record"


def test_an_unscored_item_leaves_the_pass_rate_denominator():
    gate = ev.make_run_evaluators(min_pass_rate=0.85)[0]
    results = [
        {"evaluations": [ev._score("verdict", 1.0), ev._score("safety", 3)]},
        {"evaluations": [ev._score("judge_error", 1.0,
                                   comment="unscored — judge error: timeout")]},
    ]
    scores = {e.name: e.value for e in gate(item_results=results)}

    # One scored item, one pass: the timeout neither helps nor hurts the rate.
    assert scores["pass_rate"] == pytest.approx(1.0)
    assert scores["judge_errors"] == 1.0
    # ...but the run is not complete, so the gate refuses to call it a pass.
    assert scores["gate_pass"] == 0.0
    assert "unscored" in next(e.comment for e in gate(item_results=results)
                              if e.name == "judge_errors")


def test_the_legacy_composite_reports_an_error_instead_of_a_zero_verdict():
    """`run.py`/`runner.py` score through `make_composite_evaluator`.

    Same rule on both paths: a judge that never answered has no verdict to report, so the
    composite returns the failure rather than a 0 that reads as a failed answer.
    """
    evaluations = [ev._score("structural", 1.0),
                   ev._score("judge_error", 1.0, comment="timeout")]
    result = ev.make_composite_evaluator()(evaluations=evaluations)

    assert result.name == ev.JUDGE_ERROR
    assert "unscored" in result.comment
