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
