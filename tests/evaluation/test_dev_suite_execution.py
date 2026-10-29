"""Section 4 — the checks a gated dev-suite run rests on, offline.

Gate 1 is a live run against the deployed service, so the deterministic half of the
score has to be right before it spends a call. These tests cover the three places a
canary result could be wrong without anything failing:

* the assertions the dev suite declares (verified flag, components, literal anchors),
* the scenario selection (a canary of two items out of three must not read green), and
* the rubric version stamped onto the run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evaluation.platform import run_evals  # noqa: E402
from evaluation.platform.evaluators import structural_check  # noqa: E402


def _check(expectations: dict, output: dict) -> tuple[bool, list[str]]:
    return structural_check({"expectations": expectations}, output)


# ------------------------------------------------------- dev-suite expectations


def test_a_verified_flag_must_match_when_the_item_declares_one():
    ok, _ = _check({"verified": False}, {"turn_1_response": "x", "turn_1_verified": False})
    assert ok

    ok, failures = _check({"verified": False},
                          {"turn_1_response": "x", "turn_1_verified": True})
    assert not ok and "verified: expected False, got True" in failures


def test_an_absent_verified_flag_is_not_read_as_false():
    """An envelope that does not carry the flag says nothing about grounding; treating
    absence as False would fail every turn from an older service."""
    ok, _ = _check({"verified": True}, {"turn_1_response": "x", "turn_1_verified": None})
    assert ok
    ok, _ = _check({"verified": True}, {"turn_1_response": "x"})
    assert ok


def test_a_required_component_must_be_rendered_and_a_forbidden_one_must_not():
    output = {"turn_1_response": "x",
              "turn_components": [["Column", "RiskInfluencers"]]}
    ok, _ = _check({"components": ["RiskInfluencers"]}, output)
    assert ok

    ok, failures = _check({"components": ["RiskInfluencers"]},
                          {"turn_1_response": "x", "turn_components": [["Column"]]})
    assert not ok and "RiskInfluencers missing" in failures[0]

    ok, failures = _check({"forbidden_components": ["RiskInfluencers"]}, output)
    assert not ok and "forbidden component rendered" in failures[0]


def test_components_are_collected_from_every_turn():
    output = {"turn_1_response": "a", "turn_components": [["Column"], ["CitedNarrative"]]}
    ok, _ = _check({"components": ["CitedNarrative"]}, output)
    assert ok


def test_a_literal_anchor_is_matched_across_the_whole_thread():
    output = {"turn_1_response": "Discharged home with home health.",
              "turn_responses": ["Discharged home with home health.",
                                 "She was on Coumadin and Lovenox."]}
    ok, failures = _check({"must_contain": ["coumadin", "HOME HEALTH"]}, output)
    assert ok, failures


def test_a_missing_anchor_fails_and_names_itself():
    ok, failures = _check({"must_contain": ["Coumadin"]},
                          {"turn_1_response": "no drugs documented",
                           "turn_responses": ["no drugs documented"]})
    assert not ok and "must_contain: 'Coumadin' absent" in failures[0]


def test_a_disclosure_from_another_record_fails_the_item():
    """The injection scenario's whole point: a name from the other chart is proof."""
    ok, failures = _check(
        {"must_not_disclose": ["Coumadin", "Lovenox"]},
        {"turn_1_response": "ok",
         "turn_responses": ["ok", "90000015 was discharged on Coumadin."]})
    assert not ok and "must_not_disclose: 'Coumadin' present" in failures[0]


def test_a_literature_marker_must_carry_its_citation_form():
    ok, _ = _check({"citation_marker": "PMID"},
                   {"turn_1_response": "DAPT is supported ^[PMID: 12345678]."})
    assert ok

    ok, failures = _check({"citation_marker": "PMID"},
                          {"turn_1_response": "DAPT is supported by trial 12345678."})
    assert not ok and "citation_marker" in failures[0]


def test_a_suite_two_item_is_unaffected_by_the_dev_suite_keys():
    ok, failures = _check({"command": "risk", "component": "RiskInfluencers"},
                          {"turn_1_response": "x", "turn_1_command": "risk",
                           "turn_1_components": ["RiskInfluencers"]})
    assert ok and failures == []


# ----------------------------------------------------------- scenario selection


class _Item:
    def __init__(self, item_id: str, scenario_id: str) -> None:
        self.id = item_id
        self.input = {"scenario_id": scenario_id}


def test_the_canary_selects_its_three_scenarios_in_the_order_named():
    items = [_Item("a", "dev_t1_01"), _Item("b", "dev_t2_03"), _Item("c", "dev_t7_01"),
             _Item("d", "dev_t3_01")]
    chosen = run_evals._select_scenarios(items, run_evals.CANARY_SCENARIOS)
    assert [item.id for item in chosen] == ["a", "b", "c"]


def test_a_canary_missing_one_of_its_scenarios_is_refused():
    items = [_Item("a", "dev_t1_01"), _Item("b", "dev_t2_03")]
    with pytest.raises(RuntimeError, match="not in the dataset"):
        run_evals._select_scenarios(items, run_evals.CANARY_SCENARIOS)


def test_the_canary_is_the_three_scenarios_the_phase_document_names():
    assert run_evals.CANARY_SCENARIOS == ("dev_t1_01", "dev_t2_03", "dev_t7_01")


def test_an_item_without_a_scenario_id_falls_back_to_its_dataset_id():
    class _Plain:
        id = "dev_t9_99"
        input: dict = {}

    assert run_evals._item_scenario_id(_Plain) == "dev_t9_99"


# --------------------------------------------------------------- rubric version


def test_the_run_stamps_the_rubric_version_it_is_graded_under():
    assert run_evals.rubric_version() == "v6"
    header = (REPO / "evaluation" / "rubric.md").read_text().splitlines()[0]
    assert run_evals.rubric_version() in header


def test_the_dev_suite_is_an_allowed_target_and_not_a_default_one():
    assert run_evals.DEV_SUITE_DATASET in run_evals.DATASETS or (
        run_evals.DEV_SUITE_DATASET == "ecc-tier2-dev-suite")
    assert run_evals.DEV_SUITE_DATASET not in run_evals.DATASETS


def test_the_item_input_sent_to_langfuse_validates_as_a_dataset_item():
    """The upload payload is what the task validates at run time.

    On 2026-10-23 the canary failed all three items before sending anything, because
    the input carried `hadm_id` and the contract reads `patient_id`. The payload is
    checked here instead of after a run.
    """
    from evaluation.platform.schemas import DatasetItemInput  # noqa: PLC0415
    from evaluation.platform.sync_dev_suite import build_items  # noqa: PLC0415

    payloads = build_items(manifests={
        hadm: {"encounter": "inpatient"} for hadm in
        (90000015, 90000108, 90000106, 90000105, 90000023, 90000051, 90000073)})
    validated = [DatasetItemInput.model_validate(p["input"]) for p in payloads]
    assert len(validated) == 26
    assert validated[0].patient_id == "90000015"
    assert validated[0].turns == ["/ehr"]
    # The ground truth rides on the item's own `expected_output` field, which is what
    # the run hands the evaluators alongside the input.
    assert payloads[0]["expected_output"]["facts"]
    assert payloads[0]["expected_output"]["behaviour"]
