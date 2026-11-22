"""The Dev matrix contract and the deterministic gate that holds the file to it.

Two things are tested here that nothing tested before. The first is that a row's
per-turn expectations reach the evaluator at all: `turn_expectations` was written into
the dataset and dropped by `item_payload`, so `evaluators._turn_expectation_failures`
was handed nothing and asserted nothing. The second is that the suite is a matrix: an
item that declares a cell is checked against the allocation, against the affordance
matrix the answer keys were copied from, and against the record's own section of the
fact sheet.

No network, no model, no BigQuery: every test below reads files and pure functions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.platform import dataset_audit, dev_suite, evaluators

FACT_SHEET = """## 90000015

### Manifest and risk

| field | raw | published |
|---|---|---|
| encounter | `inpatient admission` | inpatient admission |

Status post right total knee replacement secondary to degenerative joint disease.

## 90000045

### Manifest and risk

| field | raw | published |
|---|---|---|
| encounter | `inpatient admission` | inpatient admission |

Hyperglycemia with cholelithiasis and cholecystitis.
"""

MATRIX = """## 90000015 — synthetic record

| command | `[0]` | `[1]` | `[2]` | chips |
|---|---|---|---|---|
| `/ehr` | What post-discharge activity restrictions are documented? | — | — | 1 |
| `/meds` | What is the discharge dosing plan for Coumadin? | — | — | 1 |
| `/labs` | How does the abnormal discharge Hemoglobin (10.0 g/dL) compare to its reference range and trend? | — | — | 1 |

| # | feature | label | direction | chip (exact) |
|---|---|---|---|---|
| 1 | `prior inpatient days` | Prior Inpatient Days | increases | How does Prior Inpatient Days affect this patient's readmission risk? |

## 90000045 — synthetic record

| command | `[0]` | `[1]` | `[2]` | chips |
|---|---|---|---|---|
| `/ehr` | — | — | — | 0 |
| `/labs` | — | — | — | 0 |

| # | feature | label | direction | chip (exact) |
|---|---|---|---|---|
"""


def _row(**overrides):
    row = {
        "id": "dev_synthetic_01",
        "hadm_id": "90000015",
        "workflow_type": "multi_turn",
        "trajectory": "t2_dynamic_affordance_loopback",
        "command": "ehr",
        "turns": ["/ehr", "What did the operation involve?"],
        "expected_output": {"must_contain": [], "behaviour": "Plain prose."},
        "metadata": {
            "depth_tier": "2_turn",
            "modality_axis": "followup_chip",
            "rubric_version": "v8",
            "turn1_follow_ups": True,
        },
    }
    row.update(overrides)
    return row


@pytest.fixture()
def artefacts(tmp_path: Path) -> tuple[Path, Path]:
    facts = tmp_path / "fact_sheet.md"
    facts.write_text(FACT_SHEET, encoding="utf-8")
    matrix = tmp_path / "matrix.md"
    matrix.write_text(MATRIX, encoding="utf-8")
    return facts, matrix


def test_turn_expectations_reach_the_evaluator():
    """A per-turn assertion in the row must arrive in `expectations`, not only in the
    copy the judge reads."""
    turn_expectations = [{"facts": {"taper_drug": "oxycodone SR"}}]
    payload = dev_suite.item_payload(
        _row(expected_output={"must_contain": [], "behaviour": "Prose.",
                              "turn_expectations": turn_expectations}),
        {},
    )
    assert payload["expectations"]["turn_expectations"] == turn_expectations
    assert payload["expected_output"]["turn_expectations"] == turn_expectations


def test_the_matrix_is_forty_scenarios_over_one_hundred_and_twenty_turns():
    assert dev_suite.EXPECTED_ITEMS == 40
    assert dev_suite.EXPECTED_TURNS == 120
    assert len(dev_suite.DEV_MATRIX) == len(dev_suite.DEPTH_TIERS) * len(
        dev_suite.MODALITY_AXES)
    per_depth = {}
    for (depth, _axis), count in dev_suite.DEV_MATRIX.items():
        per_depth[depth] = per_depth.get(depth, 0) + count
    assert per_depth == {depth: 8 for depth in dev_suite.DEPTH_TIERS}


def test_must_contain_is_scoped_to_the_records_own_section(artefacts):
    facts, matrix = artefacts
    good = _row(expected_output={"must_contain": ["knee replacement"],
                                 "behaviour": "Prose."})
    defects = dataset_audit.dev_deterministic_defects(
        [good], fact_sheet=facts, matrix_doc=matrix)
    assert defects == {}

    bad = _row(expected_output={"must_contain": ["cholecystectomy"],
                                "behaviour": "Prose."})
    defects = dataset_audit.dev_deterministic_defects(
        [bad], fact_sheet=facts, matrix_doc=matrix)
    assert any("not in the fact sheet section for 90000015" in problem
               for problem in defects["dev_synthetic_01"])


def test_a_risk_chip_must_name_a_driver_the_matrix_lists(artefacts):
    facts, matrix = artefacts
    listed = ("How does Prior Inpatient Days affect this patient's readmission risk?")
    good = _row(metadata={"depth_tier": "2_turn", "modality_axis": "risk_chip",
                          "rubric_version": "v8"},
                command=None,
                turns=["/ehr", {"prompt": listed, "intent": "risk_driver"}])
    assert dataset_audit.dev_deterministic_defects(
        [good], fact_sheet=facts, matrix_doc=matrix) == {}

    invented = _row(metadata={"depth_tier": "2_turn", "modality_axis": "risk_chip",
                              "rubric_version": "v8"},
                    command=None,
                    turns=["/ehr", {"prompt": "Why is this patient high risk?",
                                    "intent": "risk_driver"}])
    defects = dataset_audit.dev_deterministic_defects(
        [invented], fact_sheet=facts, matrix_doc=matrix)
    assert any("not a driver chip the matrix lists" in problem
               for problem in defects["dev_synthetic_01"])


def test_a_followup_chip_must_index_a_slot_the_matrix_fills(artefacts):
    facts, matrix = artefacts
    good = _row(turns=["/ehr", "{{turn_1.follow_ups[0]}}"])
    assert dataset_audit.dev_deterministic_defects(
        [good], fact_sheet=facts, matrix_doc=matrix) == {}

    beyond = _row(turns=["/ehr", "{{turn_1.follow_ups[2]}}"])
    defects = dataset_audit.dev_deterministic_defects(
        [beyond], fact_sheet=facts, matrix_doc=matrix)
    assert any("indexes slot 2" in problem
               for problem in defects["dev_synthetic_01"])


def test_a_followup_chip_may_not_read_meds_slots(artefacts):
    facts, matrix = artefacts
    row = _row(command="meds", turns=["/meds", "{{turn_1.follow_ups[0]}}"])
    defects = dataset_audit.dev_deterministic_defects(
        [row], fact_sheet=facts, matrix_doc=matrix)
    assert any("/meds chips" in problem for problem in defects["dev_synthetic_01"])


def test_a_raw_contribution_float_in_behaviour_is_refused(artefacts):
    facts, matrix = artefacts
    row = _row(expected_output={"must_contain": [],
                                "behaviour": "The driver contributed +0.1234 to the "
                                             "score."})
    defects = dataset_audit.dev_deterministic_defects(
        [row], fact_sheet=facts, matrix_doc=matrix)
    assert any("raw contribution float" in problem
               for problem in defects["dev_synthetic_01"])


def test_a_component_claim_in_behaviour_is_refused(artefacts):
    facts, matrix = artefacts
    row = _row(expected_output={"must_contain": [],
                                "behaviour": "The answer renders the four bold labels."})
    defects = dataset_audit.dev_deterministic_defects(
        [row], fact_sheet=facts, matrix_doc=matrix)
    assert any("grades a UI component" in problem
               for problem in defects["dev_synthetic_01"])


def test_the_matrix_reports_a_cell_that_is_short(artefacts):
    defects = dev_suite.validate_dev_matrix([_row()])
    assert defects["__matrix__"]
    assert any("1 scenario(s), expected 40" in problem
               for problem in defects["__matrix__"])


def test_audit_guard_refuses_a_missing_attestation(tmp_path, monkeypatch):
    dataset = tmp_path / "dev_suite.jsonl"
    dataset.write_text("", encoding="utf-8")
    monkeypatch.setitem(dataset_audit.AUDIT_PATHS, "dev", tmp_path / "absent.json")
    monkeypatch.setitem(dataset_audit.AUDIT_SOURCES, "dev", dataset)
    with pytest.raises(SystemExit) as caught:
        dataset_audit.audit_guard("dev")
    assert "no audit attestation" in str(caught.value)


def test_audit_guard_refuses_a_stale_attestation(tmp_path, monkeypatch):
    dataset = tmp_path / "dev_suite.jsonl"
    dataset.write_text("{}", encoding="utf-8")
    attestation = tmp_path / "dev_suite.audit.json"
    attestation.write_text(json.dumps({
        "dataset_sha256": "0" * 64, "defects": {}, "items": {},
    }), encoding="utf-8")
    monkeypatch.setitem(dataset_audit.AUDIT_PATHS, "dev", attestation)
    monkeypatch.setitem(dataset_audit.AUDIT_SOURCES, "dev", dataset)
    with pytest.raises(SystemExit) as caught:
        dataset_audit.audit_guard("dev")
    assert "stale" in str(caught.value)


def test_audit_guard_refuses_a_recorded_defect(tmp_path, monkeypatch):
    dataset = tmp_path / "dev_suite.jsonl"
    dataset.write_text("{}", encoding="utf-8")
    attestation = tmp_path / "dev_suite.audit.json"
    attestation.write_text(json.dumps({
        "dataset_sha256": dataset_audit.sha256(dataset),
        "defects": {"dev_synthetic_01": ["a problem"]},
        "items": {},
    }), encoding="utf-8")
    monkeypatch.setitem(dataset_audit.AUDIT_PATHS, "dev", attestation)
    monkeypatch.setitem(dataset_audit.AUDIT_SOURCES, "dev", dataset)
    with pytest.raises(SystemExit) as caught:
        dataset_audit.audit_guard("dev")
    assert "refusing to publish" in str(caught.value)


# ------------------------------------------------- citation numbers are per turn


def _thread(second_sources, second_answer="Covered in ^[1] and ^[2]."):
    """A two-turn thread: chart passages under [1], [2], then a literature turn."""
    return {
        "turn_responses": ["Findings are in ^[1] and ^[2].", second_answer],
        "turn_sources": [
            [{"cite": 1, "label": "brief_hospital_course"},
             {"cite": 2, "label": "discharge_instructions"}],
            second_sources,
        ],
    }


def test_a_literature_turn_resolves_its_own_numbers_not_the_earlier_ones():
    """Each turn numbers its own sources from one.

    Turn 1 publishes two chart passages as [1] and [2]; the `/pubmed` turn publishes two
    abstracts as [1] and [2] of its own. Resolving with `setdefault` kept the chart
    passages in those slots, so the literature turn's markers pointed at note passages
    with no identifier and a correctly cited answer failed.
    """
    output = _thread([
        {"cite": 1, "pmid": "24108298", "label": "PMID: 24108298"},
        {"cite": 2, "pmid": "24108299", "label": "PMID: 24108299"},
    ])
    assert evaluators._citation_marker_failures("PMID", output) == []


def test_a_marker_that_addresses_nothing_still_fails():
    """The fallback must stay strict: a number no turn published is unresolvable."""
    output = _thread([{"cite": 1, "pmid": "24108298"}],
                     second_answer="See ^[1] and ^[9].")
    failures = evaluators._citation_marker_failures("PMID", output)
    assert failures and "turn 2" in failures[0]
