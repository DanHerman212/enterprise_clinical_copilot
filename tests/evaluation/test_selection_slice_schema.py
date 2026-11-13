"""The v2 selection slice: a 3 x 4 behavioural matrix, and the contracts it must keep.

Every assertion here is about the file rather than about a model. The matrix counts are
the directive's, so a row added or dropped without re-deriving them fails here; the
entity rule exists because a dosage string in `must_contain` turns a correct paraphrase
into a structural failure, which is how `ecc-model-selection-slice-v1` came to fail an
item the product had answered properly.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from evaluation import outpatient_oracle
from evaluation.platform import selection_slice, sync_selection_slice
from evaluation.platform.schemas import DatasetItemInput

REPO = Path(__file__).resolve().parents[2]
DEV = REPO / "evaluation" / "datasets" / "dev_suite.jsonl"
HOLDOUT = REPO / "evaluation" / "datasets" / "holdout_suite.jsonl"

ROWS = selection_slice.read_rows()


def _hadm_ids(path: Path) -> set[int]:
    return {int(json.loads(line)["hadm_id"])
            for line in path.read_text().splitlines() if line.strip()}


def _facts(row: dict) -> dict:
    return row["expected_output"]["facts"]


def test_the_slice_is_sixteen_scenarios_over_thirty_two_turns():
    assert len(ROWS) == 16
    assert sum(len(row["turns"]) for row in ROWS) == 32
    selection_slice.validate_rows(ROWS)


def test_every_cell_of_the_matrix_holds_the_count_the_directive_fixes():
    for (depth, axis), wanted in selection_slice.MATRIX.items():
        held = [row for row in ROWS
                if row["metadata"]["depth_tier"] == depth
                and row["metadata"]["modality_axis"] == axis]
        assert len(held) == wanted, f"{depth}/{axis}: {len(held)} != {wanted}"


def test_the_depth_tier_matches_the_number_of_turns():
    for row in ROWS:
        tier = row["metadata"]["depth_tier"]
        assert len(row["turns"]) == selection_slice.DEPTH_TIERS[tier], row["id"]


def test_no_row_names_a_command_the_product_does_not_have():
    allowed = selection_slice.COMMAND_WIRES
    for row in ROWS:
        command = row.get("command")
        if command is not None:
            assert command in allowed, row["id"]
        for turn in row["turns"]:
            text = turn if isinstance(turn, str) else str(turn.get("prompt") or "")
            word = text.split()[0] if text.split() else ""
            if word.startswith("/"):
                assert word.lstrip("/").lower() in allowed, row["id"]


def test_the_retired_risk_command_appears_nowhere():
    assert "/risk" not in json.dumps(ROWS)


def test_no_record_is_borrowed_from_another_suite():
    used = {int(row["hadm_id"]) for row in ROWS}
    assert not (used & _hadm_ids(DEV))
    assert not (used & _hadm_ids(HOLDOUT))
    assert not (used & selection_slice.SLICE_V1_HADM_IDS)
    assert not (used & selection_slice.RESERVED_HADM_IDS)


def test_the_matrix_carries_both_inpatient_and_oracle_confirmed_outpatient_records():
    used = {int(row["hadm_id"]) for row in ROWS}
    outpatient = used & set(outpatient_oracle.OUTPATIENT_HADM_IDS)
    assert outpatient, "the matrix needs an outpatient record"
    assert used - outpatient, "the matrix needs inpatient records too"


def test_the_outpatient_item_is_the_missing_section_trap():
    item = next(row for row in ROWS if int(row["hadm_id"]) == 90000040)
    assert item["metadata"]["modality_axis"] == "adversarial_perturbation"
    assert _facts(item)["discharge_medications_section_present"] is False
    assert _facts(item)["encounter"] == "outpatient"


def test_every_row_satisfies_the_harness_schema():
    for row in ROWS:
        DatasetItemInput.model_validate({
            **{key: row[key] for key in ("turns", "workflow_type")},
            "patient_id": str(row["hadm_id"]),
            "patient_manifest": {"hadm_id": str(row["hadm_id"]), "age": 0},
            "command": row.get("command"),
            "scenario_id": row["id"],
            "expected_output": row["expected_output"],
            "metadata": row["metadata"],
        })


def test_every_row_declares_rubric_v8():
    assert {row["metadata"]["rubric_version"] for row in ROWS} == {"v8"}


def test_the_workflow_type_follows_the_turn_count():
    for row in ROWS:
        assert row["workflow_type"] == selection_slice.workflow_for(len(row["turns"]))


def test_must_contain_holds_entity_names_and_never_a_dosage():
    for row in ROWS:
        for token in row["expected_output"].get("must_contain") or []:
            assert not re.search(r"\d", str(token)), f"{row['id']}: {token!r}"
            assert len(str(token).split()) <= 3, f"{row['id']}: {token!r}"


def test_an_overview_command_item_does_not_demand_a_drill_down_in_prose():
    """The v1 defect: a `/meds` item requiring the taper narrative in the answer.

    The table carries dose and frequency; a taper step belongs to the follow-up chip. So
    a single-turn `/meds` or `/labs` item may not carry a taper fact, and its behaviour
    may not require the prose to restate cells the grid already renders.
    """
    for row in ROWS:
        if len(row["turns"]) != 1 or row.get("command") not in {"meds", "labs"}:
            continue
        behaviour = row["expected_output"]["behaviour"].lower()
        assert "taper" not in behaviour, row["id"]
        assert not re.search(r"\b\d+\s?mg\b", behaviour), row["id"]


@pytest.mark.parametrize("row", ROWS, ids=[row["id"] for row in ROWS])
def test_every_row_states_facts_and_a_behaviour(row):
    expected = row["expected_output"]
    assert expected["facts"], row["id"]
    assert len(expected["behaviour"].split()) >= 25, row["id"]
    assert isinstance(expected.get("forbidden_claims", []), list)


def test_the_sync_target_is_v2_and_the_retired_dataset_is_not_written_to():
    assert sync_selection_slice.DATASET == "ecc-model-selection-slice-v2"
    source = (REPO / "evaluation" / "platform" / "sync_selection_slice.py").read_text()
    assert 'DATASET = "ecc-model-selection-slice-v1"' not in source
