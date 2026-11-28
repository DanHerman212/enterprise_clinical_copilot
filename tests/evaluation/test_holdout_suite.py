"""The holdout suite is a holdout: 30 rows, 10 records, zero dev overlap.

Phase 05's whole claim is that the suite measures generalisation to records the dev suite
never named. That claim is enforced before the file is uploaded rather than after a run
has scored it.

Nothing here runs a model or talks to BigQuery.
"""

from __future__ import annotations

import copy
import json

import pytest

from evaluation.platform import dataset_audit
from evaluation.platform import sync_holdout_suite as holdout
from evaluation.platform.dev_suite import DEV_HADM_IDS, HOLDOUT_HADM_IDS


@pytest.fixture(scope="module")
def rows():
    return holdout.read_rows()


def test_the_suite_is_thirty_scenarios_across_ninety_turns(rows):
    assert len(rows) == holdout.EXPECTED_ITEMS == 30
    assert sum(len(row["turns"]) for row in rows) == holdout.EXPECTED_TURNS == 90
    assert holdout.ITEM_ID_PREFIX == "h2_"
    assert holdout.DATASET == "ecc-tier2-holdout-suite_v2"


def test_every_trajectory_is_represented(rows):
    """T1–T7 are all covered across the 5x5 Holdout matrix."""
    counts = {
        name: sum(1 for row in rows if row["trajectory"] == name)
        for name in (
            "t1_declarative_command",
            "t2_dynamic_affordance_loopback",
            "t3_analytical_drilldown",
            "t4_cross_domain_synthesis",
            "t5_freetext_cold_start",
            "t6_adversarial_false_premise",
            "t7_adversarial_boundaries",
        )
    }
    assert all(count >= 1 for count in counts.values()), counts
    assert sum(counts.values()) == 30


def test_no_record_or_raw_text_mentions_a_dev_record(rows):
    """The comparison is dev against holdout, so the two may not share a patient."""
    records = {row["hadm_id"] for row in rows}
    raw = holdout.HOLDOUT_SUITE_PATH.read_text(encoding="utf-8")

    assert len(records) == 10
    assert records == set(HOLDOUT_HADM_IDS)
    assert not (records & set(DEV_HADM_IDS))
    for dev_hadm in DEV_HADM_IDS:
        assert dev_hadm not in raw, f"dev hadm {dev_hadm} appears in holdout_suite.jsonl"


def test_the_file_validates_and_passes_matrix_and_artefact_audit(rows):
    holdout.validate_rows(rows)
    assert holdout.validate_holdout_matrix(rows) == {}
    defects = dataset_audit.dev_deterministic_defects(
        rows,
        fact_sheet=dataset_audit.HOLDOUT_FACT_SHEET,
        matrix_doc=dataset_audit.HOLDOUT_MATRIX_DOC,
    )
    assert defects == {}, defects


def test_audit_guard_accepts_the_holdout_attestation_and_blocks_tampering(
    tmp_path, monkeypatch
):
    dataset_audit.audit_guard("holdout")

    dataset = tmp_path / "holdout_suite.jsonl"
    dataset.write_text("{}", encoding="utf-8")
    attestation = tmp_path / "holdout_suite.audit.json"
    attestation.write_text(
        json.dumps({"dataset_sha256": "0" * 64, "defects": {}, "items": {}}),
        encoding="utf-8",
    )
    monkeypatch.setitem(dataset_audit.AUDIT_PATHS, "holdout", attestation)
    monkeypatch.setitem(dataset_audit.AUDIT_SOURCES, "holdout", dataset)
    with pytest.raises(SystemExit, match="stale"):
        dataset_audit.audit_guard("holdout")


def test_a_dev_record_is_refused(rows):
    """Authored against a dev patient, the suite would not be a holdout at all."""
    poisoned = copy.deepcopy(rows)
    poisoned[0]["hadm_id"] = DEV_HADM_IDS[0]

    with pytest.raises(holdout.HoldoutSuiteError, match="is a dev record"):
        holdout.validate_rows(poisoned)


def test_an_unreserved_record_is_refused(rows):
    poisoned = copy.deepcopy(rows)
    poisoned[0]["hadm_id"] = "90000999"

    with pytest.raises(holdout.HoldoutSuiteError, match="is not a holdout record"):
        holdout.validate_rows(poisoned)


def test_a_short_suite_is_refused(rows):
    with pytest.raises(holdout.HoldoutSuiteError, match="expected 30"):
        holdout.validate_rows(rows[:-1])


def test_a_duplicate_scenario_id_is_refused(rows):
    duplicated = copy.deepcopy(rows)
    duplicated[1]["id"] = duplicated[0]["id"]

    with pytest.raises(holdout.HoldoutSuiteError, match="duplicate scenario id"):
        holdout.validate_rows(duplicated)


def test_the_dynamic_chip_rows_carry_the_placeholder(rows):
    """Loop-back chip turns resolve at run time, never as typed questions."""
    chip_rows = [
        row
        for row in rows
        if (row.get("metadata") or {}).get("modality_axis") == "followup_chip"
    ]
    assert len(chip_rows) == 5
    for row in chip_rows:
        assert any(
            "{{turn_1.follow_ups[" in str(turn) for turn in row["turns"]
        ), row["id"]
        assert (row.get("metadata") or {}).get("turn1_follow_ups") is True, row["id"]


def test_the_items_carry_what_the_harness_reads(rows):
    """Built items carry h2_-prefixed Langfuse IDs, manifests, and expectations."""
    manifests = {
        int(hadm): {"encounter_type": "inpatient admission"}
        for hadm in {row["hadm_id"] for row in rows}
    }

    payloads = holdout.build_items(manifests=manifests)

    assert len(payloads) == 30
    assert {payload["id"] for payload in payloads} == {
        f"h2_{row['id']}" for row in rows
    }
    for payload in payloads:
        item = payload["input"]
        assert item["patient_id"] in {row["hadm_id"] for row in rows}
        assert item["patient_manifest"], payload["id"]
        assert item["trajectory_index"] in range(1, 8)
        assert isinstance(payload["expected_output"], dict)
        assert payload["expected_output"].get("behaviour"), payload["id"]
