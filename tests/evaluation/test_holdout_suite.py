"""The holdout suite is a holdout: 15 rows, four records, zero dev overlap.

Phase 05's whole claim is that the suite measures generalisation to records the dev suite
never named. That claim is a property of one file, so it is checked before the file is
uploaded rather than after a run has scored it.

The two cohort rules are enforced by `sync_holdout_suite.validate_rows`, which refuses a
row for the opposite reason `dev_suite.validate_rows` refuses one: the dev validator
rejects a holdout record, and this one rejects a dev record. Nothing here runs a model.
"""

import copy

import pytest

from evaluation.platform import sync_holdout_suite as holdout
from evaluation.platform.dev_suite import DEV_HADM_IDS, HOLDOUT_HADM_IDS


@pytest.fixture(scope="module")
def rows():
    return holdout.read_rows()


def test_the_suite_is_fifteen_scenarios(rows):
    assert len(rows) == holdout.EXPECTED_ITEMS == 15


def test_every_trajectory_is_represented(rows):
    """T1–T7, in the counts the phase document specifies."""
    counts = {name: sum(1 for row in rows if row["trajectory"] == name)
              for name in ("t1_declarative_command", "t2_dynamic_affordance_loopback",
                           "t3_analytical_drilldown", "t4_cross_domain_synthesis",
                           "t5_freetext_cold_start", "t6_adversarial_false_premise",
                           "t7_adversarial_boundaries")}

    assert counts == {
        "t1_declarative_command": 3,
        "t2_dynamic_affordance_loopback": 2,
        "t3_analytical_drilldown": 3,
        "t4_cross_domain_synthesis": 2,
        "t5_freetext_cold_start": 2,
        "t6_adversarial_false_premise": 2,
        "t7_adversarial_boundaries": 1,
    }


def test_no_record_is_a_dev_record(rows):
    """The comparison is dev against holdout, so the two may not share a patient."""
    records = {row["hadm_id"] for row in rows}

    assert len(records) == 4
    assert records <= set(HOLDOUT_HADM_IDS)
    assert not (records & set(DEV_HADM_IDS))


def test_the_file_validates(rows):
    holdout.validate_rows(rows)


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
    with pytest.raises(holdout.HoldoutSuiteError, match="expected 15"):
        holdout.validate_rows(rows[:-1])


def test_a_duplicate_scenario_id_is_refused(rows):
    duplicated = copy.deepcopy(rows)
    duplicated[1]["id"] = duplicated[0]["id"]

    with pytest.raises(holdout.HoldoutSuiteError, match="duplicate scenario id"):
        holdout.validate_rows(duplicated)


def test_the_dynamic_chip_rows_carry_the_placeholder(rows):
    """T2's second turn is the chip the service produces, never a typed question."""
    for row in rows:
        if row["trajectory"] != "t2_dynamic_affordance_loopback":
            continue
        assert row["turns"][1] == "{{turn_1.follow_ups[0]}}", row["id"]
        assert (row.get("metadata") or {}).get("turn1_follow_ups") is True, row["id"]


def test_the_items_carry_what_the_harness_reads(rows):
    """A built item must carry the manifest and the deterministic expectations.

    Without `expectations` the deterministic half of the score is a silent no-op that
    reports ok — the failure mode `sync_dev_suite.py` documents.
    """
    manifests = {int(hadm): {"encounter_type": "inpatient admission"}
                 for hadm in {row["hadm_id"] for row in rows}}

    payloads = holdout.build_items(manifests=manifests)

    assert len(payloads) == 15
    assert {payload["id"] for payload in payloads} == {row["id"] for row in rows}
    for payload in payloads:
        item = payload["input"]
        assert item["patient_id"] in {row["hadm_id"] for row in rows}
        assert item["patient_manifest"], payload["id"]
        assert item["trajectory_index"] in range(1, 8)
        assert isinstance(payload["expected_output"], dict)
        assert payload["expected_output"].get("behaviour"), payload["id"]
