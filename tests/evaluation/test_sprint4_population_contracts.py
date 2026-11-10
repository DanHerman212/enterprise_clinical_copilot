"""Sprint 4: the Sprint 2 contracts hold across the whole served cohort.

Two layers, on purpose.

The offline tests read the committed provenance inventory, so they cover all 87
admissions in the default suite at no cost and fail loudly if a regeneration ever
produces a record that breaks a contract.

The live test re-derives the same facts from the tools themselves, because the
inventory is a copy: it can only prove what a previous run published. That test is
marked live and runs with the release gate, not with ``-m "not live"``.

Neither layer calls a language model.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from services.agent.ingress.contracts import SERVED_HADM_IDS

REPO = Path(__file__).resolve().parents[2]
INVENTORY = json.loads(
    (REPO / "evaluation" / "cohort_provenance_inventory.json").read_text(encoding="utf-8"))
RECORDS = {int(key): value for key, value in INVENTORY["all_records"].items()}
OUTPATIENT = frozenset(hadm for hadm, record in RECORDS.items() if record["outpatient"])
INPATIENT = frozenset(RECORDS) - OUTPATIENT

STAY_FIELDS = ("index_los_days", "discharge_location")


def _dataset(name: str) -> list[dict]:
    path = REPO / "evaluation" / "datasets" / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


# --- the boundary the sweep is about -----------------------------------------


def test_the_inventory_covers_every_served_admission():
    assert set(RECORDS) == set(SERVED_HADM_IDS)
    assert len(RECORDS) == 87


def test_the_two_suites_are_disjoint_and_inside_the_boundary():
    dev = {int(item["hadm_id"]) for item in _dataset("dev_suite.jsonl")}
    holdout = {int(item["hadm_id"]) for item in _dataset("holdout_suite.jsonl")}

    assert not dev & holdout, "a Dev admission in the blind suite is not blind"
    assert dev <= set(RECORDS) and holdout <= set(RECORDS)


# --- outpatient stay fields ---------------------------------------------------


def test_every_outpatient_record_omits_the_two_stay_fields():
    """A length of stay and a discharge destination describe a stay, and a visit is not one.

    The row carries both anyway, because the encoder built it as an admission. The
    profile is where that stops being visible to anything downstream.
    """
    assert len(OUTPATIENT) == 5, "the outpatient boundary changed; re-check the cohort"

    for hadm in sorted(OUTPATIENT):
        profile = RECORDS[hadm]["profile"]
        for field in STAY_FIELDS:
            assert field not in profile, f"{hadm} still publishes {field}"
        assert profile.get("encounter_type", "").startswith("outpatient")


def test_every_inpatient_record_keeps_them():
    assert len(INPATIENT) == 82

    for hadm in sorted(INPATIENT):
        profile = RECORDS[hadm]["profile"]
        for field in STAY_FIELDS:
            assert field in profile, f"{hadm} lost {field}, which it does have"


# --- medication provenance ----------------------------------------------------


def test_every_medication_result_states_whether_it_is_a_dedicated_list():
    """One flag per admission, always present, never inferred by the reader."""
    for hadm, record in sorted(RECORDS.items()):
        medications = record["medications"]
        assert isinstance(medications["dedicated_section_present"], bool), hadm


def test_a_record_with_no_medication_text_is_not_a_dedicated_list():
    for hadm, record in sorted(RECORDS.items()):
        medications = record["medications"]
        if not medications["section_present"]:
            assert medications["dedicated_section_present"] is False, hadm


# --- null risk drivers --------------------------------------------------------


def test_every_driver_entry_states_a_value_slot():
    """The key is always present; its value may be null, and that null is the fact.

    An omitted key and an explicit null mean different things — an aggregated parent
    has no single value, a scored feature has a recorded one that is missing — so the
    artifact keeps the slot on every entry and lets the value carry the distinction.
    """
    for hadm, record in sorted(RECORDS.items()):
        for driver in record["risk"]["drivers"]:
            assert "value" in driver, f"{hadm}/{driver.get('name')} dropped its value slot"
            assert isinstance(driver["direction"], str)


def test_the_cohort_still_contains_scored_features_with_no_value():
    """If this ever drops to zero, the null-driver case has gone from the corpus.

    It was 29 entries across 19 admissions when the contract was written. The number is
    not pinned — a feature store can fill a column — but the case must exist, because a
    test for it that no record exercises proves nothing.
    """
    nulls = [(hadm, driver["name"])
             for hadm, record in sorted(RECORDS.items())
             for driver in record["risk"]["drivers"] if driver["value"] is None]

    assert nulls, "no scored driver is missing its value; the null case is untested"


# --- the same contracts, re-derived from the tools ----------------------------


@pytest.mark.live
def test_the_prediction_tool_publishes_an_explicit_null_for_a_missing_value():
    """The source of truth behind the offline assertions above.

    Three states, and the tool has to keep them apart: a value, an explicit null for a
    column the row left empty, and no key at all for an attribution that names an
    aggregated parent. Reading the row is what makes the distinction testable — from the
    payload alone a null and an absent key look equally like "nothing to say".
    """
    from evaluation.inventory.cohort_provenance import _feature_rows
    from services.mcp.tools import predict_readmission

    served = tuple(sorted(SERVED_HADM_IDS))
    rows = _feature_rows(served)

    async def gather():
        gate = asyncio.Semaphore(6)

        async def one(hadm):
            async with gate:
                return hadm, await predict_readmission(hadm)

        return await asyncio.gather(*(one(hadm) for hadm in served))

    checked = 0
    for hadm, payload in asyncio.run(gather()):
        assert "error" not in payload, f"{hadm}: {payload.get('error')}"
        row = rows[hadm]
        for factor in payload["top_factors"]:
            feature = factor["feature"]
            if feature not in row:
                assert "value" not in factor, (
                    f"{hadm}/{feature} is an aggregated parent and has no single value")
                continue
            raw = row.get(feature)
            if raw is None:
                assert "value" in factor, (
                    f"{hadm}/{feature} has a NULL column and must say so")
                assert factor["value"] is None, (
                    f"{hadm}/{feature} was filled with {factor['value']!r}, which is not "
                    "a value the model scored")
            else:
                expected = round(float(raw), 2)
                expected = int(expected) if expected == int(expected) else expected
                assert factor["value"] == expected, f"{hadm}/{feature}"
            checked += 1

    assert checked, "no driver was examined"
