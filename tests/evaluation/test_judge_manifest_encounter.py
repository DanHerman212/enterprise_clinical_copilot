"""The judge's manifest must describe the encounter, not the model's input vector.

Canonical run 6 (`phase04-gate1-canary-20261027-184826`) failed `dev_t7_01` for the
opposite reason it had failed the run before. The product had stopped stating a length of
stay for an outpatient encounter, and the judge — handed the decoded row, which still
carried `index_los_days: "7.91667"` and an emergency `admission_type` — scored the correct
denial as a fabrication:

    "The model's answer incorrectly states that the encounter has 'no admission type and
     no length of stay.' This is a direct contradiction of the available evidence."

The "available evidence" was the answer key. These tests pin the rule that closes it: a
manifest whose record is an outpatient visit carries none of the three stay-shaped fields,
states why, and stays silent about them in the prompt the judge is actually given.

The predicate and the field list are imported from the agent rather than restated. Two
copies of "which fields only exist because there was a stay" is how the product and the
harness came to disagree in the first place.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
for _path in (_REPO, _REPO / "evaluation" / "agent"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import build_census  # noqa: E402
from services.agent.routing.admission_context import STAY_ONLY_FIELDS  # noqa: E402

#: One served row, shaped the way BigQuery returns it: every column a string, the
#: stay-shaped ones populated because the row exists to be scored.
ROW = {
    "encounter_type": "outpatient",
    "age": "67",
    "gender": "1",
    "admission_type_direct_emer": "1",
    "discharge_location_home": "1",
    "insurance_medicare": "1",
    "race_white": "1",
    "prior_admission_count": "2",
    "prior_inpatient_days": "13.4167",
    "recent_ed_visits": "0",
    "index_los_days": "7.91667",
}

#: In `evaluation.outpatient_oracle`, so `encounter_statement` says "outpatient" for it.
OUTPATIENT_HADM = 90000051


def _manifest(**overrides: object) -> dict:
    return build_census.judge_manifest(OUTPATIENT_HADM, {**ROW, **overrides})


def test_the_field_list_is_the_products_own():
    """One definition. `admission_context.STAY_ONLY_FIELDS` is what the agent drops."""
    assert STAY_ONLY_FIELDS == frozenset(
        {"admission_type", "discharge_location", "index_los_days"}
    )


def test_an_outpatient_manifest_states_none_of_the_stay_fields():
    manifest = _manifest()

    for field in ("admission_type", "discharge_location", "index_los_days"):
        assert field not in manifest, field


def test_an_outpatient_manifest_says_why_they_are_absent():
    """Omission alone invites the judge to read a gap as missing data."""
    manifest = _manifest()

    assert "not_applicable" in manifest
    note = manifest["not_applicable"].lower()
    assert "outpatient" in note
    assert "no length of stay" in note
    assert "an answer that says so is correct" in note


def test_the_encounter_statement_still_governs():
    manifest = _manifest()

    assert manifest["encounter"] == build_census.OUTPATIENT_ENCOUNTER
    assert "admission_type_source" in manifest


def test_an_inpatient_manifest_keeps_every_field_it_had():
    manifest = _manifest(encounter_type="inpatient")

    assert manifest["index_los_days"] == "7.91667"
    assert manifest["admission_type"] == "Emergency, direct admission (DIRECT EMER.)"
    assert manifest["discharge_location"] == "Home"
    assert "not_applicable" not in manifest


def test_an_unstated_encounter_keeps_the_fields_it_had():
    """A row without the column must not lose evidence on the strength of a gap: that is
    the behaviour every record had before the column existed."""
    row = {k: v for k, v in ROW.items() if k != "encounter_type"}
    manifest = build_census.judge_manifest(OUTPATIENT_HADM, row)

    assert manifest["index_los_days"] == "7.91667"
    assert "not_applicable" not in manifest


def test_the_prompt_the_judge_reads_carries_no_stay_shaped_value():
    """The end-to-end assertion: not "the key is absent" but "the number is not in the
    text the judge sees". That is the sentence run 6 quoted back as evidence."""
    from evaluation.platform.judge_call import build_user_prompt

    prompt = build_user_prompt(
        {"patient_manifest": _manifest(), "turns": ["/ehr"]},
        {"turn_1_response": "This is an outpatient visit with no length of stay.",
         "tool_evidence": []},
    )

    assert "7.91667" not in prompt
    assert "Emergency, direct admission" not in prompt
    assert "outpatient" in prompt.lower()
    assert "not_applicable" in prompt


@pytest.mark.parametrize("hadm_id", sorted(
    {90000015, 90000023, 90000106}))
def test_an_inpatient_prompt_still_carries_its_length_of_stay(hadm_id):
    """The control: the rule is about visits, not about the manifest being thin."""
    from evaluation.platform.judge_call import build_user_prompt

    prompt = build_user_prompt(
        {"patient_manifest": build_census.judge_manifest(hadm_id,
                                                        {**ROW, "encounter_type":
                                                         "inpatient"}),
         "turns": ["/ehr"]},
        {"turn_1_response": "A summary.", "tool_evidence": []},
    )

    assert "7.91667" in prompt
