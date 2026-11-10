"""Sprint 2: the evidence contract states what it means, on all three wires.

`T1` — a feature the scoring row left NULL is published as an explicit null rather than
an omitted key, and is never cross-filled from the laboratory series.

`T2` — `discharge_medications` says whether its text came from a heading the note uses
for discharge medications or from the narrative fallback, per payload and per passage.

`T3` — an outpatient profile carries no length of stay and no discharge destination.

Zero LLM, zero BigQuery: every source is patched.
"""

from __future__ import annotations

from unittest.mock import patch

import services.mcp.tools.admission_profile as ap
import services.mcp.tools.prediction as pr
import services.mcp.tools.retrieval as rs
from services.agent.engine import prompts
from services.mcp.contracts import validate_retrieval_result

# --------------------------------------------------------------------- T1


def _predict(row: dict) -> dict:
    source = type("Source", (), {"fetch": staticmethod(lambda hadm_id: dict(row))})
    reply = {
        "probability": 0.34, "threshold": 0.11, "prediction": 1, "base_value": 0.05,
        "top_factors": [
            {"feature": "monocytes_min", "attribution": -0.0836},
            {"feature": "discharge_location", "attribution": 0.0896},
        ],
    }
    with patch.object(pr, "feature_order",
                      lambda: ["age", "rdw_max", "monocytes_min"]), \
         patch.object(pr, "_source", lambda: source), \
         patch.object(pr, "predict_one", lambda instance: (reply, "model-1")):
        return pr._predict(90000012)


def test_a_feature_the_row_left_null_is_published_as_null_not_omitted():
    """Absent and null are different facts and a consumer has to be able to tell them.

    Absent means the attribution names an aggregated parent with no column of its own.
    Null means the column exists and this admission's row holds nothing for it — the model
    scored a missing value. Dropping the key made the two identical, so an answer could
    only ever say "the chart does not document the value" without distinguishing a
    laboratory feature the record never carried from a disposition encoded in thirteen
    columns.
    """
    factors = {f["feature"]: f for f in _predict(
        {"age": "80", "rdw_max": "15.5", "monocytes_min": None})["top_factors"]}

    assert "value" in factors["monocytes_min"]
    assert factors["monocytes_min"]["value"] is None
    # The attribution is still stated exactly: the direction is what the model said.
    assert factors["monocytes_min"]["direction"] == "decreases"
    assert factors["monocytes_min"]["contribution"] == -0.0836


def test_the_null_is_not_filled_from_the_laboratory_series():
    """A series' last reading is not a maximum, so a number taken from one is not the input.

    The inventory measured 29 driver instances across 19 admissions where the row holds
    NULL for a feature the model names. `get_lab_trends` carries values for the same
    analytes at a different definition, which is exactly why it must not be consulted
    here: quoting one as the scored value invents what the model read.
    """
    factors = {f["feature"]: f for f in _predict(
        {"age": "80", "rdw_max": "15.5", "monocytes_min": None})["top_factors"]}

    assert factors["monocytes_min"]["value"] is None, (
        "a null scored value must stay null; any number here came from somewhere else")


def test_an_aggregated_parent_still_omits_the_key():
    factors = {f["feature"]: f for f in _predict(
        {"age": "80", "rdw_max": "15.5", "monocytes_min": 0.21})["top_factors"]}

    assert "value" not in factors["discharge_location"], (
        "a feature with no column of its own has no value to state, null or otherwise")


def test_a_present_value_is_unaffected():
    factors = {f["feature"]: f for f in _predict(
        {"age": "80", "rdw_max": "15.5", "monocytes_min": 0.21})["top_factors"]}

    assert factors["monocytes_min"]["value"] == 0.21


def test_the_prompt_states_what_a_null_driver_means():
    # Flattened: the prompt is hard-wrapped, so a sentence spans source lines.
    text = " ".join(prompts.RISK_SPOKE.split())

    assert "no recorded value in the structured admission feature set" in text
    assert "never substitute a `get_lab_trends` reading" in text
    assert "a series is a different measurement, not the model's input" in text


# --------------------------------------------------------------------- T2

_DEDICATED_NOTE = (
    "DISCHARGE DIAGNOSIS: Congestive heart failure.\n\n"
    "DISCHARGE MEDICATIONS: Furosemide 40 mg daily.\n\n"
    "DISCHARGE INSTRUCTIONS: Low salt diet. Follow up in two weeks."
)

_NARRATIVE_NOTE = (
    "DISCHARGE DIAGNOSIS: Right hip fracture.\n\n"
    "COURSE ON ADMISSION: Treated with intravenous antibiotics and taken to theatre.\n\n"
    "HISTORY OF PRESENT ILLNESS: She fell at home and could not weight bear."
)


def _medications(note: str) -> dict:
    with patch.object(rs, "_fetch_note_row", lambda hadm: ("MT-9-DS", note)):
        return rs._search_medication_sections(90000018)


def test_a_dedicated_list_is_declared_as_one():
    result = _medications(_DEDICATED_NOTE)

    assert result["dedicated_section_present"] is True
    assert result["passages"]
    assert all(p["medication_source"] == "dedicated" for p in result["passages"])


def test_the_narrative_fallback_declares_itself_a_fallback():
    """The consumer no longer has to infer provenance from a section name.

    A drug named in a course is one the stay administered; a drug named in a discharge
    list is one the patient was sent home on. The section NAME says which, but the name is
    the thing the fallback exists to change, so the payload states it directly.
    """
    result = _medications(_NARRATIVE_NOTE)

    assert result["dedicated_section_present"] is False
    assert result["passages"]
    assert all(p["medication_source"] == "narrative" for p in result["passages"])


def test_an_absent_medication_section_declares_no_dedicated_list():
    result = _medications("DISCHARGE DIAGNOSIS: Nothing else is documented.\n")

    assert result["returned"] == 0
    assert result["dedicated_section_present"] is False


def test_the_provenance_fields_pass_the_retrieval_contract():
    validate_retrieval_result(_medications(_DEDICATED_NOTE))
    validate_retrieval_result(_medications(_NARRATIVE_NOTE))


def test_the_medications_spoke_verifies_a_transfer_list_against_both_sections():
    """"On transfer" is what the patient arrived on, and one section cannot settle it.

    The continuations are in `discharge_instructions` and the discontinuations are in
    `brief_hospital_course`; a rule that checks one and not the other reports a bridging
    drug as a take-home prescription, which is the error `dev_t3_01` and `dev_t5_01` are
    built to catch.
    """
    text = " ".join(prompts.MEDS_SPOKE.split())

    assert 'a medication list that opens "on transfer" is the list the patient arrived on' \
        in text.lower()
    assert "verify it against BOTH sections" in text
    assert "`discharge_instructions` for a continuation at home" in text
    assert "`brief_hospital_course` for a stop" in text
    assert "NOT continued" in text


# --------------------------------------------------------------------- T3

_OUTPATIENT_PROFILE = {
    "encounter_type": "outpatient encounter (no inpatient stay)",
    "age": "67",
    "gender": "Female",
    "index_los_days": "24.7917",
    "discharge_location": "Long-term acute care",
    "prior_admission_count": "28",
}

_INPATIENT_PROFILE = {
    "encounter_type": "inpatient admission",
    "age": "73",
    "index_los_days": "5.875",
    "discharge_location": "Home with home health",
}


def _profile(fields: dict) -> dict:
    source = type("Source", (), {"fetch_profile": staticmethod(lambda hadm: dict(fields))})
    with patch.object(ap, "_source", lambda: source):
        return ap._profile(90000028)


def test_an_outpatient_profile_carries_no_stay_fields():
    """The row has both columns and neither describes a visit.

    `index_los_days` is 24.79 on this outpatient row because the encoder built the row as
    though it were an admission. Stating it lets any consumer repeat a length of stay the
    encounter did not have, which is what the canary did.
    """
    result = _profile(_OUTPATIENT_PROFILE)

    assert "index_los_days" not in result["profile"]
    assert "discharge_location" not in result["profile"]
    assert result["profile"]["age"] == "67"
    assert result["profile"]["encounter_type"].startswith("outpatient")


def test_an_inpatient_profile_keeps_them():
    result = _profile(_INPATIENT_PROFILE)

    assert result["profile"]["index_los_days"] == "5.875"
    assert result["profile"]["discharge_location"] == "Home with home health"
