"""Problem 2, 2026-11-20: the clinical-output repairs, one test per repair.

The material came from a warm smoke of three admissions whose answers read like a debug
dump in places: "oncology history (contribution 0.3246; value 1)", "length of stay was
42.4583 days" three lines above "the recorded value was 42.46", a mandatory sentence at the
end of every driver answer saying the notes do not explain the mechanism, a provenance
paragraph opening every `/labs` panel, a two-arm benchmark described as one "span", and two
frequency flags raised against a note that writes `q.12h` and `b.i.d.`.

Six repairs, and each has its own section below. Two of those sections carry a second test
for a surface the first pass missed: the `/labs` free-text rules still asked for the
provenance sentence in the imperative, and the binary-encoding rule only recognised names
ending in `_flag`, so `has_procedure` was still served as the number `1`.

What is NOT tested here is the widget payload: `state["tool_calls"]` keeps the raw probability, threshold and contribution floats
for the RiskInfluencers bars, and those assertions live where they always did.
"""

from __future__ import annotations

import pytest

from services.agent.engine import deterministic
from services.agent.engine import prompts
from services.agent.engine.mcp_client import _presented_factor, _presented_for_model
from services.agent.postprocess.evidence_state import SessionEvidenceState
from services.agent.postprocess.guardrail import (
    normalise_factor_values,
    verify_med_tokens,
)
from services.agent.routing import admission_context


# ---------------------------------------------------------------------------------------
# 1. The model-facing risk projection carries no SHAP weight and no encoding
# ---------------------------------------------------------------------------------------


def test_a_driver_reaches_the_model_as_a_name_a_direction_and_a_value():
    presented = _presented_factor({
        "feature": "rdw_max", "direction": "increases",
        "contribution": 0.2592, "value": 15.5})

    assert presented == {
        "feature": "red cell distribution width (RDW)",
        "direction": "increases",
        "value": 15.5,
    }


def test_a_null_value_is_absent_rather_than_stated_as_null():
    """The sentence "the value is null" cannot be written about a key that is not there."""
    presented = _presented_factor({
        "feature": "admission_type", "direction": "increases",
        "contribution": 0.1121, "value": None})

    assert "value" not in presented
    assert "null" not in str(presented)


def test_a_database_flag_reaches_the_model_as_words():
    """`oncology_flag` is 1 because the row is one column of an encoding."""
    present = _presented_factor(
        {"feature": "oncology_flag", "direction": "increases", "value": 1})
    absent = _presented_factor(
        {"feature": "oncology_flag", "direction": "decreases", "value": 0})

    assert present["value"] == "present"
    assert absent["value"] == "not present"


def test_a_predicate_column_without_a_flag_suffix_reaches_the_model_as_words():
    """`has_procedure` is built as `1 if has_procedure else 0`; its name does not say so."""
    performed = _presented_factor(
        {"feature": "has_procedure", "direction": "increases", "value": 1})
    none_done = _presented_factor(
        {"feature": "has_procedure", "direction": "decreases", "value": 0})

    assert performed["feature"] == "procedure performed"
    assert performed["value"] == "present"
    assert none_done["value"] == "not present"


def test_a_procedure_count_is_still_a_number():
    """The encoding rule must not reach the count that lives beside it in the same row."""
    presented = _presented_factor(
        {"feature": "procedure_count", "direction": "increases", "value": 4})

    assert presented["value"] == 4


def test_the_recorded_call_still_carries_every_served_float():
    """The widget draws these bars; the projection is what the model reads, not the record."""
    served = {
        "probability": 0.332758, "threshold": 0.11, "base_value": 0.3,
        "top_factors": [{"feature": "rdw_max", "contribution": 0.316788,
                         "direction": "increases", "value": 15.5}],
    }

    projected = _presented_for_model("predict_readmission", served)

    assert served["top_factors"][0]["contribution"] == 0.316788
    assert "contribution" not in str(projected["top_factors"])


# ---------------------------------------------------------------------------------------
# 2. One duration precision across the manifest, the projection and the guardrail
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("row_value, stated", [
    ("42.458333333333336", "42.46"),
    (42.458333333333336, "42.46"),
    ("24.6667", "24.67"),
    ("16.0417", "16.04"),
    ("4.75", "4.75"),
    ("3", "3"),
    ("3.00", "3"),
])
def test_the_manifest_states_a_value_at_the_published_precision(row_value, stated):
    """Two decimals, which is what the risk result publishes for the same row."""
    block = admission_context.render(90000069, {"index_los_days": row_value})

    assert f"length_of_stay_days: {stated}" in block


def test_a_non_numeric_column_is_left_exactly_as_it_arrived():
    block = admission_context.render(90000069, {"admission_type": "Emergency (EW EMER.)",
                                                "insurance": "Not recorded"})

    assert "admission_type: Emergency (EW EMER.)" in block
    assert "insurance: Not recorded" in block


def test_the_guardrail_rewrites_to_the_same_precision_the_record_states():
    calls = [{
        "name": "predict_readmission", "args": {"hadm_id": 90000069},
        "response": {"probability": 0.075, "threshold": 0.11,
                     "top_factors": [{"feature": "index_los_days",
                                      "direction": "increases",
                                      "value": 42.458333333333336}]},
    }]

    out, flags = normalise_factor_values("Her length of stay was 42.4583 days.", calls)

    assert "42.46 days" in out
    assert "42.4583" not in out
    assert flags == ["factor_value_rounded:42.458333333333336"]


# ---------------------------------------------------------------------------------------
# 3. The risk prose rules ask for a direction, not for a weight, and no disclaimer
# ---------------------------------------------------------------------------------------


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_the_risk_spoke_no_longer_asks_for_a_contribution():
    spoke = _flat(prompts.RISK_SPOKE)

    assert "contribution" not in spoke
    assert "are not yours to state" in spoke
    # The direction is still required, but once per group rather than once per driver:
    # the answer that stated it under all five bullets repeated the same clause five times
    # (measured 2026-11-20, 90000020).
    assert "GROUPS its attribution and states it ONCE" in spoke
    assert "never repeat" in spoke


def test_the_ehr_risk_section_no_longer_asks_for_a_contribution():
    formatting = _flat(prompts.EHR_FORMATTING)

    assert "its `contribution`" not in formatting
    assert "no \"contribution\"" in formatting


def test_no_prompt_asks_for_the_mandatory_no_mechanism_sentence():
    """True of every record, so it told every reader the same nothing."""
    for name in ("CORE_HUB_PROMPT", "RISK_SPOKE", "EHR_FORMATTING"):
        text = _flat(getattr(prompts, name))
        assert "the notes do not account for the mechanism" not in text, name
        assert "do not account for it" not in text, name
    # The prohibition it belonged to survives.
    assert "NEVER SUPPLY A MECHANISM" in _flat(prompts.CORE_HUB_PROMPT)


# ---------------------------------------------------------------------------------------
# 4. The /labs provenance sentence belongs to the card, not to every answer
# ---------------------------------------------------------------------------------------


def test_the_labs_formatter_is_told_not_to_write_the_provenance_sentence():
    formatting = _flat(prompts.LABS_FORMATTING)

    assert "the card beneath this answer states that" in formatting
    assert "Write no sentence about it" in formatting
    assert "Say so once, plainly" not in formatting


def test_the_lab_spoke_leaves_the_provenance_to_the_card_too():
    """The free-text path draws the same card, so it gets the same instruction."""
    spoke = _flat(prompts.LABS_SPOKE)

    assert "say the trajectory is illustrative" not in spoke
    assert "write no sentence about where it came from" in spoke


# ---------------------------------------------------------------------------------------
# 5. A two-arm benchmark is two rates, not one span
# ---------------------------------------------------------------------------------------

_ROW = {"pmid": "40000000", "journal": "J Test Med", "year": "2024",
        "cohort_summary": "a cohort", "evidence_tier": "retrospective"}


def test_an_estimate_above_both_arms_says_so():
    block = deterministic.comparator_block(
        [dict(_ROW, reported_rate="4.6% vs. 4.2%", rate_kind="comparison")],
        patient_estimate=0.085)

    assert "relation: patient estimate (8.5%) is above both published rates " \
           "(4.6% and 4.2%)" in block


def test_an_estimate_between_the_arms_keeps_both_rates():
    block = deterministic.comparator_block(
        [dict(_ROW, reported_rate="2.6% vs. 12.0%", rate_kind="comparison")],
        patient_estimate=0.09)

    assert "sits between the two published rates (2.6% and 12.0%)" in block
    assert "span" not in block


def test_an_estimate_below_both_arms_says_so():
    block = deterministic.comparator_block(
        [dict(_ROW, reported_rate="9.0% vs. 7.5%", rate_kind="comparison")],
        patient_estimate=0.02)

    assert "is below both published rates (9.0% and 7.5%)" in block


def test_a_real_range_keeps_its_range_wording():
    block = deterministic.comparator_block(
        [dict(_ROW, reported_rate="1.0% to 1.4%", rate_kind="range")],
        patient_estimate=0.09)

    assert "is above the published range (1.0% to 1.4%)" in block


# ---------------------------------------------------------------------------------------
# 6. The frequency check reads the note's own shorthand
# ---------------------------------------------------------------------------------------

#: A discharge note as this cohort writes one: dotted and abbreviated throughout.
_DOTTED_NOTE = (
    "DISCHARGE MEDICATIONS: oxycodone SR 20 mg p.o. q.12h x5 days then 10 mg q.12h; "
    "Ativan 0.5 mg p.o. b.i.d.; trazodone 50 mg q.h.s.; Percocet 5/325 mg q.6h p.r.n."
)


def test_a_note_written_in_dotted_shorthand_matches_the_answer_written_in_words():
    answer = ("She leaves on oxycodone SR 20 mg every 12 hours, Ativan 0.5 mg twice "
              "daily, trazodone 50 mg at bedtime, and Percocet 5/325 mg every six "
              "hours as needed.")

    _, flags = verify_med_tokens(answer, _DOTTED_NOTE)

    assert flags == []


def test_a_genuinely_wrong_frequency_is_still_flagged():
    """The repair must not turn the check off: a schedule the note does not state stays out."""
    _, flags = verify_med_tokens("She leaves on Ativan 0.5 mg three times daily.",
                                 _DOTTED_NOTE)

    assert flags == ["med_freq_mismatch:TID"]


def test_a_decimal_dose_is_not_read_as_a_dotted_abbreviation():
    """`2.5 mg` is a dose; only a dot inside a word or before a digit-in-a-word goes."""
    note = _DOTTED_NOTE + " oxycodone 2.5 mg q.12h."

    _, flags = verify_med_tokens(
        "She leaves on oxycodone 2.5 mg every 12 hours.", note)

    assert flags == []


# ---------------------------------------------------------------------------------------
# 7. A count is a whole number
# ---------------------------------------------------------------------------------------


def test_a_count_reaches_the_model_as_a_whole_number():
    """`medication_order_count` is served as 4.85, and "4.85 orders" is not a count.

    The row holds an order rate under a name that promises a tally (measured 2026-11-20,
    90000020). The fraction is not information the reader is owed, so the projection states
    the number a clinician would write.
    """
    presented = _presented_factor({
        "feature": "medication_order_count", "direction": "decreases", "value": 4.85})

    assert presented["feature"] == "medication order count"
    assert presented["value"] == 5


def test_a_day_count_rounds_half_up():
    """`prior_inpatient_days` 4.75 is 5, and 4.5 is 5 rather than Python's even 4."""
    assert admission_context.clinical_value("prior_inpatient_days", 4.75) == 5
    assert admission_context.clinical_value("prior_inpatient_days", 4.5) == 5


def test_a_duration_is_a_measurement_and_keeps_two_decimals():
    """Length of stay is not a count: `42.46` is the number the risk result publishes."""
    presented = _presented_factor({
        "feature": "index_los_days", "direction": "decreases",
        "value": 42.458333333333336})

    assert presented["value"] == 42.46


def test_a_lab_value_keeps_its_own_precision():
    presented = _presented_factor(
        {"feature": "rdw_max", "direction": "increases", "value": 15.65})

    assert presented["value"] == 15.65


# ---------------------------------------------------------------------------------------
# 8. Every layer that states the count states the same number
# ---------------------------------------------------------------------------------------


def test_the_manifest_states_the_count_whole_and_the_duration_at_two_decimals():
    block = admission_context.render(90000020, {
        "prior_inpatient_days": "4.75",
        "index_los_days": "42.458333333333336",
    })

    assert "prior_inpatient_days: 5" in block
    assert "length_of_stay_days: 42.46" in block


def test_the_session_record_states_the_count_whole_too():
    """The record is read back on a later turn, so it must not restore the fraction."""
    state = SessionEvidenceState(hadm_id=90000020)
    state.ingest_tool_output("predict_readmission", {"hadm_id": 90000020}, {
        "probability_display": "8.5%", "threshold_display": "11.0%",
        "top_factors": [{"feature": "medication_order_count",
                         "direction": "decreases", "value": 4.85}],
    })

    rendered = state.render_for_prompt()

    assert "value=5" in rendered
    assert "4.85" not in rendered


def test_the_guardrail_rewrites_a_served_count_to_its_whole_number():
    """The number the model was shown is the number the guardrail now corrects toward."""
    calls = [{
        "name": "predict_readmission", "args": {"hadm_id": 90000020},
        "response": {"probability": 0.085, "threshold": 0.11,
                     "top_factors": [{"feature": "medication_order_count",
                                      "direction": "decreases", "value": 4.85}]},
    }]

    out, flags = normalise_factor_values("Her 4.85 medication orders held it down.",
                                         calls)

    assert "5 medication orders" in out
    assert "4.85" not in out
    assert flags == ["factor_value_rounded:4.85"]


# ---------------------------------------------------------------------------------------
# 9. The polish fits the prompt budgets, and the absent value is only a prohibition
# ---------------------------------------------------------------------------------------

#: The caps the prompt contract enforces, asserted beside the change that spends them.
_WORD_CAPS = {
    "CORE_HUB_PROMPT": 1500,
    "CHART_SPOKE": 650,
    "MEDS_SPOKE": 650,
    "LABS_SPOKE": 650,
    "RISK_SPOKE": 650,
    "LITERATURE_SPOKE": 650,
}


@pytest.mark.parametrize("name, cap", sorted(_WORD_CAPS.items()))
def test_the_prompt_stays_within_its_word_cap(name, cap):
    assert len(getattr(prompts, name).split()) <= cap, name


def test_the_risk_spoke_forbids_narrating_an_absent_value():
    """A rule that DESCRIBES the absence invites the sentence describing it.

    90000015's answer said admission type was "scored without a specific feature value",
    paraphrasing the rule that told the model what an absence meant (2026-11-20).
    """
    spoke = _flat(prompts.RISK_SPOKE)

    assert "gets its direction and nothing else" in spoke
    assert "write no clause about a missing, absent or unspecified value" in spoke
    assert "was scored without one" not in spoke
