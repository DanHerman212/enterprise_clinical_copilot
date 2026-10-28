"""The three deterministic repairs `00d3` asked for, at the layer each one lives in.

The directive's premise is that a negative prompt rule oscillates across runs and that a
split code path drifts silently, so every test below asserts a MECHANISM — a payload
field, a rewritten sentence, a rebinding — and none asserts a score.

  * `/ehr` rendered the estimate with no manifest cross-reference while the reasoning path
    annotated it (`holdout_t1_01`);
  * a series that bounced and returned was described as flat (`holdout_t1_02`);
  * a marker that resolved but did not hold its sentence (`holdout_t3_01`, `t4_01`,
    `t4_02`), and a judge that could not see the numbering it was scoring (`t1_03`).
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from evaluation.agent.judge import _evidence
from services.agent.engine.mcp_client import (
    ABNORMALITY_RANK,
    TRAJECTORY_SUMMARY,
    _presented_for_model,
    _presented_labs,
)
from services.agent.postprocess import claims
from services.agent.routing import admission_context

#: The `90000067` row: a driver the model weights upward for a patient with no such
#: visits, plus one protective factor the patient does have.
_MANIFEST = {"recent_ed_visits": "0", "prior_inpatient_days": "6", "age": "73"}


def _risk(*factors) -> dict:
    return {"name": "predict_readmission", "args": {}, "response": {
        "probability": 0.112419, "threshold": 0.11, "top_factors": list(factors),
    }}


# --- 1. one manifest resolution for both paths --------------------------------------

def test_the_projection_resolves_the_profile_itself_when_the_caller_omits_it():
    """`run_command` passed `manifest=None`; the reasoning loop passed the profile.

    Two callers, one of which forgot. The projection now reads the request's own profile,
    so a caller that omits it gets the same answer as one that remembers — which is what
    makes the `/ehr` path unable to drift from `tool_node` again.
    """
    admission_context.set_current_profile(_MANIFEST)
    try:
        presented = _presented_for_model("predict_readmission", {
            "probability": 0.112419, "threshold": 0.11,
            "top_factors": [{"feature": "prior_inpatient_days", "direction": "decreases"}],
        })
    finally:
        admission_context.set_current_profile(None)

    factor = presented["top_factors"][0]
    assert factor["patient_value"] == "6"
    assert "_baseline_artifact_absent_in_patient" not in factor


def test_a_record_outside_the_cohort_still_projects_without_a_profile():
    admission_context.set_current_profile(None)
    presented = _presented_for_model("predict_readmission", {
        "probability": 0.2, "threshold": 0.11,
        "top_factors": [{"feature": "prior_inpatient_days", "direction": "decreases"}],
    })

    assert presented["top_factors"] == [{"feature": "prior inpatient days",
                                         "direction": "decreases"}]


# --- 2. an absent upward driver leaves the active list ------------------------------

def test_a_zero_value_upward_driver_is_not_left_in_the_active_driver_list():
    """`00d` and `00d3`: "the primary driver increasing this risk is recent ED visits".

    The entry is not deleted — the widget still draws five bars — but it is not in the
    list of drivers this patient has, so the formatter has nothing to promote.
    """
    presented = _presented_for_model("predict_readmission", {
        "probability": 0.112419, "threshold": 0.11,
        "top_factors": [
            {"feature": "prior_inpatient_days", "direction": "decreases"},
            {"feature": "recent_ed_visits", "direction": "increases"},
        ],
    }, _MANIFEST)

    assert [f["feature"] for f in presented["top_factors"]] == ["prior inpatient days"]
    absent = presented["baseline_only_factors"]
    assert len(absent) == 1
    assert absent[0]["direction"] == "increases"
    assert absent[0]["patient_value"] == "0"
    assert absent[0]["_baseline_artifact_absent_in_patient"] is True


def test_a_driver_the_patient_has_stays_in_the_active_list():
    presented = _presented_for_model("predict_readmission", {
        "probability": 0.112419, "threshold": 0.11,
        "top_factors": [{"feature": "recent_ed_visits", "direction": "increases"}],
    }, {"recent_ed_visits": "3"})

    assert presented["top_factors"][0]["patient_value"] == "3"
    assert "baseline_only_factors" not in presented


def test_the_served_payload_keeps_every_factor():
    """A projection, not a filter: the record still says what the model returned."""
    payload = {"probability": 0.1, "threshold": 0.1, "top_factors": [
        {"feature": "recent_ed_visits", "direction": "increases", "contribution": 0.08}]}

    _presented_for_model("predict_readmission", payload, _MANIFEST)

    assert len(payload["top_factors"]) == 1
    assert payload["top_factors"][0]["contribution"] == 0.08


def test_an_asserted_absent_driver_is_rewritten_to_state_the_record():
    answer = ("The patient's readmission risk is 11.2%, which is above the 11.0% threshold. "
              "The primary driver increasing this risk is recent ED visits.")

    rewritten, flags = claims.correct_absent_drivers(answer, [_risk(
        {"feature": "recent_ed_visits", "direction": "increases"},
    )], _MANIFEST)

    assert "The primary driver increasing this risk is recent ED visits." not in rewritten
    assert "documents 0 for that factor" in rewritten
    assert "population-baseline effect" in rewritten
    # The sentence it replaced was not the only one, and the rest is untouched.
    assert rewritten.startswith("The patient's readmission risk is 11.2%")
    assert flags


def test_a_sentence_that_already_contrasts_the_two_is_left_alone():
    answer = ("The model weights recent ED visits upward, but the record documents no ED "
              "visits for this patient.")

    rewritten, flags = claims.correct_absent_drivers(answer, [_risk(
        {"feature": "recent_ed_visits", "direction": "increases"},
    )], _MANIFEST)

    assert rewritten == answer
    assert flags == []


def test_a_protective_factor_is_never_rewritten():
    """Only `increases` is corrected: a decrease for an absent factor is not a claim."""
    answer = "Prior inpatient days lower this patient's readmission risk."

    rewritten, flags = claims.correct_absent_drivers(answer, [_risk(
        {"feature": "prior_inpatient_days", "direction": "decreases"},
    )], {"prior_inpatient_days": "0"})

    assert rewritten == answer
    assert flags == []


# --- 3. a series that returned to its nadir is not flat -----------------------------

_RBC = {"key": "rbc", "label": "Red blood cells", "unit": "x10^6/uL", "decimals": 2,
        "reference_low": 4.2, "reference_high": 5.4,
        "series": [{"day": 0, "value": 4.42}, {"day": 1.4, "value": 3.06},
                   {"day": 2.8, "value": 3.47}, {"day": 4.2, "value": 3.06}]}

#: 142 -> 137 -> 139 -> 139. The last two points are equal with nothing between them, so
#: "remained at 139" is TRUE about the final interval and must not be rewritten.
_SODIUM = {"key": "sodium", "label": "Sodium", "unit": "mmol/L", "decimals": 0,
           "series": [{"day": 0, "value": 142}, {"day": 1.4, "value": 137},
                      {"day": 2.8, "value": 139}, {"day": 4.2, "value": 139}]}


def test_the_shape_is_computed_for_a_series_that_bounced_and_returned():
    shape = claims.trajectory_summary(_RBC)

    assert shape["value"] == "3.06"
    assert shape["mid"] == "3.47"
    assert shape["mid_day"] == 2.8
    assert "rose to 3.47 x10^6/uL on day 2.8" in shape["clause"]
    assert shape["summary"].startswith("did NOT remain flat at 3.06")


def test_a_flat_final_interval_has_no_shape():
    assert claims.trajectory_summary(_SODIUM) is None


def test_the_shape_travels_with_the_panel_the_model_reads():
    presented = _presented_labs({"trends": [dict(_RBC), dict(_SODIUM)]})

    assert TRAJECTORY_SUMMARY in presented["trends"][0]
    assert TRAJECTORY_SUMMARY not in presented["trends"][1]
    assert presented["trends"][0][ABNORMALITY_RANK] == 1


def test_a_flatness_clause_is_replaced_by_the_movement():
    """`00d3` `holdout_t1_02`, verbatim shape, with the verb corrected."""
    answer = ("Red blood cells reached a nadir of 3.06 x10^6/uL on day 1.4 and remained "
              "at 3.06 x10^6/uL at discharge, flagged as low.")

    rewritten, flags = claims.rewrite_flat_claims(
        answer, [{"name": "get_lab_trends", "args": {}, "response": {"trends": [dict(_RBC)]}}])

    assert "remained at" not in rewritten
    assert "rose to 3.47 x10^6/uL on day 2.8 before returning to 3.06 x10^6/uL" in rewritten
    # The claim the answer was making about the nadir is still there, unchanged.
    assert "reached a nadir of 3.06 x10^6/uL on day 1.4" in rewritten
    assert flags == ["trajectory_flattened:Red blood cells"]


def test_a_flat_claim_about_a_flat_series_is_left_alone():
    answer = "Sodium remained at 139 mmol/L at discharge."

    rewritten, flags = claims.rewrite_flat_claims(
        answer, [{"name": "get_lab_trends", "args": {}, "response":
                  {"trends": [dict(_SODIUM)]}}])

    assert rewritten == answer
    assert flags == []


# --- 4. a marker is moved to the passage that holds the sentence --------------------

_MEDS = {"name": "discharge_medications", "args": {}, "response": {"passages": [
    {"section": "discharge_medications",
     "text": "1. Lantus 25 units. 2. Cardizem CD 180 mg. 3. Aspirin 325 mg."},
    {"section": "discharge_medications",
     "text": "15. Prednisone 40 mg q.a.m. x3 days followed by 30 mg q.a.m. x3 days."},
    {"section": "discharge_medications",
     "text": "16. Clindamycin 300 mg q.i.d. x2 weeks."},
]}}


def test_a_marker_that_cannot_hold_its_sentence_is_moved_to_the_one_that_can():
    """`00d3` `holdout_t3_01`: the taper cited `^[1]`, the passage of other drugs."""
    answer = "The prednisone taper steps down from 40 mg over three days ^[1]."

    rewritten, flags = claims.rebind_citations(answer, [_MEDS])

    assert rewritten.endswith("^[2].")
    assert flags == ["citation_rebound:^1->^2"]


def test_a_marker_that_already_holds_its_sentence_is_untouched():
    answer = "The prednisone taper steps down over three days ^[2]."

    rewritten, flags = claims.rebind_citations(answer, [_MEDS])

    assert rewritten == answer
    assert flags == []


def test_a_marker_is_not_moved_when_no_passage_holds_the_words():
    """Ambiguity is not an error: if nothing matches, the model's choice stands."""
    answer = "The overall course was unremarkable ^[1]."

    rewritten, flags = claims.rebind_citations(answer, [_MEDS])

    assert rewritten == answer
    assert flags == []


# --- 5. the whole contract, in order -----------------------------------------------

def _answers(state):
    return [state["messages"][-1].content, state.get("deterministic_answer") or ""]


def test_the_settlement_hook_applies_every_correction_to_the_prose():
    from services.agent.engine.graph import sanitize_answer_markers

    state = {
        "messages": [AIMessage(content=(
            "The primary driver increasing this risk is recent ED visits. "
            "Red blood cells remained at 3.06 x10^6/uL at discharge. "
            "An unrelated claim ^[9]."
        ))],
        "tool_calls": [
            _risk({"feature": "recent_ed_visits", "direction": "increases"}),
            {"name": "get_lab_trends", "args": {}, "response": {"trends": [dict(_RBC)]}},
        ],
        "replayed_tool_calls": [],
    }
    admission_context.set_current_profile(_MANIFEST)
    try:
        sanitize_answer_markers(state)
    finally:
        admission_context.set_current_profile(None)

    text = state["messages"][-1].content
    assert "documents 0 for that factor" in text
    assert "rose to 3.47 x10^6/uL on day 2.8" in text
    assert "^[9]" not in text
    assert "^[1]" not in text and "^[2]" not in text, "a lab panel's units are 1..2 here"


def test_the_settlement_hook_is_idempotent():
    from services.agent.engine.graph import sanitize_answer_markers

    state = {
        "messages": [AIMessage(content=(
            "Red blood cells remained at 3.06 x10^6/uL at discharge."))],
        "tool_calls": [{"name": "get_lab_trends", "args": {},
                        "response": {"trends": [dict(_RBC)]}}],
        "replayed_tool_calls": [],
    }
    sanitize_answer_markers(state)
    first = state["messages"][-1].content
    sanitize_answer_markers(state)

    assert state["messages"][-1].content == first


# --- 6. the judge sees the numbering it is scoring ---------------------------------

def test_lab_analytes_reach_the_judge_as_numbered_units():
    """`holdout_t1_03` scored `^[1], ^[2], ^[3]` as pointing at an unnumbered blob."""
    evidence = _evidence({"name": "get_lab_trends", "response": {"trends": [dict(_RBC), dict(_SODIUM)]}})

    assert [entry["cite"] for entry in evidence["labs"]] == [1, 2]
    assert evidence["labs"][0]["label"] == "Red blood cells"


def test_note_passages_reach_the_judge_as_numbered_units():
    evidence = _evidence({"name": "rag_search_sections", "response": {"passages": [
        {"section": "brief_hospital_course", "text": "course"},
        {"section": "discharge_diagnosis", "text": "diagnosis"},
    ]}})

    assert [p["cite"] for p in evidence["passages"]] == [1, 2]


def test_medication_passages_reach_the_judge_as_numbered_units():
    """The taper is in the second passage, and that is only checkable if it is second."""
    evidence = _evidence(_MEDS)

    assert [p["cite"] for p in evidence["medications"]] == [1, 2, 3]
    assert "Prednisone" in evidence["medications"][1]["text"]
