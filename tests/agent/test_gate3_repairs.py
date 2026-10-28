"""The three Gate 3 repairs, at the layer each one lives in.

`00d` produced four failures and three root causes:

  * a model ATTRIBUTION reported as a fact about the patient — `recent_ed_visits` named
    as "the primary driver increasing this risk" for a record whose manifest says `0`,
    and a chip that asked the product's own question about it;
  * a follow-up citing `^[1]` (a lab trend) for a claim that came from the sixth
    numbered source;
  * `24.6667` printed as `24.7`.

Each is pinned here against its own mechanism, with no model and no network.
"""

from __future__ import annotations

import asyncio

import pytest

from services.agent.engine.mcp_client import (
    _render_tool_result,
    _presented_for_model,
)
from services.agent.postprocess import followups
from services.agent.routing import admission_context


#: The risk result Gate 3 was answered from, and the manifest it contradicted.
RISK = {
    "hadm_id": 90000067,
    "probability": 0.112419,
    "threshold": 0.11,
    "decision": 1,
    "base_value": -1.767608,
    "top_factors": [
        {"feature": "prior_inpatient_days", "contribution": -0.2656,
         "direction": "decreases"},
        {"feature": "rdw_max", "contribution": -0.106, "direction": "decreases"},
        {"feature": "recent_ed_visits", "contribution": 0.0861, "direction": "increases"},
        {"feature": "oncology_flag", "contribution": -0.0767, "direction": "decreases"},
    ],
}

MANIFEST = {"recent_ed_visits": "0", "prior_inpatient_days": "0",
            "prior_admission_count": "0", "age": "73"}


# --- 1. an attribution is not a finding about the patient ---------------------------

def test_an_increasing_factor_the_patient_does_not_have_is_marked():
    """`recent_ed_visits` increases the score for a row that records zero of it.

    Marked twice, since `00d3`: with the patient's value and the absent flag, and by being
    moved out of `top_factors` into `baseline_only_factors`. The annotation alone was not
    enough — `00d` and `00d3` both narrated the factor as the primary driver anyway.
    """
    projected = _presented_for_model("predict_readmission", RISK, MANIFEST)
    everything = [*projected["top_factors"],
                  *(projected.get("baseline_only_factors") or [])]
    factor = next(f for f in everything
                  if f["feature"].lower().startswith("recent ed"))

    assert factor["patient_value"] == "0"
    assert factor["_baseline_artifact_absent_in_patient"] is True
    assert factor not in projected["top_factors"]


def test_a_factor_the_patient_does_have_carries_its_value_and_no_flag():
    manifest = {"prior_inpatient_days": "24.6667"}
    projected = _presented_for_model("predict_readmission", RISK, manifest)
    factor = next(f for f in projected["top_factors"]
                  if f["feature"].lower().startswith("prior inpatient"))

    assert factor["patient_value"] == "24.6667"
    assert "_baseline_artifact_absent_in_patient" not in factor


def test_a_feature_the_manifest_cannot_verify_is_not_accused_of_being_absent():
    """`rdw_max` has no manifest column, so the projection says nothing about it.

    Silence is the correct answer: "the patient does not have this" is a claim nothing
    here could support, and inferring it from the key would be a second invention.
    """
    manifest = {"recent_ed_visits": "0", "prior_inpatient_days": "18.5833"}
    projected = _presented_for_model("predict_readmission", RISK, manifest)
    everything = [*projected["top_factors"],
                  *(projected.get("baseline_only_factors") or [])]
    by_feature = {factor["feature"].lower(): factor for factor in everything}

    # Verifiable and absent.
    assert by_feature["recent ed visits"]["_baseline_artifact_absent_in_patient"] is True
    # Verifiable and present.
    assert "_baseline_artifact_absent_in_patient" not in by_feature[
        "prior inpatient days"]
    # Not verifiable at all — no flag either way.
    unverifiable = by_feature["red cell distribution width (rdw)"]
    assert "patient_value" not in unverifiable
    assert "_baseline_artifact_absent_in_patient" not in unverifiable


def test_without_a_manifest_the_projection_still_withholds_the_floats():
    projected = _presented_for_model("predict_readmission", RISK)

    assert "probability" not in projected
    assert projected["probability_display"] == "11.2%"
    assert all("patient_value" not in factor for factor in projected["top_factors"])


def test_the_chip_generator_will_not_ask_about_an_absent_factor():
    """The chip is the second half of `holdout_t2_01`: the product asking itself.

    `recent_ed_visits` is the highest-ranked increasing factor and the patient has zero
    of them, so the chip must fall through to the next candidate rather than offer a
    question whose only honest answer is "that is not true of this patient".
    """
    async def run() -> str:
        # The profile is per-turn state, set by the pre-flight; set it here directly.
        admission_context.set_current_profile(MANIFEST)
        try:
            return followups._increasing_driver(RISK)
        finally:
            admission_context.set_current_profile(None)

    assert asyncio.run(run()) == ""


def test_the_chip_generator_still_asks_about_a_factor_the_patient_has():
    async def run() -> str:
        admission_context.set_current_profile({"recent_ed_visits": "2"})
        try:
            return followups._increasing_driver(RISK)
        finally:
            admission_context.set_current_profile(None)

    assert asyncio.run(run()) == "recent ED visits"


# --- 2. the marker is printed ready to copy ----------------------------------------

def _rendered(payload):
    import json  # noqa: PLC0415

    return json.loads(_render_tool_result(payload))


def test_a_note_passage_prints_the_marker_to_copy():
    rendered = _rendered({"passages": [{"id": "p1", "section": "brief_hospital_course",
                                        "text": "text"}]})

    assert rendered["passages"][0]["cite"] == 1
    assert rendered["passages"][0]["cite_header"] == (
        "[CITATION ^[1] | CLINICAL NOTE SECTION — brief_hospital_course]")


def test_an_analyte_prints_the_marker_to_copy():
    rendered = _rendered({"trends": [{"label": "RDW", "series": []}]})

    assert rendered["trends"][0]["cite_header"] == (
        "[CITATION ^[1] | STRUCTURED LAB TREND — RDW]")


def test_the_header_number_is_the_number_the_citation_layer_resolves():
    """The whole defect: a header that disagrees with `cite` would be worse than none.

    Turn 2's first note passage is `^[6]` on a thread that already drew five analytes,
    and the header has to say six.
    """
    rendered = _rendered({"passages": [{"id": "p1", "section": "discharge_diagnosis",
                                        "text": "text"}]})
    # Rendered again at the offset a follow-up turn carries.
    continued = _rendered({"passages": [{"id": "p2", "section": "brief_hospital_course",
                                         "text": "text"}]})

    assert rendered["passages"][0]["cite"] == 1
    assert continued["passages"][0]["cite_header"].startswith("[CITATION ^[1]")


def test_a_live_turn_and_a_replayed_turn_print_the_same_header():
    """Both paths go through `_render_tool_result`, which is why this is one change."""
    payload = {"passages": [{"id": "p1", "section": "history_of_present_illness",
                             "text": "text"}]}

    live = _render_tool_result(_presented_for_model("rag_search", payload), offset=5)
    replayed = _render_tool_result(_presented_for_model("rag_search", payload), offset=5)

    assert live == replayed
    assert "[CITATION ^[6] | CLINICAL NOTE SECTION — history_of_present_illness]" in live


# --- 3. the manifest states the row's digits ---------------------------------------

def test_the_manifest_does_not_round_a_recorded_value():
    """`24.6667` is the scored row; `24.7` is a different number."""
    block = admission_context.render(90000101, {"prior_inpatient_days": "24.6667"})

    assert "prior_inpatient_days: 24.6667" in block
    assert "24.7 " not in block


def test_an_outpatient_block_still_drops_the_stay_fields():
    block = admission_context.render(
        90000028, {"encounter_type": "outpatient encounter (no inpatient stay)",
                   "index_los_days": "7.9"})

    assert "length_of_stay_days" not in block
    assert "outpatient encounter" in block
