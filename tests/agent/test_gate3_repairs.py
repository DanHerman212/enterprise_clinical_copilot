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

def test_a_driver_carries_its_name_and_direction_and_nothing_else():
    """The projection is a function of the tool result, not of the patient.

    `patient_value` used to be copied into each factor as a cross-reference against the
    admission's row. It is gone: the row is stated once, in `<admission_metadata>`, and
    the projection now carries the two fields the prose and the bars share. This test is
    what keeps the third field from coming back.
    """
    projected = _presented_for_model("predict_readmission", dict(RISK))
    factor = next(f for f in projected["top_factors"]
                  if f["feature"].lower().startswith("recent ed"))

    assert factor == {"feature": "recent ED visits", "direction": "increases"}
    assert all(set(entry) <= {"feature", "direction"}
               for entry in projected["top_factors"])


def test_the_projection_takes_no_manifest():
    """A caller cannot pass one, which is the point: one fact, one place."""
    with pytest.raises(TypeError):
        _presented_for_model("predict_readmission", dict(RISK), MANIFEST)  # noqa: PLC2801


def test_the_projection_still_withholds_the_floats():
    projected = _presented_for_model("predict_readmission", dict(RISK))

    assert "probability" not in projected
    assert projected["probability_display"] == "11.2%"
    assert all("patient_value" not in factor for factor in projected["top_factors"])


def test_the_chip_generator_asks_about_the_top_increasing_factor():
    """The chip follows the payload, not the patient's row.

    This used to assert the opposite: an increasing factor the patient's profile recorded
    as zero produced no chip at all, so the next turn was never asked to explain a driver
    the record contradicts. `absent_in_manifest` and the profile plumbing it needed are
    gone, so the chip is offered again and this test says so rather than pretending
    otherwise.
    """
    admission_context.set_current_profile(MANIFEST)
    try:
        driver = followups._increasing_driver(RISK)
    finally:
        admission_context.set_current_profile(None)

    assert driver == "recent ED visits"


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
