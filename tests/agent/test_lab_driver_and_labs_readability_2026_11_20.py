"""The four defects the 2026-11-20 live audit confirmed, pinned offline.

Each class of test here corresponds to one thing a reader saw in the smoke session on
`dd01557` and one change made in response:

1. a driver question spent a tool round asking `get_lab_trends` for `monocytes_min`, a
   feature key, and was refused;
2. every driver turn after a literature search was badged General Medical Evidence under a
   PubMed caption naming a search that turn never ran;
3. two encoded numbers reached the prose as observations — `42.46` days and a raw
   `(DIRECT EMER.)` beside the words that mean it;
4. a `/labs` answer was rewritten by a guard that mistook a hospital day for a factor value.

Everything here is offline: no LLM, no MCP, no Vertex, no BigQuery.
"""

from __future__ import annotations

import asyncio

import pytest

from services.agent.engine import deterministic, mcp_client, prompts
from services.agent.postprocess import citations
from services.agent.postprocess.evidence_state import SessionEvidenceState
from services.agent.postprocess.guardrail import (
    guard_answer,
    normalise_factor_values,
    verify_lab_values,
    verify_risk_numbers,
)
from services.agent.presentation.a2ui import compose_presentation
from services.agent.routing import admission_context
from services.agent.engine.mcp_client import _presented_factor, _presented_for_model


# ---------------------------------------------------------------------------------------
# 1. A laboratory analyte is named the way the tool draws it, before dispatch
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("written, drawn", [
    # The anchors the feature build scores. `monocytes_min` is a row, `monocytes` is an
    # analyte, and the model was handing the row to the panel tool.
    ("monocytes_min", "monocytes"),
    ("rdw_max", "rdw"),
    ("hemoglobin_min", "hemoglobin"),
    ("sodium_min", "sodium"),
    ("sodium_max", "sodium"),
    ("sodium_last", "sodium"),
    ("rbc_min", "rbc"),
    ("rbc_last", "rbc"),
    # Case and stray whitespace are the same name.
    (" Monocytes ", "monocytes"),
    ("RDW_Max", "rdw"),
])
def test_a_feature_key_becomes_the_panel_analyte(written, drawn):
    out = mcp_client.normalise_lab_analyte(
        "get_lab_trends", {"analyte": written, "hadm_id": 90000015})

    assert out["analyte"] == drawn
    # Nothing else about the call is touched.
    assert out["hadm_id"] == 90000015


def test_an_analyte_the_panel_does_not_draw_is_passed_through():
    """The tool's own refusal names the analytes; a guess here would hide it.

    Only a name this module recognises is translated. `hematocrit` is not on the panel, so
    the call goes as written and the model reads the refusal's list — the message it used to
    recover from.
    """
    out = mcp_client.normalise_lab_analyte(
        "get_lab_trends", {"analyte": "hematocrit", "hadm_id": 90000015})

    assert out["analyte"] == "hematocrit"


def test_only_the_laboratory_tool_is_translated():
    assert mcp_client.normalise_lab_analyte(
        "rag_search", {"query": "monocytes_min"}) == {"query": "monocytes_min"}
    assert mcp_client.normalise_lab_analyte(
        "get_lab_trends", {"hadm_id": 90000015}) == {"hadm_id": 90000015}


class _LabBox:
    """A toolbox that records what it was dispatched and answers with one trend."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def call(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, dict(arguments)))
        return {"hadm_id": arguments.get("hadm_id"), "returned": 1, "trends": []}


def test_the_dispatch_hands_the_tool_the_analyte_not_the_feature_key():
    """The end-to-end half: what leaves the client is the name the tool accepts.

    2026-11-20, 90000015: the first call of a driver turn was
    `{"analyte": "monocytes_min"}`, refused with `unknown_analyte`; the model read the
    refusal, called again with `monocytes`, and the turn took 30 seconds instead of 3.
    """
    box = _LabBox()

    _messages, recorded = asyncio.run(mcp_client._execute_tool_calls(
        box,
        [{"name": "get_lab_trends", "args": {"analyte": "monocytes_min"}, "id": "c1"}],
        hadm_id=90000015,
    ))

    assert box.calls == [("get_lab_trends", {"analyte": "monocytes", "hadm_id": 90000015})]
    # The record states what ran, so a replay cannot reintroduce the refused name.
    assert recorded[0]["args"]["analyte"] == "monocytes"


def test_the_risk_prompt_names_analytes_and_no_warehouse_key():
    """The prompt is where the wrong name came from, so it is where the fix is asserted."""
    text = " ".join(prompts.compose_system_prompt(
        allowed_tools={"get_lab_trends", "rag_search"},
        risk_driver_question=True).split())

    assert "monocytes_min" not in text
    assert "`monocytes`, `rdw`, `hemoglobin`, `sodium`, `rbc`" in text
    # And the proxy search the RDW turn performed is forbidden by name.
    assert "never search the notes for a clinical proxy" in text


# ---------------------------------------------------------------------------------------
# 2. A chart turn is a chart turn, whatever the thread has published
# ---------------------------------------------------------------------------------------

_RECORD = {"pmid": "39165878", "title": "A study", "abstract": "A rate of 1.1%.",
           "journal": "J", "pub_date": "2024", "authors": "A B"}
_ESTIMATE = {
    "name": "predict_readmission",
    "args": {"hadm_id": 90000015},
    "response": {
        "probability": 0.090459, "threshold": 0.11,
        "probability_display": "9.0%", "threshold_display": "11.0%",
        "top_factors": [{"feature": "monocytes_min", "direction": "decreases",
                         "value": 0.2}],
    },
}
_DRIVER_ANSWER = (
    "According to the readmission risk model, lowest monocyte count (patient value: "
    "0.2 x10^3/uL) decreases this patient's 30-day readmission risk.")


def _drawn(out: dict) -> list[dict]:
    """Every component the canvas composed, in order."""
    components: list[dict] = []
    for message in (out.get("a2ui") or {}).get("messages") or []:
        update = (message or {}).get("updateComponents") or {}
        components.extend(update.get("components") or [])
    return components


def test_a_driver_turn_is_chart_grounded_even_when_the_thread_published_records():
    """The reported defect: a length-of-stay answer badged General Medical Evidence.

    The thread held a literature search from its first turn. Treating the replay as this
    turn's literature surface flipped the scope and drew a provenance caption naming that
    search — under an answer about length of stay (measured 2026-11-20, 90000069).
    """
    out = compose_presentation(
        "How does lowest monocyte count affect this patient's readmission risk?",
        _DRIVER_ANSWER, [], None, None,
        evidence=[_ESTIMATE],
        is_outpatient=False,
        literature_replay=False,
        thread_literature=[_RECORD],
    )

    assert out["scope"] == "patient_chart"
    # The estimate is the evidence, even though this turn called no tool for it.
    assert out["verified"] is True
    captions = [component for component in _drawn(out)
                if component.get("component") == "Text"]
    assert captions == []


def test_a_literature_turn_is_still_badged_as_general_evidence():
    """The rule this must not weaken: published records are somebody else's patients."""
    search = {"name": "search_literature", "args": {"query": "knee arthroplasty"},
              "response": {"query": "knee arthroplasty", "articles": [_RECORD],
                           "returned": 1}}

    out = compose_presentation(
        "What does the published literature report on total knee arthroplasty?",
        "A cohort reported 1.1% ^[PMID: 39165878].", [], None, None,
        evidence=[_ESTIMATE, search],
        is_outpatient=False,
        literature_replay=True,
    )

    assert out["scope"] == "general_evidence"


# ---------------------------------------------------------------------------------------
# 3. A risk value states its unit, and a duration states its fraction
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("feature, value, stated", [
    ("age", 68.0, "68 years"),
    ("index_los_days", 42.458333333333336, "42.46 days"),
    ("prior_inpatient_days", 4.75, 4.75),
    ("index_los_days", 1.0, "1 day"),
    ("rdw_max", 15.5879, "15.59%"),
    ("hemoglobin_min", 6.8, "6.8 g/dL"),
    ("sodium_min", 131.0, "131 mmol/L"),
    ("rbc_last", 3.09, "3.09 ×10⁶/µL"),
    ("monocytes_min", 0.2, "0.2 ×10³/µL"),
])
def test_a_measurement_states_the_unit_it_is_measured_in(feature, value, stated):
    presented = _presented_factor(
        {"feature": feature, "direction": "increases", "value": value})

    assert presented["value"] == stated


@pytest.mark.parametrize("feature, value", [
    ("prior_admission_count", 2),
    ("recent_ed_visits", 4.75),
    ("procedure_count", 1.0),
    ("medication_order_count", 4.85),
])
def test_a_count_stays_a_whole_number_with_no_unit(feature, value):
    """A tally says what it counts by its own name; a unit would be an invented word."""
    presented = _presented_factor(
        {"feature": feature, "direction": "increases", "value": value})

    assert isinstance(presented["value"], int)


def test_the_session_record_states_the_same_number_and_unit():
    """The record is re-read on later turns, so it cannot restore the raw float."""
    projected = _presented_for_model("predict_readmission", _ESTIMATE["response"])
    state = SessionEvidenceState(hadm_id=90000015)
    state.ingest_tool_output("predict_readmission", {"hadm_id": 90000015}, projected)

    rendered = state.render_for_prompt({})

    assert "value=0.2 ×10³/µL" in rendered


def test_a_value_with_its_unit_is_not_stripped_as_invented():
    """The guard compares magnitudes, so a unit beside a served value changes nothing."""
    out, flags = verify_risk_numbers(
        "Her red cell distribution width peaked at 15.59%, and her length of stay was "
        "42.46 days.", [_ESTIMATE, {"name": "predict_readmission", "args": {},
                                 "response": {"top_factors": [
                                     {"feature": "rdw_max", "value": 15.5879},
                                     {"feature": "index_los_days", "value": 42.46},
                                 ]}}])

    assert "15.59" in out
    assert "42.46 days" in out
    assert flags == []


def test_a_sentence_stating_the_rendered_value_survives_the_lab_check():
    """`6.8 g/dL` is the spelled form of a served `6.7999`, and must not be removed.

    The lab check compares the digits it finds against the corpus, so the rendered form has
    to be in that corpus — otherwise the sentence is deleted for quoting a value the turn
    had just been handed.
    """
    calls = [{"name": "predict_readmission", "args": {"hadm_id": 90000015},
              "response": {"probability": 0.09, "threshold": 0.11, "top_factors": [
                  {"feature": "hemoglobin_min", "direction": "increases",
                   "value": 6.7999}]}}]
    answer = ("According to the readmission risk model, lowest hemoglobin (patient value: "
              "6.8 g/dL) increases this patient's 30-day readmission risk.")

    out = guard_answer(answer, calls, [], state={})

    assert out["answer"] == answer
    assert out["flags"] == []


def test_the_manifest_drops_the_warehouse_code_and_keeps_the_words():
    block = admission_context.render(
        90000015, {"admission_type": "Emergency, direct admission (DIRECT EMER.)",
                   "discharge_location": "Skilled nursing facility (SNF)"})

    assert "admission_type: Emergency, direct admission" in block
    assert "DIRECT EMER." not in block
    assert "discharge_location: Skilled nursing facility" in block


def test_a_value_that_is_only_a_code_is_left_alone():
    """Stripping it would leave an empty field, which states less than the code did."""
    block = admission_context.render(90000015, {"insurance": "MEDICAID"})

    assert "insurance: MEDICAID" in block


# ---------------------------------------------------------------------------------------
# 4. A hospital day is a day, and the rounding guard leaves it alone
# ---------------------------------------------------------------------------------------


def test_a_lab_citation_states_whole_days():
    passage = citations.lab_passage({
        "key": "monocytes", "label": "Monocytes", "unit": "x10^3/uL", "decimals": 1,
        "reference_low": 0.2, "reference_high": 0.8,
        "series": [{"day": 0.0, "value": 0.6, "flag": "normal"},
                   {"day": 1.3, "value": 0.2, "flag": "normal"}],
    })

    assert "day 0)" in passage["text"]
    # An honestly fractional timepoint keeps its fraction.
    assert "day 1.3)" in passage["text"]


def test_the_lab_annotation_states_whole_days():
    recorded = [{"name": "get_lab_trends", "args": {}, "response": {"trends": [{
        "label": "Hemoglobin", "unit": "g/dL", "decimals": 1,
        "series": [{"day": 0.0, "value": 12.0, "flag": "normal"},
                   {"day": 6.0, "value": 9.0, "flag": "low"}],
    }]}}]

    block = deterministic.lab_flag_annotation(recorded)

    assert "baseline day 0 flag=normal" in block


def test_a_zero_count_does_not_rewrite_a_hospital_day():
    """`factor_value_rounded:0.0` fired on a `/labs` answer that quoted no factor at all.

    The encoded row spells zero as `0.0`; the guard read that as a fraction to correct and
    matched every `0.0` token in the answer — on `/labs`, the trajectory's own timepoints
    (measured 2026-11-20, 90000069).
    """
    calls = [{"name": "predict_readmission", "args": {"hadm_id": 90000069},
              "response": {"probability": 0.075, "threshold": 0.11, "top_factors": [
                  {"feature": "recent_ed_visits", "direction": "increases",
                   "value": 0.0}]}}]
    answer = "Hemoglobin began low at 11.9 g/dL on day 0.0, then fell on day 6.0."

    out, flags = normalise_factor_values(answer, calls)

    assert out == answer
    assert flags == []


def test_a_hospital_day_is_not_rewritten_as_a_served_count():
    """A count whose value happens to equal a timepoint must not correct the timepoint."""
    calls = [{"name": "predict_readmission", "args": {"hadm_id": 90000069},
              "response": {"probability": 0.075, "threshold": 0.11, "top_factors": [
                  {"feature": "index_los_days", "direction": "increases",
                   "value": 6.5}]}}]
    answer = "She was still on day 6.5 of the stay."

    out, flags = normalise_factor_values(answer, calls)

    assert out == answer
    assert flags == []


def test_a_count_with_a_real_fraction_is_still_corrected():
    """The rule the day exclusion must not weaken."""
    calls = [{"name": "predict_readmission", "args": {"hadm_id": 90000069},
              "response": {"probability": 0.075, "threshold": 0.11, "top_factors": [
                  {"feature": "medication_order_count", "direction": "increases",
                   "value": 4.85}]}}]

    out, flags = normalise_factor_values("She was given 4.85 orders.", calls)

    assert "5 orders" in out
    assert flags == ["factor_value_rounded:4.85"]


def test_the_lab_check_keeps_a_timepoint_sentence_intact():
    """Guards disagreeing about `day 0.0` used to delete the sentence that stated it."""
    answer = "Creatinine was 0.55 mg/dL on day 0.0 and 1.04 mg/dL on day 30.0."
    payload = {"trends": [{"key": "creatinine", "label": "Creatinine",
                           "unit": "mg/dL", "decimals": 2,
                           "series": [{"day": 0.0, "value": 0.55, "flag": "normal"},
                                      {"day": 30.0, "value": 1.04, "flag": "normal"}]}]}

    out, flags = verify_lab_values(answer, "", "", str(payload))

    assert out == answer
    assert flags == []


def test_the_labs_prompt_asks_for_a_shape_the_card_does_not_repeat():
    """One summary sentence, bullets for abnormal analytes, one closing sentence."""
    text = " ".join(prompts.LABS_FORMATTING.split())

    assert "ONE opening sentence saying whether any analyte crossed a reference limit" in text
    assert "ONE short bullet per ABNORMAL analyte" in text
    assert "Do not recite their start and end values" in text
    assert 'Write "day 6", never "day 6.0"' in text
    # The safety rules survive the reshape.
    assert "critical_low" in text and "critical_high" in text
    assert "Write no citation markers on this wire" in text
    assert "the card beneath this answer states that" in text
