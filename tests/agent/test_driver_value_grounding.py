"""A driver question must reach the patient's own value for the factor.

Reported 2026-11-02. Clicking a driver bar in the risk widget asks "What in this patient's
chart explains red cell distribution width (RDW) as a readmission risk driver?" — and the
agent answered that the chart did not document the laboratory value, and that nothing
explained the attribution.

All three statements were true of what it had been given. `predict_readmission` named the
factor, its direction and its contribution, and nothing else; the `<admission_metadata>`
block renders eleven administrative fields and no clinical input; and a discharge summary
does not contain the words "red cell distribution width", so a note search for the feature's
name returned nothing. The value existed in the scored row and stopped there.

The fix states it, once, where the attribution already is: each `top_factors` entry carries
the row's own value for that feature. These tests pin the whole path — the payload, the copy
the model reads, the guard that must let the number survive into prose, and the prompt rules
that say what to do with it.
"""

from __future__ import annotations

from unittest.mock import patch

import services.mcp.tools.prediction as pr
from services.agent.engine import graph
from services.agent.engine import mcp_client
from services.agent.engine import prompts
from services.agent.postprocess.guardrail import (
    normalise_factor_values,
    verify_risk_numbers,
)
from services.mcp.contracts import validate_prediction_result

#: The row an admission is scored on: administrative columns, the laboratory anchors the
#: booster consumes, and one aggregated parent that has no column of its own.
ROW = {
    "age": "80",
    "prior_inpatient_days": "23.5417",
    "rdw_max": "15.5",
    "rbc_last": "3.47",
    "monocytes_min": "0.21",
    "oncology_flag": 0.0,
    "medication_order_count": "17",
}


class _Source:
    def fetch(self, hadm_id):
        return dict(ROW)


def _reply(**overrides):
    payload = {
        "probability": 0.34,
        "threshold": 0.11,
        "prediction": 1,
        "base_value": 0.05,
        "top_factors": [
            {"feature": "rdw_max", "attribution": 0.2592},
            {"feature": "prior_inpatient_days", "attribution": 0.245},
            {"feature": "discharge_location", "attribution": 0.0896},
            {"feature": "monocytes_min", "attribution": -0.0836},
        ],
    }
    payload.update(overrides)
    return payload


def _predict():
    with patch.object(pr, "feature_order", lambda: ["age", "rdw_max"]), \
         patch.object(pr, "_source", lambda: _Source()), \
         patch.object(pr, "predict_one", lambda instance: (_reply(), "model-1")):
        return pr._predict(90000012)


# --- the payload ---------------------------------------------------------------

def test_every_attributed_feature_carries_the_rows_own_value():
    factors = {factor["feature"]: factor for factor in _predict()["top_factors"]}

    assert factors["rdw_max"]["value"] == 15.5
    # Published at clinical precision (2026-11-04): the row holds 23.5417, and the model
    # is told to quote a served value exactly, so the payload carries two decimals and
    # not the four the encoded column happens to hold.
    assert factors["prior_inpatient_days"]["value"] == 23.54
    assert factors["monocytes_min"]["value"] == 0.21


def test_a_count_is_a_number_rather_than_the_string_bigquery_returned():
    """Every column arrives from BigQuery as a string, so a copy would be `"17"`.

    The payload is read by a model and by the widget; a string where a number belongs is
    the kind of shape that silently fails a comparison one layer down. An integral value
    stays an integer rather than becoming `17.0`: the prompt's exact-quoting rule would
    put the trailing `.0` in front of a reader, and a count of 17 admissions is not a
    measurement to that precision.
    """
    with patch.object(pr, "feature_order", lambda: ["medication_order_count"]), \
         patch.object(pr, "_source", lambda: _Source()), \
         patch.object(pr, "predict_one", lambda instance: (
             _reply(top_factors=[{"feature": "medication_order_count",
                                  "attribution": -0.129}]), "model-1")):
        (factor,) = pr._predict(90000012)["top_factors"]

    assert factor["value"] == 17
    assert isinstance(factor["value"], int)


def test_a_repeating_float_is_published_at_clinical_precision():
    """`42.458333333333336` days is a formatting fault, not a measurement.

    The row's encoded value carries binary noise. The prompt tells the model to state a
    served value exactly, so the precision has to be decided before the model reads it:
    otherwise the prose copies fourteen digits — observed live on 90000015, "length of
    stay (42.458333333333336 days)" — or the guardrail scrapes them off afterwards with a
    regex that only fires on three or more decimals.
    """
    row = dict(ROW, index_los_days="42.458333333333336")

    class _NoisySource:
        def fetch(self, hadm_id):
            return dict(row)

    with patch.object(pr, "feature_order", lambda: ["index_los_days"]), \
         patch.object(pr, "_source", lambda: _NoisySource()), \
         patch.object(pr, "predict_one", lambda instance: (
             _reply(top_factors=[{"feature": "index_los_days",
                                  "attribution": 0.21}]), "model-1")):
        payload = pr._predict(90000012)
    (factor,) = payload["top_factors"]

    assert factor["value"] == 42.46
    # And the copy the model reads is the same number, stated as a day count with its unit
    # (2026-11-20): nobody is admitted for 42.46 days, and "(patient value: 42)" alone left
    # the reader to work out what was counted.
    presented = mcp_client._presented_for_model("predict_readmission", payload)
    assert presented["top_factors"][0]["value"] == "42 days"


def test_the_guardrail_corrects_the_fraction_to_the_whole_day():
    """The number the model is served is 42; a copied 42.46 is rewritten to it."""
    call = {"name": "predict_readmission", "response": {
        "top_factors": [{"feature": "index_los_days", "value": 42.46,
                         "direction": "increases"}],
    }}
    answer = "Her length of stay was 42.46 days, which raises her risk."

    normalised, rounding_flags = normalise_factor_values(answer, [call])
    verified, risk_flags = verify_risk_numbers(normalised, [call])

    assert normalised == "Her length of stay was 42 days, which raises her risk."
    assert rounding_flags == ["factor_value_rounded:42.46"]
    assert risk_flags == []
    assert "42 days" in verified


def test_a_parent_feature_carries_no_value_rather_than_a_guessed_one():
    """`discharge_location` is thirteen one-hot columns, so it has no single value.

    The manifest states it in words. Picking one column out of the encoding would be an
    invented fact, and stating 0 would be worse: the model would read a value the record
    never recorded.
    """
    factors = {factor["feature"]: factor for factor in _predict()["top_factors"]}

    assert "value" not in factors["discharge_location"]


def test_the_payload_still_satisfies_its_contract():
    """`value` is declared on `PredictionFactor` — an undeclared key is dropped."""
    validate_prediction_result(_predict())


def test_the_wire_schema_lets_a_factor_value_be_an_integer():
    """The TypedDict IS the MCP output schema, so `float` re-floated every count.

    `server.add_tool` derives an output schema from the return annotation, and a bare
    `float` makes this field `{"type": "number"}`. A row value of `0` for `oncology_flag`
    then reached the client as `0.0` — measured against the deployed server on
    2026-11-04 — and the prompt's rule to quote a served value exactly had a form no
    clinician writes. `int | float` derives `anyOf [integer, number]`.
    """
    import json

    from mcp.server import MCPServer

    from services.mcp.tools import predict_readmission

    server = MCPServer(name="schema-probe", version="0")
    server.add_tool(predict_readmission)
    tool = server._tool_manager.get_tool("predict_readmission")
    factor = tool.output_schema["$defs"]["PredictionFactor"]["properties"]["value"]

    options = {json.dumps(option, sort_keys=True) for option in factor["anyOf"]}
    assert json.dumps({"type": "integer"}) in options, (
        "an integer must be a valid wire value, or `0` arrives as `0.0`")
    assert json.dumps({"type": "number"}) in options, (
        "a measurement like 4.75 must still be accepted")


# --- the copy the model reads --------------------------------------------------

def test_the_model_is_shown_the_value_beside_the_direction():
    presented = mcp_client._presented_factor(
        {"feature": "rdw_max", "direction": "increases", "contribution": 0.2592,
         "value": 15.5})

    assert presented["feature"] == "red cell distribution width (RDW)"
    assert presented["direction"] == "increases"
    # The unit travels with the number, because the prose quotes this value verbatim.
    assert presented["value"] == "15.5%"
    # The weight does NOT travel. It is the model's own SHAP arithmetic, and the prose is
    # no longer asked for it: every risk answer used to read "contribution 0.2592" beside
    # a bar the reader could see (2026-11-20). The recorded call keeps it for the widget.
    assert "contribution" not in presented


def test_a_null_value_is_left_out_rather_than_named():
    presented = mcp_client._presented_factor(
        {"feature": "haemoglobin_min", "direction": "increases",
         "contribution": 0.0824, "value": None})

    # A key that is not there cannot be written into prose. Carrying `value: null` made
    # the model say "the scored input is null" about a column the reader will never look
    # up; the direction is the part of the attribution that is about the model.
    assert "value" not in presented


def test_a_flag_is_stated_as_words_rather_than_as_the_encoding():
    presented = mcp_client._presented_factor(
        {"feature": "oncology_flag", "direction": "increases", "value": 1})

    # On is the factor's own name; only absence needs saying (2026-11-20).
    assert "value" not in presented
    absent = mcp_client._presented_factor(
        {"feature": "oncology_flag", "direction": "decreases", "value": 0})
    assert absent["value"] == "not present"


def test_a_measurement_is_stated_as_the_tool_published_it():
    presented = mcp_client._presented_factor(
        {"feature": "prior_admission_count", "direction": "increases", "value": 2})

    assert presented["value"] == 2


def test_a_factor_with_no_value_given_omits_the_key():
    presented = mcp_client._presented_factor(
        {"feature": "discharge_location", "direction": "increases"})

    # Absent key is a different claim from a null value: no value was scored at all, so
    # there is nothing to report.
    assert "value" not in presented


# --- the guard that decides whether the number reaches the reader ---------------

def _calls(*, value=15.5):
    return [{
        "name": "predict_readmission",
        "args": {"hadm_id": 90000012},
        "response": {
            "probability": 0.34, "threshold": 0.11,
            "top_factors": [{"feature": "rdw_max", "direction": "increases",
                             "value": value}],
        },
    }]


def test_a_served_percentage_reaches_the_reader_intact():
    """RDW is reported in percent, so the value is written as one.

    The guard compares a percentage token against the score as a decimal, so 15.5% became
    0.155 against a probability of 0.34 and was stripped — the answer served as "her RDW
    of". A value the turn was served is supported in its own units.
    """
    answer = "Her red cell distribution width peaked at 15.5%, which the estimate weighs "
    out, flags = verify_risk_numbers(answer + "upward.", _calls())

    assert out == answer + "upward."
    assert flags == []


def test_a_served_decimal_reaches_the_reader_intact():
    """A monocyte count of 0.21 is risk-shaped and is not a risk claim."""
    calls = [{
        "name": "predict_readmission", "args": {"hadm_id": 90000012},
        "response": {"probability": 0.34, "threshold": 0.11,
                     "top_factors": [{"feature": "monocytes_min",
                                      "direction": "decreases", "value": 0.21}]},
    }]

    out, flags = verify_risk_numbers("Her lowest monocyte count was 0.21.", calls)

    assert out == "Her lowest monocyte count was 0.21."
    assert flags == []


def test_a_number_the_payload_never_stated_is_still_stripped():
    """The rule this must not weaken: an unsupported risk number is removed."""
    out, flags = verify_risk_numbers("Her RDW of 18.4% predicts readmission.", _calls())

    assert "18.4" not in out
    assert flags == ["risk_number_unsupported:18.4%"]


def test_a_served_value_is_written_at_the_published_precision():
    """The precision rule now has a value to act on, and it writes the payload's digits.

    `_factor_values` reads `top_factors[].value`; the predictor did not return one, so
    this pass had been dead in production since it was written. It rounds to the two
    decimals the risk result publishes, not to one: a rewritten sentence has to agree
    with the number beside it in the table.
    """
    out, flags = normalise_factor_values("She had 23.5417 prior inpatient days.",
                                         _calls(value=23.5417))

    assert "23.54 prior inpatient days" in out
    assert flags == ["factor_value_rounded:23.5417"]


# --- the turn ------------------------------------------------------------------

def test_the_question_the_widget_asks_binds_the_value_and_the_series():
    """The label the bar sends names the analyte, so the drill-down keeps its tools."""
    question = ("What in this patient's chart explains red cell distribution width "
                "(RDW) as a readmission risk driver?")
    lab_driver = __import__(
        "services.agent.routing.routing", fromlist=["asks_about_a_lab_analyte"]
    ).asks_about_a_lab_analyte(question)
    allowed = graph._freetext_allowed(True, False, lab_driver)

    assert lab_driver is True
    assert "predict_readmission" in allowed
    assert "get_lab_trends" in allowed
    assert "rag_search" in allowed


# --- the instructions ----------------------------------------------------------

def _prompt() -> str:
    """The system prompt with its line wrapping removed.

    Asserted this way rather than against the raw constant because a prompt is prose: its
    lines are wrapped for the file, and a rule that survives re-wrapping is the rule worth
    pinning. (The charts that assert exact substrings do so for phrases short enough to
    sit on one line; these are sentences.)
    """
    return " ".join(prompts.SYSTEM_PROMPT.split())


def test_the_prompt_states_the_payloads_value_as_the_factor_s_own_number():
    text = _prompt()

    assert "top_factors[].value" in text
    assert "`<admission_metadata>` states, EXACTLY as written" in text


def test_the_prompt_stops_the_note_search_for_the_feature_name():
    """A discharge summary does not write "red cell distribution width".

    2026-11-20: the rule used to name the warehouse key (`monocytes_min`) as an example, and
    the model handed that string to `get_lab_trends` as an analyte. The instruction now names
    the panel's own analytes and forbids the proxy search instead.
    """
    text = " ".join(_prompt().split())

    assert "Never pass an anchor key" in text
    assert "a haemoglobin and a haematocrit are not a statement about the red-cell" in text
    assert "is not the chart's account of the correlate" not in text


def test_the_prompt_keeps_the_structured_value_and_the_note_apart():
    text = _prompt()

    assert "Where they document nothing" in text
    assert "STOP after part one" in text
