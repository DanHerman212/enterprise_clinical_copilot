"""§5.5.2: a risk-driver answer about a count is not a lab turn.

Two defects were measured on `agent-00176-ngw`, both in the same turn: asked why prior
inpatient days drive the score, the loop called `get_lab_trends` — so the turn's evidence
held a panel nobody asked for, and `followups` then offered a lab chip under a
utilization answer — and the prose quoted the served feature as `16.0417 days` wrapped in
the model's own mechanics ("logit-space", "statistical indicator").

Three halves, pinned here: the tool set, the number, and the wording rules the prompt
carries.
"""

from __future__ import annotations

from services.agent.engine import graph
from services.agent.engine import prompts
from services.agent.routing import routing
from services.agent.postprocess.guardrail import normalise_factor_values

NON_LAB_DRIVER = ("Why does prior inpatient days increase this patient's readmission "
                  "risk driver profile?")
LAB_DRIVER = "Why is the lowest hemoglobin a readmission risk driver for this patient?"


def _predict(value: float = 16.0417) -> list[dict]:
    return [{
        "name": "predict_readmission",
        "args": {"hadm_id": 90000078},
        "response": {
            "probability": 0.34, "threshold": 0.11,
            "top_factors": [{"feature": "prior_inpatient_days", "value": value}],
        },
    }]


def test_a_driver_question_about_a_count_cannot_read_the_lab_series():
    allowed = graph._freetext_allowed(True, False, False)

    assert "get_lab_trends" not in allowed
    # The attribution itself is still available: withholding the series must not turn
    # the question back into a values-only answer.
    assert "predict_readmission" in allowed
    assert {"rag_search", "rag_search_sections"} <= allowed


def test_a_driver_question_about_an_analyte_keeps_the_series():
    """The lab drill-down is the reason `get_lab_trends` is admitted to free text."""
    allowed = graph._freetext_allowed(True, False, True)

    assert "get_lab_trends" in allowed


def test_an_ordinary_chart_question_keeps_the_series():
    """Scoping is for driver turns only: a plain lab question still needs the series."""
    assert "get_lab_trends" in graph._freetext_allowed(False, False)


def test_the_analyte_test_reads_both_vocabularies():
    assert routing.asks_about_a_lab_analyte(LAB_DRIVER)
    assert routing.asks_about_a_lab_analyte("why does RDW drive readmission?")
    assert routing.asks_about_a_lab_analyte("explain the red cell distribution width")
    assert not routing.asks_about_a_lab_analyte(NON_LAB_DRIVER)
    assert not routing.asks_about_a_lab_analyte("why is oncology history a driver?")


def test_a_copied_machine_float_is_written_as_a_whole_number():
    out, flags = normalise_factor_values("She had 16.0417 days as an inpatient.",
                                         _predict())

    assert "16 days" in out
    assert "16.0417" not in out
    assert flags == ["factor_value_rounded:16.0417"]


def test_a_truncated_copy_of_the_same_value_is_written_the_same_way():
    out, _ = normalise_factor_values("She had 16.04 days as an inpatient.", _predict())

    assert "16 days" in out


def test_a_lab_value_quoted_from_the_note_keeps_its_precision():
    """Only the served feature is rewritten; a measurement is not the model's arithmetic."""
    out, flags = normalise_factor_values(
        "Creatinine was 1.06 mg/dL and she had 16.0417 inpatient days.", _predict())

    assert "1.06 mg/dL" in out
    assert "16 inpatient days" in out
    assert flags == ["factor_value_rounded:16.0417"]


def test_an_answer_with_no_served_float_is_untouched():
    answer = "She had 16 days as an inpatient."

    assert normalise_factor_values(answer, _predict()) == (answer, [])


def test_the_driver_instructions_require_a_chart_citation():
    assert "cite the passage it comes from with `^[n]`" in prompts.SYSTEM_PROMPT
    assert "an attribution with no `^[n]`" in prompts.SYSTEM_PROMPT.lower()


def test_the_driver_instructions_forbid_the_model_s_own_mechanics():
    for word in ("logit", "coefficient", "statistical indicator"):
        assert word in prompts.SYSTEM_PROMPT
    assert "Never explain the model's own mechanics" in prompts.SYSTEM_PROMPT


def test_the_driver_instructions_state_the_precision_rule():
    assert "not \"16.0417 days\"" in prompts.SYSTEM_PROMPT
