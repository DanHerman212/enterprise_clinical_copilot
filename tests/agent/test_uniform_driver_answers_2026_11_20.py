"""One risk-driver answer, whichever control asked for it (2026-11-20).

Two defects met in the same place. The risk card's driver bar asked "What in this patient's
chart explains {label} as a readmission risk driver?" while the follow-up chip under an
answer asked "Why does {label} increase this patient's readmission risk?" — same evidence,
two questions, and only the bar's was answerable, because the prompt forbids explaining the
model's mechanism. And the value the answer has to state was missing for exactly the
features the manifest holds: `gender` reached the model as `1`, and the four one-hot parents
(`admission_type`, `discharge_location`, `insurance`, `race`) reached it with no value at
all, because `predict_readmission` has no single column to read for them.

Three repairs, one per section below: the value, the question, and the rule that turns both
into one sentence.
"""

from __future__ import annotations

import pytest

from services.agent.engine import prompts
from services.agent.engine.mcp_client import _presented_factor, _presented_for_model
from services.agent.postprocess.evidence_state import SessionEvidenceState
from services.agent.postprocess.guardrail import (
    normalise_factor_values,
    verify_risk_numbers,
)
from services.agent.postprocess import followups
from services.agent.routing import admission_context, routing
from shared.labels import FEATURE_LABELS, driver_question, label_is_plural


# ---------------------------------------------------------------------------------------
# 1. The value a driver answer needs is in the record, in words
# ---------------------------------------------------------------------------------------


def _block(**fields: str) -> dict[str, str]:
    """The decoded values a rendered block states, keyed by feature and by label."""
    return admission_context.administrative_values(
        admission_context.render(90000020, dict(fields)))


def test_sex_reaches_the_model_as_a_word_rather_than_a_digit():
    """`gender` is `CAST(gender = 'M' AS INT64)` in the mart; the block says Male."""
    presented = _presented_factor(
        {"feature": "gender", "direction": "decreases", "value": 1},
        _block(gender="Male"))

    assert presented["feature"] == "sex"
    assert presented["value"] == "Male"


def test_sex_is_decoded_without_a_block():
    """1 is male and 0 is female — the convention the decoder documents, applied here."""
    male = _presented_factor({"feature": "gender", "direction": "increases", "value": 1})
    female = _presented_factor({"feature": "gender", "direction": "increases", "value": 0})

    assert male["value"] == "Male"
    assert female["value"] == "Female"


@pytest.mark.parametrize("feature, stored, stated", [
    # The decoder hands back the warehouse's own code beside the words; the block states the
    # words, so the sentence reads "(patient value: Emergency)" rather than nesting a
    # parenthetical inside one (2026-11-20, 90000015).
    ("admission_type", "Emergency (EW EMER.)", "Emergency"),
    ("discharge_location", "Home", "Home"),
    ("insurance", "Medicare", "Medicare"),
    ("race", "White", "White"),
])
def test_a_one_hot_parent_carries_the_level_the_block_states(feature, stored, stated):
    """The payload omits these values; the block already holds the decoded level."""
    fields = {feature: stored}
    presented = _presented_factor(
        {"feature": feature, "direction": "increases"}, _block(**fields))

    assert presented["value"] == stated


def test_a_parent_the_block_calls_not_recorded_carries_no_value():
    """"Not recorded" is not a value; the factor keeps its direction and stops."""
    presented = _presented_factor(
        {"feature": "insurance", "direction": "increases"}, _block(insurance="Not recorded"))

    assert "value" not in presented


def test_the_session_record_states_the_same_values():
    block = admission_context.render(
        90000020, {"admission_type": "Emergency (EW EMER.)", "gender": "Female"})
    state = SessionEvidenceState(hadm_id=90000020)
    state.ingest_tool_output("predict_readmission", {"hadm_id": 90000020}, {
        "probability_display": "8.5%", "threshold_display": "11.0%", "decision": 0,
        "top_factors": [
            {"feature": "admission type", "direction": "increases"},
            {"feature": "sex", "direction": "decreases", "value": "Female"},
            {"feature": "insurance type", "direction": "increases"},
        ],
    })

    rendered = state.render_for_prompt(admission_context.administrative_values(block))

    assert "feature=admission type | direction=increases | value=Emergency" in rendered
    assert "feature=sex | direction=decreases | value=Female" in rendered
    # Nothing is invented for a factor the block does not state.
    assert "feature=insurance type | direction=increases" in rendered


def test_a_driver_answer_that_quotes_the_level_raises_no_guardrail_flag():
    """A word is not a risk-shaped number, so neither guard has anything to rewrite."""
    calls = [{
        "name": "predict_readmission", "args": {"hadm_id": 90000020},
        "response": {"probability": 0.085, "threshold": 0.11,
                     "top_factors": [{"feature": "admission_type",
                                      "direction": "increases"}]},
    }]
    answer = ("According to the readmission risk model, admission type "
              "(patient value: Emergency (EW EMER.)) increases this patient's 30-day "
              "readmission risk.")

    stripped, risk_flags = verify_risk_numbers(answer, calls, "")
    rewritten, factor_flags = normalise_factor_values(stripped, calls)

    assert risk_flags == []
    assert factor_flags == []
    assert rewritten == answer


# ---------------------------------------------------------------------------------------
# 2. One question, and it agrees with the label
# ---------------------------------------------------------------------------------------


def test_the_chip_and_the_bar_ask_the_same_question():
    """`risk_influencers.js` mirrors this helper; one string, two controls."""
    assert driver_question("oncology history") == (
        "How does oncology history affect this patient's readmission risk?")


@pytest.mark.parametrize("label", ["procedures", "prior inpatient days",
                                   "prior admissions", "recent ED visits"])
def test_a_plural_label_takes_a_plural_verb(label):
    assert label_is_plural(label)
    assert driver_question(label).startswith(f"How do {label} affect")


@pytest.mark.parametrize("label", ["oncology history", "length of stay",
                                   "red cell distribution width (RDW)",
                                   "having a procedure", "insurance type"])
def test_a_singular_label_takes_a_singular_verb(label):
    assert not label_is_plural(label)
    assert driver_question(label).startswith(f"How does {label} affect")


def test_every_curated_label_produces_a_grammatical_question():
    """The four plurals and the participle are why the set is listed, not guessed."""
    for label in FEATURE_LABELS.values():
        question = driver_question(label)
        verb = "do" if label_is_plural(label) else "does"
        assert question == (
            f"How {verb} {label} affect this patient's readmission risk?"), label


def test_the_chip_is_the_shared_question_filled_with_the_payloads_driver():
    state = {
        "tool_calls": [{"name": "predict_readmission", "args": {}, "response": {
            "probability_display": "8.5%", "threshold_display": "11.0%",
            "top_factors": [{"feature": "prior_inpatient_days", "direction": "decreases"}],
        }}],
        "command": "summarize",
    }

    chips = followups.suggest("/ehr", "An answer.", state=state)

    assert chips[0] == (
        "How do prior inpatient days affect this patient's readmission risk?")


# ---------------------------------------------------------------------------------------
# 3. The unified question routes as a driver question, declared or typed
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("question", [
    "How does oncology history affect this patient's readmission risk?",
    "How do prior inpatient days affect this patient's readmission risk?",
    "How does having a procedure affect this patient's readmission risk?",
])
@pytest.mark.parametrize("intent", ["follow_up", "risk_driver"])
def test_the_unified_question_is_the_driver_question(question, intent):
    assert routing.wants_risk_driver(question, intent) is True


@pytest.mark.parametrize("question", [
    "Why does oncology history increase this patient's readmission risk?",
    "Why does her RDW drive readmission risk?",
    "What in this patient's chart explains oncology history as a readmission risk driver?",
])
def test_the_older_wordings_still_route(question):
    """A stored thread replays a question, it does not re-compose one."""
    assert routing.wants_risk_driver(question) is True


def test_a_question_about_a_dose_is_not_a_driver_question():
    assert routing.wants_risk_driver("What dose of metoprolol was she on?") is False


# ---------------------------------------------------------------------------------------
# 4. The risk rules attach to risk turns only
# ---------------------------------------------------------------------------------------

_MED_TOOLS = frozenset({"rag_search", "discharge_medications"})


def test_a_medication_follow_up_does_not_carry_the_risk_rules():
    """`has_prediction` alone used to attach the spoke to every later turn."""
    composed = prompts.compose_system_prompt(
        allowed_tools=_MED_TOOLS,
        has_prediction=True,
        domains=frozenset({routing.DOMAIN_MEDICATIONS}),
    )

    assert "READMISSION RISK & DRIVER ATTRIBUTION" not in composed


def test_a_risk_turn_still_carries_them():
    composed = prompts.compose_system_prompt(
        allowed_tools=frozenset({"rag_search"}),
        risk_driver_question=True,
        has_prediction=True,
        domains=frozenset({routing.DOMAIN_RISK}),
    )

    assert "READMISSION RISK & DRIVER ATTRIBUTION" in composed


def test_an_unclassified_turn_keeps_them():
    """No domain named is not a domain excluded: the standing set is unchanged."""
    composed = prompts.compose_system_prompt(
        allowed_tools=frozenset({"rag_search"}), has_prediction=True, domains=frozenset())

    assert "READMISSION RISK & DRIVER ATTRIBUTION" in composed
