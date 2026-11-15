"""Conversational readmission-risk questions must bind the predictor, without overfitting.

Sprint 04f's `v2_d3_free_01` asked "And how does that leave his readmission risk looking?" —
the register a clinician actually uses — and `asks_for_risk_score` returned False, so
`predict_readmission` was never bound and the answer said the record held no estimate. The
number existed; the question was not recognised.

These cases are the contract now. Positives widen the vocabulary; negatives keep it from
admitting the tool on a question that merely mentions an admission.
"""

from __future__ import annotations

import pytest

from services.agent.engine import graph
from services.agent.routing import routing

#: Asked-for-the-estimate phrasings. Each names a readmission outcome and asks where it
#: stands, without the literal word "score".
POSITIVES = (
    "And how does that leave his readmission risk looking?",
    "What are the chances she bounces back?",
    "where does his 30-day risk stand?",
    "what is his readmission risk score?",
    "is he likely to be readmitted?",
    "does his readmission trajectory look worse than before?",
    "what are the odds she comes back in?",
)

#: Questions that must not admit the predictor. The first four are ordinary chart
#: questions; the fifth names readmission but asks about the published literature, which
#: is the literature predicate's business, not this one's.
NEGATIVES = (
    "What happened during this admission?",
    "How is his pain controlled?",
    "Walk me through the discharge plan.",
    "What are the wound care instructions?",
    "What does the literature say about readmission after ACDF?",
)


@pytest.mark.parametrize("question", POSITIVES)
def test_a_conversational_risk_question_asks_for_the_estimate(question):
    assert routing.asks_for_risk_score(question) is True


@pytest.mark.parametrize("question", NEGATIVES)
def test_an_unrelated_question_does_not_ask_for_the_estimate(question):
    assert routing.asks_for_risk_score(question) is False


def test_the_outlook_words_never_fire_without_a_readmission_anchor():
    """`leaves` and `looking` are common words; the anchor is what makes them a request."""
    for question in ("what leaves him tired?", "how is the wound looking?"):
        assert routing.asks_for_risk_score(question) is False


def test_a_risk_question_binds_the_predictor_into_free_text():
    """The predicate's only consequence: the tool crosses into the free-text set."""
    allowed = graph._freetext_allowed(False, False, risk_score_question=True)
    assert "predict_readmission" in allowed


def test_a_plain_chart_question_does_not_bind_the_predictor():
    allowed = graph._freetext_allowed(False, False)
    assert "predict_readmission" not in allowed


def test_the_driver_predicate_is_excluded_from_this_one():
    """A driver question binds the same tool through its own predicate, so this one
    declines it: two predicates admitting one call for different reasons would leave the
    binding unauditable."""
    question = "Why is prior inpatient days a readmission risk driver for this patient?"
    assert routing.asks_why_risk_driver(question) is True
    assert routing.asks_for_risk_score(question) is False
