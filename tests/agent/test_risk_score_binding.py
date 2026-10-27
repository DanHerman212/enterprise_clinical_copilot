"""A question asking for the score gets the score, not a command name.

Gate 2, `dev_t5_03` (`pilot-verify-20261027-190255`): a cold-start thread asked "What is
his 30-day readmission risk score and is it above threshold?" and was answered
"… are provided by the /ehr command." `predict_readmission` was bound only on driver
questions, so the model had nothing to call and named a command instead.

Two halves are pinned here: the wording predicate that recognises the question, and the
gate that binds the tool because of it.
"""

from __future__ import annotations

import pytest

from services.agent.engine.graph import _freetext_allowed
from services.agent.routing import routing

SCORE_QUESTIONS = (
    "What is his 30-day readmission risk score and is it above threshold?",
    "What is the readmission probability for this admission?",
    "Is her risk above the cutoff?",
    "What percentage readmission risk does the model give her?",
)

NOT_SCORE_QUESTIONS = (
    "Why does the oncology flag drive his readmission risk?",
    "What in this patient's record explains the oncology flag driving his risk?",
    "What blood thinners were prescribed at discharge?",
    "What was the length of stay?",
    "Summarise the discharge note.",
)


@pytest.mark.parametrize("question", SCORE_QUESTIONS)
def test_a_question_for_the_estimate_is_recognised(question):
    assert routing.asks_for_risk_score(question) is True


@pytest.mark.parametrize("question", NOT_SCORE_QUESTIONS)
def test_a_driver_or_chart_question_is_not(question):
    """A driver question binds the same tool for its own reason; two predicates admitting
    one call for different reasons would make the binding unauditable."""
    assert routing.asks_for_risk_score(question) is False


def test_the_gate_binds_the_estimate_on_a_score_question():
    allowed = _freetext_allowed(risk_driver_question=False, wants_literature=False,
                                risk_score_question=True)
    assert "predict_readmission" in allowed


def test_the_gate_still_withholds_it_from_an_ordinary_chart_question():
    allowed = _freetext_allowed(risk_driver_question=False, wants_literature=False)
    assert "predict_readmission" not in allowed


def test_a_declared_follow_up_does_not_re_bind_it():
    """A chip is offered only when its evidence is already in the turn that offered it,
    so the replay holds the estimate and a second call can only cost the answer."""
    allowed = _freetext_allowed(risk_driver_question=False, wants_literature=False,
                                declared_follow_up=True, risk_score_question=True)
    assert "predict_readmission" not in allowed
