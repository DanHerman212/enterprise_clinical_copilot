"""TF-1: `/risk` declines to score a record that was never an admission.

Unit-level companion to the cohort sweep's record-by-record assertion. The predicate is
tested as a truth table and the command as a shape, so a future refactor that keeps the
sweep green by accident still has to keep both facts true.
"""

from __future__ import annotations

import asyncio

import pytest

from services.agent import deterministic, routing
from services.agent.a2ui import compose_presentation

RISK = routing.command_for("risk")
OUTPATIENT = sorted(routing.OUTPATIENT_HADM_IDS)


@pytest.mark.parametrize("hadm_id", OUTPATIENT)
def test_the_predicate_holds_for_every_outpatient(hadm_id):
    assert routing.refuses_readmission_score(RISK, hadm_id) is True


@pytest.mark.parametrize("hadm_id", [90000015, 90000023, 90000106])
def test_the_predicate_does_not_hold_for_an_inpatient(hadm_id):
    assert routing.refuses_readmission_score(RISK, hadm_id) is False


def test_the_predicate_is_about_the_risk_command_only():
    """`/ehr` and `/pubmed` handle the same records their own way — no estimate, but a
    summary still runs. The gate must not turn them into refusals."""
    for wire in ("summarize", "literature", "meds", "labs"):
        command = routing.command_for(wire)
        for hadm_id in OUTPATIENT:
            assert routing.refuses_readmission_score(command, hadm_id) is False, wire


def test_no_admission_is_not_a_refusal():
    """A `/pubmed` search carries no admission and must still run."""
    assert routing.refuses_readmission_score(RISK, None) is False


@pytest.mark.parametrize("hadm_id", OUTPATIENT)
def test_the_command_returns_the_refusal_without_calling_a_tool(hadm_id):
    state = asyncio.run(deterministic.run_command(
        None, RISK, hadm_id, question="What is her readmission risk?",
        model="unused-the-gate-returns-before-any-model-call"))

    assert state["tool_calls"] == []
    assert state["deterministic_answer"] == routing.OUTPATIENT_RISK_REFUSAL
    assert state["command"] == "risk"


@pytest.mark.parametrize("hadm_id", OUTPATIENT)
def test_the_refusal_draws_no_risk_card_and_is_not_verified(hadm_id):
    state = asyncio.run(deterministic.run_command(
        None, RISK, hadm_id, question="What is her readmission risk?",
        model="unused-the-gate-returns-before-any-model-call"))

    out = compose_presentation("What is her readmission risk?",
                               state["deterministic_answer"],
                               state["tool_calls"], None, "risk")
    components = out["a2ui"]["messages"][1]["updateComponents"]["components"]
    types = {component["component"] for component in components}

    assert "RiskInfluencers" not in types
    assert out["verified"] is False
    # A refusal about a visit must not claim the chart either.
    assert out["scope"] is None
