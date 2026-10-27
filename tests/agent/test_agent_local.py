"""Tier 2: does the agent use the tool, and does it stay out of a command's way?

Tier 1 (test_mcp_stdio.py) proves the tool returns the right number. These tests
prove the agent does not quietly replace it — and, since UX Phase 1B, that it does
not reach for a tool that a command owns.

The boundary moved, so these tests moved with it. A risk score and a discharge
medication list used to be the reasoning loop's work: the model chose the tool and
reported the number, and the failure being guarded against was a confident,
plausible answer with no tool call at all. Both answers are now produced by a
command that calls its tool once, deterministically, and both tools are absent from
the loop. So the free-text path is checked for the redirect it owes, and the
tool-call assertions are made on the command path that owns them. The failure they
guard against is unchanged: a number nobody asked the service for.

Runs against either transport, so the same assertions cover the deployed
service:

    pytest tests/test_agent_local.py                       # stdio (default)
    MCP_TRANSPORT=http MCP_URL=https://... pytest tests/test_agent_local.py

Each conversation is a real LLM round trip, so the fixtures are module-scoped
and every test reads from one of the runs below.
"""

import asyncio
import json
import re
import sys
from pathlib import Path

import pytest

HARNESS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS_ROOT))

from services.agent.engine.graph import ask, final_text  # noqa: E402

# "Local" in this filename means the MCP transport, not the model: every test here
# drives `ask` through the real reasoning loop, so it needs a project with the Agent
# Platform API and credentials. A build container has neither and all eight tests
# fail with `ClientError: 403 PERMISSION_DENIED` for `unit-test-project`.
#
# Marked live rather than deleted, and worth noting as the gap it is: these are the
# free-text boundary assertions, and CI no longer exercises them. Making them
# hermetic means substituting the chat model the way `test_model_bounds.py` already
# does through `graph._build_llm` — a change of its own, not a CI fix.
pytestmark = pytest.mark.live
from services.agent.engine.mcp_client import toolbox  # noqa: E402

FIXTURE = json.loads((Path(__file__).resolve().parent / "fixtures" / "expected.json").read_text())
# The anchor is declared by the fixture rather than repeated here, so re-anchoring
# is one edit in the file that holds the values — and the suite cannot keep
# selecting an admission the served feature source no longer holds, which is what
# sorting the patient keys used to do (2026-10-12).
KNOWN_HADM_ID = int(FIXTURE["anchor_hadm_id"])
UNKNOWN_HADM_ID = 1
EXPECTED = FIXTURE["patients"][str(KNOWN_HADM_ID)]


def _run(question: str, *, kind: str | None = None, hadm_id: int | None = None) -> dict:
    """One conversation. `kind` is the command chip, when the caller chose one.

    The keyword shape mirrors the route's own call (`http._run_chain`), so these
    tests drive the deployed path rather than a convenient subset of it.
    """
    async def go():
        async with toolbox() as box:
            return await ask(box, question, question_kind=kind, hadm_id=hadm_id)

    return asyncio.run(go())


# Free text and the command, one run each: the redirect tests need a question the
# loop answers by naming a command, and the call tests need the command that runs
# the tool.
@pytest.fixture(scope="module")
def free_text_risk() -> dict:
    return _run(f"What is the readmission risk for admission {KNOWN_HADM_ID}?")


@pytest.fixture(scope="module")
def free_text_meds() -> dict:
    return _run(f"What medications is admission {KNOWN_HADM_ID} going home on?")


@pytest.fixture(scope="module")
def risk_command() -> dict:
    """The command that reaches the predictor: `/ehr`, since `/risk` was retired on
    2026-10-26. The estimate is its supplementary call, drawn beside the summary."""
    return _run("/ehr", kind="summarize", hadm_id=KNOWN_HADM_ID)


@pytest.fixture(scope="module")
def unknown_patient() -> dict:
    return _run("/ehr", kind="summarize", hadm_id=UNKNOWN_HADM_ID)


# --- the free-text boundary: a command's work is redirected, never performed ---

def test_a_free_text_risk_question_enters_the_reasoning_loop(free_text_risk):
    """Phase 01: a typed risk question is a question, not a command.

    Before the refactor it was rewritten into `/risk` and answered with the command's
    static caption — which ignores `turns`, so the same question asked as a follow-up
    got Turn 1's answer back. Now the loop receives it and decides for itself.

    What it must not do is answer from memory, and since 2026-10-22 there are two ways
    it can honour that, because the model changed which one it picks when the literature
    tool stopped being bound on every turn: it retrieves, or it names the command that
    owns the score and stops. Both are correct — `predict_readmission` is deliberately
    outside the loop, so a redirect is the honest answer to "what is her risk" — and the
    failure this guards is unchanged: a risk FIGURE with nothing behind it. So a
    redirect is accepted only when it carries no number.
    """
    assert "command" not in free_text_risk, "free text must not become a command"
    if free_text_risk["tool_calls"]:
        assert final_text(free_text_risk).strip()
        return
    text = final_text(free_text_risk)
    assert "/ehr" in text, "the loop neither looked anything up nor named /ehr"
    assert not re.search(r"\d", text), (
        "a redirect must not carry a figure: the score belongs to the /ehr turn"
    )


def test_a_free_text_medication_question_enters_the_reasoning_loop(free_text_meds):
    """Same boundary, other command — and now the medication tool is the loop's.

    It used to be withheld, so the only way to answer was to redirect to /meds. A
    follow-up that asked for a classification got the count again. The loop may call
    `discharge_medications` itself, and the guardrail still verifies the doses.
    """
    assert "command" not in free_text_meds, "free text must not become a command"
    assert any(call["name"] == "discharge_medications"
               for call in free_text_meds["tool_calls"]), free_text_meds["tool_calls"]


# --- the path that does own the tool ------------------------------------------

def test_the_risk_command_calls_the_tool(risk_command):
    """The whole point: the number comes from the service, not from memory."""
    calls = [call for call in risk_command["tool_calls"]
             if call["name"] == "predict_readmission"]
    assert calls, "the /ehr turn drew an estimate without calling the predictor"
    assert calls[0]["args"]["hadm_id"] == KNOWN_HADM_ID


def test_tool_result_matches_the_fixture(risk_command, require_served):
    """Same number the endpoint and the MCP server produce."""
    response = require_served(risk_command)
    assert abs(response["probability"] - EXPECTED["probability"]) <= EXPECTED["tolerance"]


def _risk_widget(state):
    """The widget the browser would draw for this state.

    Composed through `compose_presentation`, the same call the payload path makes, so
    these assert on what the surface actually carries rather than on a restatement of
    it written here.
    """
    from services.agent.presentation.a2ui import compose_presentation

    question = next(
        (m.content for m in state["messages"] if getattr(m, "type", "") == "human"), "")
    presentation = compose_presentation(question, "", state["tool_calls"])
    components = presentation["a2ui"]["messages"][1]["updateComponents"]["components"]
    return next(c for c in components if c["component"] == "RiskInfluencers")


def test_the_score_reaches_the_user_through_the_surface(risk_command, require_served):
    """The number has to be in the surface, and at the surface's own precision.

    `/ehr` writes prose as well, so this is no longer the only channel — but it is the
    channel a reader sees beside the narrative, and it must agree with the served value.
    Asserted against the value the service returned rather than against the fixture,
    because the rule is about the projection: this is the assertion that would have
    caught the projection handing the model a pre-rounded probability, which is how
    the prose came to say 33.0% beside a widget drawing 33.2%.
    """
    response = require_served(risk_command)

    assert _risk_widget(risk_command)["probability"] == response["probability"]


def test_the_widget_carries_the_threshold_and_the_decision(risk_command, require_served):
    response = require_served(risk_command)
    widget = _risk_widget(risk_command)

    assert widget["threshold"] == response["threshold"]
    # The model's own call travels with the number, so the widget's comparison can
    # never overrule it.
    assert widget["decision"] == response["decision"]


def test_answer_invents_no_risk_factors(risk_command, require_served):
    """Guardrail: attribute only from top_factors.

    Checked against the model's own feature vocabulary rather than a hand-list, so a
    hallucinated-but-plausible feature name fails here. Read off the widget, because
    the widget is where the factors are named now — there is no prose naming them.
    """
    response = require_served(risk_command)

    from services.agent.presentation.domain import label_for

    reported = {label_for(str(factor["feature"])) for factor in response["top_factors"]}
    shown = {factor["label"] for factor in _risk_widget(risk_command)["factors"]}
    assert shown <= reported, (
        f"the widget attributes factors the model did not: {sorted(shown - reported)}"
    )


def test_unknown_patient_is_reported_not_invented(unknown_patient):
    """Guardrail: never substitute a plausible number for a failed call.

    The call fails with the tool's own error, and what has to hold is that the surface
    draws no estimate from it — `/ehr` still writes its summary prose, so the canvas is
    no longer the only channel and the assertion is about what it must NOT carry.
    """
    calls = [call for call in unknown_patient["tool_calls"]
             if call["name"] == "predict_readmission"]
    assert calls, "the /ehr turn answered about an unknown admission without calling the tool"
    assert calls[0]["response"].get("error") == "unknown_patient"

    from services.agent.presentation.a2ui import compose_presentation

    question = next(
        (m.content for m in unknown_patient["messages"]
         if getattr(m, "type", "") == "human"), "")
    presentation = compose_presentation(question, "", calls)
    components = presentation["a2ui"]["messages"][1]["updateComponents"]["components"]

    assert "RiskInfluencers" not in {c["component"] for c in components}, (
        "a failed call must not draw an estimate"
    )
    assert str(EXPECTED["probability"]) not in presentation["a2ui"]["fallback_text"]
    assert presentation["a2ui"]["fallback_text"].strip(), "the canvas must say what happened"
