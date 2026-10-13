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
import sys
from pathlib import Path

import pytest

HARNESS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS_ROOT))

from services.agent.graph import ask, final_text  # noqa: E402
from services.agent.mcp_client import toolbox  # noqa: E402

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
    return _run("/risk", kind="risk", hadm_id=KNOWN_HADM_ID)


@pytest.fixture(scope="module")
def unknown_patient() -> dict:
    return _run("/risk", kind="risk", hadm_id=UNKNOWN_HADM_ID)


# --- the free-text boundary: a command's work is redirected, never performed ---

def test_a_free_text_risk_question_is_redirected(free_text_risk):
    """The answer is the command's name, and there is no tool call to argue with.

    Asserted on the trace, because prose cannot prove this: a model that estimated
    a risk out of its own knowledge would also produce a sentence about risk. The
    absence of a call is the guarantee, and it is why the tool is not bound rather
    than merely forbidden — a tool the model cannot call cannot answer.
    """
    assert free_text_risk["tool_calls"] == []
    assert "/risk" in final_text(free_text_risk)


def test_a_free_text_medication_question_is_redirected(free_text_meds):
    """Same boundary, other command. An assembled medication list is exactly what
    a model reading a note passage produces, and exactly what /meds replaces."""
    assert free_text_meds["tool_calls"] == []
    assert "/meds" in final_text(free_text_meds)


# --- the path that does own the tool ------------------------------------------

def test_the_risk_command_calls_the_tool(risk_command):
    """The whole point: the number comes from the service, not from memory."""
    calls = risk_command["tool_calls"]
    assert calls, "/risk answered without calling any tool"
    assert calls[0]["name"] == "predict_readmission"
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
    from services.agent.a2ui import compose_presentation

    question = next(
        (m.content for m in state["messages"] if getattr(m, "type", "") == "human"), "")
    presentation = compose_presentation(question, "", state["tool_calls"])
    components = presentation["a2ui"]["messages"][1]["updateComponents"]["components"]
    return next(c for c in components if c["component"] == "RiskInfluencers")


def test_the_score_reaches_the_user_through_the_surface(risk_command, require_served):
    """`/risk` returns the widget and no prose at all (UX ruling 2026-10-13).

    The number therefore has to be in the surface, and at the surface's own precision.
    Asserted against the value the service returned rather than against the fixture,
    because the rule is about the projection: this is the assertion that would have
    caught the projection handing the model a pre-rounded probability, which is how
    the prose came to say 33.0% beside a widget drawing 33.2%.
    """
    response = require_served(risk_command)

    assert risk_command["deterministic_answer"] == ""
    assert risk_command["answer_in_surface"] is True
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

    from services.agent.feature_labels import label_for

    reported = {label_for(str(factor["feature"])) for factor in response["top_factors"]}
    shown = {factor["label"] for factor in _risk_widget(risk_command)["factors"]}
    assert shown <= reported, (
        f"the widget attributes factors the model did not: {sorted(shown - reported)}"
    )


def test_unknown_patient_is_reported_not_invented(unknown_patient):
    """Guardrail: never substitute a plausible number for a failed call.

    The refusal is drawn rather than written now. With no prose turn there is no
    sentence to carry it, so what has to hold is that the canvas explains itself and
    carries no estimate at all.
    """
    calls = unknown_patient["tool_calls"]
    assert calls, "/risk answered without calling any tool"
    assert calls[0]["response"].get("error") == "unknown_patient"
    assert unknown_patient["deterministic_answer"] == ""

    from services.agent.a2ui import compose_presentation

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
