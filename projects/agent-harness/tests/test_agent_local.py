"""Tier 2: does the agent use the tool, and report it faithfully?

Tier 1 (test_mcp_stdio.py) proves the tool returns the right number. These
tests prove the agent does not quietly replace it. The specific failure being
guarded against: Gemini knows what readmission risk is and can produce a
confident, plausible answer with no tool call at all.

Each conversation is a real LLM round trip, so the fixtures are module-scoped
and every test reads from one of two runs.
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest

HARNESS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_ROOT))

from agent.graph import ask, final_text  # noqa: E402
from agent.mcp_client import stdio_toolbox  # noqa: E402

FIXTURE = json.loads((HARNESS_ROOT / "tests" / "fixtures" / "expected.json").read_text())
KNOWN_HADM_ID = 20924467
UNKNOWN_HADM_ID = 1
EXPECTED = FIXTURE["patients"][str(KNOWN_HADM_ID)]


def _run(question: str) -> dict:
    async def go():
        async with stdio_toolbox(python=sys.executable) as toolbox:
            return await ask(toolbox, question)

    return asyncio.run(go())


@pytest.fixture(scope="module")
def known_patient() -> dict:
    return _run(f"What is the readmission risk for admission {KNOWN_HADM_ID}?")


@pytest.fixture(scope="module")
def unknown_patient() -> dict:
    return _run(f"What is the readmission risk for admission {UNKNOWN_HADM_ID}?")


def test_agent_calls_the_tool(known_patient):
    """The whole point: it must not answer from its own knowledge."""
    calls = known_patient["tool_calls"]
    assert calls, "agent answered without calling any tool"
    assert calls[0]["name"] == "predict_readmission"
    assert calls[0]["args"]["hadm_id"] == KNOWN_HADM_ID


def test_tool_result_matches_the_fixture(known_patient):
    """Same number the endpoint and the MCP server produce."""
    response = known_patient["tool_calls"][0]["response"]
    assert abs(response["probability"] - EXPECTED["probability"]) <= EXPECTED["tolerance"]


def test_answer_states_the_exact_probability(known_patient):
    """Guardrail: report it exactly, do not round or restate."""
    text = final_text(known_patient)
    assert str(EXPECTED["probability"]) in text, text


def test_answer_gives_the_threshold_and_decision(known_patient):
    text = final_text(known_patient).lower()
    assert str(FIXTURE["threshold"]) in text, text
    assert "risk" in text


def test_answer_invents_no_risk_factors(known_patient):
    """Guardrail: attribute only from top_factors.

    Checked against the model's own feature vocabulary rather than a hand-list,
    so a hallucinated-but-plausible feature name fails here.
    """
    response = known_patient["tool_calls"][0]["response"]
    reported = {factor["feature"] for factor in response["top_factors"]}
    text = final_text(known_patient).lower()

    from mcp_server.features.manifest import groups

    absent = set(groups()) - reported
    leaked = [name for name in absent if name.lower() in text]
    assert not leaked, f"agent cited features absent from top_factors: {leaked}"


def test_unknown_patient_is_reported_not_invented(unknown_patient):
    """Guardrail: never substitute a plausible number for a failed call."""
    calls = unknown_patient["tool_calls"]
    assert calls, "agent answered without calling any tool"
    assert calls[0]["response"].get("error") == "unknown_patient"

    text = final_text(unknown_patient).lower()
    assert str(EXPECTED["probability"]) not in text
    assert any(word in text for word in ("not", "no ", "unable", "unknown", "could not"))
