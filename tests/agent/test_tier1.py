"""Tier 1 acceptance tests — the Phase 2 exit criterion.

Four criteria, one per test:

1. Known-good value   — the expected probability, decision, and non-empty factors,
                        and the narrative that carries the score at the two
                        decimals the interface draws it with
2. Graceful error     — an invalid hadm_id gives a structured error, not a crash
                        and not an invented number
3. Routing            — the call actually went through MCP, asserted on the
                        tool-call trace rather than on the prose
4. Schema contract    — the response validates against the section 6 shape

Measured on the /risk command, which is the path that owns the tool and the number
(UX Phase 1). A free-text risk question is answered by naming that command and
calling nothing, which is a boundary of its own and lives in `test_agent_local.py`.

Transport-agnostic, so the same suite gates the deployed service:

    pytest tests/test_tier1.py -v
    MCP_TRANSPORT=http MCP_URL=https://... pytest tests/test_tier1.py -v

Expected values come from tests/agent/fixtures/expected.json and are compared
within its tolerance. Nothing here hardcodes a probability: retraining rewrites
the fixture (section 20), and a hardcoded number would turn every post-retrain run
into what looks like an integration bug.
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest
from jsonschema import validate

HARNESS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS_ROOT))

from services.agent.engine.graph import ask, final_text  # noqa: E402
from services.agent.engine.mcp_client import toolbox  # noqa: E402

# This file gates the DEPLOYED service: `ask` drives the graph into the real MCP
# toolbox (`MCP_TRANSPORT=http MCP_URL=https://…`), so every test here needs a live
# endpoint and a project with the Agent Platform API enabled. In a build container
# it fails with `GooglePermissionDenied`, which is a statement about the container
# and not about the candidate revision.
#
# Marked rather than deleted: it is the only suite that checks the acceptance
# criteria against a running deployment, and Phase 7 moves the module it imports.
# `-m "not live"` is what CI runs; the marker is registered in pytest.ini.
pytestmark = pytest.mark.live

FIXTURE = json.loads((Path(__file__).resolve().parent / "fixtures" / "expected.json").read_text())
# Declared by the fixture, not repeated here: see `test_agent_local.py`.
KNOWN_HADM_ID = int(FIXTURE["anchor_hadm_id"])
UNKNOWN_HADM_ID = 1
EXPECTED = FIXTURE["patients"][str(KNOWN_HADM_ID)]

# The section 6 return shape, written out here rather than read from the
# server's advertised output_schema. The SDK derives that schema from the
# `-> dict` annotation, so it advertises {"type": "object",
# "additionalProperties": true} — which accepts literally anything. Validating
# against it would be a test that cannot fail.
#
# `additionalProperties: false` is the point of the test: a field silently
# added or renamed breaks the agent, the fixture, and A2UI, and should break
# here first.
RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "hadm_id",
        "probability",
        "threshold",
        "decision",
        "base_value",
        "top_factors",
        "model_version",
        "feature_source",
    ],
    "properties": {
        "hadm_id": {"type": "integer"},
        "probability": {"type": "number", "minimum": 0, "maximum": 1},
        "threshold": {"type": "number", "minimum": 0, "maximum": 1},
        "decision": {"type": "integer", "enum": [0, 1]},
        "base_value": {"type": "number"},
        "top_factors": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["feature", "contribution", "direction"],
                "properties": {
                    "feature": {"type": "string"},
                    "contribution": {"type": "number"},
                    "direction": {"enum": ["increases", "decreases"]},
                },
            },
        },
        "model_version": {"type": "string"},
        "feature_source": {"type": "string", "enum": ["bigquery", "feature_store"]},
    },
}

ERROR_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["hadm_id", "error", "message", "feature_source"],
    "properties": {
        "hadm_id": {"type": "integer"},
        "error": {
            "enum": [
                "unknown_patient",
                "incomplete_features",
                "feature_fetch_failed",
                "prediction_failed",
            ]
        },
        "message": {"type": "string"},
        "feature_source": {"type": "string"},
    },
}


def _run(question: str, *, kind: str | None = None, hadm_id: int | None = None) -> dict:
    """One conversation, driven the way the route drives it (`http._run_chain`)."""
    async def go():
        async with toolbox() as box:
            return await ask(box, question, question_kind=kind, hadm_id=hadm_id)

    return asyncio.run(go())


@pytest.fixture(scope="module")
def known_patient() -> dict:
    """One real round trip through Gemini, MCP, and the endpoint.

    Driven as `/ehr`, which is the command that reaches the predictor now that `/risk`
    is retired: the estimate is its supplementary call, carried beside the summary that
    explains it.
    """
    return _run("/ehr", kind="summarize", hadm_id=KNOWN_HADM_ID)


@pytest.fixture(scope="module")
def unknown_patient() -> dict:
    return _run("/ehr", kind="summarize", hadm_id=UNKNOWN_HADM_ID)


def _presentation(state):
    """The browser contract for a state, composed the way the payload path composes it."""
    from services.agent.presentation.a2ui import compose_presentation

    question = next(
        (m.content for m in state["messages"] if getattr(m, "type", "") == "human"), "")
    return compose_presentation(question, "", state["tool_calls"])


def _risk_widget(state):
    components = _presentation(state)["a2ui"]["messages"][1]["updateComponents"]["components"]
    return next(c for c in components if c["component"] == "RiskInfluencers")


def test_known_good_value(known_patient, require_served):
    """Criterion 1 — the number, the decision, and real attribution."""
    response = require_served(known_patient)

    assert "error" not in response, response
    assert abs(response["probability"] - EXPECTED["probability"]) <= EXPECTED["tolerance"], (
        f"got {response['probability']}, fixture says {EXPECTED['probability']} "
        f"(tolerance {EXPECTED['tolerance']}) — regenerate the fixture if the "
        f"model was retrained"
    )
    assert response["decision"] == EXPECTED["decision"]
    assert abs(response["base_value"] - EXPECTED["base_value"]) <= EXPECTED["tolerance"]
    assert response["threshold"] == FIXTURE["threshold"]
    assert response["top_factors"], "top_factors must not be empty"

    # The number reaches the user through the surface, and the surface is composed from
    # the same call: the served value has to BE the widget's value. Asserted against the
    # served response rather than the fixture, because the rule is about the projection.
    widget = _risk_widget(known_patient)
    assert widget["probability"] == response["probability"]
    assert widget["threshold"] == response["threshold"]


def test_graceful_error(unknown_patient):
    """Criterion 2 — structured error, no crash, no invented number.

    Driven through `graph.ask` rather than the HTTP route, so the request reaches the
    tool: the route's unserved-admission guard is a different layer and is asserted where
    it lives. What this criterion is about is that the predictor answers an unknown
    admission with the documented shape instead of a number.
    """
    calls = [call for call in unknown_patient["tool_calls"]
             if call["name"] == "predict_readmission"]
    assert calls, "agent answered about an unknown admission without calling the tool"

    response = calls[0]["response"]
    validate(instance=response, schema=ERROR_SCHEMA)
    assert response["error"] == "unknown_patient"
    assert "probability" not in response

    # The canvas has to explain itself and must not carry an estimate.
    presentation = _presentation(unknown_patient)
    types = {
        c["component"]
        for c in presentation["a2ui"]["messages"][1]["updateComponents"]["components"]
    }
    assert "RiskInfluencers" not in types, "a failed call must not draw an estimate"
    assert str(EXPECTED["probability"]) not in presentation["a2ui"]["fallback_text"]


def test_routing_went_through_mcp(known_patient):
    """Criterion 3 — assert on the trace, because the prose cannot prove it.

    A model that knew what readmission risk is could produce a confident, plausible
    answer with no tool call at all, and a grader reading only the answer would pass
    that. The command path is held to the standard the loop was held to: one call,
    the right admission, no error.
    """
    calls = [call for call in known_patient["tool_calls"]
             if call["name"] == "predict_readmission"]
    assert calls, "the answer came from the model's own knowledge — no prediction in the trace"
    assert calls[0]["args"] == {"hadm_id": KNOWN_HADM_ID}
    assert "error" not in calls[0], calls[0]


def test_schema_contract(known_patient, require_served):
    """Criterion 4 — the section 6 return shape, exactly."""
    response = require_served(known_patient)
    validate(instance=response, schema=RESPONSE_SCHEMA)

    # Provenance is load-bearing: "which model, reading from where?" must be
    # answerable from the payload alone.
    assert response["model_version"] == FIXTURE["model_version"]
    assert response["hadm_id"] == KNOWN_HADM_ID
