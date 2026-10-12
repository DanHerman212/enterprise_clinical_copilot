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

Expected values come from tests/fixtures/expected.json and are compared within
its tolerance. Nothing here hardcodes a probability: retraining rewrites the
fixture (section 20), and a hardcoded number would turn every post-retrain run
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

from services.agent.graph import ask, final_text  # noqa: E402
from services.agent.mcp_client import toolbox  # noqa: E402

FIXTURE = json.loads((Path(__file__).resolve().parent / "fixtures" / "expected.json").read_text())
KNOWN_HADM_ID = 20924467
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
    """One real round trip through Gemini, MCP, and the endpoint."""
    return _run("/risk", kind="risk", hadm_id=KNOWN_HADM_ID)


@pytest.fixture(scope="module")
def unknown_patient() -> dict:
    return _run("/risk", kind="risk", hadm_id=UNKNOWN_HADM_ID)


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

    # The number reaches the user at display precision, not only the tool. The
    # narrative used to state the raw value while the widget drew two decimals,
    # which is one score stated two ways — two numbers, as far as a reader is
    # concerned.
    text = final_text(known_patient)
    assert f"{EXPECTED['probability']:.2f}" in text, text
    assert str(EXPECTED["probability"]) not in text, text


def test_graceful_error(unknown_patient):
    """Criterion 2 — structured error, no crash, no invented number."""
    calls = unknown_patient["tool_calls"]
    assert calls, "agent answered about an unknown admission without calling the tool"

    response = calls[0]["response"]
    validate(instance=response, schema=ERROR_SCHEMA)
    assert response["error"] == "unknown_patient"
    assert "probability" not in response

    text = final_text(unknown_patient).lower()
    assert str(EXPECTED["probability"]) not in text
    assert any(w in text for w in ("not", "no ", "unable", "unknown", "could not")), text


def test_routing_went_through_mcp(known_patient):
    """Criterion 3 — assert on the trace, because the prose cannot prove it.

    A model that knew what readmission risk is could produce a confident, plausible
    answer with no tool call at all, and a grader reading only the answer would pass
    that. The command path is held to the standard the loop was held to: one call,
    the right admission, no error.
    """
    calls = known_patient["tool_calls"]
    assert calls, "the answer came from the model's own knowledge — no tool call in the trace"
    assert calls[0]["name"] == "predict_readmission"
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
