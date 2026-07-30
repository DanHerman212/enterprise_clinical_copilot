"""End-to-end check of the MCP server over stdio.

The §6 tests call `predict_readmission` in-process, which skips the entire
protocol layer. This spawns the server as a real subprocess and talks JSON-RPC
to it, so it covers what the in-process tests cannot:

  - the initialize handshake
  - tools/list — the schema and description a model actually sees
  - serialisation of the returned dict into MCP content
  - **stdout staying clean**: under stdio the transport *is* stdout, so a stray
    print() in any dependency corrupts the stream. That regression is invisible
    until a client fails to parse, which is why it is worth a test.

Requires the deployed endpoint. Uses asyncio.run rather than pytest-asyncio to
avoid a plugin dependency; the whole conversation runs once per module.
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters, stdio_client

HARNESS_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = HARNESS_ROOT / "tests" / "fixtures" / "expected.json"

# A cold call does Model.list + GCS manifest + BigQuery + Vertex predict.
CALL_TIMEOUT = 120.0

UNKNOWN_HADM_ID = 1


def _payload(result) -> dict:
    """The tool's JSON, however this SDK version chose to carry it."""
    if getattr(result, "structured_content", None):
        return result.structured_content
    return json.loads(result.content[0].text)


async def _converse() -> dict:
    """One session: list tools, score a known patient, score a bogus one."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server", "--transport", "stdio"],
        # The package dir has a hyphen, so it is not importable by path; the
        # server must be started with the harness root as cwd.
        cwd=str(HARNESS_ROOT),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            tools = await session.list_tools()

            known = await session.call_tool(
                "predict_readmission",
                {"hadm_id": int(_fixture_hadm_id())},
                read_timeout_seconds=CALL_TIMEOUT,
            )
            unknown = await session.call_tool(
                "predict_readmission",
                {"hadm_id": UNKNOWN_HADM_ID},
                read_timeout_seconds=CALL_TIMEOUT,
            )
            return {
                "server_name": init.serverInfo.name if hasattr(init, "serverInfo")
                               else init.server_info.name,
                "tools": {t.name: t for t in tools.tools},
                "known": _payload(known),
                "unknown": _payload(unknown),
            }


def _fixture() -> dict:
    if not FIXTURE_PATH.exists():
        pytest.skip(f"No fixture at {FIXTURE_PATH}; run smoke_test.py --write-fixture")
    return json.loads(FIXTURE_PATH.read_text())


def _fixture_hadm_id() -> str:
    patients = _fixture().get("patients", {})
    if not patients:
        pytest.skip("Fixture has no patients")
    return sorted(patients)[0]


@pytest.fixture(scope="module")
def session_results() -> dict:
    """Spawn the server once and reuse the results across assertions."""
    try:
        return asyncio.run(_converse())
    except Exception as exc:  # no endpoint, no credentials, no network
        pytest.skip(f"MCP stdio session failed: {type(exc).__name__}: {exc}")


def test_tool_is_advertised(session_results):
    tool = session_results["tools"].get("predict_readmission")
    assert tool is not None, "predict_readmission missing from tools/list"
    assert tool.description, "tool has no description for the model to read"
    schema = tool.input_schema
    assert "hadm_id" in schema["properties"]
    assert schema["properties"]["hadm_id"]["type"] == "integer"
    assert schema["required"] == ["hadm_id"]


def test_known_patient_matches_fixture(session_results):
    fixture = _fixture()
    hadm_id = _fixture_hadm_id()
    expected = fixture["patients"][hadm_id]
    got = session_results["known"]

    assert got.get("error") is None, got
    assert got["hadm_id"] == int(hadm_id)
    # Never assert float equality against a served model.
    assert abs(got["probability"] - expected["probability"]) <= expected["tolerance"]
    assert got["decision"] == expected["decision"]
    assert got["threshold"] == fixture["threshold"]


def test_response_carries_provenance(session_results):
    """model_version + feature_source answer "which model, reading from where?"."""
    got = session_results["known"]
    assert got["model_version"]
    assert got["feature_source"] in {"bigquery", "feature_store"}


def test_top_factors_are_well_formed(session_results):
    factors = session_results["known"]["top_factors"]
    assert factors, "no attributions returned"
    for factor in factors:
        assert factor["feature"]
        assert isinstance(factor["contribution"], float)
        expected_direction = "increases" if factor["contribution"] > 0 else "decreases"
        assert factor["direction"] == expected_direction


def test_unknown_patient_returns_structured_error(session_results):
    """A bad id must come back as data the agent can explain, not a crash."""
    got = session_results["unknown"]
    assert got["error"] == "unknown_patient"
    assert got["message"]
    assert got["hadm_id"] == UNKNOWN_HADM_ID
    assert "probability" not in got
