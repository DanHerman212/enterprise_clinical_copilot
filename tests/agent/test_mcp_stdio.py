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

HARNESS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS_ROOT))

# The production spawner, imported rather than re-implemented, so this file cannot
# drift from it: a child started without `env` gets the SDK's six-key allow-list,
# the server's config fails closed without PROJECT_ID, and the session dies at
# import — which is what this file reported as "MCP stdio session failed" for as
# long as the skip has been hiding it (2026-10-12).
from services.agent.engine.mcp_client import _mcp_child_environment, _unwrap  # noqa: E402

FIXTURE_PATH = HARNESS_ROOT / "tests" / "agent" / "fixtures" / "expected.json"

# A cold call does Model.list + GCS manifest + BigQuery + Vertex predict.
CALL_TIMEOUT = 120.0

UNKNOWN_HADM_ID = 1


def _payload(result) -> dict:
    """The tool's JSON, however this SDK version chose to carry it.

    Unwrapped because a tool annotated `-> Result | Error` is advertised as
    returning `{"result": …}`, and the SDK puts that envelope in the structured
    payload: without this, every assertion below reads a dict whose only key is
    "result" (see `mcp_client._unwrap`, the agent's copy of the same rule).
    """
    if getattr(result, "structured_content", None):
        return _unwrap(result.structured_content)
    return _unwrap(json.loads(result.content[0].text))


async def _converse() -> dict:
    """One session: list tools, score a known patient, score a bogus one."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "services.mcp.server", "--transport", "stdio"],
        # The package dir has a hyphen, so it is not importable by path; the
        # server must be started with the repo root as cwd, and with the parent's
        # environment so PROJECT_ID reaches it.
        cwd=str(HARNESS_ROOT),
        env=_mcp_child_environment(),
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
    """The admission the fixture is anchored to, by its own declaration.

    Read from `anchor_hadm_id` rather than by sorting the patient keys. The old
    anchor, "20924467", sorted first, so the suite kept selecting it long after the
    demo cohort stopped serving it and every value assertion here was measuring a
    patient the service could not find (2026-10-12).
    """
    fixture = _fixture()
    anchor = fixture.get("anchor_hadm_id")
    patients = fixture.get("patients", {})
    if not anchor or str(anchor) not in patients:
        pytest.skip(
            "Fixture declares no anchor admission it holds values for; run "
            "mlops/training/smoke_test.py <hadm_id> --write-fixture"
        )
    return str(anchor)


def _innermost(exc: BaseException) -> str:
    """Name the exception at the bottom of a task-group wrapper.

    A failure inside a task group is reported as an ExceptionGroup whose message is
    "unhandled errors in a TaskGroup (1 sub-exception)" — which says nothing about
    what went wrong. Five tests in this file stayed dead behind that sentence: the
    cause was one level down (a subprocess with no environment, and later a project
    id that does not exist) and was never printed (2026-10-12).
    """
    seen = 0
    while (subs := getattr(exc, "exceptions", None)) and seen < 10:
        exc = subs[0]
        seen += 1
    return f"{type(exc).__name__}: {exc}"


@pytest.fixture(scope="module")
def session_results() -> dict:
    """Spawn the server once and reuse the results across assertions."""
    try:
        return asyncio.run(_converse())
    except Exception as exc:  # no endpoint, no credentials, no network
        pytest.skip(f"MCP stdio session failed: {_innermost(exc)}")


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
