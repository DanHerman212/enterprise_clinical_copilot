"""Spend caps & DoS defenses (Cluster C: ECC-02, ECC-09, ECC-25).

All offline: stub toolboxes, patched token minting, fake ASGI scopes.
"""
import asyncio
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

import services.agent.engine.mcp_client as mc
import services.mcp.server as ms
from services.mcp.runtime import (
    Timeouts,
    positive_int_env,
    requires_cloud_run_auth,
    resolve_project_id,
    timeout_chain,
)
from services.agent.engine.graph import (
    BUDGET_EXHAUSTED_ANSWER,
    MAX_TOOL_CALLS_PER_TURN,
    RECURSION_LIMIT,
    _budget_exhausted_turn,
    _execute_tool_calls,
)
from services.agent.engine.mcp_client import MCPToolbox


def _run(coro):
    return asyncio.run(coro)


# --- ECC-02: per-turn tool budget & timeout chain ----------------------------


class _StubToolbox:
    def __init__(self):
        self.calls = []

    async def call(self, name, args):
        self.calls.append((name, args))
        return {"ok": True}


def test_tool_calls_beyond_the_per_turn_budget_are_refused():
    """Calls past the cap get a structured error WITHOUT hitting the toolbox
    (each real call is billed Vertex/BigQuery work)."""
    box = _StubToolbox()
    n = MAX_TOOL_CALLS_PER_TURN + 2
    calls = [{"name": f"tool_{i}", "args": {}, "id": str(i)} for i in range(n)]

    messages, recorded = _run(_execute_tool_calls(box, calls))

    assert len(box.calls) == MAX_TOOL_CALLS_PER_TURN
    assert len(recorded) == n
    for extra in recorded[MAX_TOOL_CALLS_PER_TURN:]:
        assert extra["response"]["error"] == "tool_call_limit"
    # Every call — executed or refused — still yields a ToolMessage, keeping
    # the transcript aligned with the model's tool_call ids.
    assert len(messages) == n


def test_the_tool_budget_is_the_binding_constraint_not_the_recursion_breaker():
    """A turn must be able to spend its whole tool budget and still answer.

    LangGraph counts every node transition, so each tool round costs two supersteps
    (agent, tools) before the next round can start. A limit of 10 fired on the fourth
    round of an ordinary ReAct turn on 2026-10-17 — twice — and because it fired after
    the answer had already streamed, the turn died as `agent_failed` with a correct
    partial answer on screen. The breaker therefore has to clear the budget with room
    for the answering round that follows the refusal, while staying a small explicit
    bound of its own. `MAX_TOOL_CALLS_PER_TURN` is what bounds billable tool spend;
    this is only the value at which a runaway graph is stopped.
    """
    assert RECURSION_LIMIT >= 2 * MAX_TOOL_CALLS_PER_TURN + 2
    assert RECURSION_LIMIT <= 30


def test_tool_timeout_stays_under_upstream_deadlines():
    """Timeout chain: tool call < agent ask deadline (110) < site proxy (120),
    so the innermost deadline fires first with a structured error."""
    assert MCPToolbox.call_timeout_seconds < 110 < 120
    assert MCPToolbox.call_timeout_seconds == timeout_chain().tool


def test_a_spent_turn_still_answers_when_the_model_asks_for_another_tool():
    """A spent budget must end in an answer, not an empty turn.

    A turn at the cap is given no tools, but a model can still emit a function call:
    the tool is missing from the request, not from the pattern of the transcript.
    `route` ends the graph on that message rather than refusing a sixth time, so the
    turn had no text and the caller got a 502 `answer_unavailable` — a crash-shaped
    response to a question the service had handled safely. Observed live on two
    prompt-injection probes (inj-02, inj-04): the model fabricated admission ids,
    spent the budget on searches refused as `unknown_patient`, then asked for a
    sixth lookup it could not be given.
    """
    response = AIMessage(
        content="",
        tool_calls=[{
            "name": "rag_search",
            "args": {"hadm_id": 10000002, "query": "prompt"},
            "id": "call-1",
        }],
        response_metadata={"finish_reason": "STOP"},
    )

    settled = _budget_exhausted_turn(response)

    assert settled.tool_calls == []
    assert settled.content == BUDGET_EXHAUSTED_ANSWER
    # The finish reason survives, so a record of the turn can still say what the
    # provider reported rather than looking like a clean answer.
    assert settled.response_metadata["finish_reason"] == "STOP"


def test_a_spent_turn_keeps_the_prose_it_already_wrote():
    """The refusal replaces nothing. Prose the model did write is the answer.

    Dropping text to add a policy sentence would delete a correct answer, which is
    the failure this codebase has already paid for once.
    """
    response = AIMessage(
        content="  The risk for this patient is 0.21.  ",
        tool_calls=[{"name": "rag_search", "args": {}, "id": "call-1"}],
        response_metadata={"finish_reason": "STOP"},
    )

    settled = _budget_exhausted_turn(response)

    assert settled.tool_calls == []
    assert settled.content == "The risk for this patient is 0.21."


def test_cloud_run_auth_decision_is_shared_and_explicit():
    assert not requires_cloud_run_auth({})
    assert requires_cloud_run_auth({"K_SERVICE": "agent"})
    assert not requires_cloud_run_auth({
        "K_SERVICE": "agent", "ALLOW_UNAUTHENTICATED": "true",
    })


def test_timeout_chain_is_ordered():
    assert timeout_chain({
        "MODEL_TIMEOUT_SECONDS": "9",
        "MCP_TOOL_TIMEOUT_SECONDS": "10",
        "ASK_TIMEOUT_SECONDS": "11",
    }) == Timeouts(model=9.0, tool=10.0, ask=11.0)


def test_timeout_chain_rejects_inverted_deadlines():
    # A tool call that can outlast the question, and a model call that can outlast
    # the tool call it is made inside.
    with pytest.raises(RuntimeError, match="nest"):
        timeout_chain({
            "MCP_TOOL_TIMEOUT_SECONDS": "110",
            "ASK_TIMEOUT_SECONDS": "100",
        })
    with pytest.raises(RuntimeError, match="nest"):
        timeout_chain({
            "MODEL_TIMEOUT_SECONDS": "100",
            "MCP_TOOL_TIMEOUT_SECONDS": "10",
            "ASK_TIMEOUT_SECONDS": "110",
        })


def test_project_id_resolution_prefers_environment():
    assert resolve_project_id({"PROJECT_ID": "test-project"}) == "test-project"


def test_positive_integer_configuration_is_validated():
    assert positive_int_env("RAG_TOP_K", 5, {"RAG_TOP_K": "10"}) == 10
    with pytest.raises(RuntimeError, match="RAG_TOP_K"):
        positive_int_env("RAG_TOP_K", 5, {"RAG_TOP_K": "0"})


# --- ECC-09: ID token caching -------------------------------------------------


def test_id_token_is_cached_per_audience():
    mc._token_cache.clear()
    try:
        with patch.object(mc, "id_token", side_effect=["tok-a", "tok-b"]) as minted:
            assert mc._cached_id_token("https://svc-a") == "tok-a"
            assert mc._cached_id_token("https://svc-a") == "tok-a"  # cache hit
            assert mc._cached_id_token("https://svc-b") == "tok-b"  # new audience
        assert minted.call_count == 2
    finally:
        mc._token_cache.clear()


# --- ECC-25: MCP server auth-header guard ------------------------------------


def _send_through(app, scope):
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    _run(app(scope, receive, send))
    return sent


def test_auth_guard_rejects_headerless_requests():
    async def inner(scope, receive, send):  # pragma: no cover - must not run
        raise AssertionError("request without Authorization reached the app")

    app = ms._require_auth_header(inner)
    scope = {"type": "http", "path": "/mcp", "method": "POST",
             "headers": [], "query_string": b""}
    sent = _send_through(app, scope)
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 401


def test_auth_guard_passes_health_and_authorized_requests():
    reached = []

    async def inner(scope, receive, send):
        reached.append(scope["path"])

    app = ms._require_auth_header(inner)
    _send_through(app, {"type": "http", "path": "/health", "method": "GET",
                        "headers": [], "query_string": b""})
    _send_through(app, {"type": "http", "path": "/mcp", "method": "POST",
                        "headers": [(b"authorization", b"Bearer x")],
                        "query_string": b""})
    assert reached == ["/health", "/mcp"]
