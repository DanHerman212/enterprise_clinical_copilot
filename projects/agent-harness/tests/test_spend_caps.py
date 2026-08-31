"""Spend caps & DoS defenses (Cluster C: ECC-02, ECC-09, ECC-25).

All offline: stub toolboxes, patched token minting, fake ASGI scopes.
"""
import asyncio
from unittest.mock import patch

import agent.mcp_client as mc
import mcp_server.server as ms
from agent.graph import (
    MAX_TOOL_CALLS_PER_TURN,
    RECURSION_LIMIT,
    _execute_tool_calls,
)
from agent.mcp_client import MCPToolbox


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


def test_recursion_limit_is_a_small_explicit_bound():
    assert RECURSION_LIMIT <= 10


def test_tool_timeout_stays_under_upstream_deadlines():
    """Timeout chain: tool call < agent ask deadline (110) < site proxy (120),
    so the innermost deadline fires first with a structured error."""
    assert MCPToolbox.call_timeout_seconds < 110 < 120


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
