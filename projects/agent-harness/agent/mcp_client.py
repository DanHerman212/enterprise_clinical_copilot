"""Hand-rolled MCP -> Gemini tool adapter.

Why hand-rolled rather than `langchain-mcp-adapters` (probed 2026-07-30, v0.3.1):
that package declares `mcp>=1.24.0` with **no upper bound**, so pip installs it
happily next to our `mcp==2.0.0` — then it fails at import, because
`tools.py` imports `mcp.server.fastmcp` (removed in 2.0) and reads
`tool.inputSchema` (renamed to `input_schema`). Its `sessions.py` *has* been
migrated, so the package is half-ported. The only alternative was pinning
`mcp<2.0`, which would invalidate the server built in §6-§8 and the deployed
image. The adapter surface we actually need is this file.

Transport is selected by environment so local development stays on stdio:
`MCP_TRANSPORT=stdio` (default) or `http` with `MCP_URL` pointing at the Cloud
Run service. `toolbox()` picks; the graph never knows which it got.
"""

import asyncio
import contextlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx2
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

HARNESS_ROOT = Path(__file__).resolve().parents[1]

MCP_TRANSPORT = os.environ.get("MCP_TRANSPORT", "stdio")  # "stdio" | "http"
MCP_URL = os.environ.get("MCP_URL", "")

# Gemini's Schema is a subset of JSON Schema. Pydantic emits keys it rejects
# ("title", "$schema", "additionalProperties", "default"), and an unknown key is
# a 400 at generate time, not at declaration time. Strip to the allowed set.
_ALLOWED_SCHEMA_KEYS = {
    "type", "format", "description", "nullable", "enum",
    "properties", "required", "items", "minimum", "maximum",
}


def _clean_schema(node: Any) -> Any:
    """Recursively drop schema keys Gemini does not accept."""
    if not isinstance(node, dict):
        return node
    cleaned: dict[str, Any] = {}
    for key, value in node.items():
        if key not in _ALLOWED_SCHEMA_KEYS:
            continue
        if key == "properties" and isinstance(value, dict):
            cleaned[key] = {k: _clean_schema(v) for k, v in value.items()}
        elif key == "items":
            cleaned[key] = _clean_schema(value)
        else:
            cleaned[key] = value
    return cleaned


def _payload(result: Any) -> dict:
    """Normalise a CallToolResult into the dict the tool returned."""
    if getattr(result, "structured_content", None):
        return result.structured_content
    for block in result.content or []:
        text = getattr(block, "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"result": text}
    return {}


@dataclass
class MCPToolbox:
    """The MCP tools of one session, exposed as Gemini function declarations."""

    session: ClientSession
    # Must stay below the site's 120s upstream deadline and the agent server's
    # ASK_TIMEOUT_SECONDS — a tool that outlives its caller is pure spend.
    call_timeout_seconds: float = 100.0
    _tools: dict[str, Any] = field(default_factory=dict)

    async def load(self) -> "MCPToolbox":
        result = await self.session.list_tools()
        self._tools = {tool.name: tool for tool in result.tools}
        if not self._tools:
            raise RuntimeError("MCP server advertised no tools")
        return self

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    @property
    def gemini_tool(self) -> types.Tool:
        return types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=tool.name,
                    description=(tool.description or "").strip(),
                    parameters=_clean_schema(tool.input_schema),
                )
                for tool in self._tools.values()
            ]
        )

    async def call(self, name: str, arguments: dict) -> dict:
        """Execute a tool. Transport failures become structured errors.

        The model is instructed to report errors rather than invent a number,
        so an error must reach it as data, not as an exception that collapses
        the graph.
        """
        if name not in self._tools:
            return {"error": "unknown_tool", "message": f"No MCP tool named {name!r}"}
        try:
            result = await self.session.call_tool(
                name, arguments, read_timeout_seconds=self.call_timeout_seconds
            )
        except Exception as exc:
            return {"error": "tool_call_failed", "message": f"{type(exc).__name__}: {exc}"}

        payload = _payload(result)
        if getattr(result, "is_error", False) and "error" not in payload:
            payload = {"error": "tool_call_failed", "message": str(payload)}
        return payload


@contextlib.asynccontextmanager
async def stdio_toolbox(python: str | None = None):
    """Run the MCP server as a subprocess and yield a loaded toolbox.

    No network and no auth, so a failure in §10 can only be agent-shaped.
    """
    params = StdioServerParameters(
        command=python or sys.executable,
        args=["-m", "mcp_server.server", "--transport", "stdio"],
        cwd=str(HARNESS_ROOT),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield await MCPToolbox(session).load()


def id_token(audience: str) -> str:
    """Mint an ID token for a private Cloud Run service.

    The audience must be the **service URL**, not the /mcp path. A mismatched
    audience produces a 401 identical to a missing IAM binding, and the
    audience is the more common cause.

    On Cloud Run the metadata server mints this in-process. Locally, ADC is a
    *user* credential, which cannot mint an ID token for an arbitrary audience.
    Check the credential type first rather than letting fetch_id_token fail:
    its failure path probes the GCE metadata server and stalls for seconds
    before raising DefaultCredentialsError, on every local run.
    """
    import google.auth
    import google.auth.transport.requests
    import google.oauth2.credentials
    import google.oauth2.id_token

    credentials, _ = google.auth.default()

    if not isinstance(credentials, google.oauth2.credentials.Credentials):
        # Service account or metadata server: in-process, no fork.
        request = google.auth.transport.requests.Request()
        return google.oauth2.id_token.fetch_id_token(request, audience)

    # Local user credential. NOTE: this forks. Keep it early — once
    # google-cloud-aiplatform has opened gRPC channels, fork+exec can deadlock
    # in gRPC's atfork handler. See the gsutil hang in projects/mlops.
    result = subprocess.run(
        ["gcloud", "auth", "print-identity-token"],
        capture_output=True, text=True, check=True, timeout=60,
    )
    return result.stdout.strip()


# ID tokens live one hour; refresh with headroom. Cached per audience so a
# fresh per-request toolbox does not mint (or fork gcloud for) a new token on
# every /ask.
_TOKEN_TTL_SECONDS = 45 * 60
_token_cache: dict[str, tuple[str, float]] = {}


def _cached_id_token(audience: str) -> str:
    token, fresh_until = _token_cache.get(audience, ("", 0.0))
    if time.monotonic() >= fresh_until:
        token = id_token(audience)
        _token_cache[audience] = (token, time.monotonic() + _TOKEN_TTL_SECONDS)
    return token


@contextlib.asynccontextmanager
async def http_toolbox(url: str | None = None):
    """Connect to the deployed MCP server over authenticated streamable HTTP."""
    base = (url or MCP_URL).rstrip("/")
    if not base:
        raise RuntimeError("MCP_URL is not set; required when MCP_TRANSPORT=http")

    # Token minting is blocking I/O (metadata server or a gcloud fork); off the
    # event loop so concurrent requests aren't stalled behind it (ECC-09).
    token = await asyncio.to_thread(_cached_id_token, base)
    headers = {"Authorization": f"Bearer {token}"}

    # SDK 2.0 takes an http_client, not headers — and it must be httpx2, which
    # the SDK vendors alongside the ordinary httpx other libraries pull in.
    # Transport timeout sits just above the per-call read timeout so the tool
    # deadline fires first with a structured error.
    async with httpx2.AsyncClient(headers=headers, timeout=110) as http_client:
        async with streamable_http_client(f"{base}/mcp", http_client=http_client) as (
            read,
            write,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield await MCPToolbox(session).load()


@contextlib.asynccontextmanager
async def toolbox(transport: str | None = None, url: str | None = None):
    """Transport-agnostic entry point. The graph never knows which it got."""
    chosen = (transport or MCP_TRANSPORT).lower()
    if chosen == "stdio":
        async with stdio_toolbox() as box:
            yield box
    elif chosen == "http":
        async with http_toolbox(url) as box:
            yield box
    else:
        raise ValueError(f"Unknown MCP transport {chosen!r}; expected 'stdio' or 'http'")
