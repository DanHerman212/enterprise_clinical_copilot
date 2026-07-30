"""Hand-rolled MCP -> Gemini tool adapter.

Why hand-rolled rather than `langchain-mcp-adapters` (probed 2026-07-30, v0.3.1):
that package declares `mcp>=1.24.0` with **no upper bound**, so pip installs it
happily next to our `mcp==2.0.0` — then it fails at import, because
`tools.py` imports `mcp.server.fastmcp` (removed in 2.0) and reads
`tool.inputSchema` (renamed to `input_schema`). Its `sessions.py` *has* been
migrated, so the package is half-ported. The only alternative was pinning
`mcp<2.0`, which would invalidate the server built in §6-§8 and the deployed
image. The adapter surface we actually need is this file.

Transport is stdio here (§10). §11 adds authenticated HTTP; the toolbox itself
is transport-agnostic — only the context manager changes.
"""

import contextlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HARNESS_ROOT = Path(__file__).resolve().parents[1]

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
    call_timeout_seconds: float = 180.0
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
