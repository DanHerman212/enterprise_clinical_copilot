"""Small shared wire-contract primitives.

Service-specific payloads remain owned by the agent and MCP packages. This module
contains only vocabulary shared across those boundaries.
"""

from typing import TypedDict


class ToolReference(TypedDict):
    name: str


class StructuredError(TypedDict):
    error: str
    message: str
