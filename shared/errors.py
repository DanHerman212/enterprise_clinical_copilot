"""Stable cross-service error vocabulary."""

from typing import TypedDict


class ErrorPayload(TypedDict):
    error: str
    message: str


class ContractError(ValueError):
    """A payload does not satisfy a service contract."""



class AgentResponseError(ValueError):
    """The agent produced a payload outside its HTTP success contract.

    Declared in `shared` rather than in the wire contract: the tool-execution seam
    in `services/agent/engine/mcp_client.py` validates recorded tool calls, and the
    engine may not import the layer it validates against (Phase 7).
    `ingress/contracts.py` re-exports it, so its public surface is unchanged.
    """
