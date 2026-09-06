"""Stable cross-service error vocabulary."""

from typing import TypedDict


class ErrorPayload(TypedDict):
    error: str
    message: str


class ContractError(ValueError):
    """A payload does not satisfy a service contract."""
