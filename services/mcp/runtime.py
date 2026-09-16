"""Shared runtime decisions for the agent and MCP HTTP services."""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple


def resolve_project_id(env: Mapping[str, str] | None = None) -> str:
    """Resolve GCP project identity without committing a project default."""
    values = os.environ if env is None else env
    if values.get("PROJECT_ID"):
        return values["PROJECT_ID"]

    for directory in [Path.cwd(), *Path.cwd().resolve().parents]:
        env_file = directory / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                stripped = line.strip()
                if stripped.startswith("PROJECT_ID="):
                    project_id = stripped.split("=", 1)[1].strip()
                    if project_id:
                        return project_id
            break

    raise RuntimeError(
        "PROJECT_ID is not set. Export it or put PROJECT_ID=<project> in an "
        "untracked .env at the repo root. There is deliberately no default."
    )


def requires_cloud_run_auth(env: Mapping[str, str] | None = None) -> bool:
    """Require an Authorization header only for authenticated Cloud Run mode."""
    values = os.environ if env is None else env
    return bool(values.get("K_SERVICE")) and (
        values.get("ALLOW_UNAUTHENTICATED", "").lower() != "true"
    )


class Timeouts(NamedTuple):
    """The bounded operations, shortest first.

    Named rather than positional because these are three numbers of the same kind
    and their ordering is the property that matters: an operation that may outlast
    the one wrapping it can never be seen failing before its caller gives up.
    """

    model: float
    tool: float
    ask: float


def timeout_chain(env: Mapping[str, str] | None = None) -> Timeouts:
    """One model call, one tool call, one question — and the order they nest in.

    The ordering is enforced here rather than documented, because a documented
    ordering is one nobody checks. Read once at startup, so a bad combination
    stops the service from serving instead of surfacing later as a stalled
    request.
    """
    values = os.environ if env is None else env
    try:
        model = float(values.get("MODEL_TIMEOUT_SECONDS", "60"))
        tool = float(values.get("MCP_TOOL_TIMEOUT_SECONDS", "100"))
        ask = float(values.get("ASK_TIMEOUT_SECONDS", "110"))
    except ValueError as exc:
        raise RuntimeError(
            "Model, tool and agent timeouts must be numeric."
        ) from exc
    if model <= 0 or tool <= 0 or ask <= 0 or not model < tool < ask:
        raise RuntimeError(
            "Timeouts must be positive and nest shortest to longest: "
            "MODEL_TIMEOUT_SECONDS < MCP_TOOL_TIMEOUT_SECONDS < "
            "ASK_TIMEOUT_SECONDS."
        )
    return Timeouts(model=model, tool=tool, ask=ask)


def positive_int_env(
    name: str, default: int, env: Mapping[str, str] | None = None
) -> int:
    """Read a positive integer setting and fail with its setting name."""
    values = os.environ if env is None else env
    raw = values.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer.") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer.")
    return value