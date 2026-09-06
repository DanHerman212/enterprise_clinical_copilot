"""Shared runtime decisions for the agent and MCP HTTP services."""

import os
from collections.abc import Mapping
from pathlib import Path


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


def timeout_chain(env: Mapping[str, str] | None = None) -> tuple[float, float]:
    """Return ``(tool_timeout, agent_timeout)`` and enforce their ordering."""
    values = os.environ if env is None else env
    try:
        tool_timeout = float(values.get("MCP_TOOL_TIMEOUT_SECONDS", "100"))
        agent_timeout = float(values.get("ASK_TIMEOUT_SECONDS", "110"))
    except ValueError as exc:
        raise RuntimeError("MCP and agent timeouts must be numeric.") from exc
    if tool_timeout <= 0 or agent_timeout <= 0 or tool_timeout >= agent_timeout:
        raise RuntimeError(
            "MCP_TOOL_TIMEOUT_SECONDS must be positive and shorter than "
            "ASK_TIMEOUT_SECONDS."
        )
    return tool_timeout, agent_timeout


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