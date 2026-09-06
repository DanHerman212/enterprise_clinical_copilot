"""Optional Langfuse observability for agent graph executions."""

import os
from typing import Any


LANGFUSE_ENABLED = bool(
    os.environ.get("LANGFUSE_PUBLIC_KEY")
    and os.environ.get("LANGFUSE_SECRET_KEY")
    and os.environ.get("LANGFUSE_HOST")
)


class _NoopHandler:
    """Inert handler used when Langfuse is not configured."""

    def __init__(self) -> None:
        self.last_trace_id = None


def make_handler() -> Any:
    """Return the Langfuse callback handler or an inert local equivalent."""
    if not LANGFUSE_ENABLED:
        return _NoopHandler()
    from langfuse.langchain import CallbackHandler

    return CallbackHandler()
