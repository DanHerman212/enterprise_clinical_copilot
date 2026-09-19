"""Langfuse tracing for the agent — a sink, never a dependency.

The agent does not implement tracing; it hands a callback to LangGraph and
LangChain reports the run. That is the cheapest possible integration here and it
was the design all along: the model is `ChatGoogleGenerativeAI` and the MCP tools
are LangChain `BaseTool` subclasses, so every model call and every tool call is
already a LangChain run. Nothing in `graph.py` had to change to be traced beyond
passing the handler in.

Three rules, each earned by a past failure:

1. **Tracing never fails the answer.** Every entry point here returns a usable
   value or nothing, and catches everything. The adversarial review's ECC-17 is
   the reason: with tracing enabled, a run read an attribute the callback handler
   did not expose and raised *after* the model and tool spend completed, so the
   only configuration that could fail was the one nobody exercised in dev. A
   trace is worth less than an answer.

2. **Unconfigured means off, silently.** Without `LANGFUSE_HOST`,
   `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` the SDK disables itself, and so
   does this module: no keys, no warning, no behaviour difference. Worse than the
   alternative in exactly one way — a run with no tracing looks identical to a run
   whose tracing is misconfigured, which is why `configured()` exists to be asked
   out loud (the eval collector prints it).

3. **The trace id is read defensively.** `last_trace_id` is what the handler
   exposes today, and it is fetched with `getattr(..., "")` with the client's
   `get_current_trace_id()` as a fallback, because a rename upstream must degrade
   to "no id recorded" rather than to a failed request.

The archive is not here. `services/agent/chain.py` writes the durable execution
record, and it stays the evidence of record; this module is the working surface.
A teardown of the Langfuse stack must not take the evidence with it.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# All three are required. A host without keys, or keys without a host, would
# otherwise half-configure the SDK and produce traces that go nowhere.
_ENV_VARS = ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")


def configured() -> bool:
    """True when tracing has everything it needs. Says nothing about reachability."""
    return all((os.environ.get(name) or "").strip() for name in _ENV_VARS)


def handler():
    """A LangGraph callback handler, or None when tracing is off.

    Nothing upstream learns whether tracing is on: callers pass this straight
    into the run and treat None as "no callbacks". A handler that cannot be built
    is logged once and treated as absent.
    """
    if not configured():
        return None
    try:
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()
    except Exception:  # noqa: BLE001 — see rule 1: tracing may not fail a run
        logger.warning(
            "langfuse tracing is configured but the callback handler could not be "
            "built; continuing without tracing",
            exc_info=True,
        )
        return None


def trace_id(handler=None) -> str:
    """The id of the trace this run produced, or "" when there is none.

    Two sources, because the SDK's surface for this has moved between versions:
    the handler's own `last_trace_id`, then the client's `get_current_trace_id()`.
    Either may be absent depending on timing and version; absent means an empty
    string and never an exception.
    """
    if not configured():
        return ""
    try:
        found = getattr(handler, "last_trace_id", "") or ""
        if found:
            return str(found)
        import langfuse

        return str(langfuse.get_client().get_current_trace_id() or "")
    except Exception:  # noqa: BLE001 — rule 1
        logger.warning("could not read the langfuse trace id", exc_info=True)
        return ""


def flush() -> None:
    """Deliver anything buffered. For short-lived processes, not for requests.

    A request does not call this: the SDK batches and ships on its own schedule,
    and a network round trip on the answer path would trade latency for
    promptness nobody asked for. A script that is about to exit does call it,
    because a process that exits first loses the traces it just produced.
    """
    if not configured():
        return
    try:
        import langfuse

        langfuse.get_client().flush()
    except Exception:  # noqa: BLE001 — rule 1
        logger.warning("langfuse flush failed; some traces may not have shipped",
                       exc_info=True)
