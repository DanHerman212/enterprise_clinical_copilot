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
A teardown of the Langfuse stack must not take the evidence with it."""
from __future__ import annotations

import contextlib
import contextvars
import logging
import os

from typing import Any

logger = logging.getLogger(__name__)

# All three are required. A host without keys, or keys without a host, would
# otherwise half-configure the SDK and produce traces that go nowhere.
_ENV_VARS = ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")

# Which batch run, if any, this turn belongs to. A clinician's turn has no value, and
# that absence is what makes the tag mean "this came from an evaluation". Carried in a
# context variable rather than as an `ask` argument: `ask` is the seam every test double
# already mirrors, and adding a keyword there meant sixteen fakes had to grow a field
# they do not care about.
_RUN_TAG: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "run_tag", default=None
)


def set_run_tag(tag: str | None) -> None:
    """Label the current turn. Called once per request, before any span opens."""
    _RUN_TAG.set(tag or None)


def current_run_tag() -> str | None:
    return _RUN_TAG.get()


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


@contextlib.contextmanager
def observation(name: str, *, as_type: str = "span",
                trace_metadata: dict | None = None, input: Any = None):
    """A span to hold work that LangChain does not already trace.

    `CallbackHandler` reports model and tool runs, which is everything the reasoning
    loop does. The deterministic command path is not a LangChain run at all — it calls
    its tool and then makes one formatting turn — so a `/ehr` turn produced a trace
    with exactly ONE observation and no parent. Langfuse disables its Graph view when
    there is no hierarchy to draw, so the turns most worth reviewing were the ones
    with nothing to review.

    `trace_metadata` puts the trace-level attributes on: name, session id, tags and
    metadata. They are set with `langfuse.propagate_attributes` because that is the v4
    API for trace-scoped attributes — `LangfuseSpan` has no `update_trace`, and the
    first attempt at this used one and silently stamped nothing (caught by the live
    /risk check, 2026-10-19). Previously the metadata reached Langfuse only through the
    formatting model's config, so `/risk` — which makes no LLM call at all — carried no
    tags, no session and no admission.

    Yields None when unconfigured, and never raises (rule 1). A span is a convenience
    for a reader; the answer outranks it.

    `input` is set when the span opens rather than by an `.update()` afterwards,
    because a span's argument is known at the moment it is opened — a tool span that
    names a tool and records no arguments is a row that says a search ran and not what
    for, which is the question a reader opens the trace to answer.
    """
    manager = None
    propagate = None
    if configured():
        try:
            import langfuse

            client = langfuse.get_client()
            manager = client.start_as_current_observation(
                as_type=as_type, name=name, input=input,
            )
            if trace_metadata:
                attrs = {
                    "trace_name": name,
                    "session_id": trace_metadata.get("langfuse_session_id"),
                    "tags": list(trace_metadata.get("langfuse_tags") or []) or None,
                    "metadata": {k: v for k, v in trace_metadata.items()
                                 if not k.startswith("langfuse_")} or None,
                }
                propagate = langfuse.propagate_attributes(
                    **{k: v for k, v in attrs.items() if v}
                )
        except Exception:  # noqa: BLE001 — rule 1
            logger.warning("could not open the langfuse span %r", name, exc_info=True)
            manager = None
            propagate = None
    if manager is None:
        yield None
        return
    with (propagate or contextlib.nullcontext()):
        with manager as span:
            yield span


def stamp_trace_tags(span, tags: list[str]) -> None:
    """Record the turn's outcome on the trace, best-effort.

    Trace-level tags are fixed when the trace is created, and the SDK exposes no way to
    append to them afterwards (audited against langfuse 4.14.4: the client has
    `get_current_trace_id`, `score_current_trace` and `propagate_attributes`, but no
    trace updater). So `tools:<n>` goes on the ROOT SPAN's metadata, where it is
    readable and filterable in the trace drawer, and the trace-level `tools:` tag is
    recorded as a known gap rather than faked.

    Never raises: a missing tag is a worse dashboard, a raised exception is a failed
    answer (rule 1).
    """
    if span is None:
        return
    try:
        counts = [t for t in tags if t.startswith("tools:")]
        if counts:
            span.update(metadata={"tools": counts[-1].split(":", 1)[1]})
    except Exception:  # noqa: BLE001 — rule 1
        logger.warning("could not append turn outcome to the langfuse span",
                       exc_info=True)


def set_trace_io(user_input: str | None, output: str | None, span=None) -> None:
    """Put the question and the answer on the trace, and on the root span too.

    Langfuse renders `input` and `output` as columns in the trace table. Without them
    every row shows a dash, so triage means opening the graph and reading child spans
    to discover what was even asked — which is the difference between a dashboard and
    a heap of ids. This module opens the root span, so it is the one place that knows
    both ends of the turn.

    `span` is not redundant with the trace-level call. They write to two different
    records: `set_current_trace_io` fills the table row, and the root observation kept
    `input: null, output: null` — so an exported span tree, which is what root-cause
    analysis reads, had nothing at its root. Both are written here because a caller
    that has the span must not have to remember there are two places to put it.

    Best-effort like everything here (rule 1): a missing column is a worse dashboard,
    a raised exception is a failed answer. The two writes fail independently, so a
    trace-level refusal does not cost the span its attribution.
    """
    if not configured():
        return
    try:
        import langfuse

        langfuse.get_client().set_current_trace_io(
            input=(user_input or None), output=(output or None),
        )
    except Exception:  # noqa: BLE001 — rule 1
        logger.warning("could not set the langfuse trace input/output", exc_info=True)
    if span is None:
        return
    try:
        span.update(input=(user_input or None), output=(output or None))
    except Exception:  # noqa: BLE001 — rule 1
        logger.warning("could not set the langfuse span input/output", exc_info=True)


def record_span_output(span, output, *, is_error: bool = False) -> None:
    """Record what a span produced, and whether it produced a failure.

    A tool span with no output is a row that says the tool ran and nothing about the
    result, which is the end of the turn a reviewer actually needs: whether the search
    returned passages, whether the predictor returned a score, or whether the tool
    answered with its structured `{"error": ...}`.

    An error payload is levelled ERROR rather than left DEFAULT, because the payload is
    the only place the failure appears: `MCPToolbox.call` returns errors as values, so
    no exception reaches Langfuse and a span tree that reads only its statuses shows a
    clean turn.

    Best-effort (rule 1).
    """
    if span is None:
        return
    try:
        fields: dict[str, Any] = {"output": output}
        if is_error:
            fields["level"] = "ERROR"
        span.update(**fields)
    except Exception:  # noqa: BLE001 — rule 1
        logger.warning("could not record the langfuse span output", exc_info=True)


def current_trace_id() -> str:
    """The id of the trace active in this context, or "" when there is none.

    Needed because `/risk` makes no LLM call: `trace_id(handler)` reads the LangChain
    handler's `last_trace_id`, and with no model run there is no handler id — so the
    turn reported an empty trace id and the judge had nothing to attach a score to.
    Read INSIDE the span, while the context is still open.

    Prefer `span_trace_id` when the span is in hand. This reads the CONTEXT, and the
    context is the part that can go missing: four command turns in the 22:38 run
    (2026-10-21) answered correctly and reported no trace id at all, so their scores
    had nowhere to land and the run's evidence could not be opened.
    """
    if not configured():
        return ""
    try:
        import langfuse

        return str(langfuse.get_client().get_current_trace_id() or "")
    except Exception:  # noqa: BLE001 — rule 1
        return ""


def span_trace_id(span) -> str:
    """The trace id of a span we opened ourselves, or "" if it has none.

    This asks the OBJECT rather than the ambient context. The SDK assigns the id when it
    creates the span (`LangfuseSpan.trace_id`), so an id read here cannot be missing for
    the reason `current_trace_id()` can: a context that was copied into a worker thread,
    or one that the SDK had not populated yet.

    It is the id the RESPONSE carries, so it is also the id the eval attaches scores to.
    """
    try:
        return str(getattr(span, "trace_id", "") or "")
    except Exception:  # noqa: BLE001 — rule 1
        return ""


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


# ===========================================================================
# Consolidated from `chain.py` (Phase 7).
#
# The chain's identity: which model, which code, and what each run did.
#
# Google's requirement is that the prompt template, the chain definition, the tool
# wiring and the model pin are versioned *together* as one artifact, and that every
# execution records which combination served it. This module is where that identity
# lives, so "which prompt and which model produced this answer?" has one place to
# be read from rather than four.
#
#   - `MODEL_ID` is the chain's model. Pinned in code (`services/mcp/config.py`),
#     never read from the environment, because an environment default means a
#     deploy can change the model with no commit anywhere.
#   - `CODE_REVISION` is the identity of the running code, taken from the
#     deployment rather than maintained by hand.
#   - `record_execution()` emits one structured line per execution, success or
#     failure.
#
# The identity is deliberately *not* a version number typed by hand. An earlier
# version of this module carried a `CHAIN_REVISION = "1"` that whoever edited the
# prompt was expected to bump — a claim about the code rather than a fact derived
# from it, and one that fails silently: edit the prompt, forget the bump, and two
# different behaviours report the same revision for ever, so a bad answer can no
# longer be traced to the prompt that produced it. The deployment already produces
# an exact identifier for the prompt, the tool wiring and the model together, so
# that is what is reported here (`resolve_code_revision`).
# ===========================================================================

import json
import logging
import os
from collections.abc import Iterable, Mapping, Sequence

from services.mcp.config import GEMINI_MODEL

logger = logging.getLogger(__name__)

# The chain's model. Imported rather than redefined so there is exactly one
# place the string lives; this name is what the rest of the chain reads.
MODEL_ID = GEMINI_MODEL


def resolve_code_revision(env: Mapping[str, str] | None = None) -> str:
    """The identifier of the code that is running.

    Taken from the deployment rather than typed by hand, so it cannot go stale:

      1. `CODE_REVISION` — set by whoever deploys, when they can supply the
         commit they built from.
      2. `K_REVISION` — Cloud Run's own revision name, which every Cloud Run
         container carries, so a deploy that sets nothing is still identified.
         A revision maps to an image digest, and the digest to a build.
      3. `local` — an honest label for a run that is not a deployment.

    An empty or unexpanded value counts as absent: a `$COMMIT_SHA` that never got
    substituted is not an identity, and reporting it as one would be the same
    class of mistake this replaced.
    """
    values = os.environ if env is None else env
    for name in ("CODE_REVISION", "K_REVISION"):
        value = (values.get(name) or "").strip()
        if value and not value.startswith("$"):
            return value
    return "local"


CODE_REVISION = resolve_code_revision()

# The fields the record carries, so a test can assert the shape without
# duplicating the list. It is meant to be exhaustive, and it drifted: `tool_errors`
# and `langfuse_trace_id` were written to the record and never added here, which
# left a reader comparing the two and finding a discrepancy with no way to tell
# whether the record or the list was wrong. `error` is the one conditional field
# and is deliberately not listed — it is added only when the run failed.
RECORD_FIELDS = (
    "event",
    "trace",
    "code_revision",
    "model",
    "served_model",
    "outcome",
    "finish_reason",
    "filtered",
    "tokens",
    "question_chars",
    "duration_ms",
    "stages",
    "tool_calls",
    "tool_errors",
    "guardrail_flags",
    "langfuse_trace_id",
    # The chips the turn offered. Added with the deterministic selector (Phase 05 §2):
    # the list is chosen after the turn's span has closed, so this line is the only
    # record that carries it.
    "follow_ups",
    # What a control declared this turn to be. Added 2026-10-26, after a follow-up chip
    # looped in production and neither the record nor the trace could say whether the
    # declaration had arrived — the one fact the run's routing turns on was the one fact
    # neither artifact kept.
    "intent",
)


def record_execution(
    *,
    trace: str,
    question: str,
    stages: list[dict],
    duration_ms: int,
    outcome: str,
    finish_reason: str | None = None,
    filtered: list[str] | None = None,
    served_model: str | None = None,
    tokens: dict | None = None,
    tool_calls=(),
    tool_errors=(),
    guardrail_flags: Iterable[str] = (),
    langfuse_trace_id: str = "",
    follow_ups: Sequence[str] = (),
    intent: str | None = None,
    error: str | None = None,
    model: str = MODEL_ID,
    code_revision: str = CODE_REVISION,
) -> dict:
    """Log one execution as a single structured line, and return the record.

    Emitted for failures as well as successes: "every execution logs its inputs,
    its outputs, the intermediate state of each step, and the chain configuration
    used" is the requirement, and a record that only exists for answers that
    worked cannot explain the ones that did not.

    The question and answer *text* are deliberately absent. Their shape is
    recorded instead — length, which tools ran, what the guardrails flagged —
    because the text is patient-derived and a log store has a different privacy
    regime from the repository. If the answer text is ever needed for an incident
    review, that is a decision to take deliberately, with retention, not a
    side effect of adding a log line.

    JSON on one line so it is queryable by whatever the observability layer
    eventually picks; today it lands in Cloud Logging with the rest of stdout.

    `finish_reason` is the model's own reason for stopping, and it is always
    present as a key so a failure can be queried rather than inferred from an
    empty answer: `MAX_TOKENS` is a truncation worth retrying, `SAFETY` and
    `PROHIBITED_CONTENT` are decisions that will not change.

    `tokens` is the billed usage for the whole question — see
    `model_turn.token_usage` — and `served_model` is the version that answered
    rather than the pin we asked for. Both are always present as keys, and null
    when the response reported nothing, so a missing number reads as missing
    rather than as zero. The observability layer needs this before it can have a
    token metric at all.

    `filtered` names the content-filter categories that were flagged, when any
    were. It exists so that a refusal can be counted rather than only experienced.

    `tool_errors` carries the stable codes of the tool calls that failed, and is
    always present for the same reason: an isolation refusal is an incident, and
    an incident that only exists inside a model's prompt cannot be alerted on.

    `guardrail_flags` names the guardrails that fired, not how many fired. The
    count was recorded and the names were not, so the line said a guard had
    acted without saying which one: on 2026-09-18 a follow-up logged
    `"guardrail_flags":1` and the answer it served had a number deleted, and
    the name that would have identified the guard in one reading was the one
    field the record did not keep. A count is `len` of the names, so nothing is
    lost by keeping the names instead.
    """
    record = {
        "event": "agent_execution",
        "trace": trace,
        "code_revision": code_revision,
        "model": model,
        "served_model": served_model,
        "outcome": outcome,
        "finish_reason": finish_reason,
        "filtered": filtered,
        "tokens": tokens,
        "question_chars": len(question or ""),
        "duration_ms": duration_ms,
        "stages": list(stages),
        "tool_calls": [name for name in tool_calls],
        "tool_errors": list(tool_errors),
        "guardrail_flags": sorted(guardrail_flags),
        # Empty when tracing is off, which is a supported state: the field is
        # always present so "no trace" reads as no trace rather than as a
        # missing key, and so a log line can be joined to a trace when there is
        # one. This is the join the eval loop uses — a score attaches to a trace
        # id, and the record is where that id is durable.
        "langfuse_trace_id": langfuse_trace_id,
        # The chips this turn offered. Chosen from the record after the turn's span
        # closed, so they cannot ride on the trace; this line is where they are
        # recorded against the run (Phase 05 §2, 2026-10-23). Always present, empty when
        # the caller did not ask for them or when no template fired.
        "follow_ups": list(follow_ups),
        # The declaration a control sent, verbatim, and null when none was sent. Always
        # present for the same reason `langfuse_trace_id` is: "was this turn declared?"
        # has to be answerable from the artifact, because a run that goes back for
        # evidence it already holds is either a routing defect or a declaration that
        # never arrived, and the difference is this field.
        "intent": intent,
    }
    if error:
        record["error"] = error

    logger.info(json.dumps(record, separators=(",", ":"), default=str))
    return record
