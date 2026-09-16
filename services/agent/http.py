"""HTTP surface for the agent, for Django to proxy to.

Starlette rather than FastAPI: the MCP SDK already brings Starlette in, and the
API is two routes. Adding FastAPI for that is a dependency for nothing.

A fresh MCP session is opened per request rather than held open for the life of
the instance. The MCP server runs with `stateless_http=True` (§8) precisely
because streamable-HTTP sessions are per-instance and Cloud Run's load balancer
may route a follow-up request elsewhere, so a long-lived client session buys
nothing and breaks when an instance is recycled.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Callable

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from services.agent import chain, model_turn, stages
from services.agent.a2ui import compose_presentation
from services.agent.contracts import (
    AgentRequestError,
    AgentResponseError,
    parse_agent_request,
    validate_agent_success,
)
from services.agent.graph import ask, final_message, final_text
from services.agent.guardrail import guard_answer
from services.agent.mcp_client import MCP_TRANSPORT, toolbox
from services.mcp.runtime import requires_cloud_run_auth, timeout_chain

logger = logging.getLogger(__name__)

# The service is IAM-private, but a bounded input is still the caller's contract
# rather than an assumption about it.
MAX_QUESTION_CHARS = 2000

# Hard deadline on one question, just under the site's 120s proxy timeout so
# the caller gets a structured 504 instead of a dropped connection — and a
# runaway graph cannot keep billing after the caller is gone (ECC-02).
_, ASK_TIMEOUT_SECONDS = timeout_chain()

# How long the progress stream may stay silent before it sends a comment frame.
# A tool call is allowed 100s, and an SSE connection with no bytes on it for
# that long is closed by an idle timeout somewhere in the path (Cloud Run, the
# load balancer, the caller's own client) before the answer ever arrives. The
# frame carries no data — it exists to keep the socket alive.
STREAM_KEEPALIVE_SECONDS = 15

# Defense-in-depth (ECC-07): local and Cloud Run services share this decision.
REQUIRE_AUTH_HEADER = requires_cloud_run_auth()


# --- Request handling, shared by both routes -------------------------------
# /ask and /ask/stream differ only in how the answer is delivered. They must
# not differ in what they accept, what they refuse, or when the guardrails run,
# so everything up to and including the validated payload lives here and both
# routes call it.


def _auth_error(request: Request) -> JSONResponse | None:
    """The 401 response, or None when the request may proceed."""
    if REQUIRE_AUTH_HEADER and "authorization" not in request.headers:
        return JSONResponse(
            {"error": "unauthenticated",
             "message": "This service requires an identity token."},
            status_code=401,
        )
    return None


def _trace(request: Request) -> str:
    """The Cloud Trace id, or '-' when the request did not come via Cloud Run.

    Django forwards the id Cloud Run stamped on the user's request
    (X-Cloud-Trace-Context: TRACE_ID/SPAN_ID;o=1). Logging it here pairs this
    service's lines with Django's for the same user action.
    """
    return request.headers.get("x-cloud-trace-context", "").split("/", 1)[0] or "-"


async def _question_or_error(request: Request) -> tuple[str, str | None, JSONResponse | None]:
    """The validated question and its kind, or the response to return instead."""
    try:
        body = await request.json()
    except Exception:
        return "", None, JSONResponse({"error": "invalid_json"}, status_code=400)

    try:
        parsed = parse_agent_request(body, max_question_chars=MAX_QUESTION_CHARS)
    except AgentRequestError as exc:
        return "", None, JSONResponse(
            {"error": exc.code, "message": exc.message},
            status_code=exc.status_code,
        )
    return parsed["question"], parsed["kind"], None


class AgentAnswerUnavailable(Exception):
    """The chain finished but produced nothing shippable.

    Neither route can serve this as an answer: /ask turns it into its
    `answer_unavailable` 502, and /ask/stream turns it into a terminal error
    event. It exists as an exception rather than a return value so both routes
    are forced to handle it.

    `code` separates the cases a caller can act on differently — a refusal
    (`answer_refused`) and a truncated answer (`answer_truncated`) from an answer
    that simply did not arrive — and `finish_reason` carries the model's own
    reason into the execution record so a failure can be queried rather than
    inferred.
    """

    def __init__(
        self,
        message: str,
        code: str = "answer_unavailable",
        finish_reason: str | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.finish_reason = finish_reason


def _compose_success(question: str, state: dict, trace: str) -> dict:
    """Guard the answer, compose the presentation, validate the payload.

    The whole post-model pipeline, in one place, shared by both routes: this is
    what makes a streamed answer and a single-response answer the same object
    rather than two implementations that agree today.
    """
    # Deterministic post-hoc guardrails (P4): the LLM proposes, code disposes.
    # The served answer is the guarded one; flags are returned for observability
    # and so a client can choose to render a note.
    text = final_text(state)
    if not text:
        # An empty turn is not one thing. The response says which it was, and the
        # cases need different words: a truncated answer is worth retrying, a
        # refusal by the content filters is not — at temperature 0 the same
        # question reaches the same decision (ECC-12).
        code, sentence, reason = model_turn.classify(final_message(state))
        logger.error(
            "agent produced no final answer text trace=%s code=%s finish_reason=%s",
            trace, code, reason,
        )
        raise AgentAnswerUnavailable(sentence, code=code, finish_reason=reason)

    guarded = guard_answer(text, state["tool_calls"])

    # tool_calls are trimmed to what the site's canvas composition reads
    # (name + response) — the raw arguments never need to reach the browser
    # (ECC-08). Guardrails above ran on the full records.
    trimmed_calls = [
        {"name": tc["name"], "response": tc["response"]}
        for tc in state["tool_calls"]
    ]

    # The presentation contract is composed HERE, not in the BFF: citation
    # renumbering, source resolution, and the A2UI canvas are evidence
    # semantics — they belong to the layer that ran the guardrails and saw the
    # tool evidence. Django passes them through; the browser renders them.
    presentation = compose_presentation(question, guarded["answer"], trimmed_calls)

    payload = {
        "question": question,
        "answer": presentation["answer"],
        "guardrail_flags": guarded["flags"],
        "tool_calls": trimmed_calls,
        "a2ui": presentation["a2ui"],
        "sources": presentation["sources"],
        "model": chain.MODEL_ID,
        "mcp_transport": MCP_TRANSPORT,
    }
    try:
        validate_agent_success(payload)
    except AgentResponseError:
        logger.error("agent produced a payload outside its success contract trace=%s", trace)
        raise AgentAnswerUnavailable(
            "The agent did not produce an answer. Please retry.",
            finish_reason=model_turn.finish_reason(final_message(state)),
        )

    return payload


class _StageLog:
    """Collects the stages a chain run emitted, and when each one started.

    Both routes need the same timing data for the execution record, so it is
    gathered once. `/ask` uses this for the record only; `/ask/stream` passes a
    sink as well, so the same events are also relayed to the caller. That is what
    keeps "what the user saw" and "what was recorded" the same list rather than
    two lists that agree today.
    """

    def __init__(self, sink: Callable[[dict], None] | None = None):
        self.stages: list[dict] = []
        self._sink = sink
        self._started = time.monotonic()

    def __call__(self, event: dict) -> None:
        # Only the stage, its tool and its timing are recorded. A result stage's
        # payload — a count, a failure — is deliberately not: the wire carries
        # what a waiting user needs, and the record stays as narrow as it was.
        self.stages.append({
            "stage": event["stage"],
            "tool": event.get("tool"),
            "ms": self.duration_ms,
        })
        if self._sink is not None:
            self._sink(event)

    @property
    def duration_ms(self) -> int:
        return int((time.monotonic() - self._started) * 1000)

    def record(self, trace: str, question: str, outcome: str, **kwargs) -> dict:
        """Emit the one record for this execution."""
        return chain.record_execution(
            trace=trace,
            question=question,
            stages=self.stages,
            duration_ms=self.duration_ms,
            outcome=outcome,
            **kwargs,
        )


async def _run_chain(
    question: str, question_kind: str | None = None, on_event=None
) -> dict:
    """Run the graph under the wall-clock deadline, inside one MCP session.

    Extracted because both routes need the identical bound: the deadline is a
    spend control (ECC-02), and a second copy of it is a second place for the
    two routes to disagree about how long a question may take.
    """
    async with asyncio.timeout(ASK_TIMEOUT_SECONDS):
        async with toolbox() as box:
            return await ask(
                box, question, question_kind=question_kind, on_event=on_event
            )


async def health(request: Request) -> JSONResponse:
    """Shallow by design: no Vertex, no MCP, no BigQuery.

    A deep check would bill on every probe of a scale-to-zero service and would
    mark the container unhealthy whenever a dependency blipped.

    No project/region/MCP URL (ECC-06): the route is unauthenticated at the
    app layer, and internal topology must not leak to a direct caller. The code
    revision is the one thing that does appear, because it names the deployed
    code rather than the infrastructure behind it, and without it the only way to
    answer "did my deploy land?" is to read a revision list and guess.
    """
    return JSONResponse(
        {
            "status": "ok",
            "model": chain.MODEL_ID,
            "code_revision": chain.CODE_REVISION,
            "mcp_transport": MCP_TRANSPORT,
        }
    )


async def ask_route(request: Request) -> JSONResponse:
    error = _auth_error(request)
    if error is not None:
        return error
    trace = _trace(request)
    question, question_kind, error = await _question_or_error(request)
    if error is not None:
        return error

    # Input validation is settled: from here the chain runs, and every exit below
    # emits exactly one execution record.
    log = _StageLog()
    try:
        state = await _run_chain(question, question_kind, on_event=log)
    except TimeoutError:
        logger.error("agent /ask timed out after %.0fs trace=%s", ASK_TIMEOUT_SECONDS, trace)
        log.record(trace, question, "timeout", error="timeout")
        return JSONResponse(
            {"error": "timeout",
             "message": f"The agent did not answer within {ASK_TIMEOUT_SECONDS:.0f}s."},
            status_code=504,
        )
    except Exception as exc:
        # Never let an infrastructure failure surface as a plausible answer.
        # The MCP SDK raises asyncio.ExceptionGroup when a transport task
        # fails; unwrap it so the real cause is logged instead of hiding
        # behind "unhandled errors in a TaskGroup".
        cause = exc
        if isinstance(exc, BaseExceptionGroup):
            detail = " | ".join(str(e) for e in exc.exceptions)
            cause = RuntimeError(f"{type(exc).__name__}: {detail}")
            cause.__cause__ = exc  # keep the group as the logged root cause
        # Detail stays server-side (ECC-06): exception text routinely embeds
        # the private MCP URL, IAM/audience detail and table names. The caller
        # gets a stable code + correlation id that pairs with the log line.
        correlation_id = uuid.uuid4().hex[:12]
        logger.error("agent /ask failed [%s] trace=%s", correlation_id, trace, exc_info=cause)
        log.record(trace, question, "error", error="agent_failed")
        return JSONResponse(
            {"error": "agent_failed",
             "message": "The agent failed to answer. Please retry.",
             "correlation_id": correlation_id},
            status_code=502,
        )

    try:
        payload = _compose_success(question, state, trace)
    except AgentAnswerUnavailable as exc:
        # The record says which of the three it was, in the code and in the
        # model's own finish reason.
        log.record(
            trace, question, "error",
            error=exc.code, finish_reason=exc.finish_reason,
        )
        return JSONResponse(
            {"error": exc.code, "message": str(exc)},
            status_code=502,
        )
    log.record(
        trace, question, "ok",
        finish_reason=model_turn.finish_reason(final_message(state)),
        tool_calls=[call["name"] for call in payload["tool_calls"]],
        guardrail_flags=len(payload["guardrail_flags"]),
    )
    return JSONResponse(payload)


# --- Progress streaming -----------------------------------------------------
# The answer is not streamed, and cannot be: `guard_answer` rewrites the text
# after the model finishes, so a token stream would show a draft being
# corrected — and for a clinical answer the corrected part is the part that
# matters. What streams is which step is running, followed by exactly one
# terminal frame carrying the same validated object /ask returns.


def _sse(event: str, data: dict) -> str:
    """One server-sent event: a named event, one JSON data line, a blank line."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# Queued when the chain finishes, so the relay loop can tell "the chain is
# still working but quiet" (send a keepalive) from "the chain is finished"
# (stop). A sentinel rather than a task-state check, because the check can only
# be made between `await`s and would therefore always run one keepalive late.
_CHAIN_DONE = object()


def _error_frame(code: str, message: str, correlation_id: str | None = None) -> str:
    """The terminal frame for a failure that happened after the stream opened.

    Once the first byte is on the wire the status code is spent, so failures
    that occur mid-stream cannot be a 502. They are reported here instead,
    carrying the same code, message and correlation id that /ask would have
    put in its error body, so a caller handles both the same way.
    """
    body = {"error": code, "message": message}
    if correlation_id:
        body["correlation_id"] = correlation_id
    return _sse(stages.STAGE_ERROR, body)


async def ask_stream_route(request: Request) -> Response:
    """`/ask`, with the chain's stages streamed ahead of the answer.

    Failures before the stream opens (no identity header, bad JSON, question
    too long) are returned as ordinary responses with their real status codes —
    the same ones /ask uses. Failures after it opens arrive as a terminal
    `error` event, because there is no status code left to send.
    """
    error = _auth_error(request)
    if error is not None:
        return error
    trace = _trace(request)
    question, question_kind, error = await _question_or_error(request)
    if error is not None:
        return error

    return StreamingResponse(
        _stream_chain(question, trace, question_kind),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # A cheap, explicit "do not buffer this" for any proxy in the path.
            # The classic failure without it is that every frame arrives at
            # once when the response ends, which looks exactly like no
            # streaming at all.
            "X-Accel-Buffering": "no",
        },
    )


async def _stream_chain(question: str, trace: str, question_kind: str | None = None):
    """Yield stage frames as they happen, then exactly one terminal frame.

    Nothing is composed here. The stages come from the chain (`graph._emit`) at
    the points where work actually starts, and the answer comes from the same
    `_compose_success` the single-response route uses, so the two routes cannot
    drift into serving different answers.
    """
    queue: asyncio.Queue = asyncio.Queue()

    # The events the caller sees are also the record of what ran: the sink relays
    # them, the log timestamps them.
    log = _StageLog(sink=queue.put_nowait)
    task = asyncio.create_task(_run_chain(question, question_kind, on_event=log))

    def _finished(finished: asyncio.Task) -> None:
        # Reading the exception marks it retrieved: it is logged where it
        # happened, and without this a chain that failed after the caller
        # disconnected would surface as "exception was never retrieved".
        if not finished.cancelled():
            finished.exception()
        # The sentinel is what ends the loop the moment the chain does. Waiting
        # only on the queue would instead stall the answer for up to a whole
        # keepalive interval after the work had already finished — the progress
        # stream would make the common case slower than not streaming at all.
        # Stages were queued before the task completed, so FIFO ordering keeps
        # the sentinel last.
        queue.put_nowait(_CHAIN_DONE)

    task.add_done_callback(_finished)

    try:
        while True:
            try:
                event = await asyncio.wait_for(
                    queue.get(), timeout=STREAM_KEEPALIVE_SECONDS
                )
            except asyncio.TimeoutError:
                # The chain is quiet — typically one slow tool call. Keep the
                # socket alive without pretending anything happened.
                yield ": keepalive\n\n"
                continue
            if event is _CHAIN_DONE:
                break
            yield _sse(event["stage"], event)

        try:
            state = task.result()
        except TimeoutError:
            logger.error(
                "agent /ask/stream timed out after %.0fs trace=%s",
                ASK_TIMEOUT_SECONDS, trace,
            )
            log.record(trace, question, "timeout", error="timeout")
            yield _error_frame(
                "timeout",
                f"The agent did not answer within {ASK_TIMEOUT_SECONDS:.0f}s.",
            )
            return
        except Exception as exc:
            # Same unwrapping and same policy as /ask: the detail stays in the
            # log (ECC-06), the caller gets a code and a correlation id.
            cause = exc
            if isinstance(exc, BaseExceptionGroup):
                detail = " | ".join(str(e) for e in exc.exceptions)
                cause = RuntimeError(f"{type(exc).__name__}: {detail}")
                cause.__cause__ = exc
            correlation_id = uuid.uuid4().hex[:12]
            logger.error(
                "agent /ask/stream failed [%s] trace=%s", correlation_id, trace, exc_info=cause
            )
            log.record(trace, question, "error", error="agent_failed")
            yield _error_frame(
                "agent_failed", "The agent failed to answer. Please retry.", correlation_id
            )
            return

        # The model has stopped; the guardrails have not run yet. Say so rather
        # than letting the stream go silent at the point closest to the answer.
        yield _sse(stages.STAGE_VERIFY, stages.verify_event())

        try:
            payload = _compose_success(question, state, trace)
        except AgentAnswerUnavailable as exc:
            log.record(
                trace, question, "error",
                error=exc.code, finish_reason=exc.finish_reason,
            )
            yield _error_frame(exc.code, str(exc))
            return

        log.record(
            trace, question, "ok",
            finish_reason=model_turn.finish_reason(final_message(state)),
            tool_calls=[call["name"] for call in payload["tool_calls"]],
            guardrail_flags=len(payload["guardrail_flags"]),
        )
        yield _sse(stages.STAGE_ANSWER, payload)
    finally:
        # Reached when the caller disconnects (Starlette closes the generator)
        # as well as on every normal exit. Cancelling the chain stops it before
        # its next superstep instead of billing to completion for an answer
        # nobody is waiting for; a model call already in flight is not
        # interruptible, so this bounds the waste rather than eliminating it.
        if not task.done():
            task.cancel()


app = Starlette(
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/ask", ask_route, methods=["POST"]),
        Route("/ask/stream", ask_stream_route, methods=["POST"]),
    ]
)


if __name__ == "__main__":
    import uvicorn

    # Python's root logger defaults to WARNING and nothing else in this process
    # configures logging, so the per-request INFO line in ask_route would be
    # discarded — Cloud Run collects stdout and stderr, and nothing was writing
    # there at INFO. One line per request at INFO, everything else at ERROR, is
    # the intended volume. uvicorn's own dictConfig leaves the root logger
    # alone, so this survives the call below.
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(asctime)s %(name)s %(message)s",
    )

    # Cloud Run injects PORT. Do not hardcode it.
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
