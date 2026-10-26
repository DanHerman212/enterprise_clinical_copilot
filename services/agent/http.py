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
import re
import uuid

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from services.agent import chain, model_turn, routing, stages
from services.agent import questions, turn
from services.agent.contracts import AgentRequestError, parse_agent_request
from services.agent.graph import ask, final_message
from services.agent.mcp_client import MCP_TRANSPORT, toolbox
from services.agent import observability
from services.agent.turn import (
    AgentAnswerUnavailable,
    StageLog as _StageLog,
    _response_fields,
    compose_success as _compose_success,
    unserved_admission as _unserved_admission,
)
from services.mcp.runtime import requires_cloud_run_auth, timeout_chain

logger = logging.getLogger(__name__)

# The service is IAM-private, but a bounded input is still the caller's contract
# rather than an assumption about it.
MAX_QUESTION_CHARS = 2000

# Hard deadline on one question, just under the site's 120s proxy timeout so
# the caller gets a structured 504 instead of a dropped connection — and a
# runaway graph cannot keep billing after the caller is gone (ECC-02).
#
# The route owns the number: its 504 message quotes it, and it is handed to
# `turn.run_chain` rather than read there, so a caller can bound one question
# without changing the process.
ASK_TIMEOUT_SECONDS = timeout_chain().ask


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


async def _question_or_error(
    request: Request,
) -> tuple[dict | None, JSONResponse | None]:
    """The validated request, or the response that refuses it.

    The whole parsed request is returned rather than its fields positionally: the
    shape has grown twice (the command, then the admission, now the command's
    topic), and a tuple unpacked in two routes is where the next field gets
    dropped by whichever unpack was not updated.
    """
    try:
        body = await request.json()
    except Exception:
        return None, JSONResponse(
            {"error": "invalid_json"}, status_code=400
        )

    try:
        parsed = parse_agent_request(body, max_question_chars=MAX_QUESTION_CHARS)
    except AgentRequestError as exc:
        return None, JSONResponse(
            {"error": exc.code, "message": exc.message},
            status_code=exc.status_code,
        )
    refused = _unknown_command(parsed)
    if refused is not None:
        return None, refused
    # No routing here. A command arrives as a command — a chip selection, or a leading
    # `/slash` token that `parse_agent_request` has already resolved into `kind` — and
    # everything else is a question for the reasoning loop.
    #
    # What used to stand here was a keyword router: any free-text question containing
    # "medications" or "sodium" was rewritten into `kind="meds"`/`"labs"` and sent to
    # the stateless command path, which ignores `turns` by design. On a follow-up, that
    # meant the question was answered with Turn 1's static caption — the reader asked
    # whether the discharge list held an anticoagulant and was given the medication
    # count again (2026-10-22, every turn-2 census failure). A word list cannot tell an
    # extraction request from an analytical question, so it no longer tries.
    return parsed, None


#: The shape of a typed command, anchored at the start of the question.
_SLASH_TOKEN = re.compile(r"^/([A-Za-z0-9_-]+)")


def _asked_questions(parsed: dict) -> list[str]:
    """Every question this thread has already put to the service, in order.

    The replayed turns are the caller's own transcript, and they are the only record of
    what was asked: the follow-up row a reader has clicked is a question in that list,
    so a chip matching one of them has already been used up.

    This turn's own entry is `own_words`, not `question`. `question` is the composed
    prompt — the caller's words plus `For admission 90000015.` — and a chip never
    carries that sentence, so the composed text matched no chip and the row offered back
    the question the reader had just clicked (measured in the production UI,
    `agent-00175-ttr`, 2026-10-24). The replayed turns hold the words the reader used, so
    the current turn has to be read the same way or the two ends of the comparison
    disagree.

    The replayed turns need the same reduction, and they need it for a second reason:
    the site stores the COMPOSED question on each stored turn, because that is what the
    agent was asked. Reading those raw matched on the turn a chip was clicked and matched
    nothing afterwards, so a chip asked on Turn 2 came back on Turn 3 — the row advanced
    once and then rehydrated (measured in the production UI, `agent-00176-ngw`,
    2026-10-24). `questions.words_of` removes the admission sentence from every stored
    question, which is what makes "already asked" a fact about the thread rather than
    about the most recent turn.
    """
    asked = [questions.words_of(turn.get("question"))
             for turn in parsed.get("turns") or [] if isinstance(turn, dict)]
    asked.append(str(parsed.get("own_words")
                     or questions.words_of(parsed.get("question") or "")))
    return [question for question in asked if question.strip()]


def _unknown_command(parsed: dict) -> JSONResponse | None:
    """Refuse a command-shaped question that names no command, before the chain runs.

    A slash token at the start of a question is the caller asking for a command by
    name. When no command answers to it, the honest reply is the list of names that
    do — not a reasoning turn that searches the chart, spends the tool budget and
    reports a `Reconciled` badge for what is a typo.

    Deliberately a 400 rather than a composed answer: nothing was retrieved, so there
    is no surface to draw and no citation to resolve, and the caller's request really
    was malformed. The message is user-facing text, which is what lets the site relay
    it verbatim instead of replacing it with a generic failure.

    It opens a span all the same. The answer is a decision worth counting, and the
    trace id it returns is what lets a caller (the eval, a dashboard) attach a score to
    the turn that produced it.
    """
    if parsed.get("kind"):
        # A chip or palette selection is an explicit command; it was already resolved.
        return None
    question = (parsed.get("question") or "").strip()
    # Anything that begins with a slash is a command-shaped question, including the
    # degenerate bare "/" and "/ ". The token regex needs at least one character, so
    # a bare slash used to match nothing and fall through to free text — which ran a
    # five-tool retrieval turn and answered "no command" with a chart search. A slash
    # the caller typed is a command request whether or not they finished typing it.
    if not question.startswith("/"):
        return None
    match = _SLASH_TOKEN.match(question)
    token = ("/" + match.group(1).lower()) if match else "/"
    if routing.command_for_slash(token) is not None:
        return None
    available = ", ".join(sorted(c.slash for c in routing.COMMANDS))
    with observability.observation(
        "command.unknown", trace_metadata=turn.refusal_trace_metadata(parsed),
    ) as span:
        trace_id = (observability.span_trace_id(span)
                    or observability.current_trace_id())
    return JSONResponse(
        {"error": "unknown_command",
         "message": f"There is no {token} command. Available commands: {available}.",
         "langfuse_trace_id": trace_id},
        status_code=400,
    )


async def _run_chain(
    parsed: dict,
    on_event=None,
) -> dict:
    """The seam the route tests stub through.

    Orchestration lives in `turn`; the two things it drives are passed in
    rather than imported there. `ask` and `toolbox` are rebound in THIS module
    by every route test (`patch.object(srv, "ask")` 27 times,
    `patch.object(srv, "toolbox")` 28 times), and a name imported inside
    `turn` would not be the name those patches replace: every fake would be
    bypassed, in silence, and the suite would be calling the real graph.
    """
    return await turn.run_chain(
        parsed, on_event=on_event, ask=ask, toolbox=toolbox,
        deadline=ASK_TIMEOUT_SECONDS,
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
    parsed, error = await _question_or_error(request)
    if error is not None:
        return error

    # Input validation is settled: from here the chain runs, and every exit below
    # emits exactly one execution record.
    log = _StageLog(intent=parsed.get("intent"))
    # Before the chain, because "is this admission served?" is a lookup and a lookup
    # answered deterministically cannot be talked into inventing a patient.
    refusal = await _unserved_admission(parsed, trace)
    if refusal is not None:
        return JSONResponse(refusal)
    try:
        state = await _run_chain(parsed, on_event=log)
    except TimeoutError:
        logger.error("agent /ask timed out after %.0fs trace=%s", ASK_TIMEOUT_SECONDS, trace)
        log.record(trace, parsed["question"], "timeout", error="timeout")
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
        log.record(trace, parsed["question"], "error", error="agent_failed")
        return JSONResponse(
            {"error": "agent_failed",
             "message": "The agent failed to answer. Please retry.",
             "correlation_id": correlation_id},
            status_code=502,
        )

    try:
        payload = await _compose_success(
            parsed["question"], state, trace,
            want_follow_ups=parsed.get("follow_ups") is True,
            outpatient=parsed.get("hadm_id") in routing.OUTPATIENT_HADM_IDS,
            asked=_asked_questions(parsed),
        )
    except AgentAnswerUnavailable as exc:
        # The record says which of the three it was, in the code and in the
        # model's own finish reason.
        log.record(
            trace, parsed["question"], "error",
            error=exc.code, finish_reason=exc.finish_reason,
            **_response_fields(state),
        )
        return JSONResponse(
            {"error": exc.code, "message": str(exc)},
            status_code=502,
        )
    log.record(
        trace, parsed["question"], "ok",
        finish_reason=model_turn.finish_reason(final_message(state)),
        **_response_fields(state),
        tool_calls=[call["name"] for call in payload["tool_calls"]],
        # The names, because which guard fired is the whole diagnostic: a count
        # of 1 cannot distinguish a rewritten dose from a deleted citation.
        guardrail_flags=payload["guardrail_flags"],
        # The log line and the trace describe the same run, so the line carries
        # the id that joins them: without it, "the answer looked wrong" is a
        # question about a log line or about a trace, and never about both.
        langfuse_trace_id=state.get("langfuse_trace_id") or "",
        # The chips this turn offered, when the caller asked for them. They are chosen
        # after the turn's span closed, so this line is where they are recorded against
        # the run rather than the trace (see `followups`, Phase 05 §2).
        follow_ups=state.get("follow_ups") or [],
    )
    # The command a free-text question was routed to, when it was routed at all.
    # Carried on the payload because the console draws it as a pill beside the turn:
    # the reader is told which command answered them rather than left to infer it
    # from the shape of the answer.
    if parsed.get("routed_to"):
        payload["routed_to"] = parsed["routed_to"]
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
    parsed, error = await _question_or_error(request)
    if error is not None:
        return error

    # The same seam check as /ask. A stream that opened and then answered about a
    # patient who does not exist would be no better than the blocking form.
    refusal = await _unserved_admission(parsed, trace)
    if refusal is not None:
        async def _refusal_stream():
            yield _sse(stages.STAGE_ANSWER, refusal)
        return StreamingResponse(
            _refusal_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return StreamingResponse(
        _stream_chain(parsed, trace),
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


def _keeps_the_announcement(stage: str) -> bool:
    """Whether prose announced so far still belongs to the answer that will be served.

    False for every frame that means the run went back for evidence. A tool round and a
    retrieval demand both end the pass that was speaking, and the graph answers again:
    the prose the reader was shown belongs to a draft that is not served, so comparing it
    against the settled answer would report a correction to text nobody is holding — and
    report it as a withdrawal from the record whenever a guardrail flag was set for any
    other reason.

    True for `verify` because it is emitted after the chain has finished and before
    composition; clearing on it would discard every announcement and the verdict would
    never fire at all.
    """
    return stage in (stages.STAGE_ANSWER_DELTA, stages.STAGE_VERIFY)


def _prose_settled(announced: list[str], payload: dict, trace: str) -> str | None:
    """Whether the prose a caller watched arrive survived into the served answer.

    Returns None when nothing was announced, or when nothing moved — the caller was
    shown the answer and there is nothing to reconcile. Otherwise it names the swap
    the client owes the caller, which is the whole of the Dual Swap Protocol:

      * ``"silent"`` — the prose moved and no guardrail fired. What changed is
        envelope unwrapping, whitespace, or a citation marker renumbered by the
        citation layer. The settled text is the right one and the caller does not
        need to know the two differed, because nothing was taken away.
      * ``"visible"`` — a guardrail fired, so the announcement was a draft and the
        settled answer withdrew part of it: a dose that could not be verified against
        the retrieved notes, a risk figure with nothing behind it, an age the source
        redacted, a citation that resolved to nothing. Replacing that quietly is
        exactly the dishonesty the streaming decision was about, so the caller is
        told, and the client shows it in the Evidence-Check badge.

    Why the verdict is computed here rather than by the client. The client can see
    that the text moved — it rendered the deltas — but only this process knows *why*,
    because only this process ran the guardrails. A client left to infer it from
    `guardrail_flags` would have to re-derive which flags mean content was removed
    and which mean it was reformatted, and that is a rule about the guardrails living
    in the browser.

    The difference is no longer logged as a warning. It was one while the policy was
    "announce nothing that might move", where any movement meant the assumption had
    broken; now movement is the expected outcome on the paths that stream, so a
    warning would fire on the normal case and stop meaning anything. Both outcomes are
    reported at info with the texts, which is the evidence if the swap ever needs
    judging, and the thing to watch is how often the visible one happens.
    """
    if not announced:
        return None
    watched = "".join(announced).strip()
    served = (payload.get("answer") or "").strip()
    if watched == served:
        return None

    swap = "visible" if payload.get("guardrail_flags") else "silent"
    logger.info(
        "streamed prose settled as a %s swap trace=%s (announced=%r served=%r)",
        swap, trace, watched[:160], served[:160],
    )
    return swap


async def _stream_chain(
    parsed: dict,
    trace: str,
):
    """Yield stage frames as they happen, then exactly one terminal frame.

    Nothing is composed here. The stages come from the chain (`graph._emit`) at
    the points where work actually starts, and the answer comes from the same
    `_compose_success` the single-response route uses, so the two routes cannot
    drift into serving different answers.
    """
    queue: asyncio.Queue = asyncio.Queue()

    # The events the caller sees are also the record of what ran: the sink relays
    # them, the log timestamps them.
    log = _StageLog(sink=queue.put_nowait, intent=parsed.get("intent"))
    task = asyncio.create_task(_run_chain(parsed, on_event=log))

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

    announced: list[str] = []

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
            # Kept so the settled answer can be compared with what was announced
            # once the guardrails have run (see `_prose_settled`). Cleared when the run
            # goes back for more evidence — a tool round, or the retrieval demand.
            #
            # What was announced before that belongs to a pass whose answer is not the
            # one served: the model spoke, was sent back for a retrieval, and answered
            # again. Comparing that first pass against the served text reported a
            # correction to prose the reader is no longer holding — and because the
            # verdict is `visible` whenever a guardrail flag is set, it reported it as a
            # withdrawal from the record. What the badge is for is the second pass: the
            # text on screen at the end against the text served, which is what the
            # accumulator holds once the loop has stopped moving.
            #
            # `verify` is excluded because it arrives after the chain has finished and
            # before composition — clearing on it would discard every announcement and
            # the verdict would never fire at all.
            if event.get("stage") == stages.STAGE_ANSWER_DELTA:
                announced.append(event.get("text") or "")
            elif not _keeps_the_announcement(event.get("stage")):
                announced.clear()
            yield _sse(event["stage"], event)

        try:
            state = task.result()
        except TimeoutError:
            logger.error(
                "agent /ask/stream timed out after %.0fs trace=%s",
                ASK_TIMEOUT_SECONDS, trace,
            )
            log.record(trace, parsed["question"], "timeout", error="timeout")
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
            log.record(trace, parsed["question"], "error", error="agent_failed")
            yield _error_frame(
                "agent_failed", "The agent failed to answer. Please retry.", correlation_id
            )
            return

        # The model has stopped; the guardrails have not run yet. Say so rather
        # than letting the stream go silent at the point closest to the answer.
        yield _sse(stages.STAGE_VERIFY, stages.verify_event())

        try:
            payload = await _compose_success(
                parsed["question"], state, trace,
                want_follow_ups=parsed.get("follow_ups") is True,
                outpatient=parsed.get("hadm_id") in routing.OUTPATIENT_HADM_IDS,
                asked=_asked_questions(parsed),
            )
        except AgentAnswerUnavailable as exc:
            log.record(
                trace, parsed["question"], "error",
                error=exc.code, finish_reason=exc.finish_reason,
                **_response_fields(state),
            )
            yield _error_frame(exc.code, str(exc))
            return

        # The verdict travels as its own frame rather than as a key on the answer.
        # The answer object is the contract both routes share — `/ask` returns it whole,
        # and a test asserts the streamed frame carries the same fields — and how the
        # transport performed is not part of the answer. A frame is also the safe shape:
        # `stream_prose` is opt-in, so only a caller that asked to watch the prose arrive
        # ever receives this, and that caller is the client that knows the term. An older
        # client is never sent one.
        swap = _prose_settled(announced, payload, trace)
        if swap is not None:
            yield _sse(stages.STAGE_PROSE_SETTLED, stages.prose_settled_event(swap))

        # Same field as the blocking route, and for the same reason: the console draws
        # the pill from the answer it receives, and the console receives this one.
        if parsed.get("routed_to"):
            payload["routed_to"] = parsed["routed_to"]
        log.record(
            trace, parsed["question"], "ok",
            finish_reason=model_turn.finish_reason(final_message(state)),
            **_response_fields(state),
            tool_calls=[call["name"] for call in payload["tool_calls"]],
            guardrail_flags=payload["guardrail_flags"],
            langfuse_trace_id=state.get("langfuse_trace_id") or "",
            # The same field the blocking route records, for the same reason.
            follow_ups=state.get("follow_ups") or [],
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
