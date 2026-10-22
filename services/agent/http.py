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
import time
import uuid
from typing import Callable

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from services.agent import chain, model_turn, routing, stages
from services.agent import admission_context, followups
from services.agent.a2ui import compose_presentation
from services.agent.contracts import (
    RE_DERIVABLE_TOOLS,
    AgentRequestError,
    AgentResponseError,
    parse_agent_request,
    validate_agent_success,
)
from services.agent.graph import ask, final_message, final_text
from services.agent.guardrail import guard_answer
from services.agent.mcp_client import MCP_TRANSPORT, toolbox
from services.agent import observability
from services.agent.payload import build_ui_components
from services.agent.served_admissions import is_served
from services.mcp.runtime import requires_cloud_run_auth, timeout_chain

logger = logging.getLogger(__name__)

# The service is IAM-private, but a bounded input is still the caller's contract
# rather than an assumption about it.
MAX_QUESTION_CHARS = 2000

# Hard deadline on one question, just under the site's 120s proxy timeout so
# the caller gets a structured 504 instead of a dropped connection — and a
# runaway graph cannot keep billing after the caller is gone (ECC-02).
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


#: Served when a caller names an admission this deployment cannot open.
ADMISSION_NOT_FOUND = (
    "Admission not found. This deployment can only answer for the admissions it "
    "serves, and that admission is not one of them, so nothing was retrieved and "
    "no estimate was produced."
)


async def _unserved_admission(parsed: dict, trace: str) -> dict | None:
    """A clean refusal for an admission this deployment does not serve.

    Added 2026-10-18 after probe `rob-03` asked about `90009999` and the model, given
    no tool result at all, answered "The /risk command provides the readmission risk
    for this admission." — a plausible-sounding sentence about a patient who does not
    exist. The model should never have been asked: "is this admission served?" is a
    lookup, and a lookup answered deterministically cannot drift between runs.

    A 200 rather than a 400 because nothing about the request is malformed — the
    question is well formed and the answer to it is that the record is not here. The
    composed envelope is the same one every other turn returns, so a client renders it
    without a special case, and `routed_to` says `abstain` so the console draws no
    routing pill.
    """
    hadm = parsed.get("hadm_id")
    if hadm is None or is_served(hadm):
        return None
    # Its own record: this exit happens before the route's stage log exists, and every
    # exit from a request records exactly one line.
    _StageLog().record(trace, parsed.get("question") or "", "refused",
                       error="admission_not_found")
    # Composed through the shared pipeline rather than hand-built, so this refusal
    # carries exactly the fields every other answer carries — including its trace id,
    # which the pipeline cannot supply here because no span is open around this exit.
    with observability.observation(
        "turn.refused", trace_metadata=_refusal_trace_metadata(parsed),
    ) as span:
        payload = await _compose_success(
            parsed.get("question") or "",
            {"deterministic_answer": ADMISSION_NOT_FOUND, "tool_calls": [],
             "command": None},
            trace,
        )
        # A refusal is a turn: it has a question and it has an answer, and the span
        # said neither. Without this the only trace of the decision is the `outcome`
        # tag, and a reviewer exporting the tree sees an empty root for the very turns
        # whose correctness was in question.
        observability.set_trace_io(
            parsed.get("question") or "", ADMISSION_NOT_FOUND, span=span,
        )
        payload["langfuse_trace_id"] = (
            observability.span_trace_id(span) or observability.current_trace_id()
        )
    payload["guardrail_flags"] = list(payload.get("guardrail_flags") or []) + [
        "admission_not_found"
    ]
    payload["routed_to"] = "abstain"
    return payload


def _refusal_trace_metadata(parsed: dict) -> dict:
    """Trace attributes for a turn that was refused before any work ran.

    A refusal is a decision the service made, and four of them in the 2026-10-21 run
    (`/`, `/labb`, `/medss`, `/ehrr`) were correct answers that left nothing behind: no
    span, no trace, no id in the response, so the eval had nowhere to put their scores
    and reported them unattributed. `run_tag` arrives on the request, so the trace is
    filterable exactly like an answered turn's.
    """
    run_tag = (parsed.get("run_tag") or "").strip()
    hadm = parsed.get("hadm_id")
    tags = [f"run:{run_tag}", "turn:1"] if run_tag else ["turn:1"]
    if hadm is not None:
        tags.insert(1, f"hadm:{hadm}")
    return {
        "langfuse_session_id": (f"{run_tag}:hadm_{hadm}" if run_tag and hadm else None),
        "langfuse_tags": tags,
        "hadm_id": hadm,
        "outcome": "refused",
    }


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
        "command.unknown", trace_metadata=_refusal_trace_metadata(parsed),
    ) as span:
        trace_id = (observability.span_trace_id(span)
                    or observability.current_trace_id())
    return JSONResponse(
        {"error": "unknown_command",
         "message": f"There is no {token} command. Available commands: {available}.",
         "langfuse_trace_id": trace_id},
        status_code=400,
    )


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


def _tool_error_codes(state: dict) -> list[str]:
    """The stable codes of the tool calls that failed, in call order.

    A refused call already reaches the model and the caller as a code plus a
    sentence; this is the third place it has to arrive, and the only one an
    operator can count. Note text and question text are deliberately absent from
    the record, and so is anything else a tool said about itself.
    """
    return [
        call["response"]["error"]
        for call in state.get("tool_calls") or []
        if isinstance(call.get("response"), dict) and call["response"].get("error")
    ]


def _response_fields(state: dict) -> dict:
    """What the response reported, in the shape the record carries.

    Gathered in one place because both routes and both outcomes record them, and a
    record that carries tokens on success but not on failure would make a metric
    that is wrong exactly where it is most interesting. The same is true of a
    filtered response: it has to be visible on the runs that were refused, not only
    on the ones that were not. A failed tool call is the third case.
    """
    message = final_message(state)
    return {
        "tokens": model_turn.token_usage(state),
        "served_model": model_turn.served_model(message),
        "filtered": model_turn.filtered_categories(message),
        "tool_errors": _tool_error_codes(state),
    }


async def _compose_success(
    question: str, state: dict, trace: str, *, want_follow_ups: bool = False,
    outpatient: bool = False,
) -> dict:
    """Guard the answer, compose the presentation, validate the payload.

    The whole post-model pipeline, in one place, shared by both routes: this is
    what makes a streamed answer and a single-response answer the same object
    rather than two implementations that agree today.

    Async for one reason: the follow-up questions are proposed by a model call. It is
    the last thing this does, on an answer that is already written, and it cannot fail
    the turn (`followups.suggest` returns an empty list rather than raising) — so the
    only thing it can cost is the follow-up row.
    """
    # Deterministic post-hoc guardrails (P4): the LLM proposes, code disposes.
    # The served answer is the guarded one; flags are returned for observability
    # and so a client can choose to render a note.
    # A command may bring its own answer text. `/meds` extracts a structured
    # list, and what the guardrails and the citation layer must see is the
    # narrative beside that list, not the JSON the extraction returned — the
    # extraction is payload, the narrative is the answer (see deterministic.py).
    text = state.get("deterministic_answer") or final_text(state)
    # An empty answer is normally a failed turn. It is not one when the state says the
    # surface carries the answer: `/risk` returns a widget and no prose by design (see
    # deterministic.run_command), and without this the check below would turn a
    # complete answer into a 502.
    if not text and not state.get("answer_in_surface"):
        # An empty turn is not one thing. The response says which it was, and the
        # cases need different words: a truncated answer is worth retrying, a
        # refusal by the content filters is not — at temperature 0 the same
        # question reaches the same decision (ECC-12).
        message = final_message(state)
        silent = model_turn.silent_refusal(message, outpatient=outpatient)
        if silent is not None:
            # A clean finish with nothing in it is the model choosing not to speak,
            # not the service failing. Serving it as an abstention keeps a safety
            # probe from being reported as an outage — and the sentence still tells
            # the caller what this assistant does answer.
            logger.warning(
                "agent finished without text; serving an abstention trace=%s", trace,
            )
            text = silent
        else:
            code, sentence, reason = model_turn.classify(message)
            logger.error(
                "agent produced no final answer text trace=%s code=%s finish_reason=%s",
                trace, code, reason,
            )
            raise AgentAnswerUnavailable(sentence, code=code, finish_reason=reason)

    # The guardrails see this turn's calls AND the replayed turns' calls, because
    # a follow-up is answered against both: the earlier turns are in the
    # transcript, so they are evidence for what the answer may say. Judging it by
    # this turn alone deleted the number a follow-up was repeating — a replayed
    # "risk is 0.206685, above the 0.11 threshold" was served as "the risk for
    # this patient is, which is above the operating threshold" (2026-09-18).
    # Citations stay scoped to this turn inside `guard_answer`: their numbering
    # addresses this turn's presentation, which is composed below from
    # `trimmed_calls`.
    guarded = guard_answer(
        text, state["tool_calls"], state.get("replayed_tool_calls") or []
    )

    # tool_calls carry name, arguments, response and whether the response can be
    # produced again. The arguments used to be dropped here on the grounds that
    # only the canvas composition read this list (ECC-08); that stopped being
    # true when the site began storing a turn to replay it, because a replayed
    # call with no arguments misstates what was asked. Nothing new is exposed:
    # the arguments are the caller's own question and admission, and the passages
    # the call returned already cross this boundary. Guardrails above ran on the
    # full records.
    #
    # `derivable` says whether the site should keep the response. A caller that
    # stores a turn keeps the name and arguments either way, and the response
    # only when it cannot be produced again — see `RE_DERIVABLE_TOOLS`. The flag
    # is stated by the layer that ran the tool because only that layer knows, and
    # a caller that had to guess from the response's shape would guess wrong
    # (`rag_search_sections` returns note text without a `passages` key).
    trimmed_calls = [
        {
            "name": tc["name"],
            "args": tc.get("args") or {},
            "response": tc["response"],
            "derivable": tc["name"] in RE_DERIVABLE_TOOLS,
        }
        for tc in state["tool_calls"]
    ]

    # The presentation contract is composed HERE, not in the BFF: citation
    # renumbering, source resolution, and the A2UI canvas are evidence
    # semantics — they belong to the layer that ran the guardrails and saw the
    # tool evidence. Django passes them through; the browser renders them.
    presentation = compose_presentation(
        question, guarded["answer"], trimmed_calls, state.get("command_payload"),
        # Which command this turn was, or None for free text. The canvas reserves the
        # comparative cohort table to the commands whose question it answers, so the
        # composer has to know which one it is composing for.
        state.get("command"),
    )

    # Proposed from the answer that will actually be served, not from what the model
    # first wrote: a follow-up built on a sentence a guardrail removed would ask about
    # something the reader never saw. Runs last, and cannot fail the turn —
    # `followups.suggest` returns an empty list rather than raising.
    #
    # Only when the caller asked for it. One model call per turn is a real cost, and the
    # caller that draws a follow-up row is the only one that should pay it. The two
    # derived fields below the payload are free and always present, so the shape of a
    # payload does not depend on this flag — only whether the list has anything in it.
    follow_ups = (
        await followups.suggest(
            question,
            presentation["answer"],
            model=chain.MODEL_ID,
            scope=presentation.get("scope"),
        )
        if want_follow_ups
        else []
    )

    # The guardrails' own flags, plus the one only this layer can see: a chart turn
    # whose scope was withheld because no note citation resolved. Telemetry first —
    # evals read it off the stored turn — and it also classifies such a turn's prose
    # swap as visible rather than silent, which is the honest reading of a settled
    # answer that no longer claims the chart.
    flags = list(guarded["flags"])
    if presentation.get("scope_suppressed"):
        flags.append("chart_synthesis_without_citations")

    payload = {
        "question": question,
        "answer": presentation["answer"],
        "guardrail_flags": flags,
        "tool_calls": trimmed_calls,
        "a2ui": presentation["a2ui"],
        # Routing state, published rather than kept internal. `command` is which
        # command answered; `routed_to` is the fact that it was chosen for the reader
        # by free-text routing rather than typed by them. Both routes compose through
        # here, so both carry it, and a field that exists on every turn is a field an
        # evaluation can assert on. Withheld routing state made "did the command run?"
        # unanswerable from the wire, which is how a run in which `/ehr` executed zero
        # times still reported green.
        "command": state.get("command"),
        "routed_to": None,
        "sources": presentation["sources"],
        # Published records the answer rests on, if it rests on any. Carried beside
        # `sources` rather than inside it: that list is the note-citation channel —
        # numbered, renumbered, and resolved against retrieved note passages — and a
        # journal article has no passage to resolve against. The canvas renders the
        # articles, so they have to cross this boundary or the swap has nothing to
        # draw.
        "literature": presentation["literature"],
        # The strict payload (UX Phase 1): the narrative, the citations that
        # narrative's markers resolve against, and the components to draw it
        # with. Additive on purpose — every field above still carries what the
        # console renders today, so the site, the renderer and the new
        # components can each move on their own revision (see payload.py).
        "narrative_text": presentation["answer"],
        # The same list as `sources`, under the contract's name. One source of
        # truth, two names: the console looks a footnote up in `sources`, and the
        # Phase 2 components were specified against `citations`.
        "citations": presentation["sources"],
        "ui_components": build_ui_components(
            state.get("command"), trimmed_calls, state.get("command_payload")
        ),
        "model": chain.MODEL_ID,
        # The revision that produced this answer travels with it. A stored turn
        # without it cannot be explained once the code has moved on, which is
        # what the conversation store exists to prevent.
        "code_revision": chain.CODE_REVISION,
        # The pointer from this answer to the run that produced it. The trace
        # holds what the store cannot: which tools ran, what the model was
        # shown, how long each step took. The store holds what the trace cannot:
        # the answer, after the tracing stack has been swept or torn down. An
        # operator looking at a stored turn needs to reach the run rather than
        # match it by timestamp and memory, so the id travels with the answer.
        # Empty when tracing is off, which is a supported state
        # (see `observability.py`); the caller stores the empty string and
        # renders no link rather than a link to nowhere.
        "langfuse_trace_id": state.get("langfuse_trace_id") or "",
        "mcp_transport": MCP_TRANSPORT,
        # The three questions to offer beneath this answer, proposed by the model and
        # cleaned here. Empty is a supported value and the common one on failure: the
        # console draws no follow-up row rather than an empty one.
        "follow_ups": follow_ups,
        # What the answer rested on, decided from the tool calls rather than asked of
        # the model (`a2ui.turn_scope`). `None` means it rested on nothing — a redirect
        # to a command — and the console labels no scope for it rather than claiming one.
        "scope": presentation.get("scope"),
        # The rows the console's references drawer lists. The same records the inline
        # markers resolve to, so a clicked marker and its drawer row are one record.
        "references": presentation.get("references") or [],
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
    parsed: dict,
    on_event=None,
) -> dict:
    """Run the graph under the wall-clock deadline, inside one MCP session.

    Extracted because both routes need the identical bound: the deadline is a
    spend control (ECC-02), and a second copy of it is a second place for the
    two routes to disagree about how long a question may take.

    The request arrives whole rather than as arguments. Everything in it is
    something the chain needs, and a caller that unpacks it into positionals is a
    caller that silently drops the field added after it was written.
    """
    async with asyncio.timeout(ASK_TIMEOUT_SECONDS):
        async with toolbox() as box:
            # Label this turn for telemetry before anything can emit a span. Set here
            # rather than passed down because `ask` is the boundary every route and
            # every test double already stands on: adding a keyword to it meant
            # editing sixteen fakes to accept a field they do not care about, and a
            # request-scoped value is what a context variable is for.
            observability.set_run_tag(parsed.get("run_tag"))
            # Pre-flight, before the model runs: the structured features behind
            # this admission's risk score, stated as verified ground truth. The
            # model never spends a tool call on it, and a failure here leaves the
            # prompt exactly as it was rather than failing the turn.
            context = await admission_context.hydrate(box, parsed.get("hadm_id"))
            options: dict = {
                "question_kind": parsed["kind"],
                "hadm_id": parsed["hadm_id"],
                "topic": parsed["topic"],
                "on_event": on_event,
                "turns": parsed["turns"],
                "prose_stream": parsed["stream_prose"],
            }
            if parsed.get("intent"):
                # A declared affordance, not an inference. Only present when a control
                # said what it was, so an ordinary turn stays the call it always was.
                options["intent"] = parsed["intent"]
            if context:
                # Only when there is one. A turn without a profile stays exactly
                # the call it always was, which is what keeps hydration an
                # addition rather than a required new dependency of every turn.
                options["admission_context"] = context
            return await ask(box, parsed["question"], **options)


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
    log = _StageLog()
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
    log = _StageLog(sink=queue.put_nowait)
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
            # once the guardrails have run (see `_prose_settled`).
            if event.get("stage") == stages.STAGE_ANSWER_DELTA:
                announced.append(event.get("text") or "")
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
