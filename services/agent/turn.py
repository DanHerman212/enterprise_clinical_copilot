"""One turn: run the graph, verify the answer, compose what it ships.

Layer 3 of the agent, split out of `http.py` so the transport and the turn can be
read apart. `http.py` owns sockets: IAM, request parsing, status codes, SSE
framing, keepalives. This module owns what happens between a parsed request and a
validated payload: the deadline, the MCP session, pre-flight hydration, the
guardrails, the presentation, and the one execution record per turn.

Both routes call in here, which is the point. `/ask` and `/ask/stream` differ in
how the answer is delivered and in nothing else — they must not differ in what
they accept, when the guardrails run, or what the record says — so the shared
half lives here and each route keeps only its own delivery.

The graph and the toolbox are passed IN rather than imported here. They are
rebound in `http` by every route test (`patch.object(srv, "ask")`,
`patch.object(srv, "toolbox")`), and a name imported at module scope would not be
the name those patches replace. Injecting them keeps the seam explicit and the
fakes effective; see `http._run_chain`.
"""


import asyncio
import logging
import time
from collections.abc import Callable, Sequence

from services.agent import admission_context, chain, followups, model_turn
from services.agent import observability, routing
from services.agent.a2ui import compose_presentation
from services.agent.contracts import (
    RE_DERIVABLE_TOOLS,
    AgentResponseError,
    validate_agent_success,
)
from services.agent.graph import final_message, final_text
from services.agent.guardrail import guard_answer
from services.agent.mcp_client import MCP_TRANSPORT
from services.agent.served_admissions import is_served

logger = logging.getLogger(__name__)


#: Served when a caller names an admission this deployment cannot open.
ADMISSION_NOT_FOUND = (
    "Admission not found. This deployment can only answer for the admissions it "
    "serves, and that admission is not one of them, so nothing was retrieved and "
    "no estimate was produced."
)


def refusal_trace_metadata(parsed: dict) -> dict:
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



async def compose_success(
    question: str, state: dict, trace: str, *, want_follow_ups: bool = False,
    outpatient: bool = False, asked: Sequence[str] = (),
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
        # Grounding the composer cannot see: the pre-flight manifest answered an
        # administrative question, or the replayed turns' chart calls are what a
        # follow-up is quoting. It changes `verified` and never what is drawn.
        context_grounded=bool(state.get("manifest_answered")) or any(
            call.get("name") not in routing.LITERATURE_TOOLS
            for call in (state.get("replayed_tool_calls") or [])
        ),
        # The whole conversation's evidence, in the order the model was shown it. A
        # follow-up's markers are numbered continuously across the replayed turns and
        # this one, so the source table has to be the same concatenation (TF-10).
        evidence=[*(state.get("replayed_tool_calls") or []),
                  *(state.get("tool_calls") or [])],
    )

    # Chosen from the record this turn already holds, not proposed by a model: a chip
    # is a control, and a control that asks something the chart cannot answer is worse
    # than no chip row at all (Phase 05 §2, 2026-10-23). Synchronous, in-memory, and it
    # cannot fail the turn — `followups.suggest` returns a list or an empty list.
    #
    # Only when the caller asked for it. The two derived fields below the payload are
    # free and always present, so the shape of a payload does not depend on this flag —
    # only whether the list has anything in it.
    follow_ups = (
        followups.suggest(
            question,
            presentation["answer"],
            model=chain.MODEL_ID,
            scope=presentation.get("scope"),
            state=state,
            outpatient=outpatient,
            # Every question already asked in this thread, so a chip the reader has
            # just clicked is consumed rather than offered back.
            asked=asked,
        )
        if want_follow_ups
        else []
    )
    if want_follow_ups:
        # Recorded on the turn's own record. The trace cannot hold them: this runs after
        # the turn's span has closed, and this SDK/server combination drops observation
        # writes made outside an active span (measured 2026-10-23), so the chip list
        # rides on the execution line, which already carries the trace id that joins it
        # to the run.
        state["follow_ups"] = list(follow_ups)

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
        # Whether this answer was checked against, and grounded in, the patient's own
        # record. Composed in `a2ui` from the evidence and ANDed here with the one fact
        # only this layer holds: whether the answer survived the guardrails or was
        # replaced by an abstention — an abstention is the system saying it could not
        # stand behind the answer, so it cannot be badged as verified.
        "verified": bool(presentation["verified"])
        and "abstained_after_guard" not in flags,
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


class StageLog:
    """Collects the stages a chain run emitted, and when each one started.

    Both routes need the same timing data for the execution record, so it is
    gathered once. `/ask` uses this for the record only; `/ask/stream` passes a
    sink as well, so the same events are also relayed to the caller. That is what
    keeps "what the user saw" and "what was recorded" the same list rather than
    two lists that agree today.
    """

    def __init__(self, sink: Callable[[dict], None] | None = None,
                 intent: str | None = None):
        self.stages: list[dict] = []
        self._sink = sink
        self._started = time.monotonic()
        # What a control declared this turn to be, held here rather than passed to each
        # `record` call: every exit from a request records exactly one line, and a
        # declaration that only four of five exits carried would be unqueryable beside
        # the failures it matters most for.
        self.intent = intent

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
            intent=self.intent,
            **kwargs,
        )


async def unserved_admission(parsed: dict, trace: str) -> dict | None:
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
    StageLog(intent=parsed.get("intent")).record(
        trace, parsed.get("question") or "", "refused",
        error="admission_not_found")
    # Composed through the shared pipeline rather than hand-built, so this refusal
    # carries exactly the fields every other answer carries — including its trace id,
    # which the pipeline cannot supply here because no span is open around this exit.
    with observability.observation(
        "turn.refused", trace_metadata=refusal_trace_metadata(parsed),
    ) as span:
        payload = await compose_success(
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


async def run_chain(
    parsed: dict,
    on_event=None,
    *,
    ask,
    toolbox,
    deadline: float,
) -> dict:
    """Run the graph under the wall-clock deadline, inside one MCP session.

    `ask` and `toolbox` are passed by the caller rather than imported here: the
    route tests rebind both in `http`, and a module-level import would leave those
    patches pointing at a name nothing reads. `deadline` comes with them for the
    same reason and one more: the route's 504 message quotes the number it waited,
    so the route is where it belongs.

    Extracted because both routes need the identical bound: the deadline is a
    spend control (ECC-02), and a second copy of it is a second place for the
    two routes to disagree about how long a question may take.

    The request arrives whole rather than as arguments. Everything in it is
    something the chain needs, and a caller that unpacks it into positionals is a
    caller that silently drops the field added after it was written.
    """
    async with asyncio.timeout(deadline):
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
