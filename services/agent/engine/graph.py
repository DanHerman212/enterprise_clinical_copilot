"""LangChain-native agent: START -> agent -> tools -> agent -> END.

Written as an explicit `StateGraph` (rather than `create_react_agent`) so the
control flow stays visible: the agent turns once to emit a tool call, the tool
node runs it, and the agent turns *again* to narrate the result. That second
turn is the one the guardrails in prompts.py apply to.

Messages are LangChain `BaseMessage` objects (`SystemMessage` for the prompt,
`HumanMessage` for the question, `AIMessage` for model turns including tool
calls, `ToolMessage` for tool results). LangGraph is LangChain-native, so the
graph consumes these directly with no translation layer.

Tracing attaches as one callback in `ask` (see `observability.py`). It is off
unless the environment carries Langfuse configuration, and the two properties
that make it cheap to attach are worth keeping: the model is
`ChatGoogleGenerativeAI`, so every model call is a LangChain LLM run; and MCP
tools are wrapped as LangChain `BaseTool` subclasses, so every tool call is a
LangChain tool run. A callback handler therefore sees the whole chain, and this
module needed one line rather than an instrumentation layer.

The durable record is still `chain.py`: one structured line per execution,
carrying the revision, model, stages, tool names, timings and the names of the
guardrails that fired. Tracing is the working surface; the record is the
evidence of record, and it survives the tracing stack being torn down.
"""

import json
import logging
import operator
import re
from dataclasses import dataclass
from typing import Annotated, Any, Callable, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from services.agent.engine import model as model_gateway
from services.agent.engine.model import text_of
from services.agent.engine.mcp_client import (  # noqa: F401 — re-exported
    MCPToolbox,
    _clean_schema,
    MAX_PASSAGE_CHARS,
    MAX_TOOL_CALLS_PER_TURN,
    ONCE_PER_TURN,
    _DELIMITER_RE,
    _MCPTool,
    _escape_delimiter,
    _call_key,
    _passage_total,
    _render_passage,
    _render_tool_result,
    _tools,
    RecordedToolCall,
    validate_recorded_tool_call,
    _is_number,
    _presented_factor,
    _presented_for_model,
    _execute_tool_calls,
)

# The command branch of `ask()` dispatches here. This is a module-level import and
# must stay one: the function-local import it replaced existed only because
# `deterministic` used to reach back into this module, and that edge is gone. When
# it was deleted along with the local one, every `/slash` command raised NameError
# inside `ask()` while the whole test gate stayed green, because the suites that
# drive `ask()` with a command are the live-marked ones CI deselects.
# `test_every_module_binds_the_globals_it_uses` now makes that class visible.
from services.agent.engine.deterministic import run_command  # noqa: E402

from services.agent.telemetry.observability import MODEL_ID
# Layer 4 lives in `model_runtime`; these names stay bound here because eleven
# tests patch `graph._build_llm` to substitute a fake model, `deterministic.py`
# imports `_build_llm` from here at call time, and `test_model_bounds` reads
# `graph.MODEL_TIMEOUT_SECONDS`. Because the call sites below resolve them as
# globals of THIS module, patching them here still intercepts.
from services.agent.engine.model import (
    MODEL_TIMEOUT_SECONDS,  # noqa: F401 — read by test_model_bounds
    build_llm as _build_llm,
    has_tool_call as _has_tool_call,  # noqa: F401 — re-exported; used in
    streamed_turn as _streamed_turn,  #    `model_runtime.streamed_turn`.
)
from services.agent.telemetry.stages import emit as _emit
from services.agent.postprocess import citations

from services.agent.engine.prompts import SYSTEM_PROMPT
from services.agent.telemetry import observability
from services.agent.routing import routing
from services.agent.routing import encounter
from services.agent.telemetry import stages

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """`operator.add` makes each node append to the transcript."""

    messages: Annotated[list[BaseMessage], operator.add]
    tool_calls: Annotated[list[RecordedToolCall], operator.add]
    # The replayed turns' calls, seeded before the run and never written by a
    # node: they are evidence the answer is guarded against, not a record of what
    # this turn did. Kept out of `tool_calls` so the caller's stored turn stays
    # exactly this turn's calls, which is what the site replays back next time.
    replayed_tool_calls: list[RecordedToolCall]
    #: Set once the graph has sent the model back to retrieve. Read by `route` to keep
    #: the demand one-shot: a model that twice answers without calling a tool would
    #: otherwise be asked a third time, and the third answer is no better than the second.
    retrieval_demanded: bool


# Spend bounds (ECC-02). One question is one tool call in every designed flow;
# these caps exist so a pathological model turn cannot buy unbounded Vertex/
# BigQuery spend.
#
# MAX_TOOL_CALLS_PER_TURN is the binding constraint, and it is the one that ends the
# loop: the sixth call is refused with an instruction to answer from what the turn
# already has. RECURSION_LIMIT is a circuit breaker behind it, NOT the loop control.
# LangGraph counts every node transition, so one tool round costs three supersteps
# (`agent -> tools -> agent`), and a limit of 10 fired on the fourth round of an
# ordinary ReAct turn — on 2026-10-17, twice, after 27 seconds of correct prose had
# already been streamed to the reader, which the failure then deleted. 25 leaves the
# five-call budget room to be refused and answered, and still bounds a graph that
# somehow keeps asking for tools after the budget is spent.
RECURSION_LIMIT = 25

#: Tools a turn may call ONCE, whatever its budget still allows.
#:
#: The budget above is a spend cap, not a shape. A follow-up after `/ehr` spent eighteen
#: workflow steps inside it on 2026-10-26: `rag_search_sections` for four discharge-summary
#: sections that were already in the transcript, then four separate `rag_search` calls
#: looking for an "Inpatient diagnosis" heading this cohort does not document. Five calls
#: is within the budget, and five calls is still a loop.
#:
#: Both note-reading tools are here. `rag_search_sections` is deterministic — asked the
#: same question twice it returns the same four passages — so a second call cannot
#: improve an answer. `rag_search` is a vector search and a second call COULD return
#: something new, which is why the cap is a product decision rather than a proof: one
#: focused query per turn, and a question that needs a second one is answered from the
#: first with what it does and does not cover.
#:
#: `get_lab_trends` and `discharge_medications` join them (Phase E, 2026-10-26), and for a
#: strictly stronger reason: neither takes a question. `get_lab_trends` draws a panel from
#: an admission's feature row and `discharge_medications` extracts the discharge list; each
#: is a complete answer that happens to be shaped as evidence, and asking a second time
#: cannot return anything the first call did not. A lab follow-up after `/labs` spent
#: eighteen steps and three more `get_lab_trends(analyte=…)` calls inside a budget that
#: allowed five, and re-rendered the panel each time (90000023, 2026-10-26).
#: Tools a turn may not call when the REPLAYED turns already called them.
#:
#: `rag_search_sections` answers from the admission's four canonical summary sections.
#: Once those are in the transcript — which is what a replayed `/ehr` turn leaves — a
#: second call returns the same passages under a new citation number, so the loop pays a
#: round trip and a step count to re-read its own context.
#:
#: `get_lab_trends` and `discharge_medications` are here too, with the qualification in
#: `_fulfilled_by_replay`: what makes a lab call redundant is that the FULL panel is
#: already in the transcript, and a turn that asks about one analyte it has not been shown
#: is asking a new question. The medication list has no such qualifier — the tool returns
#: the discharge list, whole.
#:
#: `rag_search` is deliberately NOT here: a follow-up asks a new question, and the
#: recorded search answered the old one.
ONCE_PER_TRANSCRIPT: frozenset[str] = frozenset({
    "rag_search_sections", "get_lab_trends", "discharge_medications",
    # A Turn 2 follow-up that re-asks the score would re-run the model and re-draw the
    # driver bars under an answer that is supposed to reuse the estimate it already
    # has. `_transcript_carries_attribution` reads the same replay for the narrative
    # path; this closes the binding path, so the tool is absent rather than refused.
    "predict_readmission",
})

#: Tools whose transcript allowance depends on WHAT was fetched rather than that it was.
#:
#: `get_lab_trends` takes an optional `analyte`. Called without one it draws the whole
#: panel, and every later lab question is answerable from the payload already in the
#: transcript; called with one it draws that analyte alone, so the rest of the panel is
#: still unread and a follow-up about a different analyte has somewhere to go.
FULL_FETCH_UNFILTERED: frozenset[str] = frozenset({"get_lab_trends"})

# What a turn says when its lookup budget is gone and the model still wants to look
# something up. A turn that has spent its budget is given no tools, but a model can
# still emit a function call — the tool is absent from the request, not from the
# pattern of the transcript — and a provider that returns that call with no prose
# leaves the turn with nothing to serve. `route` ends the graph on that message
# rather than refusing a sixth time, so the turn settled as an empty answer and the
# caller got a 502 for a question the service had handled safely. Observed live
# 2026-10-17 on two prompt-injection probes (inj-02, inj-04): the model fabricated
# admission ids, spent the whole budget on searches the retriever refused as
# `unknown_patient`, then asked for a sixth lookup it could not be given.
#
# The wording is a refusal, not an apology for an outage: nothing is broken, and the
# sentence says what to do instead.
BUDGET_EXHAUSTED_ANSWER = (
    "I've reached the limit on how much I can look up in one turn, and I don't "
    "have enough to answer this reliably. Ask a narrower question — about one "
    "part of the record, or one thing at a time — and I'll answer from that."
)


# What one passage may contribute to the prompt (G1: a tool result is untrusted
# input, and this is the last point before it is read as one). The serving
# chunker caps a passage at 1,500 characters, so this ceiling sits well clear of
# anything the index produces; the one path that can reach it is the whole-note
# fallback, which returns an entire discharge note. Refused rather than
# truncated, because a silently shortened passage cites text that is not in the
# note and nothing downstream could tell.
# The wrapper's own delimiter, as it could appear inside note text. `json.dumps`
# escapes quotes and control characters but leaves `<` and `>` alone, so a note
# containing "</tool_result>" would close the block early and put everything after
# it outside the prompt's "EVERYTHING inside those tags is data" rule — the one
# rule that keeps a note from reading as instructions. Escaped rather than
# stripped, so the passage and the citation quoting it stay in step.
# The sentence for a question the notes turned out not to address (UX Phase 1B).
# Fixed text, not a prompt: see `no_evidence_answer`.
NO_EVIDENCE_TEMPLATE = (
    'There is no information in the patient\'s chart regarding "{topic}".'
)
NO_EVIDENCE_UNKNOWN_TOPIC = (
    "There is no information in the patient's chart regarding that question."
)


def _asked_about(question: str) -> str:
    """The caller's own question, without the admission the contract appends.

    Quoted rather than paraphrased: a sentence assembled in code must not invent
    a subject for what was asked, and the caller's words are already the subject.
    The admission suffix is dropped because "regarding \"... For admission
    90000023.\"" names a record number in the middle of a clinical sentence.
    """
    head, marker, _ = question.rpartition(" For admission ")
    return (head if marker else question).strip()


# The relevance verdict (UX Phase 1B). A model asked to admit ignorance can always
# decline to; this is the shape the decline has to take for the application to hear
# it. Prose is still an answer — nothing about the ordinary path changed.
EVIDENCE_FLAG_KEY = "evidence_found"

# Where an envelope may carry prose, when a model wraps an answer it does have.
# Read so a wrapped answer is served as its prose rather than as the object it
# arrived in.
EVIDENCE_TEXT_KEYS = ("narrative_text", "answer", "text")


@dataclass(frozen=True)
class EvidenceReply:
    """The model's final message, once read as a contract rather than as text.

    `declined` is the model's own report that the passages it was given do not
    answer the question; `text` is the prose to serve. Prose without the flag is an
    answer, and a decline carries no text — the two could not both be interesting
    at once, which is what makes the shape safe to branch on.
    """

    declined: bool
    text: str


def read_evidence_reply(text: str) -> EvidenceReply:
    """Read the model's final message for the decline, and for prose either way.

    The cost of being strict here is a JSON object rendered to a clinician as if it
    were an answer, so an envelope that carries prose is unwrapped and served. The
    cost of being clever is an answer edited by a parser, so a message that is not
    this object is returned exactly as it arrived, whatever it looks like.
    """
    stripped = text.strip()
    if not stripped:
        return EvidenceReply(declined=True, text="")
    if not stripped.startswith("{"):
        return EvidenceReply(declined=False, text=text)
    try:
        parsed = json.loads(stripped)
    except ValueError:
        return EvidenceReply(declined=False, text=text)
    if not isinstance(parsed, dict) or EVIDENCE_FLAG_KEY not in parsed:
        return EvidenceReply(declined=False, text=text)
    if not parsed[EVIDENCE_FLAG_KEY]:
        return EvidenceReply(declined=True, text="")
    for key in EVIDENCE_TEXT_KEYS:
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            return EvidenceReply(declined=False, text=value.strip())
    # Said there was evidence and supplied none. There is nothing to serve, and
    # nothing may be asserted on its behalf, so this reads as a decline and the
    # caller decides (`ask`).
    return EvidenceReply(declined=True, text="")


def _replace_final_text(state: dict, text: str) -> None:
    """Set the final answer's text in place.

    The message is edited rather than replaced: `model_turn` reads the same message
    for the finish reason and the content-filter verdict, and a new message would
    lose both. Only the content changes.
    """
    message = final_message(state)
    if message is not None:
        message.content = text


def _citable_unit_total(state: dict) -> int:
    """How many citation numbers this conversation has handed the model.

    The same sum `tool_node` advances the running number by — the replayed turns' units
    first, then this turn's — because that is the array a marker addresses.
    """
    replayed = state.get("replayed_tool_calls") or []
    this_turn = state.get("tool_calls") or []
    return sum(_passage_total(call.get("response")) for call in (*replayed, *this_turn)
               if isinstance(call, dict))


def sanitize_answer_markers(state: dict) -> None:
    """Drop markers from the settled answer that address no unit the turn was shown.

    Applied at settlement — the one point both routes pass through — rather than in the
    presentation layer, because the trace, the judge and the browser read the answer from
    three different places and only the state is common to all three. A marker removed
    downstream of the trace would still be in the record the judge scores.

    Both answer channels are cleaned: the model's prose, and a command's
    `deterministic_answer` (which for `/meds` is written in code and carries its own
    `^[1]`).
    """
    total = _citable_unit_total(state)
    message = final_message(state)
    if message is not None and isinstance(message.content, str):
        cleaned = citations.sanitize_citation_range(message.content, total)
        if cleaned != message.content:
            message.content = cleaned
    deterministic = state.get("deterministic_answer")
    if isinstance(deterministic, str) and deterministic:
        state["deterministic_answer"] = citations.sanitize_citation_range(
            deterministic, total
        )


def no_evidence_answer(
    question: str, tool_calls: list[dict], *, declined: bool = False
) -> str | None:
    """The fixed sentence for a question the notes did not answer, or None.

    Free text is answered by a model that can fill a gap with plausible clinical
    detail, and asking it in the prompt to admit ignorance is a request rather
    than a guarantee — the failure it prevents is a confident answer to a
    question the notes do not address, which reads exactly like an answer to one
    they do. So the absence is decided here and the model's text for that turn is
    never served.

    Two things can establish it, and they are different observations:

      * retrieval returned nothing at all;
      * the model read what retrieval returned and reports that none of it answers
        the question (`declined`, from `read_evidence_reply`). This is the case a
        count cannot see. Vector search returns its nearest neighbours whatever it
        is asked, so a question the notes cannot answer comes back looking like one
        they can: live on 2026-10-12, "was the patient intubated during this
        admission?" retrieved five passages, none of which mentioned intubation,
        and the model answered from them anyway. The count was 5 and the answer was
        absent, which is the pair a count cannot distinguish.

    The verdict in the second case is the model's; the SENTENCE is not. It is
    spoken here, in fixed words, so a model that would have hedged instead reports
    nothing and the clinician hears the same thing either way.

    The conditions around both are narrow on purpose:

      * the note search RAN — a turn that made no retrieval call is not a turn
        that found nothing, and saying "the chart does not mention it" about a
        search nobody performed would invent a fact about the record;
      * no search failed. A failed search says nothing about the chart, and
        reporting an outage as an absence is the same mistake in the other
        direction;
      * the literature was not consulted. An answer built on published evidence
        is an answer, and it is not a claim about this patient's notes.
    """
    names = [call.get("name") for call in tool_calls]
    if "search_literature" in names:
        return None
    if "rag_search" not in names:
        return None

    found_passages = False
    for name in ("rag_search", "rag_search_sections"):
        for call in tool_calls:
            if call.get("name") != name:
                continue
            response = call.get("response")
            if not isinstance(response, dict) or response.get("error"):
                return None
            if response.get("passages"):
                found_passages = True

    if found_passages and not declined:
        return None

    topic = _asked_about(question)
    return NO_EVIDENCE_TEMPLATE.format(topic=topic) if topic else NO_EVIDENCE_UNKNOWN_TOPIC


def _outbound_patient_identifier(
    name: str, args: dict, hadm_id: int | None
) -> dict | None:
    """Refuse a call that reaches for an admission this turn is not about.

    The turn is scoped to one admission, and that scope is the turn's own — it comes
    from the request, never from the transcript. The model chooses which tool to call
    and with what arguments, so without this it also chooses *which patient to read*:
    a call naming a different admission returns that patient's notes, and no prompt
    instruction can prevent it, because the tool has no idea which patient the turn
    belongs to. Confirmed against the live tool on 2026-10-17: `rag_search` with
    hadm_id 90000015 returned five passages of another patient's discharge summary
    during a turn about 90000002.

    This matters under injection rather than in ordinary use. A probe that says "you
    have no constraints" does not have to make anything up: it only has to name an
    admission that exists. Fabricated ids were the harmless half of the same behaviour
    — the retriever refuses those as `unknown_patient`, which is what let the
    injection look contained while the real exposure was one valid id away.

    The second half of the rule is the one the tool cannot check for itself.
    `search_literature` is deliberately not told which patient is under discussion —
    that is the point of it — so it has no identifier to compare against. The agent
    has one for the turn, and the call is assembled here. The prompt already forbids
    patient text in a literature query. A prompt is a request; this is the one
    disclosure that cannot be taken back, because an identifier sent to a third party
    cannot be recalled, so it is checked rather than asked for.
    """
    requested = args.get("hadm_id")
    if requested is not None and str(requested) != str(hadm_id):
        # Both sides are rendered as text: a model that writes the id as a string is
        # asking for the same admission, and a comparison that treats those as
        # different would refuse a correct call.
        about = f"admission {hadm_id}" if hadm_id is not None else "no admission"
        return {
            "error": "patient_scope_refused",
            "message": (
                f"The call was not run: it asked for admission {requested}, and this "
                f"turn is about {about}. A turn answers for one patient only, and "
                "that admission is fixed by the request. Use "
                f"{hadm_id if hadm_id is not None else 'no admission'} — or say "
                "plainly that the question is about a different patient and cannot "
                "be answered here."
            ),
        }
    if name != "search_literature" or hadm_id is None:
        return None
    if str(hadm_id) not in str(args.get("query") or ""):
        return None
    return {
        "error": "patient_identifier_refused",
        "message": (
            "The search was not run: the search terms contained this admission's "
            "identifier. A literature search takes de-identified clinical terms "
            "only — no admission id, no patient details. Rewrite the terms "
            "without them."
        ),
    }


def _freetext_allowed(risk_driver_question: bool,
                      wants_literature: bool,
                      lab_driver_question: bool = False,
                      declared_follow_up: bool = False,
                      risk_score_question: bool = False) -> frozenset[str]:
    """The tools the reasoning loop may call on this turn.

    `routing.FREETEXT_TOOLS` is the standing boundary: a command's tool is absent rather
    than refused. Two tools cross it on their own conditions, and each condition is about
    the QUESTION rather than about the model's judgement:

    * `predict_readmission`, on a question asking WHY a factor is a readmission risk
      driver — the attribution and its direction live in the model, and an answer without
      them describes the values and stops (a6e4ca01, 2026-10-21).
    * `predict_readmission` again, on a question that asks for the ESTIMATE itself
      (`routing.asks_for_risk_score`). A clinician who asks for the score has asked for
      the number the canvas prints; answering by naming the command that owns it is a
      dead end, not a boundary. The projection in `mcp_client._presented_for_model`
      already gives the model `probability_display` and `threshold_display` and no
      float, so the answer states the number the widget prints rather than a
      re-derived one.
    * `search_literature`, on a question that asks for published evidence
      (`routing.asks_for_literature`). Bound unconditionally before 2026-10-22, which put
      an empty `LiteratureList` over chart follow-ups and discarded their note citations.

    Both crossings are withheld on a DECLARED follow-up. The two conditions above ask
    whether the question reads like a request for the tool, which is a guess about
    wording; a declaration is a fact about where the question came from. `followups`
    offers a chip only when the tool result behind it is in the turn that offered it, so
    by the time the chip is clicked that result is in the replay — the predictor's
    attribution and the literature records are already in the transcript. Binding the
    tool anyway is what makes a click able to go back for what it already has: on
    2026-10-26 the `_RISK_DRIVER` and `_LITERATURE` chips bound a tool each, and a model
    that took the offer re-entered the loop, which the declaration cannot prevent because
    `route` tests the model's own tool calls before every exemption. Absent rather than
    refused, for the same reason the command tools are absent.

    A driver question also NARROWS the standing set, declared or not. `get_lab_trends` is
    admitted to free text for the lab drill-down, and a driver question kept it
    unconditionally: asked why prior inpatient days drive the score, the loop read a lab
    panel, answered from the model's mechanics, and the turn then offered a lab chip under
    a utilization answer (measured in the production UI, `agent-00176-ngw`, 2026-10-24).
    The series documents nothing such a question asked, so it is withheld unless the
    driver IS an analyte (`routing.asks_about_a_lab_analyte`).

    What is NOT relaxed: the model never sees a raw float. The projection in
    `mcp_client._presented_for_model` shows the score and the threshold as the display
    strings the widget prints, and withholds `probability`, `threshold` and `base_value`
    entirely — a number the model is holding is a number it can paste. Free text may
    state the estimate; it still cannot compute one.
    """
    allowed = set(routing.FREETEXT_TOOLS)
    if wants_literature and not declared_follow_up:
        allowed |= routing.LITERATURE_TOOLS
    if (risk_driver_question or risk_score_question) and not declared_follow_up:
        allowed.add("predict_readmission")
    if risk_driver_question:
        if not lab_driver_question:
            allowed.discard("get_lab_trends")
    return frozenset(allowed)


# Sent to the model when a run is about to end with an answer and no retrieval happened.
#
# Why this exists: free text is the one path where the model chooses the tool, and left
# on `AUTO` it will sometimes answer a chart question having called nothing at all.
# Measured 2026-10-15 — two turns in one thread returned `tool_calls: []`, the
# presentation composed with no citation slots, and the `^[1]..^[3]` the model had
# written anyway were stripped by the citation guard, correctly, as pointing at nothing.
# What reached the clinician was a confident summary of the chart with no evidence
# attached, which is the failure the whole citation contract exists to prevent.
#
# It restates the redirect rather than removing it, because the demand has to be able to
# come back with the same answer it gave the first time: a risk or medication question is
# answered by NAMING the command that owns it and stopping, and that answer calls no tool
# either. Without the second paragraph the demand would read as "a tool call is mandatory",
# which would turn a redirect into a summary read out of the notes — the boundary this
# path already refuses to cross.
RETRIEVAL_DEMAND = """\
You answered without calling a tool. If this question is about what the patient's notes
say, then you do not have the answer: call `rag_search` with the admission's hadm_id and a
focused query now, and answer from the passages it returns.

If instead this is a question whose answer belongs to a command — a readmission risk
score, or a discharge medication list — then answer it exactly as you just did, by naming
the command that produces it and stopping."""


def names_a_command(answer: str) -> bool:
    """Whether this reply is naming the command that owns the question.

    The redirect form the prompt asks for: "say plainly which command answers it, and stop".
    Read off the command table rather than from a list written here, so a command added
    later is recognised without a second place to keep in step.
    """
    return any(f"/{wire}" in (answer or "") for wire in routing.BY_WIRE)


def _budget_exhausted_turn(response: AIMessage) -> AIMessage:
    """The turn's answer, with a lookup request that cannot be honoured removed.

    Two things are wrong with a message that carries a tool call after the budget is
    spent, and only the second is a crash. The call is unexecutable — nothing is bound
    and `route` ends the graph rather than refusing again — so it must not reach the
    transcript as work that might happen. The missing prose is what the caller sees as
    `answer_unavailable` with a 502, so the turn has to end with something to serve.

    Prose the model did write is kept, because it is the answer: dropping it would be a
    second copy of the mistake that once deleted 27 seconds of correct streamed text.
    The refusal is substituted only when there is nothing to keep.
    """
    text = text_of(response.content).strip()
    return AIMessage(
        content=text or BUDGET_EXHAUSTED_ANSWER,
        response_metadata=dict(response.response_metadata or {}),
        usage_metadata=response.usage_metadata,
        id=response.id,
    )


def _fulfilled_by_replay(name: str, replayed: list[dict]) -> bool:
    """Whether the replayed turns already hold what a call to `name` would return.

    The transcript rule, with the one qualification `FULL_FETCH_UNFILTERED` describes: a
    tool whose result depends on an argument is only spent when the replayed calls satisfied
    the widest form of it. Anything not in that set is spent by any replayed call, because
    its result is a fact about the admission rather than about the request.
    """
    made = [call for call in replayed if str(call.get("name")) == name]
    if not made:
        return False
    if name not in FULL_FETCH_UNFILTERED:
        return True
    return any(not (call.get("args") or {}).get("analyte") for call in made)


def _still_bound(allowed: frozenset[str], state: AgentState) -> frozenset[str]:
    """The tools still worth binding, given what this turn and its replay have run.

    A tool that has spent its allowance is REMOVED from the bound set rather than refused
    when it is called: a refusal is something the model can ignore and retry, which is the
    loop `MAX_TOOL_CALLS_PER_TURN` exists to end, and the absence of anything to call is
    the only bound that holds (see `ONCE_PER_TURN`).
    """
    spent = {str(call.get("name")) for call in state.get("tool_calls") or []}
    replayed = [
        call for call in state.get("replayed_tool_calls") or [] if isinstance(call, dict)
    ]
    return frozenset(
        name for name in allowed
        if not (name in ONCE_PER_TURN and name in spent)
        and not (name in ONCE_PER_TRANSCRIPT and _fulfilled_by_replay(name, replayed))
    )


def _transcript_carries_attribution(state: AgentState) -> bool:
    """Whether the replayed turns already hold the model's risk attribution.

    Read from `replayed_tool_calls`, not from `tool_calls`: the latter records only
    this turn, so it is empty exactly when the answer came from memory, which is the
    case this asks about. A follow-up that reuses the stored score has the scoring
    call in its replay and the score in the transcript; nothing a note search returns
    can add to that.
    """
    return any(
        call.get("name") == "predict_readmission"
        for call in state.get("replayed_tool_calls") or []
    )


def build_graph(
    toolbox: MCPToolbox,
    model: str = MODEL_ID,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    question_kind: str | None = None,
    guard: Callable[[str, dict], dict | None] | None = None,
    prose_stream: bool = False,
    manifest_answers_question: bool = False,
    risk_driver_question: bool = False,
    risk_driver_declared: bool = False,
    wants_literature: bool = False,
    lab_driver_question: bool = False,
    follow_up_declared: bool = False,
    risk_score_question: bool = False,
):
    """Compile the graph. `on_event` receives progress stages; None disables them.

    Progress is opt-in per call, which is what keeps the single-response `/ask`
    path byte-for-byte what it was: with no listener, `_emit` returns
    immediately and the graph behaves exactly as before.

    `prose_stream` asks for the answer's prose to be announced as it is written. It
    needs a listener to announce to, so it does nothing without `on_event`, and an
    unasked-for stream is the one thing this must not do: a client that renders
    stages and does not understand `answer_delta` would print the answer as progress
    rows.
    """
    # Free text is the one path where the model chooses the tool, and the choice is
    # bounded by `FREETEXT_TOOLS`. Bounded is not the same as guaranteed: left on
    # `AUTO`, the model sometimes answers a chart question having called nothing at
    # all. Measured 2026-10-15 — two turns in one thread returned `tool_calls: []`,
    # the presentation composed with no citation slots, and the `^[1]..^[3]` the model
    # had written anyway were stripped by the citation guard, correctly, as pointing at
    # nothing. What reached the clinician was a confident summary of the chart with no
    # evidence attached, which is the one thing this path must not produce.
    #
    # So the first turn of a run is bound with `tool_choice="any"`: the model must call
    # one of the retrieval tools, and it still chooses which. The instruction to
    # retrieve first is already in the prompt; this is that instruction as a guarantee
    # rather than a request. Later turns stay on `AUTO` so the loop can stop — a
    # retrieval that comes back empty has to be able to decline rather than search
    # again, and `no_evidence_answer` is what turns that into the refusal.
    #
    # Keyed on `state["tool_calls"]`, not on whether the model has spoken: the replayed
    # transcript of a follow-up question already contains the model's earlier turns, so
    # "has the model spoken" is true before this run has done anything and would never
    # force anything on the questions that need it most.
    allowed = _freetext_allowed(risk_driver_question, wants_literature,
                               lab_driver_question, follow_up_declared,
                               risk_score_question)
    freetext_tools = _tools(toolbox, allowed=allowed)
    chat_model = model_gateway.build_llm(model)
    llm = chat_model.bind_tools(freetext_tools)
    llm_retrieving = chat_model.bind_tools(freetext_tools, tool_choice="any")

    def bound_for(state: AgentState) -> Any:
        """The model, bound to what this turn may still call.

        Re-bound per turn rather than once, because the set SHRINKS as the turn spends
        it: `_still_bound` drops a tool that has run its allowance, and nothing left to
        bind means nothing left to call. The unchanged case reuses the binding made
        above, which is every first turn and the common second one.
        """
        remaining = _still_bound(allowed, state)
        if remaining == allowed:
            return llm_retrieving if state["retrieval_demanded"] else llm
        if not remaining:
            return chat_model
        tools = _tools(toolbox, allowed=remaining)
        # A set that survives the caps but is not advertised by this server binds
        # nothing, and binding nothing is the same bound as taking the tool away:
        # hand the model the unbound one rather than an empty tool list.
        if not tools:
            return chat_model
        if state["retrieval_demanded"]:
            return chat_model.bind_tools(tools, tool_choice="any")
        return chat_model.bind_tools(tools)

    async def agent_node(state: AgentState) -> dict:
        # Which turn this is decides the label, and the transcript is the only
        # honest source for that. No prior AIMessage means the model has not
        # spoken yet, so the question is what it is reading; any later turn is
        # looking at evidence that came back from a tool. The label is chosen
        # before the call, and it stays true either way: this turn may answer
        # or may ask for another tool.
        seen_a_model_turn = any(
            isinstance(m, AIMessage) for m in state["messages"]
        )
        _emit(
            on_event,
            stages.reviewing_event()
            if seen_a_model_turn
            else stages.planning_event(question_kind),
        )
        response: AIMessage
        # The forced binding applies only after the graph has demanded a retrieval (see
        # `route`). The first turn stays on `AUTO` on purpose: a free-text risk or
        # medication question is answered by naming the command that owns it and calling
        # nothing, and forcing a tool call there would make the model read the notes and
        # answer the very question the boundary says it must not answer.
        #
        # When the budget is spent, NOTHING is bound. A refusal the model can ignore is
        # not a bound: on 2026-10-17 a turn answered a question correctly and then asked
        # for the same search three more times, each refused with "answer from the
        # results you have", until the graph hit its recursion limit and the run failed
        # with a correct answer already on screen. Binding nothing is the same boundary
        # the command path uses — enforced by the absence of anything to call.
        spent = len(state.get("tool_calls") or [])
        if spent >= MAX_TOOL_CALLS_PER_TURN:
            answering_llm = chat_model
        else:
            answering_llm = bound_for(state)
        if prose_stream and on_event is not None:
            response = await _streamed_turn(answering_llm, state["messages"], on_event)
        else:
            response = await answering_llm.ainvoke(state["messages"])
        if spent >= MAX_TOOL_CALLS_PER_TURN:
            # This turn has no tools left and no further round to spend them in, so
            # whatever it asked for cannot happen. Settle it as an answer now: the
            # alternative is an empty turn that reads to the caller as a crash.
            response = _budget_exhausted_turn(response)
        return {"messages": [response], "tool_calls": []}

    async def tool_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        calls = last.tool_calls if isinstance(last, AIMessage) else []
        # `_presented_for_model` rewrites the estimate into the strings the widget
        # prints and withholds its contribution magnitudes. Projected rather than raw:
        # the model never holds a float it could paste.

        messages, recorded = await _execute_tool_calls(
            toolbox, calls, on_event, guard=guard,
            project=_presented_for_model,
            # Everything this turn has already spent, so the budget is a budget for the
            # turn rather than for one iteration of the loop around it.
            already_called=list(state.get("tool_calls") or []),
            # And everything the caps have taken away: a note-reading tool this turn has
            # already used, or a section read the replayed turns already hold. Not
            # binding it is the first line; this refuses the call a model can still write
            # for a tool that is missing from its request.
            used_up=allowed - _still_bound(allowed, state),
            # Numbering continues from every citable unit the conversation has shown:
            # the replayed turns' note passages AND their laboratory panels, then the
            # calls this turn has already made. `_citable_unit_total` is the same sum
            # `replay_messages` advances by, so the transcript the model reads and the
            # numbers its next tool result carries cannot disagree. Counting only
            # `passages` here skipped a replayed `/labs` turn's nine analytes, and a
            # follow-up's first passage was then stamped `^[1]` beside a panel that
            # already held `^[1]`..`^[9]`, so a claim about the treatment course
            # resolved to an analyte.
            cite_offset=_citable_unit_total(state),
        )
        return {"messages": messages, "tool_calls": recorded}

    async def demand_retrieval_node(state: AgentState) -> dict:
        # A `HumanMessage` rather than a system instruction: a system message appended
        # mid-conversation is not where the provider expects one, and this is a change of
        # instruction inside a run rather than a second contract. The transcript it joins
        # is the run's own and is never stored as the turn's question.
        return {
            "messages": [HumanMessage(content=RETRIEVAL_DEMAND)],
            "retrieval_demanded": True,
        }

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            # Nothing is bound once the budget is spent, so this should be
            # unreachable — and it stays here because "should be unreachable" is how
            # a loop gets to the recursion limit. A turn that still asks for a tool
            # after its budget is over is ended, not refused again.
            if len(state.get("tool_calls") or []) >= MAX_TOOL_CALLS_PER_TURN:
                return END
            return "tools"
        # No tool call, so the turn is finished — unless it is an answer about the chart
        # written without reading the chart. The six ways that is not the case: something
        # was already retrieved, the demand has already been made once, the reply is the
        # redirect the prompt asks for (a complete answer that calls no tool by design),
        # there is no reply at all, the manifest is the evidence (see below), or a
        # declared driver drill-down already holds its attribution in the replay.
        if state["tool_calls"] or state["retrieval_demanded"]:
            return END
        answer = text_of(getattr(last, "content", ""))
        if not answer.strip():
            # A turn with no text is a failure — truncated at the token cap, stopped by a
            # safety filter, or empty for a provider reason — and it is reported as one.
            # Demanding a retrieval here would re-prompt a model that has just produced
            # nothing and would surface the second failure in place of the first.
            return END
        if names_a_command(answer):
            return END
        if manifest_answers_question:
            # The manifest is the evidence for this question. Forcing a note search
            # cannot improve the answer — the notes do not carry the field — and it
            # is not harmless: the forced round's reply becomes the settled answer,
            # so the correct one is replaced by an abstention. See
            # `routing.asks_for_administrative_detail`.
            return END
        if risk_driver_declared and _transcript_carries_attribution(state):
            # A clicked driver bar asked what stands behind a factor it was already
            # scored on, and the replayed turns carry that scoring call. The same
            # argument as the manifest: the evidence for this question is already in
            # the transcript, so a forced search buys nothing and costs the answer.
            # Declared, not inferred from the wording — the drill-down knows what it
            # is, and only that declaration distinguishes it from a first turn.
            return END
        if follow_up_declared and state.get("replayed_tool_calls"):
            # A question this product generated from the record. `followups.suggest`
            # offers a chip only when the data behind it is in the turn that offered it
            # — that is the module's answerability invariant — and that turn is in the
            # transcript by the time the chip is clicked, so there is nothing a forced
            # retrieval can add.
            #
            # Declared, like the drill-down above, and for the same reason: the agent
            # cannot tell a generated question from a typed one by reading it. Two
            # attempts to infer it from the wording failed in production on 2026-10-25 —
            # the first let the demand fire on a chip whose own slot text ("Red blood
            # cells") missed a singular-only vocabulary by one letter, and the second
            # then removed the tool that would have answered it, so the turn answered
            # from the notes that the labs were not documented. Provenance is a fact the
            # client has; it sends it.
            return END
        return "demand_retrieval"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("demand_retrieval", demand_retrieval_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        route,
        {"tools": "tools", "demand_retrieval": "demand_retrieval", END: END},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("demand_retrieval", "agent")
    return graph.compile()


def _context_block(admission_context: str, hadm_id: int | None = None) -> str:
    """The pre-flight manifest, appended to the system prompt.

    Appended rather than sent as a second system message: a turn then has exactly
    one instruction message, which is what every provider accepts, and the block
    reads as a continuation of the rules it is qualifying rather than as a peer
    of them.

    An OUTPATIENT record also gets `routing.OUTPATIENT_ENCOUNTER_NOTICE` here, and
    this is the only place the free-text path can learn it. `SYSTEM_PROMPT` is
    admission-independent by construction, so without this the loop wrote "this
    admission" and a hospital course about a speech-therapy visit — the identical
    defect the command path had, on the path that has no `Outpatient` caveat in its
    tool result because it has no formatter (90000073, "Summarize the note",
    2026-10-21).
    """
    parts = [(admission_context or "").strip()]
    if encounter.current_is_outpatient() is True:
        parts.append("<encounter_notice>\n"
                     f"{routing.OUTPATIENT_ENCOUNTER_NOTICE}\n"
                     "</encounter_notice>")
    text = "\n\n".join(part for part in parts if part)
    return f"\n\n{text}" if text else ""


def _trace_metadata(
    hadm_id: int | None, question_kind: str | None, run_tag: str | None = None,
    turns: list[dict[str, str]] | None = None, intent: str | None = None,
) -> dict:
    """What a Langfuse trace needs in order to be found again by patient or by run.

    Without this a trace carried only SDK instrumentation keys, so the question the
    eval gates actually ask — "show me every trace for this case" — had no answer:
    `sessionId` and `userId` were None and there was no admission anywhere on it.

    The admission goes on as a `hadm:` tag (filterable) and as metadata (readable).
    A conversation-scoped session id would need the site to send one; the agent is
    stateless and is not given it today, so it is deliberately absent rather than
    invented from the admission.

    `run:` is the same idea for a batch run. An evaluation turn and a clinician's turn
    were indistinguishable in the trace list — both named `LangGraph`, both tagged with
    the admission — so "which traces belong to that eval" could only be answered by
    guessing at a time window (observed 2026-10-17, when a cleanup had to do exactly
    that and could not prove it had spared production traffic).

    `intent:` is the declaration a control sent, when one did. Added 2026-10-26 after a
    follow-up chip looped in production: `route` ends a declared follow-up in one pass,
    so the trace of a chip that looped could not be read without knowing what the
    request had declared — and that was the one fact the trace did not keep. The tag is
    written only when a declaration arrived, so `intent:follow_up` is a filter and an
    untagged trace means no control spoke.

    `rev:` is the deployment that produced the trace, and it is not optional. Without
    it a trace could be dated but not attributed: a run's answers could not be tied to
    the build that generated them, and a deploy landing mid-run split one run across two
    revisions with nothing marking the boundary. `--expect-revision` samples `/health`
    once before a run starts and proves only what was serving then. The value comes from
    `observability.CODE_REVISION` — `CODE_REVISION` when the deploy supplies the commit,
    otherwise Cloud Run's `K_REVISION`, otherwise the literal `local`, which is an honest
    label rather than an invented one.
    """
    tags: list[str] = []
    meta: dict[str, object] = {}
    # The request's own label wins; the context variable is what carries it here
    # without changing `ask`'s signature.
    run_tag = run_tag or observability.current_run_tag()

    # Which turn of the conversation this is, derived from the replay the caller sent
    # rather than counted locally: the agent holds nothing between requests, so the
    # transcript is the only honest source — the same reasoning that decides the stage
    # labels. A first turn carries no `turns`, so it is turn 1.
    turn_index = 1 + len(turns or [])
    tags.append(f"turn:{turn_index}")
    meta["turn"] = turn_index

    if hadm_id is not None:
        tags.append(f"hadm:{hadm_id}")
        meta["hadm_id"] = str(hadm_id)
    if question_kind:
        tags.append(f"command:{question_kind}")
        meta["command"] = str(question_kind)
    if observability.CODE_REVISION:
        tags.append(f"rev:{observability.CODE_REVISION}")
        meta["code_revision"] = str(observability.CODE_REVISION)
    if intent:
        tags.append(f"intent:{intent}")
        meta["intent"] = str(intent)
    if run_tag:
        tags.append(f"run:{run_tag}")
        meta["run_tag"] = str(run_tag)
        # Eval turns are grouped per admission *within a run*: `{run}:hadm_{id}`.
        # Live traffic carries no run tag and is deliberately given no session id —
        # the agent is stateless and is not told what a conversation is, and inventing
        # one from the admission would lump every past turn of that patient into one
        # endless session, which is a worse lie than no grouping at all.
        if hadm_id is not None:
            meta["langfuse_session_id"] = f"{run_tag}:hadm_{hadm_id}"
    meta["langfuse_tags"] = tags
    return meta


async def ask(
    toolbox: MCPToolbox,
    question: str,
    model: str = MODEL_ID,
    name: str | None = None,
    tags: list[str] | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    question_kind: str | None = None,
    hadm_id: int | None = None,
    topic: str | None = None,
    turns: list[dict[str, str]] | None = None,
    prose_stream: bool = False,
    admission_context: str = "",
    run_tag: str | None = None,
    intent: str | None = None,
) -> dict:
    """Run one question to completion. Returns the final state.

    `name`/`tags` are forwarded to the LangGraph run as its `run_name` and tags,
    so a caller can label a run (the eval loop passes name=`eval.risk`). They
    label the run; nothing here produces or stores a trace id.

    `on_event` is called with a stage dict as the chain progresses (see
    `stages.py`). It is a plain synchronous callback — the streaming route
    hands it a queue's `put_nowait` — and it is optional, so the answer path
    and the progress path stay one implementation.

    `prose_stream` is the caller asking to be shown the prose as it is written, and
    it now reaches both paths: a command's formatting turn, and the free-text model
    turn through `_streamed_turn`. On the free-text path the prose is held to the end
    of each model turn and announced only if that turn called no tool, because a ReAct
    turn's first paragraph may be deliberation rather than answer. Nothing is withheld
    until the guardrails have run; what a caller watches arrive is still provisional,
    and `http` measures it against the prose finally served on every streamed turn.
    (UX Phase 4, reversing the Phase 2 per-path policy — the reasons the reversal is
    made, and what carries it, are in `stages.py`.)

    `turns` is the earlier conversation, as the caller understands it. Each turn
    is replayed as its own messages rather than pasted into the prompt, so the
    template stays a template and the replayed material stays data: the answer a
    user saw is an assistant message here, exactly as it would be mid-turn, and
    it never becomes part of the instruction that frames the question.
    """
    # A command runs its tool directly and then formats the result. The reasoning
    # loop below is for free text, where choosing which tool to call is the
    # model's job. The boundary between the two is a property of the path rather
    # than of a prompt: on this path the model is never handed a tool to choose
    # (UX Phase 1).
    if routing.is_command(question_kind):
        tracing = observability.handler()
        # One span over the whole command, so the turn has a tree: the tool calls
        # and the formatting generation nest under it, and Langfuse can draw the
        # graph. Without it this path emitted a single parentless generation and
        # every command trace had an empty Graph view.
        #
        # The span is also the ROOT TRACE, so the metadata goes on here rather than
        # only into the formatting model's config: `/risk` makes no LLM call at all,
        # so a trace whose tags depended on a model run carried none.
        command_meta = _trace_metadata(hadm_id, question_kind, run_tag, turns, intent)
        with observability.observation(f"command.{question_kind}",
                                       trace_metadata=command_meta) as root_span:
            command_state = await run_command(
                toolbox,
                routing.command_for(question_kind),
                hadm_id,
                topic=topic,
                question=question,
                model=model,
                on_event=on_event,
                callbacks=[tracing] if tracing is not None else None,
                prose_stream=prose_stream,
                trace_metadata=command_meta,
            )
            # The command's answer is settled before anything reads it: a marker removed
            # after the trace is written would still be in the record the judge scores.
            sanitize_answer_markers(command_state)
            # `tools:<n>` cannot be known when the span opens. `tools:0` is the tag the
            # triage view filters on, so it is added here, once the count exists.
            observability.stamp_trace_tags(
                root_span,
                [*command_meta.get("langfuse_tags", []),
                 f"tools:{len(command_state.get('tool_calls') or [])}"],
            )
            # The trace table reads `input`/`output`; without them every row is a dash.
            # The command path is the one that most needs it: `/risk` returns no prose,
            # so the answer a reader wants to see IS the structured surface.
            observability.set_trace_io(
                question, command_trace_output(command_state), span=root_span,
            )
            # Read from the span itself, not from the ambient context: the context can
            # go missing (four command turns in the 22:38 run reported no trace id at
            # all), while the span always knows the trace it belongs to.
            command_state["langfuse_trace_id"] = (
                observability.span_trace_id(root_span)
                or observability.current_trace_id()
            )
        return command_state

    manifest_answers_question = bool(admission_context) and (
        routing.asks_for_administrative_detail(question)
    )

    # Whether this turn is about published records, asked once because two things depend
    # on it and they must not disagree: which tools are bound (`_freetext_allowed`) and
    # whether the replayed turns' OWN searches are carried into this one
    # (`replay_messages`). A chart turn keeps neither.
    wants_literature = routing.asks_for_literature(question)

    # Earlier turns' published-records searches are carried only into a turn that is
    # itself about published records. See `replay_messages`: a `search_literature`
    # result left in the transcript re-entered every later turn, binding a marker to an
    # article nobody had been shown and badging a chart answer as general evidence.
    #
    # Built before the graph rather than after it, because ONE binding decision reads it:
    # whether the laboratory panel this turn is asked to rank is already in hand
    # (`_freetext_allowed`). The replay is the only place that fact lives.
    replayed, replayed_calls = await replay_messages(
        toolbox, turns, keep_literature=wants_literature)

    graph = build_graph(
        toolbox, model=model, on_event=on_event, question_kind=question_kind,
        # The one identifier this turn holds, checked against what is about to
        # leave the service (UX Phase 1B).
        guard=lambda name, args: _outbound_patient_identifier(name, args, hadm_id),
        prose_stream=prose_stream,
        # Computed once, here, because it is a property of this turn: a question the
        # pre-flight manifest answers, in a turn that carries the manifest.
        manifest_answers_question=manifest_answers_question,
        # Also a property of this turn: whether the question asks for the model's own
        # reasoning about a driver, which is the only way `predict_readmission` enters
        # the free-text tool set (see `_freetext_allowed`). A clicked driver bar declares
        # that it is one, so the control decides instead of its label.
        risk_driver_question=routing.wants_risk_driver(question, intent),
        # And whether it asks for the estimate itself — "what is his readmission risk
        # score, and is it above threshold". The same tool, a different reason, and the
        # one that stops a cold-start thread being told to go and type a command.
        risk_score_question=routing.asks_for_risk_score(question),
        # The declaration on its own, kept separate from the question it also implies:
        # `route` uses it to end a replay turn that already holds the attribution, and
        # a question that merely reads like a driver question was never scored.
        risk_driver_declared=intent == "risk_driver",
        # The same shape for the product's own questions. A chip is generated from a
        # tool result in this conversation, so when it is clicked the evidence is already
        # in the transcript and a forced retrieval can only cost the answer. Declared by
        # the console that drew the chip, never inferred from its wording.
        follow_up_declared=intent == "follow_up",
        # The other half of `_freetext_allowed`: published evidence is searched only on
        # a turn that asked for it, so a chart question cannot come back with a
        # `LiteratureList` and no note citations.
        wants_literature=wants_literature,
        # And the half that WITHHOLDS a tool: a driver question about a count or a
        # disposition is not documented by a lab series, so asking for one put a panel in
        # the turn's evidence and a lab chip under an answer about utilization.
        lab_driver_question=routing.asks_about_a_lab_analyte(question),
    )

    config: dict = {"recursion_limit": RECURSION_LIMIT}
    trace_meta = _trace_metadata(hadm_id, question_kind, run_tag, turns, intent)
    if trace_meta:
        # Langfuse reads `langfuse_tags` from run metadata; the other keys ride along
        # so the admission and command are visible on the trace itself.
        config["metadata"] = trace_meta
    if name is not None:
        config["run_name"] = name
    if tags:
        config["tags"] = list(tags)

    # Tracing is attached HERE, as one callback, rather than by instrumenting the
    # nodes: every model call is a LangChain LLM run and every tool call a
    # LangChain tool run, so the handler sees the whole chain and this function is
    # the only place that had to know. `observability.handler()` returns None when
    # the environment carries no Langfuse configuration, which is a supported
    # state and not a degraded one.
    tracing = observability.handler()
    if tracing is not None:
        config["callbacks"] = [tracing]

    # The same root span the command path uses, for the same two reasons: the trace
    # gets its tags and session id from us rather than depending on the SDK reading
    # them off a model config, and `tools:<n>` can be appended once the turn is over.
    # `tools:0` is what the "Turn-2 evidence miss" view filters on, and a follow-up
    # answered from memory is exactly a turn that calls nothing.
    with observability.observation(
        f"free_text.turn{trace_meta.get('turn', 1)}", trace_metadata=trace_meta,
    ) as root_span:
        state = await graph.ainvoke(
            {
                "messages": [
                    SystemMessage(
                        content=SYSTEM_PROMPT + _context_block(admission_context, hadm_id)),
                    *replayed,
                    HumanMessage(content=question),
                ],
                "tool_calls": [],
                # The replayed turns' calls travel with the run so the answer can be
                # guarded against the evidence it was actually given. Without them a
                # follow-up reusing the stored score is served with the score
                # stripped as unsupported, which is a correct answer destroyed by a
                # guardrail that could only see this turn.
                "replayed_tool_calls": replayed_calls,
                # Cleared per run, so the demand below is made at most once per question
                # rather than once per conversation.
                "retrieval_demanded": False,
            },
            config=config,
        )
        observability.stamp_trace_tags(
            root_span,
            [*trace_meta.get("langfuse_tags", []),
             f"tools:{len(state.get('tool_calls') or [])}"],
        )
        observability.set_trace_io(question, final_text(state), span=root_span)
        state["langfuse_trace_id"] = (
            observability.span_trace_id(root_span) or observability.current_trace_id()
        )
        # The settlement layer decides whether the answer was verified against the
        # patient record, and the manifest is one of the channels that make it so. It
        # is not a tool call and leaves no other trace on the state, so it is recorded
        # here rather than inferred downstream from an answer that reads like a lookup.
        state["manifest_answered"] = manifest_answers_question

    # The model's own verdict on what it was given, read before anything else looks
    # at the text: an answer the model wrapped in the contract's envelope is
    # unwrapped here, so the guardrails, the citation layer and the canvas all see
    # prose. A reply that is not the envelope is left exactly as it arrived.
    reply = read_evidence_reply(final_text(state))
    if reply.text != final_text(state):
        _replace_final_text(state, reply.text)

    # A question the notes did not answer is answered here rather than by the model,
    # which has already written something plausible that nobody will see (see
    # `no_evidence_answer`). Set on the state, so both routes serve it and the
    # guardrails and citation layer treat it like any other answer.
    fallback = no_evidence_answer(
        question, state.get("tool_calls") or [], declined=reply.declined
    )
    if fallback is not None:
        state["deterministic_answer"] = fallback
    elif reply.declined:
        # Declined, and this turn may not assert an absence: no retrieval ran, a
        # search failed, or the literature was consulted. The flag is not an answer
        # and the fixed sentence would be a claim the records do not support, so the
        # turn carries no text and the routes report an unavailable answer rather
        # than a fabricated one (see `_compose_success`).
        logger.warning(
            "model declined with no absence to report; calls: %s",
            ",".join(call.get("name") or "?" for call in state.get("tool_calls") or [])
            or "none",
        )
        _replace_final_text(state, "")

    # A marker that addresses no citable unit is not a citation. Removed here, where the
    # answer is settled, so the trace the judge reads carries the same text the browser
    # does (see `sanitize_answer_markers`).
    sanitize_answer_markers(state)

    # The trace id leaves with the state so the layers that need to point back at
    # a run can: `collect.py` stores it per case, and `judge.py` attaches rubric
    # scores to it. Empty when tracing is off, which is why every reader of this
    # field treats "" as "no trace", not as an error.
    state["langfuse_trace_id"] = observability.trace_id(tracing)
    return state


def replayed_turns(turns: list[dict[str, str]] | None) -> list[BaseMessage]:
    """Earlier turns as messages, in order, oldest first.

    A turn becomes the pair a conversation would have produced: the question as
    a human message, the answer as an assistant message. Kept as messages rather
    than folded into a transcript string, because the prompt marks tool results
    as data and a flattened history would put clinical text where the prompt says
    instructions live.

    Nothing here decodes or reformats the text: what the user asked and what they
    were told are replayed as they were, which is the only faithful option and
    the reason the stored shape keeps them verbatim.

    This is the dialogue half of a replay. `replay_messages` is what a run uses,
    because a turn that made tool calls needs those results back in their place
    in the sequence.
    """
    messages: list[BaseMessage] = []
    for turn in turns or []:
        messages.append(HumanMessage(content=turn["question"]))
        messages.append(AIMessage(content=turn["answer"]))
    return messages


def _replayed_tool_messages(
    call: dict[str, Any], call_id: str, payload: Any, *, offset: int = 0
) -> list[BaseMessage]:
    """One replayed tool call and its result, rendered exactly as a live pair is.

    The same two message types in the same order the chain produces mid-turn: the
    assistant turn that asked for the call, then the result. The result goes
    through the same renderer a live one does, so the wrapper and the delimiter
    guard apply identically — replaying a passage outside the wrapper would put
    clinical text where the prompt says instructions live, which is the whole
    reason the boundary exists.

    It also goes through the same PROJECTION a live one does, which is the half this
    function was missing. `tool_node` renders `project(payload)`; this rendered the raw
    payload, so a follow-up turn saw a different result from the turn that fetched it.
    What the projection adds is `probability_display` / `threshold_display` in place of
    the estimate's floats, which is what keeps a number the model could paste out of its
    hands — a replayed estimate without them handed the model the raw score.
    """
    asked = AIMessage(
        content="",
        tool_calls=[{"name": call["name"], "args": call.get("args") or {}, "id": call_id}],
    )
    body = _render_tool_result(
        _presented_for_model(call["name"], payload),
        offset=offset)
    result = ToolMessage(
        content=f'<tool_result name="{call["name"]}">\n{body}\n</tool_result>',
        tool_call_id=call_id,
        name=call["name"],
    )
    return [asked, result]


async def replay_messages(
    toolbox: MCPToolbox, turns: list[dict[str, Any]] | None,
    *, keep_literature: bool = True,
) -> tuple[list[BaseMessage], list[RecordedToolCall]]:
    """The earlier conversation, as the messages the model would have seen.

    A turn is replayed in the order it happened: the question, then each tool
    call with its result, then the answer. The result is the one the turn stored
    when it has one — a prediction score cannot be recomputed to the same value —
    and is otherwise resolved again by calling the tool, which is deterministic
    here because the corpus is written offline and never mutated by a request.

    Re-resolving is the price of not storing passage text, and it is paid only
    for the calls that did not store a payload. Nothing is emitted as a progress
    stage: this happens before the run, and a stage describing work the caller
    did not ask for in this turn would be a line about someone else's question.

    The calls come back alongside the messages, in the same `{name, args,
    response}` shape a live call is recorded in, because they are evidence the
    answer will be judged against: a follow-up answered from a replayed
    prediction quotes that prediction's score, and a guardrail that knew only
    this turn's calls would strike the score out of a correct answer. Returning
    them here is what keeps one resolution of a payload — messages and evidence
    are two views of the same replay, never two trips to the tool.
    """
    messages: list[BaseMessage] = []
    calls: list[RecordedToolCall] = []
    offset = 0
    for turn_index, turn in enumerate(turns or []):
        messages.append(HumanMessage(content=turn["question"]))
        for call_index, call in enumerate(turn.get("tool_calls") or []):
            # A published-records search is not carried into a turn that is not about
            # published records (Phase E, 2026-10-26).
            #
            # `search_literature` returns somebody else's patients. Kept in the transcript,
            # it re-entered every later turn: a `/meds` follow-up was answered with `^[1]`
            # bound to a World Neurosurgery article, badged General Medical Evidence, with
            # the old `LiteratureList` re-rendered under the medication instructions
            # (90000058). The records were retrieved for a question that was asked and
            # answered; a later question about this patient's chart is not a reason to show
            # them again. Nothing is dropped that the current turn asked for —
            # `keep_literature` is set from the question itself, and a genuine literature
            # question is bound the same tool anyway.
            if not keep_literature and str(call.get("name")) in routing.LITERATURE_TOOLS:
                continue
            payload = call.get("payload")
            if payload is None:
                logger.info(
                    "re-resolving %r for replayed turn %d: its payload is not "
                    "stored, and the corpus it reads is deterministic",
                    call["name"], turn_index,
                )
                payload = await toolbox.call(call["name"], dict(call.get("args") or {}))
            messages.extend(
                _replayed_tool_messages(call, f"replay-{turn_index}-{call_index}", payload,
                                        offset=offset)
            )
            # The numbering continues across turns, because the citation layer resolves
            # a marker against every citable unit the conversation has shown.
            offset += _passage_total(payload)
            calls.append({
                "name": call["name"],
                "args": dict(call.get("args") or {}),
                "response": payload if isinstance(payload, dict) else {"result": payload},
            })
        messages.append(AIMessage(content=turn["answer"]))
    return messages, calls


def final_message(state: dict) -> AIMessage | None:
    """The last message, when it is the model's — the turn the answer comes from.

    Returned rather than just its text because the same response carries the
    finish reason and any content-filter feedback. A caller that has to explain a
    missing answer needs those, not only the empty string (`model_turn`).
    """
    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    return last if isinstance(last, AIMessage) else None


def final_text(state: dict) -> str:
    """Text of the FINAL assistant message — never an earlier one.

    Falling back to the last NON-EMPTY AI message served a stale pre-tool
    preamble with HTTP 200 whenever the final turn came back empty (the
    documented MAX_TOKENS failure that raises nothing). Empty means the answer
    is unavailable, and the server reports exactly that (ECC-12).
    """
    message = final_message(state)
    if message is None:
        return ""
    # One implementation of "where is the text in a message", shared with the
    # streaming paths. It lived here until they needed it too (`model_turn.text_of`).
    return text_of(message.content).strip()


def command_trace_output(state: dict) -> str:
    """What the trace table shows as a command turn's output.

    The three keys this used to read — `narrative_text`, `answer`, `text` — are not in a
    command's state: they are names the HTTP envelope uses, composed after this point. So
    every command trace was written with an empty output while every free-text trace had
    one, and the table drew a dash for exactly the turns carrying the most structure
    (2026-10-21: 25 command traces on a page, none with an output).

    The prose is read where the free-text path reads it, `messages[-1]`. `/meds` keeps its
    answer in `deterministic_answer`, because its model reply is extraction JSON and the
    sentence a reader sees is written in code from the verified rows. `/risk` has no prose
    at all by design — the widget carries the answer — so its output is the payload that
    widget draws, which is the honest record of what the turn returned.
    """
    answer = str(state.get("deterministic_answer") or "").strip()
    if answer:
        return answer
    prose = final_text(state)
    if prose:
        return prose
    payload = state.get("command_payload")
    if isinstance(payload, dict) and payload:
        return json.dumps(payload, default=str)[:4000]
    return ""