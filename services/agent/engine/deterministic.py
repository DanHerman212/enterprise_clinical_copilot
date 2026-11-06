"""The deterministic path: a command runs its tool, then formats the result.

UX Phase 1. A slash command does not reason about which tool to call — it names
one tool, this module calls it, and the model's only job is to turn the result
into prose the interface can render.

Two properties make the boundary hold, and neither is a matter of instruction:

  1. **The tool is called here, not chosen by the model.** The command's tool and
     arguments come from `routing.py`; the model is not asked.
  2. **The formatting turn has no tools bound.** `_build_llm` returns the model
     with no `bind_tools`, so a secondary search is not forbidden, it is
     impossible — there is nothing to call.

Everything downstream is unchanged: the formatting turn's text goes back through
the same post-model pipeline (`guardrail.py`, `citations.py`, `a2ui.py`) that the
reasoning loop's answer does, so a command's answer is guarded exactly like any
other.

Medication extraction
---------------------
`/meds` is the one command whose formatting is structured rather than prose: the
model acts as a data-entry clerk over the section text and returns JSON. Clinical
prose is too irregular for a deterministic parser, so the model does the
extraction — and every row it returns is then checked against the source text,
because an extraction that invents a medication is the one failure a clinician
cannot see.

Topics
------
`/pubmed` needs the caller's own words, and they reach the tool unaltered: the
command declares which argument they fill, and this module passes the request's
topic straight through. Nothing here invents one — a search string the agent
composed would be a search nobody asked for and nobody can see.

Field names
-----------
A command's result is rendered as prose by a model that has never seen the
service's identifiers. Where a name the interface draws with already exists
(`feature_labels.py`), the model is shown that name rather than the key, so the
prose and the widget cannot describe one factor two ways.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from services.agent.routing import routing
from services.agent.routing import encounter
from services.agent.telemetry import stages
from services.agent.postprocess import citations
from services.agent.postprocess import answer as answer_text
from services.agent.postprocess import evidence_state as session_evidence
from shared.labels import humanize
from services.agent.engine.model import text_of
from services.agent.engine import model as model_gateway
from services.agent.engine import model as model_turn
from services.agent.engine import prompts
from services.agent.telemetry.stages import emit as _emit
from services.agent.engine.mcp_client import (
    MCPToolbox,
    RecordedToolCall,
    _execute_tool_calls,
    _presented_call,
    _presented_for_model,
    # Re-exported for the tests that assert the estimate's projection directly. The
    # module no longer calls it — it calls the dispatcher — but the seam it names is
    # still the one those tests stand on.
    _presented_prediction,
)
from services.agent.engine.prompts import SYSTEM_PROMPT
from services.agent.engine.prompts import (
    EHR_RECORD_QUERY,
    FORMATTER_PREAMBLE,
    MEDS_FALLBACK_FORMATTING,
    MEDS_FORMATTING,
    OUTPATIENT_ENCOUNTER_NOTICE,
    formatting_for,
)
from services.agent.routing.routing import (
    Command,
    supplementary_arguments,
    tool_arguments,
)
from services.mcp.retrieval.sections import ADMISSION_DIAGNOSIS

logger = logging.getLogger(__name__)


# A medication line's dose, as it appears in note text: a number followed by a
# unit, optionally with a route and a frequency. Used only to VERIFY an extracted
# row, never to extract one — a parser good enough to extract would be a parser
# bad enough to fail on the irregular lines, which is why the model extracts.
_DOSE_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?|iu|meq|%)", re.IGNORECASE
)


def _source_text(recorded: list[RecordedToolCall]) -> str:
    """Every passage the tool returned, as one searchable string."""
    parts: list[str] = []
    for call in recorded:
        response = call.get("response")
        if not isinstance(response, dict):
            continue
        for passage in response.get("passages") or []:
            if isinstance(passage, dict) and isinstance(passage.get("text"), str):
                parts.append(passage["text"])
    return "\n".join(parts)


def _normalise(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def _medication_sections(recorded: list[RecordedToolCall]) -> tuple[str, ...]:
    """The note sections this turn's medication retrieval actually returned, in order.

    The distinction the narrative depends on: a `discharge_medications` section is a
    formal list and may be counted as one, while a fallback retrieval that landed in the
    hospital course may not. `discharge_medications` returns the tools' own `section`
    labels, so this is the retrieval's answer rather than a guess about the prose.
    """
    sections: list[str] = []
    for call in recorded:
        response = call.get("response")
        if not isinstance(response, dict):
            continue
        for passage in response.get("passages") or []:
            if not isinstance(passage, dict):
                continue
            section = str(passage.get("section") or "").strip()
            if section and section not in sections:
                sections.append(section)
    return tuple(sections)


#: The section whose presence makes `/meds` a discharge-list question.
_DISCHARGE_MEDS_SECTION = "discharge_medications"


def meds_formatting_for(sections: tuple[str, ...]) -> str:
    """The extraction instruction this record's own retrieval calls for.

    One wire, two questions. Where a `discharge_medications` section came back, the
    question is what the patient was sent home on and the arrival-list exclusion applies.
    Where it did not, the retrieval fell back to narrative sections and the same exclusion
    discards most of what they name — so the question becomes what those sections NAME.

    Decided from the sections the tool returned rather than from the note's admission or
    its prose: the retrieval is the one thing this layer can see without guessing, and the
    narrative that follows is written from the rows this instruction produced.
    """
    if _DISCHARGE_MEDS_SECTION in sections:
        return MEDS_FORMATTING
    return MEDS_FALLBACK_FORMATTING


def parse_medications(
    model_output: str, source_text: str
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """The extracted rows, and the rows refused because the note does not say them.

    Returns `(verified, dropped)`. A row survives when its medication name and,
    if stated, its dosage both appear in the section text. The check is
    deliberately lenient about formatting — the note writes "Celebrex 200 mg
    daily" and an extraction may return "celebrex" and "200 mg" — and strict
    about substance: a name that is nowhere in the text is dropped, because a
    medication the note does not mention is the one error that reads as
    authoritative.
    """
    payload = _extract_json_object(model_output)
    rows = payload.get("medications") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        logger.warning("medication extraction returned no 'medications' array")
        return [], []

    source = _normalise(source_text)
    verified: list[dict[str, str]] = []
    dropped: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = _normalise(str(row.get("medication") or ""))
        dosage = _normalise(str(row.get("dosage") or ""))
        frequency = _normalise(str(row.get("frequency") or ""))
        if not name:
            continue
        reason = None
        if name not in source:
            reason = "medication not present in the note text"
        elif dosage and not _dose_supported(dosage, source):
            reason = "dosage not present in the note text"
        if reason:
            dropped.append({"medication": row.get("medication", ""), "reason": reason})
            continue
        verified.append({
            "medication": str(row.get("medication") or "").strip(),
            "dosage": str(row.get("dosage") or "").strip(),
            "frequency": str(row.get("frequency") or "").strip(),
        })
    return verified, dropped


def _dose_supported(dosage: str, source: str) -> bool:
    """Whether a dosage is stated in the note.

    A dosage is supported when its number-and-unit appears in the text, allowing
    the extraction to drop the surrounding words ("200 mg daily" -> "200 mg").
    """
    match = _DOSE_RE.search(dosage)
    if match:
        return _normalise(match.group(0)) in source
    return dosage in source


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """The JSON object in a model's reply, tolerating a fenced code block."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
    except ValueError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start == -1 or end <= start:
            logger.warning("no JSON object found in the extraction reply")
            return None
        try:
            parsed = json.loads(stripped[start:end + 1])
        except ValueError:
            logger.warning("extraction reply was not valid JSON")
            return None
    return parsed if isinstance(parsed, dict) else None


def _passage_count(recorded: list[RecordedToolCall]) -> int:
    """How many passages this turn's tool calls actually returned."""
    return sum(
        len((call.get("response") or {}).get("passages") or [])
        for call in recorded
        if isinstance(call.get("response"), dict)
    )


def _retrieved_records(recorded: list[RecordedToolCall]) -> list[Any]:
    """Everything the turn retrieved that a reader can read on the page.

    A note passage or a published record. Asked in one place because the answer needs a
    yes/no — may the withheld-summary sentence point at records? — and the log line wants
    a count, and two implementations of "what counts as a record" would eventually
    disagree about a tool that returns neither (`predict_readmission`, a medication
    extraction) and quietly blame the model for an empty result.
    """
    records: list[Any] = []
    for call in recorded or []:
        response = (call or {}).get("response")
        if not isinstance(response, dict):
            continue
        records.extend(
            passage for passage in response.get("passages") or []
            if isinstance(passage, dict)
        )
        records.extend(
            article for article in response.get("articles") or []
            if isinstance(article, dict)
        )
    return records


def _consolidate_medication_passages(recorded: list[RecordedToolCall]) -> None:
    """Put the whole medications text in one passage, at index 0.

    `discharge_medications` returns every chunk of the medications section and then
    every chunk of the discharge instructions — deliberately, because the names are in
    the first and the doses are in the second, and the extraction needs both. But
    `/meds` cites its narrative with a single `^[1]`, and `^[1]` addresses
    `passages[0]`: the FIRST chunk of the medications section. A reader who clicked it
    saw a fragment of the list the grid had just shown them in full, and the guardrail
    compared the whole answer against one chunk of its evidence.

    Joined here, once, before anything reads the passages — the citation layer, the
    guardrail's source text and the payload all see the same complete text. The labels
    inside the joined body still say which section each part came from, so nothing is
    lost by merging the containers. The model was already given the per-section
    rendering above; this changes what the answer CITES, not what the model read.
    """
    for call in recorded:
        response = call.get("response")
        if not isinstance(response, dict):
            continue
        passages = response.get("passages")
        if not isinstance(passages, list) or len(passages) < 2:
            continue
        text = "\n\n".join(
            str(passage.get("text") or "") for passage in passages
            if isinstance(passage, dict)
        ).strip()
        if not text:
            continue
        # The first passage's own keys are kept; only the text is replaced. The
        # `section` key is deliberately NOT relabelled: citation resolution matches on
        # the machine name (`discharge_medications`), and a human label here would make
        # `resolve_source` miss it and serve the "no information" sentence instead of the
        # passage. What the reader sees as a heading comes from the key, upstream.
        merged = dict(passages[0])
        merged["text"] = text
        response["passages"] = [merged]
        response["returned"] = 1


def medication_narrative(verified: list[dict[str, str]], returned_passages: int) -> str:
    """Deprecated forward: the narrative is written in `postprocess.answer`.

    Moved on 2026-10-29 (`00d7` Row 8). Answer prose belongs to `postprocess/`, and the
    engine's formatting turn only orders the sections. This stays one line so callers that
    still import it from here get identical prose; nothing in the engine calls it.
    """
    return answer_text.medication_narrative(verified, returned_passages)


# The wording lives in `routing`, beside the set that decides which admissions are
# outpatient, because the free-text path appends the same instruction (`graph._context_block`).
# The name is kept here because this is the path that has always carried it and the tests
# read it from here.
_NO_READMISSION = OUTPATIENT_ENCOUNTER_NOTICE


def _context_block(admission_context: str, hadm_id: int | None = None) -> str:
    """The pre-flight manifest, as the text that qualifies a prompt.

    Appended rather than sent as a second system message: a turn then has exactly
    one instruction message, which is what every provider accepts, and the block
    reads as a continuation of the rules it is qualifying rather than as a peer
    of them.

    It lives here, in the layer that composes prompts, and BOTH paths use it: the
    free-text loop appends it to `SYSTEM_PROMPT`, and `/ehr` appends it to the
    formatting turn's input. Until 2026-11-05 it lived in `graph` and reached only the
    free-text loop, so a command was formatted without the manifest and then graded
    against it — two items with byte-identical `/ehr` answers and identical manifests
    scored fidelity 3 and fidelity 1 in one run (`dev_t1_02`, `dev_t2_01`).

    An OUTPATIENT record also gets `prompts.OUTPATIENT_ENCOUNTER_NOTICE` here, and
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
                     f"{prompts.OUTPATIENT_ENCOUNTER_NOTICE}\n"
                     "</encounter_notice>")
    text = "\n\n".join(part for part in parts if part)
    return f"\n\n{text}" if text else ""


def _formatter_input(tool_messages: list[Any], outpatient: bool,
                     question: str | None = None,
                     comparators: list[dict] | None = None,
                     context_block: str = "",
                     session_block: str = "") -> str:
    """What the formatting turn reads: the question, the tool results, the caveat.

    The QUESTION goes first, and it is not decoration. Until Phase 01 this turn could be
    reached by a free-text question that a keyword router had rewritten into `/labs` or
    `/meds`, with the caller's words consumed by the router — so a formatter built from
    the tool result alone answered a question it had never been shown. Observed
    2026-10-21: "What were her hemoglobin and sodium trends?" (90000015) and "Show me
    her hemoglobin and sodium labs" (90000023) were routed to `/labs`, and the model
    summarised all ten analytes because ten was all it had. The router is gone; a command
    now arrives only because the caller named it, and the question is that name.

    `question` defaults to None so the shape this function had before the question was
    threaded through is still expressible, and a turn without one formats exactly as it did.

    `comparators` is the half of the comparison the note passages cannot contain: the
    published cohorts retrieval offered and an extraction turn verified. The table draws
    them under this answer and the references list numbers them, and until 2026-11-02 the
    prose between the two had no idea they existed — the briefing stated this patient's
    estimate and stopped, beside a table comparing it against published rates (UX gap 3).

    `context_block` is the pre-flight manifest, already rendered. It comes last because
    it is the authority the passages are read against: the discharge disposition in it
    outranks a facility a note names from an earlier encounter, and the rule that says so
    (`EHR_FORMATTING`) can only be applied to a block the turn has been shown.

    `session_block` is the session record as it stood BEFORE this command ran — what the
    thread already retrieved, and which sections it already read or found empty. It
    comes before the manifest rather than after it for the reason above: it is more
    evidence for the same question, while the manifest is the rule the evidence is read
    under, and the rule goes last. This turn's own units are not in it; they are in
    `tool_messages`, already numbered.
    """
    text = "\n\n".join(str(m.content) for m in tool_messages)
    if outpatient:
        text = f"{text}\n\n{_NO_READMISSION}"
    block = comparator_block(comparators)
    if block:
        text = f"{text}\n\n{block}"
    if (session_block or "").strip():
        text = f"{text}\n\n{session_block.strip()}"
    if (context_block or "").strip():
        text = f"{text}{context_block}"
    asked = (question or "").strip()
    return f"Question asked: {asked}\n\n{text}" if asked else text


def comparator_block(comparators: list[dict] | None) -> str:
    """The verified cohorts as the formatting turn reads them, or "".

    One line per row, with everything the prose needs and nothing it does not: the PMID
    it must cite, the journal and year the table prints, the rate with its shape, and the
    cohort the rate belongs to. The `rate_sentence` is left out on purpose — it is the
    evidence the row was verified against, not a sentence to be quoted at a reader who
    has not opened the record.

    Empty when there are no rows, and then nothing is appended at all. A header saying
    "no published cohorts" would invite the model to explain the absence, and the prompt
    already says what to write when the block is missing.

    Stated here rather than sent as a tool result because no tool returned these rows:
    retrieval returned twenty-five candidates and an extraction turn verified these.
    """
    rows = [row for row in comparators or [] if isinstance(row, dict)]
    if not rows:
        return ""
    lines = [
        "<verified_comparators>",
        "Published 30-day readmission cohorts for this admission's condition. Retrieval "
        "found these candidates and each row was checked against the abstract it was read "
        "from. Cite a row with its PMID, written ^[PMID: <pmid>], immediately after the "
        "sentence that draws on it. These are POPULATION benchmarks, not predictions "
        "about this patient.",
    ]
    for row in rows:
        pmid = str(row.get("pmid") or "").strip()
        if not pmid:
            continue
        journal = str(row.get("journal") or "").strip()
        year = str(row.get("year") or "").strip()
        source = " · ".join(part for part in (journal, year) if part) or "record"
        rate = str(row.get("reported_rate") or "").strip()
        kind = str(row.get("rate_kind") or "").strip()
        cohort = str(row.get("cohort_summary") or "").strip()
        tier = str(row.get("evidence_tier") or "").strip()
        shape = {
            "comparison": "two arms, published as two figures — quote both",
            "range": "a range, quote it as stated",
        }.get(kind, "a single figure")
        detail = f" · {tier}" if tier else ""
        lines.append(f"- PMID {pmid} · {source} · {rate} ({shape}){detail}")
        if cohort:
            lines.append(f"  cohort: {cohort}")
    lines.append("</verified_comparators>")
    return "\n".join(lines)


async def _formatting_turn(
    llm: Any,
    prompt: list[Any],
    *,
    config: dict | None,
    announce: Callable[[str], None] | None = None,
) -> Any:
    """The formatting model turn.

    Returns the model's message in the shape `ainvoke` returns — the accumulated
    chunks are an `AIMessageChunk`, which subclasses `AIMessage` — so `final_text`
    and `model_turn` are unchanged either way.

    `announce` asks for the prose to be shown as the model writes it; with it None
    this is the single blocking call it has always been. This turn is the one place in
    the agent where announcing on arrival needs no lookahead: no tool is bound
    (`_build_llm` is called without `bind_tools`), so the turn cannot be deliberation
    choosing a tool. There is nothing for the text to turn out to have been instead,
    so there is nothing to hold back.

    Whether the announced text is safe to *show* is a different question and not one
    this function answers. `Command.prose_is_model_output` decides whether the caller
    is offered this turn's output at all, and the guardrails' verdict on it lands
    after the turn has finished — which is what the swap protocol in `http` exists for.
    """
    if announce is None:
        return await llm.ainvoke(prompt, config=config)

    turn: Any = None
    async for chunk in llm.astream(prompt, config=config):
        turn = chunk if turn is None else turn + chunk
        # The chunk's own text, not its repr: Gemini answers in content blocks, so
        # announcing `str(chunk.content)` would put the envelope on screen instead of
        # the summary (see `model_turn.text_of`).
        piece = text_of(chunk.content)
        if piece:
            announce(piece)
    if turn is None:
        raise RuntimeError("the formatting model returned no chunks")
    return turn


#: The client the comparator step extracts with, replaceable so a test can run this step
#: offline. `tests/agent/conftest.py` sets it to `lambda model: None`, and a `None` client
#: means "this environment provides no extraction" — the step is skipped, no model is
#: called, and the production path learns nothing about tests.
extraction_llm_factory: Callable[[str], Any] = model_gateway.build_extraction_llm


async def _readmission_comparators(
    toolbox: Any, model: str, recorded: list[RecordedToolCall], on_event,
    *, extraction_llm: Any = None, evidence_state: Any = None,
) -> tuple[list[dict], dict | None, list[RecordedToolCall], list[dict]]:
    """The published cohorts this admission's estimate is compared against.

    Three steps, and each is somebody else's decision already stated: the condition comes
    from the chart (`derive_target`), the records come from PubMed through the traced
    `search_literature` tool exactly as `/pubmed` uses it, and the rows come from
    `extract_comparators`, which verifies every claim against the abstract it was read
    from.

    The tool call is executed through `_execute_tool_calls`, so it lands in `recorded` with
    its arguments and its response and appears in Langfuse as a tool span. That is the rule
    the whole step exists to satisfy: a cohort table whose rows cannot be traced to a call
    this turn made is the static table again, with a network request in front of it.

    Returns (rows, target, calls, passes). Empty rows is a real answer and is reported as
    one — the reader is told the literature stated no 30-day rate for this condition.
    """
    sections = note_sections(recorded)
    if not sections:
        return [], None, [], []

    llm = extraction_llm if extraction_llm is not None else extraction_llm_factory(model)
    if llm is None:
        # No extraction client in this environment. Stated on the state rather than left
        # silent: `comparators: []` and "nobody looked" are different answers, and the
        # surface says which one it is drawing.
        return [], {"source": "unavailable",
                     "reason": "no extraction client is configured here"}, [], []
    derived = await derive_target(llm, sections)
    target = derived.target
    if target is None:
        return [], {"source": derived.source, "reason": derived.reason}, [], []

    calls_made: list[RecordedToolCall] = []
    counter = {"calls": 0}

    async def search(terms, wide: bool) -> dict:
        # One query, however many terms it was given: the condition leads and the anchors
        # follow as synonyms, which is the one call `search_literature` turns into an OR of
        # quoted phrases against the broadened outcome clause. Duplicates are dropped
        # because the condition is usually one of its own anchors.
        ordered: list[str] = []
        for term in [terms] if isinstance(terms, str) else list(terms or []):
            cleaned = " ".join(str(term or "").split())
            if cleaned and cleaned.lower() not in {entry.lower() for entry in ordered}:
                ordered.append(cleaned)
        if not ordered:
            return {}
        counter["calls"] += 1
        args: dict[str, Any] = {
            "query": ordered[0],
            "max_results": 10,
            "scope": "readmission",
        }
        if len(ordered) > 1:
            args["synonyms"] = ordered[1:]
        call = {
            "name": "search_literature",
            "args": args,
            "id": f"summarize-literature-{counter['calls']}",
        }
        _, recorded_calls = await _execute_tool_calls(
            toolbox, [call], on_event, project=_presented_for_model, traced=True,
            evidence_state=evidence_state,
        )
        calls_made.extend(recorded_calls)
        response = recorded_calls[-1].get("response") if recorded_calls else {}
        return response if isinstance(response, dict) else {}

    result = await extract_comparators(llm, target, search=search)
    payload = {
        "condition": target.condition,
        "procedure": target.procedure,
        "anchors": list(target.anchors),
        "source": derived.source,
    }
    passes = [
        {"query": entry.query, "wide": entry.wide, "candidates": entry.candidates,
         "aligned": entry.aligned, "verified": entry.verified,
         "excluded": entry.excluded}
        for entry in result.passes
    ]
    return result.rows, payload, calls_made, passes


async def run_command(
    toolbox,
    command: Command,
    hadm_id: int | None,
    *,
    topic: str | None = None,
    question: str,
    model: str,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    callbacks: list[Any] | None = None,
    replayed_calls: list[RecordedToolCall] | None = None,
    prose_stream: bool = False,
    trace_metadata: dict | None = None,
    extraction_llm: Any = None,
    admission_context: str = "",
) -> dict[str, Any]:
    """Run one command to completion and return a state the pipeline understands.

    The returned dict has the same shape the reasoning loop's final state does —
    `messages`, `tool_calls`, `replayed_tool_calls` — so `guardrail.py`,
    `citations.py` and `a2ui.py` cannot tell which path produced it, which is
    what keeps a command's answer subject to the same checks as any other.

    `topic` is the caller's own words for a command that takes them, and it is the
    only input here that the command does not supply itself.

    `prose_stream` is the caller asking to be shown the prose as it is written. It is
    honoured only for a command whose prose is pre-validated
    (`routing.Command.streams_prose`), and it defaults off: a caller that does not
    ask receives exactly the frames it received before streaming existed.

    `admission_context` is the pre-flight manifest the request already carries. It is the
    same string the free-text loop appends to its system prompt, and it reaches the
    formatting turn here because a command is graded against it: an answer that named a
    facility from an earlier encounter instead of the discharge disposition in the block
    was scored as a fidelity failure twice in one run, on identical prose, and no prompt
    on this path had ever been shown the block (`dev_t1_02`, `dev_t2_01`, 2026-11-05).
    """

    arguments = tool_arguments(command, hadm_id, topic)
    _emit(on_event, stages.planning_event(command.wire))

    # The calls, named by the table rather than chosen by a model: the command's own
    # tool, then any the command declares alongside it. `/ehr` is the only command with
    # a second, and it takes one so a single turn can put the readmission score beside
    # the narrative it qualifies.
    calls = [{"name": command.tool, "args": arguments, "id": f"{command.wire}-1"}]
    supplementary = supplementary_arguments(command, hadm_id, topic)
    calls += [
        {"name": name, "args": extra, "id": f"{command.wire}-{index}"}
        for index, (name, extra) in enumerate(supplementary, start=2)
    ]
    # An outpatient note has no readmission to estimate, so the estimate was not
    # requested. Say so in the formatter's input: without it the model writes the
    # fourth section from the notes, which is how a speech-therapy progress note came
    # back with a "Readmission risk & trajectory" paragraph (2026-10-17).
    outpatient = (
        not encounter.states_inpatient_fields()
        and any(spec.tool == "predict_readmission" for spec in command.supplementary)
    )
    # Every passage reaches the formatting turn and the citation ledger in the order the
    # retrieval tool returned it, which for `/ehr` is the note's own document order. There
    # used to be a re-ordering hook here that put a summary's passages into the canonical
    # four-section order, because the tool returned them scored rather than read. The tool
    # now returns the note's sections as the note states them, so a hook that sorted them
    # into a second order is a second opinion about how a chart reads (2026-11-03).
    #
    # The session's evidence, hydrated from the replay BEFORE a single tool of this
    # command runs. It is the numbering authority for the whole turn: `_execute_tool_calls`
    # ingests each payload into it and stamps the positions it assigned, so a passage the
    # thread already holds keeps its position instead of taking a new one.
    #
    # `prior_session_block` is captured now, not at the formatter, because by then the
    # record also holds this turn's own units — and those are already in `tool_messages`,
    # numbered. The formatter needs the evidence THIS turn has not shown it.
    evidence_state = (
        session_evidence.SessionEvidenceState.hydrate_from_calls(
            hadm_id, [_presented_call(call) for call in (replayed_calls or [])])
        if hadm_id is not None else None
    )
    prior_session_block = ""
    if evidence_state is not None and (
        evidence_state.position_count()
        or evidence_state.checked_sections
        or evidence_state.prediction
        or evidence_state.literature
    ):
        prior_session_block = evidence_state.render_for_prompt()
    # Where this turn's own units begin, for `turn.compose_success`: the stored
    # `citation_map` is measured from it, and a later replay adds it back to put the
    # markers on the positions the record gives that evidence.
    replay_positions = evidence_state.position_count() if evidence_state is not None else 0

    tool_messages, recorded = await _execute_tool_calls(
        toolbox, calls, on_event, project=_presented_for_model,
        # This path opens `command.<wire>` around the whole turn, so its tool spans
        # have a parent to nest under. The reasoning loop does not, and spanning there
        # produced a separate root trace per tool.
        traced=True,
        order=None,
        cite_offset=(evidence_state.position_count() if evidence_state is not None
                     else 0),
        evidence_state=evidence_state,
    )

    # `/ehr` asks for four named discharge-summary sections. An outpatient note has
    # none of them, so the sectioned call succeeds and returns nothing — and the
    # formatter, told only that, reported the RECORD as empty ("No information was
    # found in the provided record to summarize", 90000051, 2026-10-21). One
    # record-wide search settles which of the two it is: passages means there was
    # something to summarize, and nothing means the absence is real and sayable.
    if command.wire == "summarize" and hadm_id is not None and not _passage_count(recorded):
        fallback = [{
            "name": "rag_search",
            "args": {"hadm_id": hadm_id, "query": EHR_RECORD_QUERY},
            "id": f"{command.wire}-record",
        }]
        extra_messages, extra_recorded = await _execute_tool_calls(
            toolbox, fallback, on_event, project=_presented_for_model, traced=True,
            order=None,
            # The record already holds whatever the numbered call registered, so its
            # length is where the fallback's units begin.
            cite_offset=(evidence_state.position_count()
                         if evidence_state is not None else _passage_count(recorded)),
            evidence_state=evidence_state,
        )
        tool_messages += extra_messages
        recorded += extra_recorded

    # Whether this record is a stay at all. Broader than `outpatient` above, which only
    # asks whether the risk estimate was withheld: every command on an outpatient record
    # owes the reader the same statement about what the record is, not only the one that
    # would have drawn a risk card.
    outpatient_record = (
        hadm_id is not None and encounter.current_is_outpatient() is True
    )

    def _session_evidence() -> Any:
        """This turn's evidence record, or `None` when there is no admission to scope it to.

        The object built before the first tool ran, now holding everything the command
        executed: `_execute_tool_calls` ingests into it rather than into a copy, so the
        three exits below each carry the same record and the numbering they hand the
        caller is the numbering the renderer stamped.
        """
        return evidence_state

    # `/ehr` on an outpatient record. The four headings describe a stay, and this record
    # is a visit, so the command's answer is the encounter notice itself — the sentence
    # stating what the record is and what it therefore cannot hold.
    #
    # It is decided here, in code, and not by the formatter. The instruction used to tell
    # the model to write a renamed heading instead (`EHR_FORMATTING`: "write **Clinical
    # course.** instead and omit the other three sections"), which the rubric calls a
    # responsiveness defect — a command's answer shape is not the model's to renegotiate,
    # and an item scored 1 for obeying its prompt is the clearest evidence of that
    # (`dev_t7_01`, 2026-11-05). Gated on the RECORD, not on whether the record-wide
    # search found text: a single speech-therapy passage is exactly what produced the
    # summary this branch exists to replace.
    #
    # The tool calls stay on the state. They are what the turn retrieved, the trace row
    # and the drawer show them, and a turn whose evidence vanished when its answer became
    # deterministic would be a turn an audit cannot follow.
    if command.wire == "summarize" and outpatient_record:
        return {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=question),
                *tool_messages,
            ],
            "tool_calls": recorded,
            "replayed_tool_calls": replayed_calls or [],
            "command": command.wire,
            "deterministic_answer": answer_text.outpatient_ehr_notice(),
            # The manifest answered this turn. `encounter: outpatient` is the fact the
            # notice states, and it came from the pre-flight block rather than from a
            # passage — so the composer has to be told, or it reads a citation-free
            # chart answer as synthesis nobody can vouch for and withholds `verified`
            # from the one turn whose answer IS the record's type (`dev_t7_01`).
            "manifest_answered": True,
            **
            {"hadm_id": hadm_id, "evidence_state": _session_evidence(),
             "replay_positions": replay_positions},
        }

    # `/meds` cites with one marker, so the passages it cites have to be one passage.
    # Done before the formatting turn's state is built, because everything downstream —
    # the answer's citation, the guardrail's source text and the grid — reads `recorded`.
    if command.wire == "meds":
        _consolidate_medication_passages(recorded)

    # The published cohorts this admission's estimate is compared against, retrieved live
    # and recorded as tool calls like every other piece of evidence this turn rests on.
    command_extras: dict[str, Any] = {}
    if command.wire == "summarize" and not outpatient_record:
        comparators, target_payload, literature_recorded, passes = \
            await _readmission_comparators(
                toolbox, model, recorded, on_event, extraction_llm=extraction_llm,
                evidence_state=evidence_state)
        recorded += literature_recorded
        command_extras["comparators"] = comparators
        command_extras["clinical_target"] = target_payload
        command_extras["comparator_passes"] = passes

    # Every result this turn produced goes to the formatting turn, supplementary calls
    # included.
    #
    # This reverses the withholding that stood here earlier the same day, and the reason
    # is what the withholding actually bought. It did not stop the model writing about
    # risk; it stopped the model *explaining* risk. The reader got a chart summary and a
    # score side by side with nothing joining them — two artifacts stacked, which is what
    # a reader notices first. What replaces it is a division of labour rather than a
    # shared number: the widget owns the quantitative score and the driver bars, and the
    # narrative's fourth section owns how those drivers relate to this patient's course.
    # `routing.OUTPATIENT_EHR_NOTICE` is the sentence served for an outpatient record
    # with no readable content; it lives in `postprocess.answer` now (`00d7` Row 4).
    # records the rule, so the three cannot drift apart quietly.
    _emit(on_event, stages.reviewing_event())

    if not command.prose:
        # No formatting turn, and therefore no LLM call at all: this command's answer
        # IS its surface. Everything downstream — the guardrails, the citation layer,
        # the envelope composer — reads the same state shape it always did, so the
        # command stays subject to the same checks as any other; the difference is
        # that there is no prose for them to inspect.
        return {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=question),
                *tool_messages,
            ],
            "tool_calls": recorded,
            "replayed_tool_calls": replayed_calls or [],
            "command": command.wire,
            # Empty, and declared as such: `_compose_success` treats an empty answer
            # as a failed turn unless the state says the surface carries it.
            "deterministic_answer": "",
            "answer_in_surface": True,
            **{"hadm_id": hadm_id, "evidence_state": _session_evidence(),
               "replay_positions": replay_positions},
            **command_extras,
        }

    # The manifest, for the one command whose prose is graded against it. `/ehr` is the
    # briefing: it states where the patient went and on what regimen, and both of those are
    # administrative facts the block holds and the notes only sometimes state. Rendered
    # here rather than passed in pre-rendered so that the block and the free-text path's
    # block are one function's output — the outpatient notice included.
    context_block = (
        _context_block(admission_context, hadm_id)
        if command.wire == "summarize" and not outpatient_record
        else ""
    )

    llm = model_gateway.build_llm(model)  # no bind_tools: a formatting turn cannot call anything
    config = {"callbacks": list(callbacks)} if callbacks else {}
    if trace_metadata:
        config["metadata"] = dict(trace_metadata)
    # Streaming is a capability the CALLER asks for (`routing.Command.streams_prose`).
    # A caller that has not asked gets exactly the frames it got before streaming
    # existed — which matters while a canvas that renders stages would print an
    # announcement as a progress row.
    streams = bool(prose_stream and command.streams_prose and on_event is not None)

    def _announce(piece: str) -> None:
        _emit(on_event, stages.answer_delta_event(piece))

    # Two prose sources, and the turn streams for one of them.
    #
    # `/meds`'s model reply is the extraction JSON — announcing it would put
    # `{"medications": …}` on a clinician's screen — so its answer is the narrative
    # written in code from verified rows and this turn is not asked to stream.
    #
    # `/ehr` and `/pubmed` are the other way round: the model's reply IS the answer, so
    # the turn streams and each chunk is announced on arrival.
    #
    # Which instruction the turn receives is decided BEFORE it runs, and for `/meds` that
    # decision is the record's: a dedicated medication section is asked for the regimen to
    # continue at home, a fallback narrative is asked what those sections name. Asking the
    # discharge question of a record with no discharge list discarded most of the evidence
    # and left the narrative counting what survived (2026-11-04).
    sections = _medication_sections(recorded)
    instruction = formatting_for(command.wire)
    if command.wire == "meds":
        instruction = meds_formatting_for(sections)
    response = await _formatting_turn(
        llm,
        [
            SystemMessage(content=f"{FORMATTER_PREAMBLE}\n\n{instruction}"),
            HumanMessage(content=_formatter_input(
                tool_messages, outpatient, question,
                command_extras.get("comparators"),
                # Only `/ehr` is shown the manifest. The command whose prose is graded
                # against the block is the one that has to read it, and giving `/meds`
                # or `/labs` an administrative header they have no use for is evidence
                # in the window that no sentence may rest on.
                context_block=context_block,
                # What the thread held before this command ran, so a follow-up command
                # can answer from evidence an earlier turn retrieved instead of reporting
                # the record as undocumented.
                session_block=prior_session_block,
            )),
        ],
        config=config,
        announce=_announce if streams and command.prose_is_model_output else None,
    )
    text = response.text() if hasattr(response, "text") else str(response.content)

    # `/labs` publishes no citations, so it carries no markers (2026-11-03). Its evidence
    # is the trajectory table drawn under the sentence — every reading with its own
    # reference interval and flag — so a pill in the prose opens a card restating the row
    # beside it. The instruction asks for none; this removes any the model wrote anyway,
    # because a marker with no citation behind it renders as a dead badge.
    if command.wire == "labs":
        text = citations.strip_markers(text)

    state: dict[str, Any] = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=question),
            *tool_messages,
            response,
        ],
        "tool_calls": recorded,
        "replayed_tool_calls": replayed_calls or [],
        # Which command this was. The payload composer reads it to decide what
        # the answer is drawn as, and it is the only marker that distinguishes a
        # command's state from a free-text one downstream.
        "command": command.wire,
        **{"hadm_id": hadm_id, "evidence_state": _session_evidence(),
           "replay_positions": replay_positions},
        **command_extras,
    }

    if command.prose and not str(text).strip():
        # A formatting turn that finished with nothing in it. The sentence is put on the
        # state here rather than left to the envelope's empty-answer branch, because the
        # trace's output is read off this state (`graph.command_trace_output`) and an
        # empty answer wrote the turn's trace row as a dash — the one record an audit
        # opens first (TF-6, 2026-10-23). The wording is the same one the envelope
        # would have served, so the answer itself is unchanged.
        #
        # Empty has two causes and they are not the same statement. A model that finished
        # cleanly and said nothing has no answer; a model whose reply was withheld by a
        # provider filter never got to say one, and the records it retrieved are sitting
        # under the sentence. `/pubmed Hypothermia` on 2026-10-30 was the second case and
        # was served as the first: three PMIDs in the drawer beneath "I don't have an
        # answer for that", with `finish_reason=SAFETY` and
        # `HARM_CATEGORY_DANGEROUS_CONTENT` in the trace the whole time.
        withheld = answer_text.reply_withheld(
            model_turn.finish_reason(response),
            blocked=bool(model_turn.blocked_reason(response)),
        )
        records = _retrieved_records(recorded)
        if withheld and records:
            logger.warning(
                "command.%s: the formatting reply was withheld (%s%s); serving the %d "
                "retrieved record(s) without a summary",
                command.wire,
                model_turn.finish_reason(response) or "unreported",
                "".join(f", {category}"
                        for category in model_turn.filtered_categories(response) or []),
                len(records),
            )
        # Which sentence that is, and why, is `postprocess.answer`'s decision: this route
        # knows the facts of the turn and nothing about the wording or the reader's
        # situation. It is invoked rather than selected (Charter 13 Law 2).
        state["deterministic_answer"] = answer_text.empty_turn_sentence(
            finish_reason=model_turn.finish_reason(response),
            blocked=bool(model_turn.blocked_reason(response)),
            records_retrieved=len(records),
            outpatient=outpatient_record,
        )

    if command.wire == "meds":
        source = _source_text(recorded)
        verified, dropped = parse_medications(text, source)
        returned = _passage_count(recorded)
        # `sections` was read above, where it chose the instruction this turn was given;
        # the narrative reports the same sections rather than re-reading them.
        state["command_payload"] = {
            "medications": verified,
            "dropped": dropped,
            "returned_passages": returned,
            "sections": list(sections),
        }
        # The narrative replaces the raw JSON: the grid renders the list, and the
        # prose is what the guardrails and the citation layer expect to see.
        state["deterministic_answer"] = answer_text.medication_narrative(
            verified, returned, sections=sections)
        if streams:
            # Generated here from verified rows, so announcing it is not a preview of
            # unverified text: this IS the settled answer, arriving as soon as it
            # exists rather than after the envelope is composed.
            _announce(state["deterministic_answer"])

    return state


# ---------------------------------------------------------------------------
# Live literature comparators
# ---------------------------------------------------------------------------
#
# The comparative table on `/ehr` sets this patient's 30-day estimate against published
# cohorts. Until now those cohorts came from a 9-row static dictionary in `presentation/`,
# selected by a keyword classifier — a curated table that was honest about being curated
# and still, structurally, a table that could disagree with the patient in front of it.
#
# What follows is the replacement, and it holds to three rules that are the reason it lives
# here rather than in the presentation layer:
#
#   * nothing about a condition, a study, a rate or a PMID is written down anywhere in this
#     module. The condition comes from the chart, in this request; the studies come from
#     PubMed, in this request; the rate comes from the abstract that was retrieved.
#   * the model PROPOSES and this module DISPOSES. Every field that can be checked is
#     checked against the retrieved record — the sentence verbatim, the figure inside that
#     sentence, the horizon, the PMID — and a row that fails any check is dropped rather
#     than corrected.
#   * a cohort for another condition is not a cohort. The alignment gate is deterministic
#     and runs on the patient's own vocabulary: an abstract that does not name this
#     patient's condition is not this patient's comparison, whatever the model thought.
#
# The rule the table used to break is the third one. A knee arthroplasty cohort beside a
# hip fracture patient reads as a comparison and is not one, and no amount of downstream
# checking could tell it from a correct one because both are well-formed rows.

#: The sections a target is derived from, in the order the derivation reads them: what the
#: admission WAS, then what happened during it. Canonical names from
#: `services.mcp.retrieval.sections`, named here because the order is a reading decision
#: rather than the retrieval contract's.
_TARGET_SECTION_ORDER = ("discharge_diagnosis", "brief_hospital_course")

#: An anchor shorter than this is an organ or a limb, not a condition. "hip" and "knee"
#: both appear in a paper about the other one; "COPD" and "sepsis" do not need the length
#: rule to be safe, but it costs nothing to apply it uniformly.
_MIN_ANCHOR_CHARS = 4

#: Words that name a region, a symptom, a setting or a generic operation class rather than
#: a condition. A single word from this set cannot identify a cohort — "hip" is in a paper
#: about knee surgery and "arthroplasty" is in a paper about shoulders — so it is refused
#: as an anchor on its own. Multi-word anchors are never checked against this set: "hip
#: fracture" and "cervical fusion" are specific, and the phrase is what makes them so.
#:
#: This is a structural list, not a clinical one: it holds no disease, no procedure, no
#: rate and no study, and it cannot admit or refuse a cohort by itself. It only says which
#: single words are too general to carry an alignment decision alone.
_AMBIGUOUS_ANCHOR_WORDS = frozenset({
    "hip", "knee", "spine", "back", "pain", "joint", "bone", "wound",
    "surgery", "care", "stay", "cohort", "adult", "adults", "elderly",
    "patient", "patients", "hospital", "arthroplasty", "replacement",
    "fusion", "decompression", "fixation", "repair", "resection",
})

#: How many anchors may reach the query. Eight or so named synonyms is already a wide OR;
#: past that the term list starts matching on incidental vocabulary rather than on the
#: condition.
MAX_ANCHORS = 8

#: How many verified rows make a pass good enough not to need the second one. Two, because
#: one record is not a comparison — and the second pass costs two E-utilities calls and a
#: model call, which is worth spending on a turn that has nothing to show.
MIN_COMPARATORS = 2

#: The horizon the comparison is made at. Stated as a constant so the check below reads as
#: the rule it is: a five-year rate beside a 30-day estimate is two unrelated numbers in
#: adjacent cells.
_COMPARATOR_HORIZON_DAYS = 30

_THIRTY_DAY = re.compile(
    r"\b30[-\s]?day\b|\bthirty[-\s]day\b|\bwithin 30 days\b", re.IGNORECASE
)
#: A horizon that is NOT the one being compared. A record titled "predictors of
#: thirty-day readmission" can still quote a sentence about its 90-day figure, and the
#: quote is what the row publishes, so a conflicting horizon inside the quoted sentence is
#: a refusal rather than a pass.
_OTHER_HORIZON = re.compile(
    r"\b(?:60|90|180|365)[-\s]?day\b"
    r"|\b(?:six|nine|twelve)[-\s]month\b"
    r"|\b(?:one|two|three|five|1|2|3|5)[-\s]year\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_PERCENT_IN_TEXT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
#: A rate that is stated as a span. Kept as one verified row rather than split into two,
#: because the abstract states it as one figure and a row is a quotation — but the surface
#: has to know, since a cell holding "1.0% to 1.4%" is not the same object as one holding
#: "8.4%" (`rate_kind`).
_RANGE_RATE = re.compile(r"\d\s*%?\s*(?:to|through|\u2013|\u2014|-)\s*\d", re.IGNORECASE)


def _horizon_context(article: dict, sentence: str) -> str:
    """The record's own statement of its horizon: its title, and the text just before the
    quoted sentence.

    A paper names its horizon where a paper names its subject — "Predictors of thirty-day
    readmission after anterior cervical fusion" — and then reports the number in a
    sentence that says only "the readmission rate was 4.1%". Requiring the horizon inside
    the quote rejected exactly that record on 2026-11-02 (PMID 24108298), and two more
    like it.
    """
    abstract = " ".join(str(article.get("abstract") or "").split())
    index = abstract.find(sentence)
    window = abstract[max(0, index - 240):index] if index > 0 else ""
    return f"{article.get('title') or ''} {window}".strip()


def rate_kind(reported_rate: Any) -> str:
    """Whether a verified rate is one figure, a span, or a two-arm contrast.

    A range is a legitimate quotation — the Spinal 2017 cohort reports "1.0% to 1.4%" and
    a reader is owed exactly that — so it is kept. What it must not be is indistinguishable
    from a point estimate downstream, where a comparison against one predicted probability
    is drawn.
    """
    return normalise_rate(reported_rate)[1]


#: A two-arm result: "7.8% vs. 11.5%", "15% and 12%". The connector is what makes it a
#: contrast rather than two findings, which is why the model's "and" has to be repaired
#: before publication.
_COMPARISON_RATE = re.compile(
    r"\d\s*%\s*(?:vs\.?|versus|and)\s*\d", re.IGNORECASE
)


def normalise_rate(reported_rate: Any) -> tuple[str, str]:
    """The rate as it should be published, and its kind: point, range, or comparison.

    Three shapes, and only two were recognised before 2026-11-02. The third is the two-arm
    result, and it was being published as a single figure: `37120515` contrasted two care
    models (7.8% vs. 11.5%) and the row said "7.8% and 11.5%", which reads as two
    unrelated findings rather than one contrast, and which the surface would draw as a
    point estimate beside one predicted probability.

    So the connector is normalised to the abstract's own word, "vs.", and the kind is
    reported with it. A range is left byte-for-byte as the abstract states it: "1.0% to
    1.4%" is a quotation and no rewriting is owed.
    """
    text = " ".join(str(reported_rate or "").split())
    figures = _NUMBER.findall(text)
    if len(figures) >= 2 and _COMPARISON_RATE.search(text):
        return f"{figures[0]}% vs. {figures[1]}%", "comparison"
    if _RANGE_RATE.search(text):
        return text, "range"
    return text, "point"


#: Words that mark a cohort as a different POPULATION rather than a narrower band of the
#: same one. Age bands are benchmarks — octogenarians are the same operation in an older
#: body; children are a different patient.
#:
#: Named as vocabulary because a population cannot be recognised any other way, and kept
#: strictly to population words: no disease, no procedure, no rate, so it can never admit
#: or refuse a cohort on clinical grounds. Measured 2026-11-02: a paediatric empyema cohort
#: passed every check and was served as an adult's comparison.
_PEDIATRIC_MARKERS = re.compile(
    r"\b(?:pediatric|paediatric|children|child|infant|infants|neonatal|neonate|"
    r"adolescent|adolescents|juvenile|under 18|< ?18)\b",
    re.IGNORECASE,
)


def names_pediatric_population(*parts: Any) -> bool:
    """Whether any of these texts describes a cohort of children."""
    return any(_PEDIATRIC_MARKERS.search(f" {str(part or '')} ") for part in parts)


def is_pediatric_target(target: ClinicalTarget) -> bool:
    """Whether this admission says of itself that it is a child's.

    Read from the target's own words, which come from the chart. A paediatric hospital
    carries pediatric admissions and this guard turns itself off for them without a
    setting; an adult service leaves it on.
    """
    return names_pediatric_population(target.condition, target.procedure, *target.anchors)


def adult_cohorts(articles: list[dict] | None,
                  target: ClinicalTarget) -> tuple[list[dict], list[dict]]:
    """The pool without cohorts of a population this admission is not in.

    The one population rule this layer keeps. It is structural rather than clinical — an
    abstract naming children is a cohort of children — and it is applied to the whole pool
    before ranking so a paediatric cohort cannot take a slot from an adult one.

    What used to sit beside it here, and no longer does, was a gate that refused any record
    whose abstract did not repeat one of the patient's anchor phrases. That is a token test
    pretending to be a clinical one: an orthopaedic paper that writes "arthroplasty" where
    the chart wrote "replacement" is the right cohort and was refused, and the refusal was
    invisible to the reader because it happened before the model ever saw the abstract.
    Which cohort describes this patient is a judgement, and the model that reads the
    abstract is the layer entitled to make it.
    """
    pool = [article for article in articles or [] if isinstance(article, dict)]
    if is_pediatric_target(target):
        return pool, []
    kept: list[dict] = []
    refused: list[dict] = []
    for article in pool:
        if names_pediatric_population(
            article.get("title"), article.get("journal"), article.get("abstract")
        ):
            refused.append({
                "pmid": str(article.get("pmid") or "").strip(),
                "reason": "cohort is a paediatric population and this admission is not",
            })
            continue
        kept.append(article)
    return kept, refused



@dataclass(frozen=True)
class ClinicalTarget:
    """What this admission was, in the terms a literature search can be run with.

    Built from the chart and nothing else. `anchors` are the phrases an abstract has to
    contain for its cohort to be this patient's cohort, and they are the deterministic
    half of the alignment gate: the model's `clinical_match` is a judgement, and this is
    the part that can be re-read a year later and still mean the same thing.
    """

    condition: str
    procedure: str = ""
    anchors: tuple[str, ...] = ()

    @property
    def search_terms(self) -> str:
        """The clinical half of the E-utilities query: condition, then procedure."""
        return " ".join(part for part in (self.condition, self.procedure) if part)


def _normalise_text(value: Any) -> str:
    return " ".join(str(value or "").split()).lower()


def _usable_anchor(phrase: Any) -> str:
    """One candidate anchor, or empty when it could not identify a condition.

    Synonyms are the point of this function, so an anchor does NOT have to be a phrase the
    chart used — the chart says "uterine leiomyoma" and the papers say "fibroids", and a
    gate that only knew the chart's words refused the five records that mattered on
    2026-11-02. What it does have to be is specific: a single word drawn from
    `_AMBIGUOUS_ANCHOR_WORDS` names a region, a symptom or an operation class, and would
    admit a cohort for a different condition on the same limb.
    """
    text = _normalise_text(phrase).strip(" .,;:")
    if len(text) < _MIN_ANCHOR_CHARS:
        return ""
    if " " not in text and text in _AMBIGUOUS_ANCHOR_WORDS:
        return ""
    return text


def build_target(payload: dict | None) -> ClinicalTarget | None:
    """The `ClinicalTarget` a reply describes, or None when it names no condition.

    None is a real answer: a record whose note names no condition has no search to run,
    and running one anyway would be searching for the patient's demographics. The caller
    draws no comparison, which is what "we have nothing to compare this against" looks
    like when it is honest.
    """
    if not isinstance(payload, dict):
        return None
    condition = " ".join(str(payload.get("condition") or "").split())
    procedure = " ".join(str(payload.get("procedure") or "").split())
    if not condition:
        return None
    proposed = payload.get("anchor_terms")
    if not isinstance(proposed, list):
        proposed = []
    anchors: list[str] = []
    for candidate in (*(proposed or []), condition, procedure):
        anchor = _usable_anchor(candidate)
        if anchor and anchor not in anchors:
            anchors.append(anchor)
        if len(anchors) >= MAX_ANCHORS:
            break
    if not anchors:
        return None
    return ClinicalTarget(condition=condition, procedure=procedure, anchors=tuple(anchors))


def note_sections(recorded: list[RecordedToolCall] | None) -> dict[str, str]:
    """The note sections this turn's retrieval returned, in reading order.

    Concatenated per section rather than replaced: `rag_search_sections` can return more
    than one passage for a section, and the last one is not the section. Ordering puts the
    diagnosis first when it is there, because that is what the admission WAS.
    """
    collected: dict[str, str] = {}
    for call in recorded or []:
        if not isinstance(call, dict):
            continue
        payload = call.get("response")
        for passage in (payload or {}).get("passages") or []:
            if not isinstance(passage, dict):
                continue
            name = str(passage.get("section") or "").strip()
            text = str(passage.get("text") or "").strip()
            if not name or not text:
                continue
            collected[name] = f"{collected[name]} {text}".strip() if name in collected else text
    ordered: dict[str, str] = {}
    for name in _TARGET_SECTION_ORDER:
        if name in collected:
            ordered[name] = collected.pop(name)
    ordered.update(collected)
    return ordered


#: Why a structured reply produced nothing. Reported rather than collapsed into "no
#: answer", because the three call for opposite responses: a withheld reply is the
#: provider's filter acting on the pair (prompt, record), a non-JSON reply is the model
#: ignoring the format, and an empty condition is a fact about the chart.
#:
#: Measured 2026-11-02: the elaborated target prompt returned `finish_reason=SAFETY` with
#: `HARM_CATEGORY_DANGEROUS_CONTENT` on `90000090`, reproducibly across three calls, and the
#: report said "the chart named no condition" — the exact class of silently-wrong
#: measurement this whole step exists to remove.
REPLY_WITHHELD = "withheld"
REPLY_NOT_JSON = "not_json"


async def _structured_reply(
    llm, system: str, human: str, config: dict | None,
) -> tuple[dict | None, str]:
    """One model reply as (JSON object, "") or (None, why it produced nothing).

    The reply is asked for as JSON because everything downstream of it is checked rather
    than read: a condition becomes a query string, and a row becomes a claim that has to
    match the abstract it came from. Prose in either place would be unfalsifiable.
    """
    response = await llm.ainvoke(
        [SystemMessage(content=system), HumanMessage(content=human)],
        config=config or {},
    )
    if answer_text.reply_withheld(
        model_turn.finish_reason(response),
        blocked=bool(model_turn.blocked_reason(response)),
    ):
        logger.warning(
            "literature extraction: the reply was withheld (%s%s)",
            model_turn.finish_reason(response) or "unreported",
            "".join(f", {category}"
                    for category in model_turn.filtered_categories(response) or []),
        )
        return None, REPLY_WITHHELD
    content = getattr(response, "content", response)
    payload = _extract_json_object(text_of(content))
    return (payload, "") if payload is not None else (None, REPLY_NOT_JSON)


#: Why a structured reply produced nothing, in the words a report can print.
_REPLY_PROBLEM = {
    REPLY_WITHHELD: "was withheld by the model's safety filter",
    REPLY_NOT_JSON: "was not a JSON object",
}


def reply_problem(reason: str) -> str:
    """The sentence form of a `_structured_reply` reason."""
    return _REPLY_PROBLEM.get(reason, "produced no reply")


@dataclass(frozen=True)
class TargetResult:
    """The target, where it came from, and the reason there is none.

    `reason` is empty when a target was produced, and otherwise names the failure rather
    than the absence. `source` says which mechanism produced it, because the three are not
    equivalent evidence: a model reading the chart, the same model asked more plainly, and
    a parser reading the chart's own headers.
    """

    target: ClinicalTarget | None
    reason: str = ""
    source: str = "model"
    #: "model", "model_short_prompt", or "chart_headers".


#: A line that opens a block of a discharge summary — "OPERATIONS:", "DISCHARGE
#: DIAGNOSIS:" — as opposed to a sentence. Named as a shape rather than a vocabulary: a
#: run of capitals and separators followed by a colon.
_SECTION_HEADER = re.compile(r"^\s*([A-Z][A-Z0-9 ,/&()'.-]{2,}):\s*(.*)$")

#: What a note puts in front of a diagnosis or an operation: "1.", "2)", "-", "*", a
#: bullet.
_ENUMERATION = re.compile(r"^\s*(?:\d+[.)]|[-*\u2022\u00b7])\s+")

#: A date a note puts in front of an operation: "July 25, 2006: ".
_LEADING_DATE = re.compile(
    r"^\s*(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{1,2},\s+\d{4}\s*:\s*",
    re.IGNORECASE,
)

#: A candidate that opens with one of these is a sentence about the patient rather than a
#: name for a condition. English function words, not clinical ones.
_SENTENCE_STARTS = frozenset({
    "the", "a", "an", "she", "he", "they", "this", "that", "there",
    "her", "his", "we", "it", "patient", "patient's",
})


def _first_clause(line: Any) -> str:
    """The leading noun phrase of one line: the first sentence, up to its first separator."""
    text = _LEADING_DATE.sub("", _ENUMERATION.sub("", str(line or "").strip()))
    sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
    for separator in (",", ";", "("):
        index = sentence.find(separator)
        if index > 0:
            sentence = sentence[:index]
    return " ".join(sentence.split()).strip(" .:-").strip()


def _reads_as_condition(phrase: str) -> bool:
    words = phrase.split()
    if not words or len(phrase) < 4:
        return False
    if words[0].lower() in _SENTENCE_STARTS:
        return False
    return any(character.isalpha() for character in phrase)


def fallback_target_from_sections(sections: dict | None) -> ClinicalTarget | None:
    """The condition a chart names, read off its own headers, with no model involved.

    The last resort of `derive_target`, and the reason it exists: a discharge note is a
    document with structure, and a content filter acting on a model's reply must not be
    able to turn "this admission was for X" into "this chart names no condition". Measured
    2026-11-02, on two admissions whose notes name the diagnosis in their first line.

    Nothing here knows any condition. It reads the shape every discharge summary has — a
    diagnosis block whose first entry is the admission's problem, or an all-caps block
    header naming an operation — strips the numbering, the leading date and the trailing
    comorbidities, and takes what is left. It is deliberately dumb: it produces a search
    phrase, and every check downstream still has to agree with it before a row is served.
    """
    blocks: list[str] = []
    # Both diagnosis blocks, because a note may carry only one of them. Until 2026-11-03
    # the admission and discharge diagnoses shared a single key, so reading one read the
    # other; now that they are separate sections, a note documenting only the diagnosis it
    # admitted has to be read here or this last resort would find no condition in a note
    # that states one in its first line.
    for key in ("discharge_diagnosis", ADMISSION_DIAGNOSIS):
        blocks.extend(str((sections or {}).get(key) or "").splitlines())
    for line in str((sections or {}).get("brief_hospital_course") or "").splitlines():
        header = _SECTION_HEADER.match(line)
        if header:
            blocks.append(header.group(2) or "")
    for block in blocks:
        phrase = _first_clause(block)
        if not _reads_as_condition(phrase):
            continue
        anchor = _usable_anchor(phrase)
        if anchor:
            return ClinicalTarget(condition=phrase, procedure="", anchors=(anchor,))
    return None


async def derive_target(llm, sections: dict | None, *,
                        config: dict | None = None) -> TargetResult:
    """The admission's primary condition and index procedure, read from its own note.

    Three attempts, in descending order of what they can understand and ascending order of
    what they depend on:

      1. the elaborated instruction;
      2. the same rules asked more plainly, for the case where the provider's content
         filter withholds the first (measured 2026-11-02);
      3. `fallback_target_from_sections`, which reads the chart's own headers and needs no
         model at all.

    The third is not a degradation of quality but of ambition: it names the first
    diagnosis the note states and searches for that, where the model would have chosen the
    principal one. A chart that reaches it still gets a comparison attempt, and every check
    downstream still has to agree with the phrase it produced.
    """
    payload, why = await _structured_reply(
        llm,
        prompts.LITERATURE_TARGET_PROMPT,
        prompts.literature_target_input(sections),
        config,
    )
    if payload is not None:
        target = build_target(payload)
        if target is not None:
            return TargetResult(target=target)
        why = "the reply named no condition"

    retry, retry_why = await _structured_reply(
        llm,
        prompts.LITERATURE_TARGET_FALLBACK_PROMPT,
        prompts.literature_target_input(sections),
        config,
    )
    if retry is not None:
        target = build_target(retry)
        if target is not None:
            return TargetResult(target=target, source="model_short_prompt")

    salvaged = fallback_target_from_sections(sections)
    if salvaged is not None:
        return TargetResult(
            target=salvaged,
            reason=(
                "the model produced no condition "
                f"(first reply {reply_problem(why)}; retry {reply_problem(retry_why)}); "
                "the condition was read from the chart's own headers"
            ),
            source="chart_headers",
        )
    return TargetResult(
        None,
        f"first reply {reply_problem(why)}; retry {reply_problem(retry_why)}; "
        "and the chart states no condition header",
    )


def verify_comparator(row: dict | None, articles: dict[str, dict],
                      target: ClinicalTarget) -> tuple[dict | None, str]:
    """One proposed row, verified against the record it claims to come from.

    Returns (row, "") when every check passes, or (None, reason) when any fails. The
    published fields — journal, date, title, authors, url — are taken from the RECORD
    rather than from the model's row, so a citation line cannot be assembled from a
    plausible-looking string the abstract never supported.

    The checks, and each one has caught something:

      * the PMID is one this turn retrieved;
      * the model did not itself decline the match;
      * the cohort is a population this admission is not locked out of — children;
      * the horizon is 30 days, and the quoted sentence says so;
      * `rate_sentence` appears in the abstract character for character;
      * `reported_rate` is the figure that sentence states.

    What is NOT checked here any more is whether the abstract names one of the patient's
    anchor phrases, and whether its operation was as major as this admission's. Both were
    token tests over clinical prose, and both refused correct cohorts for saying the same
    thing in other words: an orthopaedic cohort that wrote "arthroplasty" where the target
    wrote "replacement", a spine cohort whose operation was named by approach rather than
    by procedure. The question "is this the right cohort for this patient" is a clinical
    judgement, it belongs to the model that reads the abstract, and the parts of it this
    layer CAN check without reading are the ones left: a paediatric population is a
    population this admission is not in, a horizon is a number, and a sentence either
    appears in its abstract or it does not.
    """
    if not isinstance(row, dict):
        return None, "row is not an object"
    pmid = str(row.get("pmid") or "").strip()
    article = articles.get(pmid)
    if article is None:
        return None, f"pmid {pmid or '(none)'} is not one this search retrieved"
    if not is_pediatric_target(target) and names_pediatric_population(
        article.get("title"), article.get("journal"),
        row.get("cohort_summary"), article.get("abstract"),
    ):
        return None, f"pmid {pmid} is a cohort of children and this admission is not"
    if row.get("clinical_match") is False:
        declined = " ".join(str(row.get("rationale") or "").split())[:160]
        return None, f"pmid {pmid} was declined: {declined or 'no rationale given'}"
    try:
        horizon = int(row.get("horizon_days"))
    except (TypeError, ValueError):
        return None, f"pmid {pmid} states no horizon"
    if horizon != _COMPARATOR_HORIZON_DAYS:
        return None, f"pmid {pmid} horizon is {horizon} days, not 30"

    abstract = " ".join(str(article.get("abstract") or "").split())
    sentence = " ".join(str(row.get("rate_sentence") or "").split())
    if not sentence:
        return None, f"pmid {pmid} quotes no sentence"
    if sentence not in abstract:
        return None, f"pmid {pmid} rate_sentence is not verbatim from the abstract"
    if not _THIRTY_DAY.search(sentence):
        # The quote does not say the horizon; the record may. What it may not do is state a
        # different one in the sentence the row publishes.
        if _OTHER_HORIZON.search(sentence):
            return None, f"pmid {pmid} rate_sentence states a different horizon"
        if not _THIRTY_DAY.search(_horizon_context(article, sentence)):
            return None, (
                f"pmid {pmid} states no 30-day horizon in its sentence, its title "
                "or the lines before it"
            )

    rate, kind = normalise_rate(row.get("reported_rate"))
    figures = _NUMBER.findall(rate)
    if not figures:
        return None, f"pmid {pmid} states no numeric rate"
    if figures[0] not in _PERCENT_IN_TEXT.findall(sentence):
        return None, f"pmid {pmid} rate {rate} is not the figure its own sentence states"

    return {
        "pmid": pmid,
        "reported_rate": rate,
        "rate_kind": kind,
        "rate_sentence": sentence,
        "cohort_summary": " ".join(str(row.get("cohort_summary") or "").split()),
        "evidence_tier": " ".join(str(row.get("evidence_tier") or "").split()),
        "horizon_days": _COMPARATOR_HORIZON_DAYS,
        # From the record, never from the row: these are the citation line, and a
        # citation line the abstract did not support is the defect this whole step is
        # about.
        "title": str(article.get("title") or "").strip(),
        "journal": str(article.get("journal") or "").strip(),
        # The compact journal, so the table's pill and the prose pill can name the record
        # the same way. From the record, like the rest of the citation line.
        "journal_abbrev": str(article.get("journal_abbrev") or "").strip(),
        "pub_date": str(article.get("pub_date") or "").strip(),
        "authors": str(article.get("authors") or "").strip(),
        "url": str(article.get("url") or "").strip(),
    }, ""


def verify_comparators(rows: list[dict] | None, articles: list[dict] | None,
                       target: ClinicalTarget) -> tuple[list[dict], list[dict]]:
    """Every proposed row, verified; the failures come back with their reasons."""
    by_pmid = {
        str(article.get("pmid") or "").strip(): article
        for article in articles or [] if isinstance(article, dict)
    }
    verified: list[dict] = []
    rejected: list[dict] = []
    for row in rows or []:
        kept, reason = verify_comparator(row, by_pmid, target)
        if kept is not None:
            if kept["pmid"] not in {entry["pmid"] for entry in verified}:
                verified.append(kept)
        else:
            rejected.append({
                "pmid": str((row or {}).get("pmid") or "").strip(),
                "reason": reason,
            })
    return verified, rejected


#: At most two comparators reach the table. The surface draws a comparison, not a
#: literature review, and a third row pushes the patient's own baseline off the fold.
MAX_COMPARATORS = 2


@dataclass(frozen=True)
class ComparatorPass:
    """What one query contributed, so a thin result can be read as a retrieval fact.

    The distinction this preserves is the one the first run could not make from its
    printed output: a pass that returned six records for this patient's condition and
    verified none of them is a fact about the literature, while a pass that returned
    nothing is a fact about the query. Both are empty tables and they call for opposite
    responses.
    """

    query: str
    wide: bool
    candidates: int
    aligned: int
    verified: int
    rejected: tuple[dict, ...] = ()
    #: Candidates removed for being a population this admission is not in. Reported
    #: separately from the verified count because it is a retrieval fact: a pool that lost
    #: half its records to the population filter is thin for a different reason than a pool
    #: whose records were all about another condition.
    excluded: int = 0
    #: Whether this pass's records were merged with another's before the model read them.
    #: Two passes, one extraction: without this a report showing two passes and one call
    #: would read as a lost pass.
    pooled: bool = False


@dataclass(frozen=True)
class ComparatorResult:
    """Verified rows, every refusal with its reason, and the passes that produced them."""

    rows: list[dict]
    rejected: list[dict]
    passes: list[ComparatorPass]


async def extract_comparators(
    llm, target: ClinicalTarget, *, search, config: dict | None = None,
    max_rows: int = MAX_COMPARATORS,
) -> ComparatorResult:
    """Comparator rows for this patient: ONE retrieval, ONE extraction.

    `search(terms, wide)` is the caller's retrieval callable — the tool function, or a fake
    in a test — and it is passed in rather than imported so that this module keeps no
    dependency on the tool surface and the policy can be exercised without NCBI.

    ONE query and ONE model call, which is a reversal of the two-pass design and worth
    stating plainly. That design asked the condition alone first, with the exact-phrase
    outcome clause, and only asked the broadened question if the narrow pass failed to
    produce two rows. It was built for precision and it cost the reader time: measured
    2026-11-02, a `/ehr` turn showed `Searching the Literature` twice in the workflow and
    two five-second extraction gaps in the trace, 25.6s for a briefing that had already
    done its chart work.

    The precision the narrow pass was built for is no longer bought locally at all: the
    pool from one broadened query is handed to the extraction turn whole, in PubMed's
    order, and the model that reads the abstracts decides which of them is this patient's
    cohort. Ordering the pool in Python before the model saw it only decided which records
    the model was allowed to consider. What the reader gets is the union the two passes
    were trying to approximate, read once.

    Empty stays a real answer at every stage: a query that returns nothing, a pool that is
    all another population, and a pool that is all a different condition are three
    different facts, and each is reported rather than collapsed into "no rows".
    """
    candidates = await candidate_pass(search, target)
    verified, refused_rows = await _extract_pass(
        llm, target, candidates.aligned, config)
    rejected = [*candidates.refused, *refused_rows]
    return ComparatorResult(
        rows=verified[:max_rows], rejected=rejected,
        passes=[candidates.report(verified=len(verified),
                                  rejected=candidates.refused)],
    )


@dataclass(frozen=True)
class CandidatePass:
    """One query's contribution to the pool the model reads, before it has read it.

    A record rather than a dict because the retrieval facts and the model's verdict are
    reported together and must not be confused: `candidates` is what PubMed returned,
    `aligned` is what named this admission's condition, `verified` is what the extraction
    turn could quote a sentence for, and `excluded` is what the population filter refused.
    """

    query: str
    wide: bool
    candidates: int
    aligned: tuple[dict, ...]
    refused: tuple[dict, ...]
    excluded: int

    def report(self, *, verified: int, rejected: tuple[dict, ...]) -> ComparatorPass:
        return ComparatorPass(
            query=self.query, wide=self.wide, candidates=self.candidates,
            aligned=len(self.aligned), verified=verified,
            rejected=tuple(rejected), excluded=self.excluded,
        )


async def candidate_pass(search, target: ClinicalTarget) -> CandidatePass:
    """The one readmission query this turn runs, and what it left to read.

    The clinical half is the condition OR its anchors, which is what the tool builds from
    a query plus its synonyms; the outcome half is the broadened clause, because a record
    that says "readmission within 30 days" matches none of the four exact phrases and the
    reader is owed the record that says it. One query rather than two: the second existed to
    catch what the first missed, and asking once with the broader clause and ranking the
    result catches it without a second round trip.

    Nothing here decides whether a record is ABOUT this patient's condition. Every adult
    record the search returned goes to the extraction turn, in the order PubMed ranked
    them — no local ordering and no local cut. A Python sort over titles and abstracts was
    deciding which records the model was allowed to read, and each of its rules — name the
    condition, state a percentage, name the operation — was a guess about clinical
    relevance standing where reading the abstract belongs. A token gate used to refuse
    records before the model saw them, which turned a synonym into a silent refusal.
    """
    result = await search([target.condition, *target.anchors], True)
    result = result if isinstance(result, dict) else {}
    query = str(result.get("query") or "")
    articles = [article for article in (result.get("articles") or [])
                if isinstance(article, dict)]
    pool, refused = adult_cohorts(articles, target)
    return CandidatePass(
        query=query, wide=True, candidates=len(articles),
        aligned=tuple(pool),
        refused=tuple(refused),
        excluded=len(refused),
    )



async def _extract_pass(
    llm, target: ClinicalTarget, aligned: tuple[dict, ...] | list[dict],
    config: dict | None,
) -> tuple[list[dict], list[dict]]:
    """One model call over `aligned`, verified against the abstracts it came from.

    Returns `(rows, rejections)`. A withheld or malformed reply is a rejection with its
    reason, never an exception and never an empty success: "the model declined" and "the
    literature stated nothing" are different statements to the reader.
    """
    if not aligned:
        return [], []
    payload, why = await _structured_reply(
        llm,
        prompts.LITERATURE_COMPARATOR_PROMPT,
        prompts.literature_comparator_input(target, aligned),
        config,
    )
    if payload is None:
        return [], [{"pmid": "", "reason": f"the extraction reply {reply_problem(why)}"}]
    verified, rejected_rows = verify_comparators(
        payload.get("comparators"), aligned, target)
    for item in payload.get("rejected") or []:
        if isinstance(item, dict):
            rejected_rows.append({
                "pmid": str(item.get("pmid") or "").strip(),
                "reason": " ".join(str(item.get("reason") or "").split())[:200],
            })
    return verified, rejected_rows

