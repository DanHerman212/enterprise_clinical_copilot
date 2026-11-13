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

import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass, field, replace
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
from services.agent.engine import gateway
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
from services.mcp import config as mcp_config
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
        route = _normalise(str(row.get("route") or ""))
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
        setting = _normalise(str(row.get("setting") or ""))
        if setting not in ("discharge", "inpatient"):
            setting = ""
        # A route or indication is kept only when the passage states it. "IV" and "for
        # pain" are short enough to appear in any clinical note by accident, so the same
        # rule the dose gets applies: what the note does not say is not carried.
        verified.append({
            "medication": str(row.get("medication") or "").strip(),
            "dosage": str(row.get("dosage") or "").strip(),
            "frequency": str(row.get("frequency") or "").strip(),
            "route": str(row.get("route") or "").strip() if route and route in source else "",
            "setting": setting,
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


def _passage_supports(passage: dict, row: dict[str, str]) -> bool:
    """Whether this passage states the drug a row names, and the dose it states for it.

    The same leniency the extraction check uses, applied to one passage at a time: a name
    must appear in the passage body, and a dosage must be supported when the row carries
    one. A drug named in one section and dosed in another is a row no single passage
    supports, which is what makes the pair of citations necessary rather than decorative.
    """
    body = _normalise(str(passage.get("text") or ""))
    name = _normalise(str(row.get("medication") or ""))
    if not name or name not in body:
        return False
    dosage = _normalise(str(row.get("dosage") or ""))
    return not dosage or _dose_supported(dosage, body)


def _passage_names(passage: dict, row: dict[str, str]) -> bool:
    """Whether this passage names the drug at all, whatever it says about the dose.

    The looser half of `_passage_supports`, and the one that decides whether a passage
    CONTRIBUTES to the answer. A discharge list states names and the instructions state the
    regimen, so the list names every drug the answer reports and doses almost none of them:
    asking a dose of it would leave the passage that named the drug uncited.
    """
    body = _normalise(str(passage.get("text") or ""))
    name = _normalise(str(row.get("medication") or ""))
    return bool(name) and name in body


def _medication_passage_positions(recorded: list[RecordedToolCall],
                                  evidence_state: Any,
                                  verified: list[dict[str, str]]) -> tuple[int, ...]:
    """The record positions of the passages the narrative reports on, in payload order.

    Resolved from the record rather than counted here, because the two disagree the moment
    the conversation has evidence in it: `/meds` retrieves one or two passages, and either
    may be one the thread already read. `_execute_tool_calls` stamps the position it
    assigned or REUSED, and the marker the narrative writes has to be that number, not
    this turn's first.

    Which passages, though, is a question about the ROWS. A discharge list that names only
    the drugs the patient arrived on and a set of instructions that prescribe and dose the
    take-home regimen are two different statements, and a single marker on the first one
    asks the reader to believe a list it does not contain. So:

      * a passage that supports every row is cited alone — one marker, one source, and the
        reader lands on text that states what the sentence says;
      * rows that span passages cite each contributing passage, in payload order;
      * with no verified rows there is nothing to attribute, and every returned passage is
        cited, because the sentence then reports on the retrieval itself.
    """
    if evidence_state is None:
        return ()
    located: list[tuple[int, dict]] = []
    for call in recorded or []:
        response = call.get("response")
        if not isinstance(response, dict):
            continue
        for passage in response.get("passages") or []:
            if not isinstance(passage, dict):
                continue
            passage_id = str(passage.get("id") or "").strip()
            if not passage_id:
                continue
            position = evidence_state.position_of_note_passage(passage_id)
            if position is not None:
                located.append((position, passage))
    if not located:
        return ()
    rows = [row for row in verified or [] if isinstance(row, dict)]
    if rows:
        complete = [(position, passage) for position, passage in located
                    if all(_passage_supports(passage, row) for row in rows)]
        if complete:
            return (complete[0][0],)
        partial = [position for position, passage in located
                   if any(_passage_names(passage, row) for row in rows)]
        if partial:
            return tuple(dict.fromkeys(partial))
    return tuple(dict.fromkeys(position for position, _ in located))


def medication_narrative(verified: list[dict[str, str]], returned_passages: int,
                         *, cite: int | None = None) -> str:
    """Deprecated forward: the narrative is written in `postprocess.answer`.

    Moved on 2026-10-29 (`00d7` Row 8). Answer prose belongs to `postprocess/`, and the
    engine's formatting turn only orders the sections. This stays one line so callers that
    still import it from here get identical prose; nothing in the engine calls it.
    """
    return answer_text.medication_narrative(verified, returned_passages, cite=cite)


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


def _patient_estimate(recorded: list[RecordedToolCall] | None) -> float | None:
    """The probability `predict_readmission` returned for this admission, or None.

    Read off the recorded call rather than carried in a variable, because the estimate is
    a tool result like every other fact this turn states: it reaches the prose as a tool
    message, and the comparison's direction has to be computed from the same response the
    widget draws. A second copy passed alongside the call is a second answer to "what is
    this patient's risk" and the two would eventually disagree.

    None when no tool ran or the response carries no usable figure, and then no relation
    line is written: a direction needs two numbers.
    """
    for call in recorded or []:
        if call.get("name") != "predict_readmission":
            continue
        response = call.get("response") or {}
        value = response.get("probability") if isinstance(response, dict) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        return float(value)
    return None


def lab_flag_annotation(recorded: list[RecordedToolCall]) -> str:
    """Per analyte: how many readings are flagged, and the flag on the baseline.

    The trajectory payload already carries a flag on every timepoint, and the formatter is
    told to read them. What it cannot see is the SHAPE of the series — whether the
    abnormal readings are all of them or three of eleven, and whether the reading the
    admission opened with (day 0.0) was already abnormal. Two answers failed on exactly
    that: one said an analyte "remained low throughout" whose first reading was normal,
    and one said "all readings" were abnormal for a series of six with one flagged value.

    Counted here rather than left to the model, because it is arithmetic over the payload
    and the prompt's job is to say what to do with the number, not to work it out.
    """
    lines: list[str] = []
    for call in recorded or []:
        response = call.get("response")
        if not isinstance(response, dict):
            continue
        for trend in response.get("trends") or []:
            if not isinstance(trend, dict):
                continue
            series = [point for point in trend.get("series") or []
                      if isinstance(point, dict)]
            if not series:
                continue
            label = str(trend.get("label") or trend.get("key") or "value").strip()
            unit = str(trend.get("unit") or "").strip()
            flags = [str(point.get("flag") or "normal").strip().lower()
                     for point in series]
            abnormal = [flag for flag in flags if flag != "normal"]
            baseline = series[0]
            lines.append(
                f"- {label}{f' ({unit})' if unit else ''}: {len(abnormal)} of "
                f"{len(series)} readings flagged abnormal; baseline "
                f"day {baseline.get('day')} flag={flags[0]}."
            )
    if not lines:
        return ""
    return ("<lab_flags>\nHow many readings of each analyte the panel flags abnormal, "
            "and the flag on the baseline (first) reading:\n"
            + "\n".join(lines) + "\n</lab_flags>")


def _formatter_input(tool_messages: list[Any], outpatient: bool,
                     question: str | None = None,
                     comparators: list[dict] | None = None,
                     context_block: str = "",
                     session_block: str = "",
                     patient_estimate: float | None = None,
                     fallback_literature: list[dict] | None = None,
                     lab_flags: str = "") -> str:
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
    block = comparator_block(comparators, patient_estimate=patient_estimate)
    if block:
        text = f"{text}\n\n{block}"
    else:
        # No verified cohort, but records were read. The second block is what keeps a turn
        # that could not verify a rate from publishing nothing at all about the literature
        # it retrieved: it asks for what those studies report, and says in its own header
        # that there is no benchmark among them. Exactly one of the two blocks is ever
        # given, so the prompt's rules for "a block is present" and "no block is present"
        # cannot both be in play at once.
        fallback = fallback_literature_block(fallback_literature)
        if fallback:
            text = f"{text}\n\n{fallback}"
    if (session_block or "").strip():
        text = f"{text}\n\n{session_block.strip()}"
    if (lab_flags or "").strip():
        text = f"{text}\n\n{lab_flags.strip()}"
    if (context_block or "").strip():
        text = f"{text}{context_block}"
    asked = routing.strip_admission_suffix((question or "").strip())
    return f"Question asked: {asked}\n\n{text}" if asked else text


def comparator_block(comparators: list[dict] | None,
                     *, patient_estimate: float | None = None) -> str:
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

    `patient_estimate` is the model's own predicted probability for this admission, and
    passing it adds a `relation:` line to each row: the direction of the comparison, worked
    out here rather than by the model. It was worked out by the model until 2026-11-08, and
    a briefing stated that 9.0% sat "below" a published 3.8% — two figures in adjacent cells
    and a word connecting them the wrong way round. A direction is arithmetic, so it is
    computed where arithmetic is exact, and the prompt is told to use the field verbatim.
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
        relation = _rate_relation(patient_estimate, rate, kind)
        if relation:
            lines.append(f"  relation: {relation}")
    lines.append("</verified_comparators>")
    return "\n".join(lines)


def _rate_relation(patient_estimate: float | None, reported_rate: str,
                   rate_kind_value: str = "") -> str:
    """Where this patient's estimate sits against one published rate, in one clause.

    Arithmetic over the numbers each side already publishes, and nothing else: no rounding
    invented, no averaging, no difference the record does not state. A two-arm row and a
    range are compared against their SPAN rather than against one of their figures, because
    "higher than 2.6% vs. 0.4%" is not a statement about either arm.

    Empty when the estimate or the rate is missing, which is the honest answer: a direction
    needs two figures, and a turn holding one of them has no comparison to state.
    """
    if patient_estimate is None:
        return ""
    figures = [float(value) for value in _NUMBER.findall(str(reported_rate or ""))]
    if not figures:
        return ""
    estimate = patient_estimate * 100.0
    shown = f"{estimate:.1f}%"
    low, high = min(figures), max(figures)
    if str(rate_kind_value or "") in ("range", "comparison") and len(figures) > 1:
        span = " ".join(str(reported_rate or "").split())
        if estimate > high:
            return f"patient estimate ({shown}) is above the whole published span ({span})"
        if estimate < low:
            return f"patient estimate ({shown}) is below the whole published span ({span})"
        return f"patient estimate ({shown}) is within the published span ({span})"
    rate = f"{low:g}%"
    if estimate > low:
        return f"patient estimate ({shown}) is higher than the published rate ({rate})"
    if estimate < low:
        return f"patient estimate ({shown}) is lower than the published rate ({rate})"
    return f"patient estimate ({shown}) equals the published rate ({rate})"


#: How much of an abstract the fallback block carries.
#:
#: The formatter summarises what these studies report about readmission risk factors, so it
#: needs the finding rather than the citation line — but a briefing sentence is not a
#: literature review, and two full abstracts is a third of the turn's input. Six hundred
#: characters reaches the results clause of a structured abstract and stops.
_FALLBACK_ABSTRACT_CHARS = 600


def fallback_literature_block(articles: list[dict] | None) -> str:
    """The retrieved studies, as the formatting turn reads them when no cohort verified.

    The other side of `comparator_block`, and the reason it exists: a `/ehr` turn on an
    admission whose abstracts state no 30-day percentage used to publish NOTHING from the
    literature. Eighteen records were retrieved, filtered, read by an extraction turn and
    then dropped, because the only route a readmission-scope search had to the reader was a
    verified matrix row (`90000030`, `90000047` — measured 2026-11-08). The reader was shown
    a chart summary and a risk score with no literature at all, and no statement that any
    had been found.

    What this block does NOT do is imply a benchmark. Its header says in as many words that
    none of these states a 30-day rate, because the section that draws on them must say that
    too: a rate-shaped sentence beside a risk score, sourced to a paper that reported no
    rate, is the failure the whole verification path exists to prevent.

    Empty when there are no records, and then nothing is appended: a turn with no literature
    says nothing about the literature.
    """
    records = [article for article in articles or [] if isinstance(article, dict)]
    if not records:
        return ""
    lines = [
        "<retrieved_readmission_literature>",
        "Peer-reviewed studies retrieved from PubMed for this admission's condition and "
        "procedure. NONE of them states a single 30-day readmission percentage for a "
        "cohort this patient belongs to, so there is NO benchmark figure to set this "
        "patient's estimate against. Summarise what they report about readmission risk, "
        "post-discharge outcomes or clinical trajectory, and cite each study you use with "
        "its PMID, written ^[PMID: <pmid>], immediately after the sentence that draws on "
        "it.",
    ]
    for article in records:
        pmid = str(article.get("pmid") or "").strip()
        if not pmid:
            continue
        journal = str(article.get("journal") or "").strip()
        year = str(article.get("pub_date") or "").strip()
        title = " ".join(str(article.get("title") or "").split())
        source = " · ".join(part for part in (journal, year) if part) or "record"
        lines.append(f"- PMID {pmid} · {source}")
        if title:
            lines.append(f"  title: {title}")
        abstract = " ".join(str(article.get("abstract") or "").split())
        if abstract:
            if len(abstract) > _FALLBACK_ABSTRACT_CHARS:
                abstract = abstract[:_FALLBACK_ABSTRACT_CHARS].rstrip() + " …"
            lines.append(f"  reports: {abstract}")
    lines.append("</retrieved_readmission_literature>")
    return "\n".join(lines)


#: How hard this module tries before it gives up on a capacity refusal.
#:
#: `gemini-3.1-flash-lite` is reached on the multi-region endpoint (`GEMINI_LOCATION`),
#: where capacity is shared and `429 RESOURCE_EXHAUSTED` arrives in bursts. Measured
#: 2026-11-08: an `/ehr` turn whose formatting call took 5.19 s with identical token
#: counts took 64.16 s under that pressure, and a run of turns failed outright — because
#: `langchain_google_genai` retries 5xx and raises 429 immediately, so a capacity blip
#: reached the clinician as "the agent failed to answer".
#:
#: Four attempts over roughly ten seconds turns a burst into a slower answer. It is
#: bounded because the alternative is worse: an answer that arrives after the request has
#: already timed out is not an answer.
RETRY_MAX_ATTEMPTS = 4
RETRY_BASE_SECONDS = 1.5
RETRY_MAX_SECONDS = 20.0


#: What a capacity refusal looks like in an exception, without importing the provider.
#:
#: The table moved to `engine/gateway.py`, which is the one module allowed to know how a
#: provider phrases a refusal: the same question is asked of the judge client and of the
#: adapter's own callers, and three copies of a marker list is how one of them ends up
#: missing a phrase. What stays here is the decision — a capacity refusal is retried, and
#: nothing else is.
def _rate_limited(exc: BaseException) -> bool:
    """Whether the provider refused this call for capacity rather than for content."""
    return gateway.classify_error(exc) == gateway.CAPACITY


async def _pause_before_retry(attempt: int, what: str) -> None:
    """Wait, with jitter, before attempt `attempt + 1`.

    Jitter matters more than the delay: four concurrent turns throttled at the same instant
    would otherwise retry in lockstep and exhaust themselves together, which is the burst
    they were waiting out.
    """
    ceiling = min(RETRY_BASE_SECONDS * (2 ** (attempt - 1)), RETRY_MAX_SECONDS)
    wait = ceiling * (0.5 + random.random())
    logger.warning(
        "%s was refused for capacity (attempt %d of %d); retrying in %.1fs",
        what, attempt, RETRY_MAX_ATTEMPTS, wait,
    )
    await asyncio.sleep(wait)


async def _retrying(call: Callable[[], Any], *, what: str,
                    attempts: int = RETRY_MAX_ATTEMPTS) -> Any:
    """Await `call()`, retrying only a capacity refusal, up to `attempts` times.

    Anything that is not a 429 is re-raised on the first attempt: a content filter, a
    malformed request and a real defect all need to surface as themselves, and retrying
    them would turn a clear failure into a slow one.

    `attempts` is per-call because the two callers have opposite economics. The formatting
    turn is the answer the reader is waiting on, so it is worth four attempts over ten
    seconds. A structured extraction is an ENRICHMENT: if it cannot run, the turn still
    answers, and spending thirty-five seconds retrying a step whose failure is already
    handled is the reader's time spent on nothing — measured 2026-11-08, `90000069` spent
    34.7 s of a 67.6 s turn on five refused extraction attempts before degrading to the
    fallback that was waiting for it all along.
    """
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return await call()
        except Exception as exc:
            if attempt >= max(1, attempts) or not _rate_limited(exc):
                raise
            await _pause_before_retry(attempt, what)
    raise AssertionError("unreachable: the loop returns or raises")


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

    A capacity refusal is retried, because this is the call the reader is waiting on and
    it is 83% of an `/ehr` turn (64.16 s of 77.18 s, `826289ea`, 2026-11-08). Streaming
    is retried only BEFORE the first frame: once a piece has been announced, a second
    attempt would print the prose twice.
    """
    if announce is None:
        return await _retrying(lambda: llm.ainvoke(prompt, config=config),
                               what="the formatting turn")

    attempt = 1
    while True:
        turn: Any = None
        try:
            async for chunk in llm.astream(prompt, config=config):
                turn = chunk if turn is None else turn + chunk
                # The chunk's own text, not its repr: Gemini answers in content blocks, so
                # announcing `str(chunk.content)` would put the envelope on screen instead
                # of the summary (see `model_turn.text_of`).
                piece = text_of(chunk.content)
                if piece:
                    announce(piece)
        except Exception as exc:
            # Safe to retry only while nothing has been announced; see the docstring.
            if (turn is not None or attempt >= RETRY_MAX_ATTEMPTS
                    or not _rate_limited(exc)):
                raise
            await _pause_before_retry(attempt, "the formatting turn")
            attempt += 1
            continue
        if turn is None:
            raise RuntimeError("the formatting model returned no chunks")
        return turn


#: The client the comparator step extracts with, replaceable so a test can run this step
#: offline. `tests/agent/conftest.py` sets it to `lambda model: None`, and a `None` client
#: means "this environment provides no extraction" — the step is skipped, no model is
#: called, and the production path learns nothing about tests.
extraction_llm_factory: Callable[[str], Any] = model_gateway.build_extraction_llm


def _pass_rows(passes: list["ComparatorPass"]) -> list[dict]:
    """The passes as the turn state carries them, and the shape a log reader depends on.

    `retrieved -> pool_size -> aligned -> verified` is the whole diagnostic: `candidates`
    is what PubMed returned, `pool_size` what survived the population filter, `aligned`
    what the extraction turn was given (the cap, so it stops at
    `MAX_EXTRACTION_CANDIDATES`) and `verified` what could be quoted. Without `pool_size`
    the line read "aligned=10, excluded=0", which looks like ten matches and is really ten
    reads (`cr`, 2026-11-08).
    """
    return [
        {"query": entry.query, "wide": entry.wide, "candidates": entry.candidates,
         "aligned": entry.aligned, "verified": entry.verified,
         "excluded": entry.excluded, "outcome": entry.outcome,
         "pool_size": entry.pool_size}
        for entry in passes or []
    ]


def _fallback_literature(rows: list[dict] | None,
                         pool: list[dict] | None) -> list[dict]:
    """The records a zero-row turn still publishes, or none.

    Only when there are no verified rows. A turn WITH a cohort states the benchmark and
    cites it; adding studies that reported no 30-day rate to the same paragraph would put
    an unquantified paper beside a quantified one and invite the prose to average what one
    of them never said.

    The head of the RANKED pool, not of PubMed's reply: the ranking is the one that puts a
    record about this admission's operation and condition first
    (`_comparator_relevance`), which is the same order a reader would choose two papers in.
    """
    if rows:
        return []
    return [article for article in (pool or [])
            if isinstance(article, dict)][:MAX_FALLBACK_LITERATURE]


async def _readmission_comparators(
    toolbox: Any, model: str, recorded: list[RecordedToolCall], on_event,
    *, extraction_llm: Any = None, evidence_state: Any = None,
    admission_context: str = "", callbacks: list[Any] | None = None,
) -> tuple[list[dict], dict | None, list[RecordedToolCall], list[dict], str,
           list[dict]]:
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

    `admission_context` is the pre-flight manifest, read here for the index patient's age
    and sex and for nothing else. The comparison is a population match, and until those two
    facts reached it the only thing matched was the condition: a knee-arthroplasty cohort
    of octogenarians was accepted beside a 41-year-old with no objection from any layer.

    `callbacks` are the turn's tracing callbacks, and they are threaded into BOTH model
    calls below. Neither used to receive them (`config=None`), so the two steps that decide
    whether the table draws at all — the target extraction and the comparator extraction —
    produced no Langfuse generation at all. On `826289ea` that left 8.59 s and 2.16 s of
    model time visible only as gaps between stage events.
    """
    sections = note_sections(recorded)
    if not sections:
        # No chart text, so no condition and no query. Reported as a degraded benchmark
        # rather than as a literature that offered nothing: nothing was asked of it.
        return [], None, [], [], routing.OUTCOME_EXTRACTION_DEGRADED, []

    # Whose cohort this has to be, read from the manifest first and the note second. An
    # unknown age filters nothing downstream, which is the honest behaviour: refusing a
    # cohort because a number could not be read would trade a wrong cohort for no cohort.
    patient_age, patient_sex = patient_demographics(admission_context, sections)

    llm = extraction_llm if extraction_llm is not None else extraction_llm_factory(model)
    if llm is None:
        # No extraction client in this environment. Stated on the state rather than left
        # silent: `comparators: []` and "nobody looked" are different answers, and the
        # surface says which one it is drawing.
        return ([], {"source": "unavailable",
                     "reason": "no extraction client is configured here"}, [], [],
                routing.OUTCOME_EXTRACTION_DEGRADED, [])
    derived = await derive_target(
        llm, sections, config={"callbacks": list(callbacks)} if callbacks else None,
        patient_age=patient_age, patient_sex=patient_sex,
    )
    target = derived.target
    if target is None:
        logger.warning(
            "comparators: no clinical target (%s: %s)", derived.source, derived.reason,
        )
        return ([], {"source": derived.source, "reason": derived.reason}, [], [],
                routing.OUTCOME_EXTRACTION_DEGRADED, [])

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

    result = await extract_comparators(
        llm, target, search=search,
        config={"callbacks": list(callbacks)} if callbacks else None,
    )
    payload = {
        "condition": target.condition,
        "procedure": target.procedure,
        "anchors": list(target.anchors),
        "patient_age": target.patient_age,
        "patient_sex": target.patient_sex,
        "source": derived.source,
    }
    passes = _pass_rows(result.passes)
    # What the briefing is shown when nothing verified. The abstracts were retrieved,
    # filtered by population and read; a turn whose verification produced no row has still
    # found this patient's literature, and publishing none of it is the defect this
    # directive exists to close (`90000030`: eighteen retrieved, zero published).
    fallback_literature = _fallback_literature(result.rows, result.pool)
    if fallback_literature:
        # Say so on the call that retrieved them, because settlement runs before the
        # guardrail and before presentation, and it is a pure function of the turn's own
        # tool calls (graph `_settle` passes those and nothing else). Without this the
        # records are published by `presentation` — which reads the array directly — and
        # deleted from the prose by `enforce_citations`, because the allowance it builds
        # from `tool_calls` cannot see an array that lives on the state. That is exactly
        # how both badges died on `90000069` and `90000020` while the drawer kept the
        # papers (2026-11-08).
        #
        # This is a declaration, not evidence: the pool stays uncitable and only the two
        # records the narrative was actually given are named here.
        announced = [str(article.get("pmid") or "").strip()
                     for article in fallback_literature
                     if str(article.get("pmid") or "").strip()]
        for call in calls_made:
            if call.get("name") not in routing.LITERATURE_TOOLS:
                continue
            response = call.get("response")
            if isinstance(response, dict):
                call["response"] = {**response, "fallback_pmids": announced}
    if fallback_literature:
        logger.info(
            "comparators: no verified row; %d retrieved record(s) published for the "
            "narrative to cite: %s",
            len(fallback_literature),
            ", ".join(str(article.get("pmid") or "") for article in fallback_literature),
        )
    outcome = benchmark_outcome(result.passes)
    # The pass, in one line, whether or not it produced a row.
    #
    # "The table drew nothing" is three different facts — the abstracts stated no rate, the
    # model proposed rows the verifier refused, or the extraction never read the abstracts —
    # and until this line only the first was visible anywhere, because it is the one the
    # matrix prints. `_reply_rows` records what the model proposed; this records what the
    # retrieval and the outcome were, so the two together are enough to tell the three
    # apart from a Cloud Run log alone.
    for entry in result.passes:
        logger.info(
            "comparators: query=%r wide=%s candidates=%d pool_size=%d "
            "sent_to_extractor=%d aligned=%d verified=%d "
            "excluded=%d outcome=%s",
            entry.query, entry.wide, entry.candidates, entry.pool_size,
            entry.aligned, entry.aligned,
            entry.verified, entry.excluded, entry.outcome,
        )
    logger.info(
        "comparators: condition=%r procedure=%r patient=%s/%s rows=%d outcome=%s",
        target.condition, target.procedure, target.patient_age or "?",
        target.patient_sex or "?", len(result.rows), outcome,
    )
    for rejection in result.rejected:
        logger.info("comparators: refused %s — %s",
                    rejection.get("pmid") or "(no pmid)",
                    rejection.get("reason") or "")
    return result.rows, payload, calls_made, passes, outcome, fallback_literature


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
    evidence_state: Any = None,
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
    # `evidence_state` is the record the caller's replay already built, when it has one.
    # A mid-thread command runs against the thread's evidence, and the replay that restored
    # that evidence has already ingested it — so hydrating the same calls a second time here
    # would parse a whole note's worth of passages for nothing, before any tool of this
    # command has run. The caller passes the record it made; a command with no thread, or a
    # caller that holds no record, gets the hydration as before.
    if evidence_state is None and hadm_id is not None:
        evidence_state = session_evidence.SessionEvidenceState.hydrate_from_calls(
            hadm_id, [_presented_call(call) for call in (replayed_calls or [])]
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

    # `/meds` cites with markers on the passages its rows came from, so the passages stay
    # as the tool returned them. They used to be joined into the first one here, which put
    # the record and the command's answer on two different numbering schemes: the record
    # held two positions and the merged payload held one, so the marker named the first
    # passage while the model had extracted from both.

    # The published cohorts this admission's estimate is compared against, retrieved live
    # and recorded as tool calls like every other piece of evidence this turn rests on.
    command_extras: dict[str, Any] = {}
    if command.wire == "summarize" and not outpatient_record:
        (comparators, target_payload, literature_recorded, passes, outcome,
         fallback_literature) = \
            await _readmission_comparators(
                toolbox, model, recorded, on_event, extraction_llm=extraction_llm,
                evidence_state=evidence_state,
                admission_context=admission_context,
                # The turn's tracing callbacks, so the target and comparator extractions
                # are generations in the trace rather than gaps between stage events.
                callbacks=callbacks)
        recorded += literature_recorded
        command_extras["comparators"] = comparators
        command_extras["clinical_target"] = target_payload
        command_extras["comparator_passes"] = passes
        # The studies this turn retrieved and could not reduce to a 30-day rate. Read by
        # the formatter (so the risk section can say what they report), by the guardrail
        # (so their PMIDs survive `enforce_citations`) and by the composer (so they reach
        # the reader's pills and references drawer). One array, three readers.
        command_extras["fallback_literature"] = fallback_literature
        # What the benchmark step came to, as a word rather than as a length. `[]` rows is
        # a real answer and the surface has to know WHICH real answer: the orchestrator
        # turns this into the one bit `presentation` is given, so the table can say the
        # abstracts offered no cohort when that is what happened, and that the extraction
        # did not complete when that is what happened.
        command_extras["comparator_outcome"] = outcome

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
                # The model's own estimate for this admission, so the comparison's
                # DIRECTION is computed here rather than inferred by the prose. A section
                # that states two figures and the wrong word between them is the defect
                # this argument closes.
                patient_estimate=_patient_estimate(recorded),
                # And, when no cohort verified, the studies that were retrieved anyway.
                fallback_literature=command_extras.get("fallback_literature"),
                # `/labs` only: the count of flagged readings per analyte and the flag on
                # its baseline, counted here rather than left to the prose.
                lab_flags=(lab_flag_annotation(recorded)
                           if command.wire == "labs" else ""),
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
        #
        # The grid is the home prescription list, so a narrative fallback's ward
        # administrations do not belong in it. Rows the extractor marked `inpatient` are
        # withheld whenever the record also documents something to take home; with no
        # discharge row at all the grid keeps them, because a record whose only
        # medications were given on the ward is still a record with medications, and the
        # narrative states which of them is which (2026-11-10).
        home_rows = [row for row in verified
                     if str(row.get("setting") or "").strip().lower() != "inpatient"]
        grid_rows = home_rows or verified
        state["command_payload"] = {
            "medications": grid_rows,
            "dropped": dropped,
            "returned_passages": returned,
            "sections": list(sections),
        }
        # The narrative replaces the raw JSON: the grid renders the list, and the
        # prose is what the guardrails and the citation layer expect to see. The markers
        # are the positions the RECORD gave the passages these rows were read from, not
        # this turn's first: a passage the thread already read keeps the number it had,
        # and a marker that names a different passage is worse than no marker at all
        # (2026-11-06), while a marker that names only one of two passages leaves half the
        # sentence unverifiable (2026-11-10).
        state["deterministic_answer"] = answer_text.medication_narrative(
            verified, returned, sections=sections,
            cite=_medication_passage_positions(recorded, evidence_state, verified))
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

#: How many ranked records a turn with no verified 30-day cohort still publishes. Two,
#: because the section that draws on them is one paragraph: one study is an anecdote and
#: three is a literature review, and the reader of a briefing wants the two most relevant
#: papers, cited, not a survey.
MAX_FALLBACK_LITERATURE = 2

#: The horizon the comparison is made at. Stated as a constant so the check below reads as
#: the rule it is: a five-year rate beside a 30-day estimate is two unrelated numbers in
#: adjacent cells.
_COMPARATOR_HORIZON_DAYS = 30

_THIRTY_DAY = re.compile(
    r"\b30[\s-]?days?\b|\bthirty[\s-]?days?\b|\bwithin 30 days\b", re.IGNORECASE
)
#: A horizon that is NOT the one being compared. A record titled "predictors of
#: thirty-day readmission" can still quote a sentence about its 90-day figure, and the
#: quote is what the row publishes, so a conflicting horizon inside the quoted sentence is
#: a refusal rather than a pass.
_OTHER_HORIZON = re.compile(
    r"\b(?:60|90|180|365)[\s-]?days?\b"
    r"|\b(?:six|nine|twelve|eighteen|twenty[\s-]?four)[\s-]?months?\b"
    r"|\b(?:one|two|three|four|five|ten|\d{1,2})[\s-]?years?\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_PERCENT_IN_TEXT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
#: A readmission count stated as ZERO in words rather than as a figure.
#:
#: "…no patient was readmitted within 30 days after discharge" is a real result, and
#: `38504222` states it that way. `_PERCENT_IN_TEXT` needs a numeral, so a true zero was
#: unverifiable and the row was refused with "rate 0% is not the figure its own sentence
#: states" — the check reporting that a sentence states no percentage when what it states
#: is no readmissions (2026-11-08). Narrow on purpose: the negation AND a readmission word
#: in the same clause, so "no difference in readmission" is not read as a zero rate.
_VERBAL_ZERO = re.compile(
    r"\bno\s+(?:patients?|cases?|subjects?|individuals?|persons?)\b[^.;]{0,80}?"
    r"\b(?:readmitted|readmissions?)\b"
    r"|\b(?:there\s+were|had)\s+no\s+readmissions?\b"
    r"|\bzero\s+readmissions?\b",
    re.IGNORECASE,
)


def _shares_a_percent_sign(figures: list[str], sentence: str) -> bool:
    """Whether every figure is stated in a clause that carries a percentage marker.

    "The 30-day readmission rate decreased to 1.2 from 1.3%" states two figures and one
    `%`. `_PERCENT_IN_TEXT` needs the sign on the figure itself, so `1.2` was "not the
    figure its own sentence states" and the row was refused on both `cp` runs
    (2026-11-08) — a true invariant reported through a surface-grammar rule.

    The bound stays tight in the direction that matters: every figure the row publishes
    has to appear in the sentence, and the sentence has to state a percentage at all. A
    sentence quoting bare integers is not a rate.
    """
    if not figures:
        return False
    if "%" not in sentence and not re.search(r"\bpercent\b", sentence, re.IGNORECASE):
        return False
    return all(
        re.search(rf"(?<![\d.]){re.escape(figure)}(?![\d])", sentence)
        for figure in figures
    )
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
    local = f"{article.get('title') or ''} {window}".strip()
    if _THIRTY_DAY.search(local):
        return local
    # A paper states its horizon ONCE, in Methods ("30-day outcomes were assessed…"), and
    # then reports figures in Results sentences that never repeat it. `33049410` has that
    # shape: "unplanned readmissions (4.96% vs 3.72%)", with the horizon declared a
    # paragraph earlier and outside the 240-character window — so the record was refused
    # with "states no 30-day horizon" on 2026-11-08 while its Methods say 30-day outcomes.
    #
    # The horizon is a property of the STUDY and a quote is a sentence from it, so the
    # whole abstract is consulted before refusing. What this does not relax: a sentence
    # carrying its OWN conflicting horizon is refused before this is reached (see
    # `verify_comparator`), so a record reporting both can never publish its 90-day figure
    # as a 30-day rate.
    return f"{local} {abstract}".strip()


def _thirty_day_rate(sentence: str) -> str | None:
    """The percentage the sentence attaches to its 30-day horizon, when it states one.

    A range sentence carries two horizons and two figures: "Readmission rates for ACDF
    ranged from 5.1% (30 days) to 7.7% (90 days postoperatively)" (`31528438`). The row
    publishes ONE number, so it has to be the number the 30-day horizon owns — taking
    whichever figure happens to sit beside the sentence would publish the 90-day rate as a
    30-day one, which is the defect the whole check exists to prevent.

    Nearest figure to the marker, by distance. That handles the range form (5.1% is seven
    characters from "30 days", 7.7% is thirty), the ordinary form (one figure, one
    horizon), and the sentence that leads with one horizon and reports the other later.
    A sentence with no 30-day marker returns None and nothing is checked beyond what
    already was.
    """
    marker = _THIRTY_DAY.search(sentence)
    if marker is None:
        return None
    figures = [(match.start(), match.group(1))
               for match in _PERCENT_IN_TEXT.finditer(sentence)]
    if not figures:
        return None
    return min(figures, key=lambda entry: abs(entry[0] - marker.start()))[1]


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


#: A cohort restricted to an older population: the words a paper uses when the rate it
#: reports is not a general adult rate.
#:
#: Structural vocabulary only — no disease, no procedure, no rate — so this can refuse a
#: cohort for being the wrong population and never for being the wrong condition. Measured
#: 2026-11-08: a 41-year-old cervical-fusion admission was compared against a rate
#: published for "Elderly patients (age >=65 years)", and every layer accepted it because
#: the condition matched and nothing on the path knew the patient's age.
#:
#: The unit may be "an age not this one": a cohort of octogenarians is a real benchmark for
#: an 82-year-old and the wrong one for a 41-year-old, so the decision is made against
#: `patient_age` and never against the words alone.
_ELDERLY_COHORT = re.compile(
    r"\b(?:elderly|geriatric|octogenarian|nonagenarian|centenarian"
    r"|(?:age|aged)\s*(?:>=|≥|>|over|above|of)?\s*(?:6[5-9]|7\d|8\d|9\d)"
    r"|(?:6[5-9]|7\d|8\d)\s*(?:years?\s*)?(?:and\s*older|or\s*older|\+)"
    r"|older\s+adults?|very\s+old)\b",
    re.IGNORECASE,
)

#: A cohort restricted to a younger population, which is the same mismatch the other way
#: round. Only the unambiguous forms: an age band that starts at 18 and runs to 40 is a
#: young-adult cohort, and so is "adolescents and young adults".
_YOUNG_ADULT_COHORT = re.compile(
    r"\b(?:young\s+adults?|younger\s+adults?|adolescents?|pediatric|paediatric"
    r"|under\s+(?:2\d|3\d|40|4\d|50)\b"
    r"|(?:18|19|2\d)\s*(?:-|–|to|through)\s*(?:2\d|3\d|4\d)\b"
    r"|non-?elderly)\b",
    re.IGNORECASE,
)


def _cohort_matches_patient_age(cohort_text: Any, snippet_text: Any,
                                patient_age: int | None) -> bool:
    """Whether a published cohort is a population this patient could belong to.

    The check `adult_cohorts` makes for children, made for the other two population
    mismatches: an age-restricted subgroup that EXCLUDES the index patient. A rate
    reported for the over-65 subgroup of a study is not the rate for a 41-year-old, even
    though the study is about the same operation — and the estimate and the cohort sit in
    adjacent cells, so the reader reads them as the same claim.

    Read from the row's own `cohort_summary` and from the sentence it quotes, and nothing
    else. The abstract is not scanned: a general-adult study that mentions the elderly in
    its discussion is a valid comparison, and refusing it would be the token test this
    module already removed once.

    `patient_age` None is unknown, and an unknown age filters nothing. Only the
    unambiguous forms are refused, because a false refusal costs the reader the comparison
    entirely while a false acceptance is corrected by the extraction prompt.
    """
    if patient_age is None:
        return True
    text = f"{cohort_text or ''} {snippet_text or ''}"
    if patient_age < 65 and _ELDERLY_COHORT.search(text):
        return False
    if patient_age >= 65 and _YOUNG_ADULT_COHORT.search(text):
        return False
    return True


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

    `patient_age` and `patient_sex` are the index patient's own demographics, carried
    beside the condition for one reason: a readmission cohort is a population, and a rate
    published for octogenarians is not this patient's rate when this patient is 41. They
    are facts the session already holds — the pre-flight manifest states both, and a
    discharge note states them in its first line — and until they were carried here the
    comparison was matched on condition alone.
    """

    condition: str
    procedure: str = ""
    anchors: tuple[str, ...] = ()
    #: The index patient's age in years, when the session knows it. None means unknown, and
    #: an unknown age filters nothing: a check that guessed would refuse correct cohorts.
    patient_age: int | None = None
    #: `male`, `female`, or empty when the record does not state it.
    patient_sex: str = ""

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


#: The age and sex a cohort has to be compatible with, read from whatever this turn holds.
#:
#: Two sources, in descending order of authority. The `<admission_metadata>` block is the
#: feature row the readmission model was scored on, so an age it states is the age the
#: system was given. A chart that states the age in its own first line is the same fact
#: from the document, and is used only when the manifest is absent — a record whose
#: manifest was unreadable still has a patient in it.
_MANIFEST_AGE = re.compile(r"^\s*-\s*age:\s*(\d{1,3})\s*$", re.MULTILINE | re.IGNORECASE)
_MANIFEST_SEX = re.compile(r"^\s*-\s*gender:\s*(\S+)", re.MULTILINE | re.IGNORECASE)
_NOTE_AGE = re.compile(r"\b(\d{2,3})[-\s]year[-\s]old\b", re.IGNORECASE)
_NOTE_SEX = re.compile(
    r"\b(\d{2,3})[-\s]year[-\s]old\s+(male|female|man|woman|gentleman|lady|boy|girl)\b",
    re.IGNORECASE,
)

#: Words that name a sex the way a discharge summary writes one.
_SEX_WORDS: dict[str, str] = {
    "male": "male", "man": "male", "gentleman": "male", "boy": "male",
    "female": "female", "woman": "female", "lady": "female", "girl": "female",
}


def _normalise_sex(value: Any) -> str:
    word = str(value or "").strip().lower()
    return _SEX_WORDS.get(word, word if word in ("male", "female") else "")


def patient_demographics(admission_context: str,
                         sections: dict | None) -> tuple[int | None, str]:
    """The index patient's age and sex, from the manifest first and the note second.

    Exists because the comparator step matched a population on condition alone, and a
    thirty-day readmission rate published for octogenarians is not the rate a 41-year-old's
    estimate should be drawn beside. Both facts are already in the turn — the pre-flight
    manifest states them as fields, and a discharge note states them in its opening line —
    so nothing here is inferred; the only decision is which source to read first.

    Age is bounded to 0..120 before it is returned: a regex over prose can find a number
    that looks like an age and is not one, and a wrong age would filter the right cohort
    out rather than merely failing to filter.
    """
    context = str(admission_context or "")
    age: int | None = None
    sex = ""
    manifest_age = _MANIFEST_AGE.search(context)
    if manifest_age:
        age = int(manifest_age.group(1))
    manifest_sex = _MANIFEST_SEX.search(context)
    if manifest_sex:
        sex = _normalise_sex(manifest_sex.group(1))

    if age is None or not sex:
        note_text = " ".join(str(body or "") for body in (sections or {}).values())
        if age is None:
            stated = _NOTE_AGE.search(note_text)
            if stated:
                age = int(stated.group(1))
        if not sex:
            named = _NOTE_SEX.search(note_text)
            if named:
                if age is None:
                    age = int(named.group(1))
                sex = _normalise_sex(named.group(2))

    if age is not None and not 0 <= age <= 120:
        age = None
    return age, sex


def build_target(payload: dict | None, *, patient_age: int | None = None,
                 patient_sex: str = "") -> ClinicalTarget | None:
    """The `ClinicalTarget` a reply describes, or None when it names no condition.

    None is a real answer: a record whose note names no condition has no search to run,
    and running one anyway would be searching for the patient's demographics. The caller
    draws no comparison, which is what "we have nothing to compare this against" looks
    like when it is honest.

    The demographics are the caller's rather than the model's: the reply names a condition,
    and age and sex are read from the record by `patient_demographics`.
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
    return ClinicalTarget(condition=condition, procedure=procedure, anchors=tuple(anchors),
                          patient_age=patient_age, patient_sex=patient_sex)


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
#: The endpoint refused for capacity, and it refused every attempt this module makes.
#:
#: A fourth reason rather than an exception, because the two callers below must DEGRADE on
#: it: a briefing whose target extraction was throttled still has a chart with a diagnosis
#: in its headers, and a comparison whose extraction was throttled still has an estimate to
#: show alone. Letting the 429 propagate turned a capacity blip into "the agent failed to
#: answer" (2026-11-08), which is the one outcome a clinician cannot act on.
REPLY_RATE_LIMITED = "rate_limited"

#: How many attempts a structured extraction gets before the step degrades.
#:
#: Two, and the number is a latency decision rather than a reliability one. The step is an
#: enrichment: `derive_target` can fall back to the chart's own headers and
#: `extract_comparators` can fall back to publishing the studies it did read, and both of
#: those are waiting whether or not the model answers. So a refusal that survives one
#: backoff is treated as "this step is not available right now" and the turn moves on.
#:
#: Measured 2026-11-08 on `agent-00266-kbs`: the four-attempt ceiling spent 34.71 s of a
#: 67.62 s `/ehr` (`90000069`) and 60.65 s of an 88.66 s one (`90000020`) — every attempt
#: refused, nothing produced, and the fallback published in the end either way.
STRUCTURED_MAX_ATTEMPTS = 2

#: The generation parameters a structured reply is asked for under.
#:
#: The extraction effort, because the two questions are extraction, not reasoning — "which
#: condition is this admission for" and "which abstract states a 30-day rate" are answered
#: by reading, and the tokens spent deliberating them are spent before the first character
#: of JSON. Measured on `4ed74ac0` (2026-11-08): 3928 and 3933 reasoning tokens for 164
#: and 159 tokens of output, 36.8 s of an 88.7 s turn, and BOTH replies failed to parse —
#: the allowance went to thoughts and the answer arrived truncated.
#:
#: Stated as an EFFORT rather than as a provider's level. This once read
#: `{"thinking_level": "minimal", "response_mime_type": "application/json"}`, which was the
#: pin's spelling spoken by a module that was never told which model would serve the call.
#: The pin accepts `minimal`; `gemini-3.8-flash` and `gemini-3.1-pro` refuse it with a 400
#: INVALID_ARGUMENT, so on 2026-11-13 (Sprint 04 pre-flight, `sel_t2_01`) every structured
#: call made on a tool-bound client failed, and the failure surfaced as `agent_failed` with
#: no trace id. The level a provider maps this effort onto is a property of the model, so it
#: is answered in `engine/gateway.py` and nowhere else.
#:
#: `json_mode` makes the reply a JSON object by contract rather than by instruction. Which
#: wire parameter expresses that — a response mime type, an OpenAI-style `response_format`,
#: or nothing at all but the prompt — is a provider fact for the same reason.
#:
#: Set per call rather than at construction: the client is built once and shared with the
#: formatting turn, which must keep its thinking.
_STRUCTURED_CALL_PARAMS: dict[str, Any] = {
    "effort": gateway.EFFORT_EXTRACTION,
    "json_mode": True,
}


async def _structured_reply(
    llm, system: str, human: str, config: dict | None,
) -> tuple[dict | None, str]:
    """One model reply as (JSON object, "") or (None, why it produced nothing).

    The reply is asked for as JSON because everything downstream of it is checked rather
    than read: a condition becomes a query string, and a row becomes a claim that has to
    match the abstract it came from. Prose in either place would be unfalsifiable.

    The call carries `_STRUCTURED_CALL_PARAMS`: the extraction effort and a JSON reply.
    Both are latency and reliability, not fidelity — the model is being asked to transcribe
    what it read, and the two failure modes observed in production were a reply that spent
    its allowance on deliberation (`4ed74ac0`) and one that arrived as prose.

    A capacity refusal is retried ONCE (`STRUCTURED_MAX_ATTEMPTS`) and, exhausted, reported
    as `REPLY_RATE_LIMITED` rather than raised: both callers have a defined answer for a
    step that could not run, and neither has one for an exception.
    """
    messages = [SystemMessage(content=system), HumanMessage(content=human)]
    bound = _bounded_call(llm, _STRUCTURED_CALL_PARAMS)
    try:
        response = await _retrying(
            lambda: bound.ainvoke(messages, config=config or {}),
            what="the literature extraction",
            attempts=STRUCTURED_MAX_ATTEMPTS,
        )
    except Exception as exc:  # noqa: BLE001 - re-raised below unless it is capacity
        if not _rate_limited(exc):
            raise
        logger.warning(
            "literature extraction: the endpoint was rate limited on every attempt; "
            "degrading this step instead of failing the turn (%s)",
            str(exc)[:160],
        )
        return None, REPLY_RATE_LIMITED
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
    REPLY_RATE_LIMITED: "was refused for capacity on every attempt",
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


async def derive_target(llm, sections: dict | None, *, config: dict | None = None,
                        patient_age: int | None = None,
                        patient_sex: str = "") -> TargetResult:
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

    The model goes FIRST even where the headers state a diagnosis, and the reason is the
    index procedure (2026-11-08). `fallback_target_from_sections` names a condition and
    nothing else, and `candidate_pass` searches `[condition, *anchors]`: without the
    procedure the query loses "anterior cervical discectomy and fusion" and every synonym
    the literature tool builds from it, which is what filled the 20-abstract pool on
    `90000069`. Reading the header instead saved 12-17 s and removed the comparison the
    `/ehr` canvas exists to draw. The cost in `_extract_once` was OUTPUT generation — a
    written refusal per abstract — not input prefill, so the read that had to get shorter
    is the reply, and the target extraction is not the place to spend the saving.

    `patient_age` and `patient_sex` are stamped on whichever target survives, whatever
    produced it: the demographics are a fact about the patient, not about the reply, so
    the salvage path — which returns a target built with no knowledge of them — carries
    them too.
    """
    def _with_demographics(target: ClinicalTarget) -> ClinicalTarget:
        return replace(target, patient_age=patient_age, patient_sex=patient_sex)

    payload, why = await _structured_reply(
        llm,
        prompts.LITERATURE_TARGET_PROMPT,
        prompts.literature_target_input(sections),
        config,
    )
    if payload is not None:
        target = build_target(payload, patient_age=patient_age, patient_sex=patient_sex)
        if target is not None:
            return TargetResult(target=target)
        why = "the reply named no condition"

    # Read the chart's own headers BEFORE paying for a second model call. The second call
    # exists for one failure — a provider filter withholding a reply that the plainer
    # wording can still elicit — and a malformed reply from a model that was not filtered is
    # not that failure. On a note whose diagnosis block is right there, the retry was twenty
    # seconds spent to produce what `fallback_target_from_sections` reads for nothing, on
    # every `/ehr` (2026-11-08).
    if why != REPLY_WITHHELD:
        salvaged = fallback_target_from_sections(sections)
        if salvaged is not None:
            return TargetResult(
                target=_with_demographics(salvaged),
                reason=(
                    f"the model produced no condition ({reply_problem(why)}); "
                    "the condition was read from the chart's own headers"
                ),
                source="chart_headers",
            )

    retry, retry_why = await _structured_reply(
        llm,
        prompts.LITERATURE_TARGET_FALLBACK_PROMPT,
        prompts.literature_target_input(sections),
        config,
    )
    if retry is not None:
        target = build_target(retry, patient_age=patient_age, patient_sex=patient_sex)
        if target is not None:
            return TargetResult(target=target, source="model_short_prompt")

    salvaged = fallback_target_from_sections(sections)
    if salvaged is not None:
        return TargetResult(
            target=_with_demographics(salvaged),
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
    # The population check the prompt asks the model to make and this layer makes again:
    # a rate reported for an age-restricted subgroup the index patient is not in is not
    # this patient's comparison, whatever the condition says. Deterministic because the
    # failure is a number in a cell beside another number, and a judgement that can be
    # read off the row costs nothing to apply twice.
    if not _cohort_matches_patient_age(
        row.get("cohort_summary"), row.get("rate_sentence"), target.patient_age
    ):
        return None, (
            f"pmid {pmid} reports its rate for an age-restricted population that "
            f"excludes this patient (age {target.patient_age})"
        )
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
        # A zero stated in words is a real result and publishes no numeral to match.
        # `38504222` reports that no patient was readmitted within 30 days; the check
        # called that "not the figure its own sentence states", which is true and beside
        # the point — the sentence states a zero, in the only way it can (2026-11-08).
        # And a figure inside a clause that carries one sign for both numbers is stated.
        if not (figures[0] in {"0", "0.0"} and _VERBAL_ZERO.search(sentence)) \
                and not _shares_a_percent_sign(figures, sentence):
            return None, f"pmid {pmid} rate {rate} is not the figure its own sentence states"
    # When the sentence states more than one horizon, the published number must be the one
    # the 30-day horizon owns. Only then: a sentence carrying ONE horizon has nothing to
    # disambiguate, and a span rate ("1.0% to 1.4% within 30 days") is one horizon with two
    # figures — reading its second figure as "the 30-day one" refused a correct row on
    # 2026-11-08 (`test_a_span_rate_is_kept_and_marked_as_a_range`).
    owned = _thirty_day_rate(sentence) if _OTHER_HORIZON.search(sentence) else None
    if owned is not None and figures[0] != owned:
        return None, (
            f"pmid {pmid} rate {rate} is not the figure its 30-day horizon states "
            f"({owned}%)"
        )

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
                # The quote goes with the refusal, because the reason alone cannot say
                # whether the model quoted a bad sentence or the check misread a good one.
                # `31528438` and `33049410` were refused on 2026-11-08 and neither refusal
                # was answerable from the log without the sentence it referred to.
                "sentence": " ".join(
                    str((row or {}).get("rate_sentence") or "").split())[:300],
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
    #: Which of the tool's two passes the caller asked for, as the tool-call contract
    #: spells it. Since Stage 9A the tool runs the horizon-pinned pass and broadens it on
    #: its own, so this records the request and not the retrieval: read `query` for what
    #: was actually asked of PubMed, and `pool_size` for what came back.
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
    #: How many adult candidates there were before the extraction cap.
    #:
    #: `aligned` used to be the only number reported beside `candidates`, and it is the
    #: CAP (`pool[:MAX_EXTRACTION_CANDIDATES]`), not a match count — which made "aligned=10,
    #: excluded=0" read as "ten matched" when it meant "ten were read and nothing was
    #: filtered". Reported separately so a thin pool and a capped read cannot be confused
    #: again (`cr`, 2026-11-08).
    pool_size: int = 0
    #: Whether this pass's records were merged with another's before the model read them.
    #: Two passes, one extraction: without this a report showing two passes and one call
    #: would read as a lost pass.
    pooled: bool = False
    #: What the pass came to — `verified`, `verified_empty` or `extraction_degraded`.
    #: Carried so the surface can tell a literature that offered no comparable cohort
    #: from an extraction that never read the abstracts.
    outcome: str = routing.OUTCOME_VERIFIED_EMPTY


def benchmark_outcome(passes: list["ComparatorPass"]) -> str:
    """What the turn's benchmark retrieval came to, across every pass.

    One row verified is `verified`, whatever happened in a pass that failed — the table
    has something to draw. With no row verified, a single failed pass makes the whole turn
    `extraction_degraded`, because a reader told "the abstracts stated no rate" would be
    reading a statement about the literature that this turn never established.
    """
    if any(pass_.outcome == routing.OUTCOME_VERIFIED for pass_ in passes):
        return routing.OUTCOME_VERIFIED
    if any(pass_.outcome == routing.OUTCOME_EXTRACTION_DEGRADED for pass_ in passes):
        return routing.OUTCOME_EXTRACTION_DEGRADED
    return routing.OUTCOME_VERIFIED_EMPTY


@dataclass(frozen=True)
class ComparatorResult:
    """Verified rows, every refusal with its reason, and the passes that produced them.

    `pool` is the ranked adult candidate array the extraction turn was given — the same
    records, in the same order, that `CandidatePass.aligned` holds. It travels out of here
    because a turn that verified NOTHING is not a turn that read nothing: the abstracts
    were retrieved, filtered and read, and when no row survived there is still peer-reviewed
    evidence about this patient's condition that the briefing can summarise and cite. The
    caller decides how much of it to publish; this layer only says what was read.
    """

    rows: list[dict]
    rejected: list[dict]
    passes: list[ComparatorPass]
    pool: list[dict] = field(default_factory=list)


async def extract_comparators(
    llm, target: ClinicalTarget, *, search, config: dict | None = None,
    max_rows: int = MAX_COMPARATORS,
) -> ComparatorResult:
    """Comparator rows for this patient: ONE retrieval, ONE extraction.

    `search(terms, wide)` is the caller's retrieval callable — the tool function, or a fake
    in a test — and it is passed in rather than imported so that this module keeps no
    dependency on the tool surface and the policy can be exercised without NCBI.

    ONE retrieval call and ONE model call. The two passes are not gone, they moved: the
    strict horizon-pinned query and the broadened fallback both run inside
    `services.mcp.tools.literature.search_readmission_benchmarks`, so a turn still shows
    `Searching the Literature` once and the broadening it performs is invisible to the
    workflow. Measuring it here cost the reader time — 2026-11-02, two extraction gaps and
    25.6s — and measuring it there costs nothing.

    Empty stays a real answer at every stage: a query that returns nothing, a pool that is
    all another population, and a pool that is all a different condition are three
    different facts, and each is reported rather than collapsed into "no rows".
    """
    candidates = await candidate_pass(search, target)
    verified, refused_rows, outcome = await _extract_pass(
        llm, target, candidates.aligned, config)
    rejected = [*candidates.refused, *refused_rows]
    return ComparatorResult(
        rows=_by_evidence_scale(verified)[:max_rows], rejected=rejected,
        passes=[candidates.report(verified=len(verified),
                                  rejected=candidates.refused, outcome=outcome)],
        # The array the extraction read, not the rows it produced. A caller that publishes
        # the survivors of a failed verification is publishing what this patient's
        # literature says about readmission, which is a different claim from a 30-day rate
        # and is labelled as one wherever it is drawn.
        pool=[article for article in candidates.aligned if isinstance(article, dict)],
    )


def _by_evidence_scale(rows: list[dict] | None) -> list[dict]:
    """Verified rows, the strongest evidence first.

    The table draws at most two rows, so which two is a clinical statement: a numeric rate
    from a registry is a benchmark, and "no patient was readmitted" from a 33-patient series
    is a finding. When the model proposes both, the order it proposes them in is the order
    it happened to read them.

    Sorted on three facts the row now carries from the pool it came out of — whether its
    abstract states a numeric rate, the cohort size, and the retrieval score — with the
    model's order kept as the tie-breaker so a run stays reproducible. A row with none of
    them (a verified row whose pool article is missing) sorts last rather than being
    dropped: it passed verification, so it is a real row.
    """
    return sorted(
        [row for row in rows or [] if isinstance(row, dict)],
        key=lambda row: (
            not row.get("has_readmission_rate_sentence"),
            -(int(row.get("sample_size_n") or 0)),
            -float(row.get("rank_score") or 0.0),
        ),
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
    #: The scope the caller asked for. The tool decides which clause to ask; see
    #: `ComparatorPass.wide`.
    wide: bool
    candidates: int
    aligned: tuple[dict, ...]
    refused: tuple[dict, ...]
    excluded: int
    #: How many adult candidates there were before the extraction cap. `aligned` is the
    #: cap, not a match count, and reporting only that made "aligned=10, excluded=0" read
    #: as "ten matched" when it meant "ten were read, nothing was filtered" (`cr`,
    #: 2026-11-08).
    pool_size: int = 0

    def report(self, *, verified: int, rejected: tuple[dict, ...],
               outcome: str) -> ComparatorPass:
        return ComparatorPass(
            query=self.query, wide=self.wide, candidates=self.candidates,
            aligned=len(self.aligned), verified=verified,
            rejected=tuple(rejected), excluded=self.excluded,
            pool_size=self.pool_size, outcome=outcome,
        )


#: How many candidate abstracts one extraction call is given.
#:
#: Not a relevance cut and not a local ranking — the pool arrives in PubMed's order and the
#: model reads what it is given. It is an INPUT-SIZE bound, and it exists because the
#: extraction reply grows one row per abstract: twenty-five abstracts is a reply long enough
#: to be cut off by the output ceiling, and a cut-off reply is discarded whole. Ten fills
#: the table twice over — the surface draws at most two rows — and a ten-abstract reply fits
#: comfortably inside `COMPARATOR_REPLY_TOKENS` now that it carries rows alone.
#:
#: It was briefly cut to six (2026-11-08) on the theory that a smaller input makes a smaller
#: reply. The reply did shrink, but for the reason it was going to shrink anyway — the
#: `rejected` essays are gone — and six cost two cohorts that had already been read and
#: verified correctly on `90000069`, because PubMed's ranking does not put them first. The
#: cost is OUTPUT generation, not input prefill; the input bound is not where the time is.
#:
#: What is refused is reported, not silent: `ComparatorPass.candidates` still states what
#: PubMed returned, so a table built from the top ten of twenty-five is visible as such.
MAX_EXTRACTION_CANDIDATES = 5

#: How many of those slots are reserved for abstracts that actually state a rate.
#:
#: The whole five, so the reservation is stated rather than implied: an abstract with no
#: readmission figure cannot produce a row, and one that can is worth a slot ahead of one
#: that cannot. When fewer than five qualify the remainder fills from the ranked pool.
_RATE_BEARING_SLOTS = 5


def _comparator_relevance(article: dict, target: ClinicalTarget) -> tuple:
    """How likely this abstract is to state a usable 30-day rate for this admission.

    Three facts, in the order a clinician would apply them, and no clinical judgement:
    does it state a 30-day rate at all; is it about the operation this admission had;
    is it about the diagnosis. Every term is a substring of text the record already
    publishes, so two runs on the same payload rank identically and PubMed's own order
    survives as the tie-breaker.
    """
    text = f"{article.get('title') or ''} {article.get('abstract') or ''}".lower()
    states_rate = (bool(_THIRTY_DAY.search(text))
                   and bool(_PERCENT_IN_TEXT.search(text)))
    procedure = str(target.procedure or "").strip().lower()
    condition = str(target.condition or "").strip().lower()
    return (
        not states_rate,
        not (procedure and procedure in text),
        not (condition and condition in text),
    )


def _articles_of(response: dict | None) -> list[dict]:
    """The records a `search_literature` response carries, as a fresh list.

    A named reader rather than an inline comprehension because the pool is now built from
    two responses and the two must be read the same way; a second spelling of "the
    articles, if it has any" is how one of them ends up accepting a malformed element.
    """
    return [article for article in (response or {}).get("articles") or []
            if isinstance(article, dict)]


async def candidate_pass(search, target: ClinicalTarget) -> CandidatePass:
    """The one readmission query this turn runs, and what it left to read.

    The clinical half is the condition OR its anchors, which is what the tool builds from
    a query plus its synonyms. ONE query: retrieval, broadening, the population filter and
    the ranking all live in `services.mcp.tools.literature.search_readmission_benchmarks`
    as of Stage 9A, because they were split between that module and this one and the two
    halves drifted — this side ranked on three booleans with no notion of cohort size, so a
    33-patient series that readmitted nobody sorted above a 1,140-patient cohort reporting
    3.4% (`90000069`, 2026-11-09).

    What is left here is the part only this side can do: choosing how many abstracts to put
    in one prompt, and putting the ones that can produce a row first.

    RATE-BEARING FIRST, THEN THE REST. `MAX_EXTRACTION_CANDIDATES` used to be ten records
    in the ranker's order, which on `90000090` meant one rate-bearing abstract and nine
    that could not produce a row — and the model proposed exactly one column, the wrong one.
    A prompt is a budget, and spending it on abstracts that cannot yield a figure is how a
    matrix stays empty over a pool that contained two usable cohorts.
    """
    result = await search([target.condition, *target.anchors], True)
    result = result if isinstance(result, dict) else {}
    query = str(result.get("query") or "")
    articles = _articles_of(result)
    pool, refused = adult_cohorts(articles, target)
    # The tool returns its pool already ranked, with the score and the reason on each
    # record. A pool without them was injected by a test double, and then this side ranks
    # for itself — the same question, asked of the mock, so a fixture keeps its meaning.
    if any("rank_score" in article for article in pool):
        ranked = sorted(pool, key=lambda article: -float(article.get("rank_score") or 0.0))
    else:
        ranked = sorted(pool, key=lambda article: _comparator_relevance(article, target))
    rate_bearing = [article for article in ranked
                    if article.get("has_readmission_rate_sentence")]
    selected = (rate_bearing[:_RATE_BEARING_SLOTS]
                + [article for article in ranked if article not in rate_bearing])
    selected = selected[:MAX_EXTRACTION_CANDIDATES]
    logger.info(
        "comparators: reading %d abstract(s), %d of them rate-bearing: %s",
        len(selected), len([a for a in selected if a.get("has_readmission_rate_sentence")]),
        ", ".join(
            f"{article.get('pmid')}(score={article.get('rank_score')},"
            f"n={article.get('sample_size_n')},"
            f"rate={'y' if article.get('has_readmission_rate_sentence') else 'n'})"
            for article in selected),
    )
    return CandidatePass(
        query=query, wide=True, candidates=len(articles),
        aligned=tuple(selected),
        refused=tuple(refused),
        pool_size=len(pool),
        excluded=len(refused),
    )



#: The output ceiling for one comparator extraction, in tokens.
#:
#: Derived from the client's own default (`GEMINI_MAX_OUTPUT_TOKENS`) rather than restated,
#: because the two numbers were 4096 twice and the second one was the lower of the pair the
#: moment the default moved. A ceiling here that is smaller than the client's would be a
#: cap nobody wrote down as a decision: the bind succeeds, the model truncates, and the
#: failure reads as an unparseable reply. It is enough now that the pool is bounded at
#: `MAX_EXTRACTION_CANDIDATES`. It was doubled once when one call could be handed
#: twenty-five abstracts: measured 2026-11-06 on `90000069`, the reply came back
#: `finish_reason=MAX_TOKENS`, `_extract_json_object` found no object, and a table that had
#: two verifiable cohorts in it drew nothing. Bounding the input fixes the cause; the retry
#: over halves stays for the residue.
COMPARATOR_REPLY_TOKENS = mcp_config.GEMINI_MAX_OUTPUT_TOKENS

#: How many abstracts one extraction call is given before the pool is split.
#:
#: Not a relevance cut and not an ordering — the pool is passed whole, and a split is the
#: same pool read in two halves only when the first read failed. Twenty records is what one
#: query returns; a split is used only to keep a truncated reply from deleting the pass.
COMPARATOR_POOL_SPLIT_THRESHOLD = 1


def _bounded_call(llm, params: dict[str, Any]):
    """The model with one call's generation parameters bound.

    The parameters are semantic — an effort level and "answer in JSON" — and the adapter
    re-expresses them for whichever model is actually serving the call. That indirection is
    the point: a candidate that refuses a level, or spells "answer in JSON" differently,
    gets its own spelling without this module learning a single provider fact.

    A client with no `bind` is returned unchanged, which is what keeps a test double usable
    here without implementing LangChain's binding surface.
    """
    return gateway.bind(llm, params)


def _bounded(llm, tokens: int):
    """The model with an explicit output ceiling, when it can carry one.

    `max_output_tokens` is the one generation parameter every provider this deployment can
    reach spells the same way, so it goes through the same binder for consistency rather
    than because it needs translating.
    """
    return gateway.bind(llm, {"max_output_tokens": tokens})


def _reply_rows(payload: dict, pool: list[dict], target: ClinicalTarget,
                rejected_rows: list[dict]) -> list[dict]:
    """The rows one parsed reply verified, with its own refusals appended to the caller's.

    The extraction PROPOSES and `verify_comparators` DISPOSES, exactly as before this was
    split: a row survives only if its sentence appears in the abstract it names. The one
    difference is that the refusals are collected across calls, because a retry over halves
    reads the same pool and a reader is owed every reason it was refused.
    """
    verified, refused = verify_comparators(
        payload.get("comparators"), pool, target)
    # The retrieval's own judgement of the record, carried onto the row it produced: the
    # cohort size the abstract stated, the score it was ranked by, and whether it states a
    # numeric rate at all. The table draws two rows and the section quotes them, so WHICH
    # two survive is a clinical statement, and it cannot be made from a row that knows only
    # the rate it reports (`90000069`: a 33-patient series at 0% was the only row).
    by_pmid = {str(article.get("pmid") or "").strip(): article
               for article in pool or [] if isinstance(article, dict)}
    for row in verified:
        article = by_pmid.get(str(row.get("pmid") or "").strip())
        if not isinstance(article, dict):
            continue
        for field in ("sample_size_n", "rank_score", "has_readmission_rate_sentence"):
            if field in article and field not in row:
                row[field] = article[field]
    # `rejected` is OPTIONAL and empty by contract: the prompt stopped asking the model for
    # a written refusal per abstract on 2026-11-08, because the essay was most of the reply
    # and a long reply is one that gets cut off. The read stays for two reasons — a reply
    # from a model that still writes one is not an error, and the refusal list is where the
    # verifier's own reasons accumulate, which is what a reader is owed either way.
    for item in payload.get("rejected") or []:
        if isinstance(item, dict):
            rejected_rows.append({
                "pmid": str(item.get("pmid") or "").strip(),
                "reason": " ".join(str(item.get("reason") or "").split())[:200],
            })
    rejected_rows.extend(refused)
    # What the model proposed and what survived, in one line.
    #
    # This is the line whose absence made an empty `ComparativeRiskMatrix` undiagnosable
    # (`826289ea`, 2026-11-08). Twenty-one abstracts came back, extraction ran for 8.6 s,
    # and the table drew nothing — and nothing anywhere recorded whether the model had
    # answered `{"comparators": []}` or proposed rows the verifier refused. The two call
    # for opposite responses, and the reply that would have told them apart is discarded by
    # this function. Now it is written down before it is discarded.
    proposed = payload.get("comparators")
    logger.info(
        "comparator extraction: %d proposed, %d verified, %d refused, over %d abstract(s)",
        len(proposed) if isinstance(proposed, list) else 0,
        len(verified), len(refused), len(pool),
    )
    for rejection in refused:
        logger.info(
            "comparator extraction: refused %s — %s | quoted=%r",
            rejection.get("pmid") or "(no pmid)", rejection.get("reason") or "",
            rejection.get("sentence") or "",
        )
    return verified


async def _extract_pass(
    llm, target: ClinicalTarget, aligned: tuple[dict, ...] | list[dict],
    config: dict | None,
) -> tuple[list[dict], list[dict], str]:
    """Read `aligned` and return `(rows, rejections, outcome)`.

    A withheld, truncated or malformed reply is a rejection with its reason AND the
    `extraction_degraded` outcome, never an empty success: "the model declined" and "the
    literature stated nothing" are different statements to the reader, and a turn that
    cannot tell them apart tells the reader the second when it means the first.

    One retry, over two halves of the same pool. The failure this exists for is a reply cut
    off by the output ceiling, which is a function of how much there is to say: half the
    abstracts is half the reply, and each half is verified against its own records exactly
    as the whole was. A half that fails too leaves the pass degraded — reported, not
    collapsed.
    """
    pool = [article for article in aligned or [] if isinstance(article, dict)]
    if not pool:
        # Nothing was retrieved, so nothing was read. This is not a failed extraction: the
        # query is what came back empty, and the pass reports it as such.
        return [], [], routing.OUTCOME_VERIFIED_EMPTY
    rows, rejected, why = await _extract_once(llm, target, pool, config)
    if not why:
        return rows, rejected, (routing.OUTCOME_VERIFIED if rows
                                else routing.OUTCOME_VERIFIED_EMPTY)
    # The retry halves the input so a reply that was CUT OFF becomes two replies that fit.
    # That is the only failure it can repair, so it is the only failure it is spent on: a
    # withheld reply is the provider's filter acting on the abstracts, and asking twice more
    # in halves buys two more blocked calls and another twenty seconds of the reader's
    # time (2026-11-08).
    if len(pool) > COMPARATOR_POOL_SPLIT_THRESHOLD and why == REPLY_NOT_JSON:
        middle = len(pool) // 2
        halves = (pool[:middle], pool[middle:])
        merged: list[dict] = []
        every_rejection = list(rejected)
        any_read = False
        for half in halves:
            if not half:
                continue
            half_rows, half_rejected, half_why = await _extract_once(
                llm, target, half, config)
            every_rejection.extend(half_rejected)
            merged.extend(half_rows)
            any_read = any_read or not half_why
        if any_read:
            seen: set[str] = set()
            unique = [row for row in merged
                      if not (row["pmid"] in seen or seen.add(row["pmid"]))]
            return (unique, every_rejection,
                    routing.OUTCOME_VERIFIED if unique
                    else routing.OUTCOME_VERIFIED_EMPTY)
        return [], every_rejection, routing.OUTCOME_EXTRACTION_DEGRADED
    return [], rejected, routing.OUTCOME_EXTRACTION_DEGRADED


async def _extract_once(
    llm, target: ClinicalTarget, pool: list[dict], config: dict | None,
) -> tuple[list[dict], list[dict], str]:
    """One extraction call. Returns `(rows, rejections, why_it_produced_nothing)`.

    `why` is empty on a reply that parsed, whatever it contained: "the model read these
    abstracts and found no cohort" is a successful pass with no rows, and conflating it
    with a failure is what makes a table say the literature offered nothing when the truth
    is that extraction never ran.
    """
    payload, why = await _structured_reply(
        _bounded(llm, COMPARATOR_REPLY_TOKENS),
        prompts.LITERATURE_COMPARATOR_PROMPT,
        prompts.literature_comparator_input(target, pool),
        config,
    )
    if payload is None:
        return [], [{"pmid": "", "reason": f"the extraction reply {reply_problem(why)}"}], why
    rejected: list[dict] = []
    rows = _reply_rows(payload, pool, target, rejected)
    return rows, rejected, ""

