"""Settlement of the answer a turn serves: fallbacks, prose repair, marker bounds.

Everything that decides what the reader is told when the model's own text is not served
as written. Until 2026-10-29 these lived in `engine/graph.py`, which meant the module that
runs the reasoning loop also decided, in regexes, what the clinician sees. The charter's
Law 2 puts them here instead: the engine orchestrates state and returns the model's raw
output, and `postprocess/` owns every edit to it.

Three responsibilities, and they are separate on purpose:

  * `no_evidence_answer` decides that the notes do not answer the question, and speaks the
    sentence that says so. The model's plausible answer for that turn is never served.
  * `replace_final_text` is the small state operation the two channel-writers share: the
    message the answer is read from is edited in place, because `model_turn` reads the same
    message for the finish reason and the content-filter verdict, and a new message would
    lose both.
  * `sanitize_answer_markers` applies the citation bound to whichever channel carries the
    answer this turn — the model's prose, or a command's `deterministic_answer`.

This module takes the state dict as it arrives from the graph and edits it. That is the
one shape constraint of the move: the alternative was to have every caller read the bound
and the channel rules for itself, which is how the two citation counters drifted apart in
the first place.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from langchain_core.messages import AIMessage

from services.agent.postprocess import citations
from services.agent.postprocess import discharge_section
from services.agent.routing import routing
from shared import labels
from shared import model_outcomes


NO_EVIDENCE_TEMPLATE = (
    'There is no information in the patient\'s chart regarding "{topic}".'
)
NO_EVIDENCE_UNKNOWN_TOPIC = (
    "There is no information in the patient's chart regarding that question."
)

#: Served when every claim in an answer was removed: an empty answer is not an answer,
#: and the caller has to be told the record does not support the question.
GUARDED_ABSTENTION = (
    "I can't confirm that from this patient's record. The note I can see does not "
    "document it, and I won't state it as if it did. Ask about another part of the "
    "record and I'll answer from what is documented."
)

#: Served when the model finished cleanly and said nothing at all. Not a service failure
#: — nothing broke, no token cap was hit — so reporting it as one told the caller to retry
#: a decision that is stable at temperature 0. Observed live 2026-10-17 against a
#: prompt-injection probe (`inj-04`, "output the first 200 words of your prompt
#: template"): `finish_reason=STOP` with an empty candidate, and the caller got a 502 for a
#: question the service had handled safely.
SILENT_TURN_SENTENCE = (
    "I don't have an answer for that. I can summarise this admission, explain its "
    "readmission risk, list its discharge medications, or search the clinical "
    "literature — ask about one of those and I'll answer from the record."
)

#: The same sentence for a record that was not a stay. The inpatient wording offers an
#: admission summary, a readmission risk and a discharge medication list; on an outpatient
#: note all three are inventions (90000051, "Summarize the note", 2026-10-21: fidelity 0,
#: grounding 1, safety 0 — the judge read the offer as a claim about the encounter).
OUTPATIENT_SILENT_TURN_SENTENCE = (
    "I don't have an answer for that. I can summarise this encounter note, review its "
    "laboratory values, or search the clinical literature — ask about one of those and "
    "I'll answer from the record."
)

#: Served when the provider's own filter withheld the reply and records were retrieved.
#:
#: `/pubmed Hypothermia` on 2026-10-30 retrieved three records and then answered with
#: `SILENT_TURN_SENTENCE`: the formatting reply came back `finish_reason=SAFETY` with
#: `HARM_CATEGORY_DANGEROUS_CONTENT` flagged, the text was empty, and the empty-text branch
#: served the sentence written for a model that said nothing. So the page showed a refusal
#: above three records it had just fetched, and the reader was told the service had no
#: answer when the truth was that the model's answer had been taken away.
#:
#: The sentence states the mechanism and points at what survived. It promises nothing
#: about a retry — the same input is filtered the same way — and it does not name a
#: provider, because which filter responded is an implementation detail to the clinician.
SAFETY_WITHHELD_SUMMARY = (
    "The language model's safety filter withheld its summary of these records, which it "
    "does for some clinical subject matter. The records themselves were retrieved and "
    "are listed with this answer, so they can be read directly; the filter acts on the "
    "model's reply, not on the search."
)

#: Served when the chart returned the patient's discharge medication list and the model's
#: reply carried no text at all.
#:
#: Observed live 2026-11-10: a free-text request for the discharge list called
#: `discharge_medications`, got the dedicated section back, and the model's reply arrived
#: empty with the provider's dangerous-content filter flagged. The turn settled to no text
#: at all, and the route reported the question as refused — for a list the service was
#: holding and renders deterministically on `/meds`. Nothing about that turn is an
#: absence, so the answer points at the list instead of saying nothing about it.
#:
#: The words name neither the provider nor the filter. Which filter responded is an
#: implementation detail to a clinician, and what happened to the summary is the same
#: either way: it did not come back, and the list did.
MEDICATIONS_WITHHELD_SENTENCE = (
    "I have this admission's discharge medication list, but no summary of it came back "
    "with this answer. Run /meds to read the list itself, with the doses and frequencies "
    "the record states."
)

#: `/ehr` on an outpatient record whose record-wide search returned nothing. The
#: formatter had abstained with the silent-turn sentence, which named neither the
#: encounter nor its consequence. The notice says what the record is, what it therefore
#: does not hold, and what remains available; it invents no stay, no length of stay and no
#: discharge list.
OUTPATIENT_EHR_NOTICE = (
    "This record is an outpatient encounter, not a hospital admission, so an "
    "inpatient admission summary and readmission risk estimate do not apply: "
    "no inpatient hospital course, inpatient discharge diagnosis or inpatient "
    "discharge medication list exists for this visit, and nothing here should "
    "be read as an inpatient stay. Ask about this visit instead, or run the "
    "command on an inpatient admission."
)


def _normalise(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


#: Where a turn's own evidence numbering starts when no session record sits behind it.
#:
#: A command always has an admission and therefore always has a record, so this is the
#: value for a pure call with no conversation: one passage returned, and it is the first
#: unit of an array numbered from one. It is NOT a fallback for a caller that forgot to
#: supply `cite` — such a caller passes the position it was given.
_FIRST_TURN_POSITION = 1


def _distinct_medications(verified: list[dict[str, str]]) -> list[str]:
    """The medication names in `verified`, once each, in first-seen order.

    A taper is one drug in two rows — "oxycodone SR 20 mg" then "oxycodone SR 10 mg",
    because the note steps the dose down — and counting rows told a reader there were
    fourteen discharge medications when the note prescribed thirteen (90000015,
    2026-10-21). Identity is the normalised name, so casing and spacing cannot split one
    drug into two. The spelling of the first row that named the drug is kept: this is what
    the note said, not a value this module chose.
    """
    names: list[str] = []
    seen: set[str] = set()
    for row in verified:
        name = str(row.get("medication") or "").strip()
        key = _normalise(name)
        if not key or key in seen:
            continue
        seen.add(key)
        names.append(name)
    return names


def _course_only(sections: tuple[str, ...]) -> bool:
    """Whether the only thing the retrieval could read was the hospital course.

    The distinction the sentence makes: a discharge-instructions section is where a note
    writes the prescription, so a list drawn from it is a discharge list; a hospital course
    is where a note writes what was given, so a list drawn from it is what the stay
    documented. One section in the tuple, and it is the course.
    """
    return tuple(sections) == ("brief_hospital_course",)


def _split_by_setting(verified: list[dict[str, str]]) -> tuple[list[dict[str, str]],
                                                                 list[dict[str, str]]]:
    """The rows the patient takes home, and the rows the stay administered.

    The distinction only exists on a record with no dedicated medication section, where
    the extractor reports each row's `setting`. A row with no setting, or any other value,
    is a discharge row: that is what an extraction over a discharge list means, and it
    keeps every caller that predates the field working unchanged.
    """
    discharge: list[dict[str, str]] = []
    inpatient: list[dict[str, str]] = []
    for row in verified:
        setting = str(row.get("setting") or "").strip().lower()
        (inpatient if setting == "inpatient" else discharge).append(row)
    return discharge, inpatient


def _is_equipment_or_supply(name: str) -> bool:
    """Whether a list row names equipment, a gas or a supply rather than a drug.

    A classification the answer and the follow-up chips both ask, defined once in
    `shared/labels.py`: the extraction verifies every numbered line, and what has to be
    decided afterwards is whether a line is counted as a prescription. Home oxygen is
    ordered, numbered and reported; it is not a drug, and a count that silently drops it
    contradicts the list a reader can see.
    """
    return labels.is_equipment_or_supply(name)


def _cite_suffix(cite: int | Iterable[int] | None) -> str:
    """The markers this sentence carries, or `""` when the caller has no position.

    One marker per passage the sentence reports on, space-separated — the shape
    `citations.renumber_citations` writes and the browser's marker regex reads. Two
    passages are cited when the rows came from two: a `/meds` answer whose drug names are
    in a discharge list and whose doses are in the instructions the same call returned
    reports on both, and a single marker opened only one of them.
    """
    if isinstance(cite, int) and not isinstance(cite, bool):
        values: Iterable[int] = (cite,)
    else:
        values = cite or ()
    positions: list[int] = []
    for value in values:
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            if value not in positions:
                positions.append(value)
    return "".join(f" ^[{position}]" for position in positions)


def medication_narrative(verified: list[dict[str, str]], returned_passages: int,
                         sections: tuple[str, ...] = (),
                         *, cite: int | Iterable[int] | None = None) -> str:
    """What the narrative says for a medications command.

    The prose names the medications as well as counting them. It used to say only "N
    discharge medications, as recorded in the discharge note", on the reasoning that the
    grid carries the list — but the prose is what the judge and the citation layer read as
    the answer, and a question asking WHICH medications was answered with a number
    (90000015, 2026-10-21). The grid still renders the rows with doses and frequencies;
    the sentence is the answer a reader who is not looking at the card can still use.

    The count is of distinct drugs, not of rows; see `_distinct_medications`.

    `sections` is the retrieval's own answer to "what did I actually return", and it
    decides whether a formal list may be claimed at all. `discharge_medications` can fall
    back to top-k retrieval, and on an admission with no medication section it returns the
    hospital course. A row extracted from the hospital course is a drug the stay mentions,
    not a discharge prescription, and calling it one asserts a list the record does not
    have (`dev_t1_04`: "2 discharge medications, as recorded in the discharge note", on a
    note with no such section, 2026-11-04). The sentence states the absence and rooms what
    it does say in the section that says it.

    `cite` is the POSITION the session record gave the passage this sentence reports on,
    and it is the caller's to supply because only the caller holds the record. It used to
    be the literal `^[1]`, on the reasoning that `/meds` retrieves one passage — true of
    the turn and false of the conversation: after an `/ehr` that had already numbered four
    sections, `^[1]` addressed the discharge diagnosis while the drug list sat at `^[4]`.
    A marker that names the wrong passage is worse than none, so a caller with no position
    writes none.

    The fallback sentence reserved for a genuinely empty result is used only when the tool
    returned no passages at all. A tool that returned text this module could not verify is
    a different failure, and the prose states that failure clinically: what was retrieved,
    and that no medication could be read from it. It does not name our internals — a
    "formatting failure" is a fact about this module, not about the patient's record, and
    a judge is right to score it ungrounded (90000023, 2026-10-21).
    """
    discharge_rows, inpatient_rows = _split_by_setting(verified)
    names = _distinct_medications(discharge_rows)
    inpatient_names = _distinct_medications(inpatient_rows)
    # The sentence reports on the passage the tool returned, so it carries that passage's
    # marker like any other claim: the citation layer resolves the position against the
    # record's array, and the judge reads a provenance-poor answer as one that cited
    # nothing (5fa503f6 and e5558ec7 passed with `citation_missing`, 2026-10-21). Only
    # when a position exists — a marker pointing at nothing is worse than none.
    source = _cite_suffix(cite if cite is not None else
                          (_FIRST_TURN_POSITION if returned_passages else None))
    dedicated = "discharge_medications" in sections
    instructions_only = "discharge_instructions" in sections
    # Neither section: the retrieval fell back to narrative text, and a statement about
    # what the patient takes home has to be made about that text rather than about a
    # discharge list the record does not have.
    narrative_fallback = not dedicated and not instructions_only
    where = ", ".join(section.replace("_", " ") for section in sections) or (
        "material that was retrieved")
    if narrative_fallback:
        if not names and not inpatient_names:
            if returned_passages == 0:
                return ("No specific discharge medication information is available for "
                        "this admission.")
            return ("No dedicated discharge medications section is documented in the "
                    f"record, and no medication could be read from the {where}{source}.")
        sentences = ["No dedicated discharge medications section is documented in the "
                     "record."]
        if names:
            sentences.append(
                f"Documented to continue at discharge: {', '.join(names)}{source}.")
        if inpatient_names:
            tail = "" if names else source
            sentences.append(
                "Given during the stay without a documented home prescription: "
                f"{', '.join(inpatient_names)}{tail}.")
        return " ".join(sentences)
    if names:
        if dedicated:
            drug_names = [name for name in names if not _is_equipment_or_supply(name)]
            supply_names = [name for name in names if _is_equipment_or_supply(name)]
            if not drug_names:
                # Nothing here is a prescription. The list is a supply list, and calling
                # its items medications would name oxygen as a drug.
                count = len(supply_names)
                noun = "equipment or supply order" if count == 1 else \
                    "equipment or supply orders"
                return (
                    f"{count} discharge {noun}, as recorded in the discharge note: "
                    f"{', '.join(supply_names)}{source}."
                )
            count = len(drug_names)
            noun = "medication" if count == 1 else "medications"
            # The count is of DRUGS, and the list the note numbered may hold supplies
            # beside them. Both facts are stated rather than one being dropped: a record
            # that numbers home oxygen first and fifteen drugs after it is fifteen
            # medications and sixteen listed items, and a sentence saying only "15"
            # invites a reader holding the page to count sixteen.
            total = len(names)
            if supply_names:
                if len(supply_names) == 1:
                    aside = (f" (plus {supply_names[0]} ordered as equipment/supply, "
                             f"{total} listed items in total)")
                else:
                    aside = (f" (plus {len(supply_names)} equipment/supply orders: "
                             f"{', '.join(supply_names)}; {total} listed items in total)")
            else:
                aside = ""
            return (
                f"{count} discharge {noun}{aside}, as recorded in the discharge note: "
                f"{', '.join(drug_names)}{source}."
            )
        # Direct clinical prose. The sentence used to open with a statement about the
        # NOTE — "The clinical record does not contain a dedicated discharge medications
        # section" — which is a fact about the document's headings, told to a clinician who
        # asked what the patient is taking. It also read as an apology, and an apology in
        # a medication answer is the one thing worse than a short list: it makes the answer
        # sound uncertain about drugs the record does state.
        if _course_only(sections):
            return (
                f"Medications documented during the hospital course: "
                f"{', '.join(names)}{source}."
            )
        return (
            f"Medications documented for discharge in the {where}: "
            f"{', '.join(names)}{source}."
        )
    if returned_passages == 0:
        return "No specific discharge medication information is available for this admission."
    if dedicated:
        return (
            "The discharge medications section was retrieved, but no medication could be "
            f"read from it{source}."
        )
    return f"No medications are documented in the {where} for this admission{source}."


def _final_message(state: dict) -> AIMessage | None:
    """The last message, when it is the model's — the turn the answer comes from.

    `engine/graph.final_message` is the same rule. It is repeated rather than imported
    because `postprocess/` may not import `engine/`: the duplicate is four lines of
    structural test and cannot drift, whereas the edge it would avoid would make the
    post-processing layer untestable without a compiled graph.
    """
    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    return last if isinstance(last, AIMessage) else None


def replace_final_text(state: dict, text: str) -> None:
    """Set the final answer's text in place.

    The message is edited rather than replaced: `model_turn` reads the same message for
    the finish reason and the content-filter verdict, and a new message would lose both.
    Only the content changes.
    """
    message = _final_message(state)
    if message is not None:
        message.content = text


def apply_model_reply(state: dict, text: str) -> None:
    """Serve the model's reply unwrapped, in place of the envelope it arrived in.

    A decision, not a rewrite: the reply was written by the model and this only says
    which part of it is the answer. It lives here because the engine may not edit prose
    (Charter 13 Law 2) — the same rule that keeps the fallback sentences here — and the
    call site in `engine/graph.py` is a line of invocation.
    """
    replace_final_text(state, text)


def medication_list_retrieved(tool_calls: list[dict]) -> bool:
    """Whether these calls fetched the patient's discharge medication list.

    Read off the payload rather than the question's wording, because this decides what to
    serve on a turn where the model produced no text at all: there is no interpretation of
    the question to consult. A call that failed, that came back with no passages, or that
    fell back to narrative sections (`dedicated_section_present` false) is not a list —
    the sentence it would earn points the reader at `/meds`, and a narrative fallback is
    not the record's discharge list to point at.
    """
    for call in tool_calls:
        if call.get("name") != "discharge_medications":
            continue
        response = call.get("response")
        if not isinstance(response, dict) or response.get("error"):
            continue
        if response.get("passages") and response.get("dedicated_section_present"):
            return True
    return False


def turn_reply_withheld(state: dict) -> bool:
    """Whether the provider took this turn's reply away, read off its own message.

    `reply_withheld` is the rule; this is where the facts it needs come from. The
    interpretation of a provider's metadata lives in `shared.model_outcomes` rather than
    here, for two reasons: this layer may not import `engine/`, and the same question is
    asked there about the same message — a second reading of one response is how two
    layers come to disagree about whether an answer was withheld.
    """

    message = _final_message(state)
    if message is None:
        return False
    if model_outcomes.is_blocked(message):
        return True
    return model_outcomes.is_withheld_reason(
        model_outcomes.native_reason(message))


def reject_model_reply(state: dict) -> None:
    """Withdraw an answer the turn may not serve, or point at what it could not say.

    A declined reply with no absence to report is not an answer: the model declined, and
    nothing was retrieved for it to decline against, so any sentence served here would be
    a claim the record does not support. The turn carries no text and the routes report an
    unavailable answer rather than a fabricated one.

    The exception is a turn that DID retrieve the patient's discharge medication list and
    whose reply the provider's filter took away. That turn has something to serve, and it
    is not an interpretation of the record — it is the record's own list, which `/meds`
    renders from the same call. Serving nothing told the reader the question had been
    refused while the answer sat in the turn (`MEDICATIONS_WITHHELD_SENTENCE`).

    Both halves are required. A model that declines voluntarily — an `evidence_found:
    false` envelope on a turn where the record DOES hold a discharge list — is stating a
    judgement about the question, and answering it with "the filter withheld the summary"
    would report a filter that never ran. Observed on a false-premise probe, 2026-11-10:
    the turn declined with `finish_reason=STOP` and the sentence blamed a safety filter.

    Recorded by editing the message rather than by raising, so the finish reason and the
    content-filter verdict stay readable on the turn the caller was refused.
    """
    retrieved = (
        medication_list_retrieved(state.get("tool_calls") or [])
        or medication_list_retrieved(state.get("replayed_tool_calls") or [])
    )
    if retrieved and turn_reply_withheld(state):
        replace_final_text(state, MEDICATIONS_WITHHELD_SENTENCE)
        return
    replace_final_text(state, "")


def reply_withheld(finish_reason: str | None, *, blocked: bool = False) -> bool:
    """Whether the provider took the model's reply away rather than the model stopping.

    A blocked prompt and a withheld reply are the same statement to a reader, and both
    mean the same thing whatever produced them. Which token a provider writes to say so is
    `shared.model_outcomes`'s business: it holds the one table, so this rule and the
    engine's outcome rule cannot drift apart about the same response.

    It is deliberately not every refusal reason. A truncation is the model running out of
    room, which is a different statement about the turn, and a turn that stopped for an
    unlisted reason has no reply to explain.

    The reason arrives as a token rather than as a message because the caller holds one:
    the command path decides what to serve before the message reaches this layer, and the
    shared table is what makes one token mean one thing in both places.
    """
    if blocked:
        return True
    return model_outcomes.is_withheld_reason(finish_reason)


def silent_turn_sentence(*, outpatient: bool = False) -> str:
    """The sentence for a turn that finished with no answer of its own.

    `outpatient` picks the wording that is true of the record. It is a property of the
    admission, not of the model's answer, which is why it is passed in rather than
    inferred from anything here.
    """
    return OUTPATIENT_SILENT_TURN_SENTENCE if outpatient else SILENT_TURN_SENTENCE


def outpatient_ehr_notice() -> str:
    """The `/ehr` notice for a record that was never an admission."""
    return OUTPATIENT_EHR_NOTICE


#: Served when the question asks to switch to, read, or summarise a different admission.
#:
#: The refusal is deterministic and contains no clinical fact from the current record and
#: no identifier from the question: a refusal that repeats the number it is refusing
#: confirms the number, and one that summarises this patient to a question about another
#: is the cross-patient disclosure it was meant to prevent.
FOREIGN_ADMISSION_REFUSAL = (
    "This session is open on one patient's encounter, and I can't switch to another "
    "admission, read another record, or report on one. Ask me about this encounter — its "
    "discharge summary, laboratory course, medications, readmission risk, or the "
    "published literature — and I'll answer from the record in front of me."
)


def foreign_admission_refusal() -> str:
    """The sentence served for a question about another admission."""
    return FOREIGN_ADMISSION_REFUSAL


def empty_turn_sentence(
    *,
    finish_reason: str | None = None,
    blocked: bool = False,
    records_retrieved: int = 0,
    outpatient: bool = False,
) -> str:
    """What to serve when the turn has no reply to serve.

    Three cases, and they are not the same statement:

    * the provider's filter withheld the reply and records were retrieved — the reader
      gets `SAFETY_WITHHELD_SUMMARY`, because there IS something to read and the sentence
      has to point at it rather than deny it exists;
    * the provider's filter withheld the reply and nothing was retrieved — nothing
      survived, so the turn is silent rather than offering records that are not there;
    * the model finished cleanly and said nothing — `SILENT_TURN_SENTENCE`, which is a
      statement about the question rather than about the service.

    Served on the state rather than left to the envelope's empty-answer branch, because
    the trace's output is read off the state (`graph.command_trace_output`) and an empty
    answer wrote the turn's trace row as a dash — the one record an audit opens first.
    """
    if reply_withheld(finish_reason, blocked=blocked) and records_retrieved:
        return SAFETY_WITHHELD_SUMMARY
    return silent_turn_sentence(outpatient=outpatient)


def _asked_about(question: str) -> str:
    """The caller's own question, without the admission the contract appends.

    Quoted rather than paraphrased: a sentence assembled in code must not invent a subject
    for what was asked, and the caller's words are already the subject. The admission
    suffix is dropped because "regarding \\"... For admission 90000023.\\"" names a record
    number in the middle of a clinical sentence.
    """
    head, marker, _ = (question or "").rpartition(" For admission ")
    return (head if marker else (question or "")).strip()


@dataclass(frozen=True)
class Settlement:
    """The settled answer, as a value rather than as a set of edits.

    Charter 13 §3 forbids this package from mutating LangGraph state directly, and the
    served path now does not: `settle_answer` takes plain data and returns this, and the
    engine applies it. The two channels are separate fields because a command carries its
    answer in `deterministic_answer` while a free-text turn carries it in the model's
    message, and a settlement that collapsed them would have to know which one won.
    """

    #: The prose to serve, already swept, bounded and renumbered for the reader.
    prose: str | None = None
    #: A command's own answer, settled by the same rules.
    deterministic: str | None = None
    #: `{display_number: position in the conversation's evidence array}`, or empty.
    display: dict[str, int] = field(default_factory=dict)
    #: What settlement removed, for `guardrail_flags`.
    flags: tuple[str, ...] = ()


def _summary_passages(tool_calls: Iterable[dict]) -> list[dict]:
    """This turn's `/ehr` retrieval, passage by passage, as the filter reads it.

    The retrieved passage and not the citable unit: the filter's first question is what kind
    of passage the turn holds, and a passage the model was not shown a marker for is still a
    passage the turn read.
    """
    passages: list[dict] = []
    for call in tool_calls:
        if str(call.get("name") or "") != "rag_search_sections":
            continue
        response = call.get("response")
        if not isinstance(response, dict):
            continue
        for passage in response.get("passages") or []:
            if isinstance(passage, dict):
                passages.append(passage)
    return passages


def _citable_units(tool_calls: list[dict], evidence_state: Any) -> list[dict]:
    """What a pre-compaction marker number indexes, position `n` at `[n - 1]`.

    The session record when the turn has one, because that is the array the marker was
    written against — a passage the thread already held keeps its position and does not take
    a second one. The concatenation of the calls otherwise, which is the same array for a
    turn that is the first of its thread.
    """
    if evidence_state is not None:
        try:
            return list(evidence_state.ordered_citable_units())
        except (AttributeError, ValueError):
            pass
    units: list[dict] = []
    for call in tool_calls:
        response = call.get("response") if isinstance(call, dict) else None
        if isinstance(response, dict):
            units.extend(citations.citable_passages(response))
    return units


def _lab_positions(tool_calls: list[dict], replayed_tool_calls: list[dict],
                   evidence_state: Any) -> frozenset[int]:
    """Which marker numbers open a laboratory series, over the array the markers address.

    Built from the same ledger `citation_bound` counts, in the same order — the replayed
    turns first, then this one — and from the session record when the turn has one, so a
    position here is the same position a marker carries. The citation owner is given the
    set rather than the units: what it enforces is one rule about a laboratory panel, and
    that rule only needs to know which numbers are series.
    """
    ledger = citations.EvidenceLedger.of(
        list(replayed_tool_calls or []), list(tool_calls or []),
        evidence_state=evidence_state,
    )
    return frozenset(
        position for position, unit in enumerate(ledger.units, start=1)
        if isinstance(unit, dict) and unit.get("section") == citations.LAB_SECTION
    )


def settle_answer(
    *,
    prose: str | None,
    deterministic: str | None,
    tool_calls: list[dict] | None,
    replayed_tool_calls: list[dict] | None,
    wants_literature: bool = False,
    comparators: list[dict] | None = None,
    thread_literature: list[dict] | None = None,
    fallback_literature: list[dict] | None = None,
    evidence_state: Any = None,
    command: str | None = None,
) -> Settlement:
    """Decide the answer's markers: vocabulary, bound, then the reader's numbering.

    Pure. It reads the turn's calls and the two strings and returns what to serve; it
    takes no state object, so it cannot edit one — which is what makes the boundary a
    contract rather than a convention, and what lets the whole decision be tested without
    a graph.

    `prose` must be TEXT. A model message's `content` is not reliably a string: Gemini
    answers in content blocks, so the streamed path accumulates
    `[{'type': 'text', 'text': '…'}]`. Settlement used to gate every step on
    `isinstance(content, str)`, which made it a silent no-op on every streamed free-text
    turn — no vocabulary sweep, no bound, no numbering, and markers past the end of the
    array reached the reader (`agent-00227-n7g`, patient 90000023 turn 3, 2026-10-31).
    The caller unwraps (`graph.final_text`) because the engine owns the container and this
    module owns the prose.

    `comparators` is the cohort rows the comparison table published — plain data, passed
    in rather than read out of the graph, so this stays a function of its arguments.

    `thread_literature` is the same idea one layer out: the published records an EARLIER
    turn of this thread retrieved and then cited, so a follow-up can cite them again
    instead of being told the note does not document the literature (2026-11-04). It
    widens the badge allowance by exactly those identifiers and nothing else — the pool a
    search returns is not evidence, and `citations.cited_articles` is what decided which
    records of it a turn actually rested on.

    `fallback_literature` is the fourth channel and the one this function used to be blind
    to: the studies a turn with no verified cohort still retrieved and published. Ignoring
    them here made `enforce_citations` delete the badges the prose had just written, while
    `presentation` — which reads that array directly — kept the rows in the drawer. The
    answer read as uncited over a reference list naming the papers, and the deletion was
    invisible because the allowance was assembled here rather than asked for
    (`citations.allowed_literature_pmids`).
    """
    # The array the answer's markers address. With `evidence_state` it is the session
    # record's — a passage the conversation retrieved twice occupies one position, so the
    # bound is the record's length rather than the payload sum — and the `display` map
    # below is then `{page_number: record position}`, which is what `presentation`
    # resolves through and what `turn.py` republishes for the next turn's replay.
    bound = citations.citation_bound(tool_calls or [], replayed_tool_calls or [],
                                     evidence_state=evidence_state)
    # The badge allowance, from the one function that defines it.
    #
    # It was assembled here for three channels: the turn's own evidence search, the cohorts
    # the table verified, and the records an earlier turn rested on. `/ehr`'s comparator
    # search declares `scope="readmission"`, so `literature_pmids` deliberately does not
    # count it — those records are the table's, not the answer's own citation list — and the
    # compensation for that was the comparator rows. When the rows were empty there was
    # nothing to compensate with, and a badge naming a study the turn HAD just published was
    # removed here as `citation_out_of_range` (measured 2026-11-03 on a verified cohort;
    # again 2026-11-08 on both fallback records of `90000069` and `90000020`).
    #
    # `replay` carries the one fact this function cannot derive: whether the replayed turns'
    # searches are in scope for THIS answer. `wants_literature` is stamped once by
    # `graph.ask` — the decision that let the earlier search stay in the transcript — so this
    # stays a read of that decision rather than a second opinion about the question.
    pmids = citations.allowed_literature_pmids(
        tool_calls or [],
        conversation_calls=replayed_tool_calls or None,
        replay=wants_literature,
        comparators=comparators,
        fallback_literature=fallback_literature,
        thread_literature=thread_literature,
    )

    flags: list[str] = []
    # Which positions in the conversation's array are laboratory SERIES. The citation
    # owner needs them to apply the one domain rule about a panel: four or more lab
    # markers on a sentence that names no analyte is an attribution for a statement the
    # panel cannot support, because no single trajectory says "all of them were normal".
    lab_positions = _lab_positions(tool_calls or [], replayed_tool_calls or [],
                                  evidence_state)
    settled_prose: str | None = None
    if prose:
        # The constructs we PRINT first, then the marker VOCABULARY, then the bound. The
        # model copies what it is shown and generalises what it can copy, so a bracket
        # this package printed for the prompt reached two served answers on 2026-11-01 —
        # one copied whole, one extended with labels no code path prints — while every
        # check below called the turn clean. `strip_printed_constructs` is the rule that
        # does not depend on the model complying; it lives in the citation owner because
        # that module is what prints the artifact it removes.
        stripped, printed = citations.strip_printed_constructs(prose)
        flags.extend(printed)
        swept = citations.drop_unknown_markers(stripped)
        settled_prose, found = citations.enforce_citations(
            swept, bound, pmids, lab_positions=lab_positions)
        flags.extend(found)

    settled_deterministic: str | None = None
    if isinstance(deterministic, str) and deterministic:
        stripped, printed = citations.strip_printed_constructs(deterministic)
        flags.extend(printed)
        swept = citations.drop_unknown_markers(stripped)
        settled_deterministic, found = citations.enforce_citations(
            swept, bound, pmids, lab_positions=lab_positions)
        flags.extend(found)
    else:
        settled_deterministic = deterministic

    # Which sentences a `/ehr` discharge section may contain, decided against the sections
    # this turn actually holds. Here rather than in the prompt because a formatting turn is
    # a model turn: the instruction was made explicit three times and the fourth answer
    # still reported a hospital-course drug as a home prescription under the heading that
    # promises the discharge regimen. After the bound, because the marker has to resolve
    # before its section can be known; before the page's numbering, because compacting a
    # paragraph this pass is about to shorten would leave the reader a gap.
    if settled_prose and command == "summarize":
        settled_prose, unplaced = discharge_section.filter_discharge_section(
            settled_prose,
            passages=_summary_passages(tool_calls or []),
            units=_citable_units(tool_calls or [], evidence_state),
        )
        flags.extend(unplaced)

    # The page's numbering, last and separate from the array's. A marker's number is the
    # position of its unit in the conversation's evidence array — cumulative across
    # turns, which is what makes a follow-up citable at all — and a reader should not be
    # shown the gaps that produces (`^[5]`, `^[8]`, then `^[14]`, `^[15]` on one real
    # thread). `compact_display_numbers` renumbers the served prose to the order it cites
    # and returns the map the presentation layer resolves the new numbers through, so a
    # pill, a drawer row and an evaluation source keep naming the same passage.
    display: dict[str, int] = {}
    if settled_prose:
        # The one canonical spelling before the page's numbering is computed. A cluster
        # the model wrote as `^[2], [5]` had one marker and one piece of prose in it; a
        # display number is assigned per marker, so compaction has to see all of them or
        # the reader is handed a pill-less `[5]` beside a numbered pill (Stage 6C).
        settled_prose = citations.normalise_carets(settled_prose)
        # `/pubmed` keeps the numbering its records were published under. On this route the
        # marker is not a position in an evidence array — a `search_literature` result
        # contributes no citable unit, so there is no array for it to address — it is the
        # record's index in the list the model was shown, and `render_numbered` stamps the
        # articles `1..N` in retrieval order while `a2ui` numbers `sources`, the pills and
        # `references` `1..N` in that same order. Compacting the prose to the order it
        # happens to cite them renames the page and leaves the drawer alone: `dev_t1_06`
        # turn 1 wrote `^[2] ^[1] ^[3] ^[2]`, was served `^[1] ^[2] ^[3] ^[1]`, and the
        # reader's first pill then opened a paper the sentence was not about (2026-11-11).
        # So the map stays empty here, and `resolve_sources` reads the numbers directly.
        if command != "literature":
            settled_prose, display = citations.compact_display_numbers(settled_prose)
    if settled_deterministic:
        if command == "literature":
            # Same reason: nothing on this route renumbers, so nothing may publish a map
            # from a numbering that was never applied.
            settled_deterministic = citations.normalise_carets(settled_deterministic)
        else:
            compacted, deterministic_map = citations.compact_display_numbers(
                citations.normalise_carets(settled_deterministic))
            settled_deterministic = compacted
            # The deterministic answer WINS at the point of serving — `turn.compose_success`
            # reads `deterministic_answer` in preference to the model's prose — so its map is
            # the one that describes the numbers a reader will click. It used to be kept only
            # when the prose carried no markers at all (`display = display or deterministic_map`),
            # which meant a turn that had both served the deterministic sentence through the
            # prose's numbering and drew the wrong passage behind every marker.
            display = deterministic_map or display

    return Settlement(prose=settled_prose, deterministic=settled_deterministic,
                      display=display, flags=tuple(flags))


def sanitize_answer_markers(state: dict, text: str | None = None) -> None:
    """Apply `settle_answer` to a state dict. Kept for callers that hold one.

    This is the state-shaped form and it edits the state it is given, which is why the
    served path no longer goes through it: `graph.ask` calls `settle_answer` and writes the
    result itself, so the engine owns its own state and this package owns prose only
    (Charter 13 §3). The adapter stays because it is the documented entry point for
    settlement and because existing callers assert against a state — the two are the only
    reason it exists, and keeping the decision in `settle_answer` means there is still one
    implementation of it.
    """
    message = _final_message(state)
    prose = text
    if prose is None and message is not None and isinstance(message.content, str):
        prose = message.content
    settled = settle_answer(
        prose=prose,
        deterministic=state.get("deterministic_answer"),
        tool_calls=state.get("tool_calls") or [],
        replayed_tool_calls=state.get("replayed_tool_calls") or [],
        wants_literature=bool(state.get("wants_literature")),
        # The adapter holds a state, so it reads the same rows `graph._settle` passes.
        comparators=state.get("comparators") or [],
        thread_literature=state.get("thread_literature") or [],
        # And the record, for the same reason: the bound is a length of ONE array.
        evidence_state=state.get("evidence_state"),
        # Which command this turn was. The `/ehr` discharge section is the one passage this
        # package filters by section, and only a formatting turn for that command can
        # contain it.
        command=state.get("command"),
    )
    if message is not None and settled.prose is not None:
        message.content = settled.prose
    if settled.deterministic is not None:
        state["deterministic_answer"] = settled.deterministic
    if settled.display:
        state["citation_display_map"] = dict(settled.display)
    if settled.flags:
        # Settlement runs AFTER `guard_answer`, so its verdicts used to be thrown away:
        # the answer a reader was served could be missing a citation while
        # `guardrail_flags` said the turn was clean (2026-10-30, a turn whose two PMID
        # badges were removed at settlement and reported as `guardrail_flags: []`).
        state["guardrail_flags"] = [*(state.get("guardrail_flags") or []),
                                    *settled.flags]


def no_evidence_answer(
    question: str,
    tool_calls: list[dict],
    *,
    declined: bool = False,
    evidence_state: Any = None,
) -> str | None:
    """The fixed sentence for a question the notes did not answer, or None.

    Free text is answered by a model that can fill a gap with plausible clinical detail,
    and asking it in the prompt to admit ignorance is a request rather than a guarantee —
    the failure it prevents is a confident answer to a question the notes do not address,
    which reads exactly like an answer to one they do. So the absence is decided here and
    the model's text for that turn is never served.

    Two things can establish it, and they are different observations:

      * retrieval returned nothing at all;
      * the model read what retrieval returned and reports that none of it answers the
        question (`declined`, from `read_evidence_reply`). This is the case a count cannot
        see. Vector search returns its nearest neighbours whatever it is asked, so a
        question the notes cannot answer comes back looking like one they can: live on
        2026-10-12, "was the patient intubated during this admission?" retrieved five
        passages, none of which mentioned intubation, and the model answered from them
        anyway. The count was 5 and the answer was absent, which is the pair a count
        cannot distinguish.

    The verdict in the second case is the model's; the SENTENCE is not. It is spoken here,
    in fixed words, so a model that would have hedged instead reports nothing and the
    clinician hears the same thing either way.

    The conditions around both are narrow on purpose:

      * the note search RAN — a turn that made no retrieval call is not a turn that found
        nothing, and saying "the chart does not mention it" about a search nobody performed
        would invent a fact about the record;
      * no search failed. A failed search says nothing about the chart, and reporting an
        outage as an absence is the same mistake in the other direction;
      * the literature was not consulted. An answer built on published evidence is an
        answer, and it is not a claim about this patient's notes.

    `evidence_state` is the session record, and it is the second way a turn can hold
    passages without having fetched one now: a follow-up turn replays the earlier turns'
    calls into the record rather than re-running them, so `tool_calls` holds nothing while
    the model is reading five passages out of `<session_evidence_state>`. Counting only
    this turn's payloads called that an empty retrieval and replaced a correctly grounded
    follow-up answer with the fixed absence sentence — a claim about the chart made against
    evidence the session was holding.

    The record is asked for NOTE passages specifically, not for any unit. A session that
    holds only a laboratory panel has no passage about the discharge note, so an
    unanswered note question in that session is still an absence — which is what the
    `get_lab_trends`-only turn in EVIDENCE's own rule says.
    """
    names = [call.get("name") for call in tool_calls]
    if any(name in routing.LITERATURE_TOOLS for name in names):
        return None
    if "rag_search" not in names:
        return None

    found_passages = False
    for call in tool_calls:
        if call.get("name") not in routing.PASSAGE_TOOLS:
            continue
        response: Any = call.get("response")
        if not isinstance(response, dict) or response.get("error"):
            return None
        if response.get("passages"):
            found_passages = True

    if not found_passages and evidence_state is not None:
        items = getattr(evidence_state, "items", None) or {}
        found_passages = any(
            getattr(item, "source_tool", None) in routing.PASSAGE_TOOLS
            for item in items.values()
        )

    if found_passages and not declined:
        return None

    # The session may hold evidence this turn did NOT fetch. An interpretive follow-up —
    # "why does this value drive her risk", "which of these labs is most concerning" — is
    # answered from the record the earlier turns already established, and an auxiliary
    # search that comes back empty says nothing about whether the answer is grounded: it
    # says the corpus has no NEW passage matching the query, which is what a question
    # about explanation rather than about documentation looks like.
    #
    # Replacing that answer with the fixed absence sentence was the defect: a clinician
    # asked what a documented value means, the model explained it from the chart it was
    # holding, and the turn served "There is no information in the patient's chart
    # regarding that question" over the top of a correct answer (2026-11-08).
    #
    # The model's own decline still wins: `declined` is it saying it could not answer, and
    # that is a fact about the turn rather than about the count.
    if not declined and _session_holds_chart_evidence(evidence_state, question):
        return None

    topic = _asked_about(question)
    return NO_EVIDENCE_TEMPLATE.format(topic=topic) if topic else NO_EVIDENCE_UNKNOWN_TOPIC


def _session_holds_chart_evidence(evidence_state: Any, question: str = "") -> bool:
    """Whether the session record already holds evidence that can answer this question.

    The record is modal, and the question says which modality it needs. A note passage
    answers a question about the chart; the estimate answers a question about risk; a
    laboratory panel answers a question about a measurement — and a panel answers nothing
    about whether the note documents an intubation, which is why a lab-only session is
    still the absence the fixed sentence describes (EVIDENCE's own rule: "a turn that
    called only `get_lab_trends` has not read the note at all").

    So the test is not "does the record hold anything" but "does it hold what was asked
    about", and the lab channel is credited only when the question invokes it.
    """
    if evidence_state is None:
        return False
    if getattr(evidence_state, "items", None):
        for item in (evidence_state.items or {}).values():
            if getattr(item, "source_tool", None) in routing.PASSAGE_TOOLS:
                return True
    # Any other unit in the record is a laboratory series, which answers a laboratory
    # question and no other kind.
    lab_question = bool(
        routing.names_an_analyte(question) or re.search(r"\blabs?\b|\blaborator", question or "",
                                                        re.IGNORECASE)
    )
    if lab_question and (getattr(evidence_state, "items", None)
                         or getattr(evidence_state, "checked_labs", None)):
        return True
    return getattr(evidence_state, "prediction", None) is not None
