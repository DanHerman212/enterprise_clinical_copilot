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
from typing import Any

from langchain_core.messages import AIMessage

from services.agent.postprocess import citations
from services.agent.routing import routing


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


def medication_narrative(verified: list[dict[str, str]], returned_passages: int,
                         sections: tuple[str, ...] = ()) -> str:
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

    The fallback sentence reserved for a genuinely empty result is used only when the tool
    returned no passages at all. A tool that returned text this module could not verify is
    a different failure, and the prose states that failure clinically: what was retrieved,
    and that no medication could be read from it. It does not name our internals — a
    "formatting failure" is a fact about this module, not about the patient's record, and
    a judge is right to score it ungrounded (90000023, 2026-10-21).
    """
    names = _distinct_medications(verified)
    # The sentence reports on the passage the tool returned, so it carries that passage's
    # marker like any other claim: the citation layer resolves `^[1]` against this turn's
    # retrieved passages, and the judge reads a provenance-poor answer as one that cited
    # nothing (5fa503f6 and e5558ec7 passed with `citation_missing`, 2026-10-21). Only
    # when a passage exists — a marker pointing at nothing is worse than none.
    source = " ^[1]" if returned_passages else ""
    dedicated = "discharge_medications" in sections
    if names:
        if dedicated:
            count = len(names)
            noun = "medication" if count == 1 else "medications"
            return (
                f"{count} discharge {noun}, as recorded in the discharge note: "
                f"{', '.join(names)}{source}."
            )
        where = ", ".join(section.replace("_", " ") for section in sections) or (
            "material that was retrieved")
        return (
            "The clinical record does not contain a dedicated discharge medications "
            f"section; the {where} documents the following medications administered or "
            f"mentioned during the stay rather than a formal discharge medication list: "
            f"{', '.join(names)}{source}."
        )
    if returned_passages == 0:
        return "No specific discharge medication information is available for this admission."
    if dedicated:
        return (
            "The discharge medications section was retrieved, but no medication could be "
            f"read from it{source}."
        )
    where = ", ".join(section.replace("_", " ") for section in sections) or (
        "material that was retrieved")
    return (
        "The clinical record does not contain a dedicated discharge medications section, "
        f"and no medication list is recorded in the {where}{source} (any medications "
        "administered or prescribed elsewhere in the stay are documented in the clinical "
        "course narrative rather than a dedicated medication section)."
    )


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


def reject_model_reply(state: dict) -> None:
    """Withdraw an answer the turn may not serve.

    A declined reply with no absence to report is not an answer: the model declined, and
    nothing was retrieved for it to decline against, so any sentence served here would be
    a claim the record does not support. The turn carries no text and the routes report an
    unavailable answer rather than a fabricated one.

    Recorded by editing the message rather than by raising, so the finish reason and the
    content-filter verdict stay readable on the turn the caller was refused.
    """
    replace_final_text(state, "")


def reply_withheld(finish_reason: str | None, *, blocked: bool = False) -> bool:
    """Whether the provider took the model's reply away rather than the model stopping.

    `SAFETY` and a blocked prompt are the two forms measured live, and they are the two
    this answers for. It is deliberately not every refusal reason: a truncation is the
    model running out of room, which is a different statement about the turn, and a turn
    that stopped for an unlisted reason has no reply to explain.

    The reason is read by whoever holds the response; this function is the one place that
    decides what the reason means, so the two routes cannot disagree about it.
    """
    return blocked or finish_reason == "SAFETY"


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


def settle_answer(
    *,
    prose: str | None,
    deterministic: str | None,
    tool_calls: list[dict] | None,
    replayed_tool_calls: list[dict] | None,
    wants_literature: bool = False,
    comparators: list[dict] | None = None,
    thread_literature: list[dict] | None = None,
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
    """
    bound = citations.citation_bound(tool_calls or [], replayed_tool_calls or [])
    # The badge allowance follows the same rule the bound does, on the one axis where it
    # may: a turn that is itself about published records can cite the records the replay
    # showed it. `wants_literature` is stamped once by `graph.ask` — the decision that let
    # the earlier search stay in the transcript in the first place — so this stays a read
    # of that decision rather than a second opinion about the question, and a chart turn
    # still cites nothing it was not shown.
    pmids = citations.literature_pmids(
        tool_calls or [], (replayed_tool_calls or None) if wants_literature else None,
    )
    # And the cohorts the comparison table published, which is a third channel beside the
    # note passages and the answer's own literature — the same union `guardrail` builds
    # (`citations.comparator_pmids`), made here so both passes allow the same set.
    #
    # `/ehr`'s comparator search declares `scope="readmission"`, so `literature_pmids`
    # deliberately does NOT count it: those records are the table's, not the answer's own
    # citation list, and counting them would draw the briefing as a literature turn with
    # twenty-five candidates. The consequence was that a `^[PMID: …]` badge naming a
    # cohort the table had just verified resolved against an empty allowance and was
    # deleted here — so it never reached the guardrail below, which already knew the row.
    # Measured 2026-11-03: `^[PMID: 24108298]` on 90000069 and both comparator badges on
    # 90000015 were removed as `citation_out_of_range` while the references drawer listed
    # the records they named.
    pmids = pmids | citations.comparator_pmids(comparators)
    # And the records an earlier turn of the thread verified. Same rule, one layer out: a
    # follow-up answering from the thread's own published benchmark is not naming an
    # identifier nobody retrieved — the thread retrieved it, and `replay_messages`
    # carried back the records that turn cited.
    pmids = pmids | citations.record_pmids(thread_literature)

    flags: list[str] = []
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
        settled_prose, found = citations.enforce_citations(swept, bound, pmids)
        flags.extend(found)

    settled_deterministic: str | None = None
    if isinstance(deterministic, str) and deterministic:
        stripped, printed = citations.strip_printed_constructs(deterministic)
        flags.extend(printed)
        swept = citations.drop_unknown_markers(stripped)
        settled_deterministic, found = citations.enforce_citations(swept, bound, pmids)
        flags.extend(found)
    else:
        settled_deterministic = deterministic

    # The page's numbering, last and separate from the array's. A marker's number is the
    # position of its unit in the conversation's evidence array — cumulative across
    # turns, which is what makes a follow-up citable at all — and a reader should not be
    # shown the gaps that produces (`^[5]`, `^[8]`, then `^[14]`, `^[15]` on one real
    # thread). `compact_display_numbers` renumbers the served prose to the order it cites
    # and returns the map the presentation layer resolves the new numbers through, so a
    # pill, a drawer row and an evaluation source keep naming the same passage.
    display: dict[str, int] = {}
    if settled_prose:
        settled_prose, display = citations.compact_display_numbers(settled_prose)
    if settled_deterministic:
        compacted, deterministic_map = citations.compact_display_numbers(
            settled_deterministic)
        settled_deterministic = compacted
        display = display or deterministic_map

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
    question: str, tool_calls: list[dict], *, declined: bool = False
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

    if found_passages and not declined:
        return None

    topic = _asked_about(question)
    return NO_EVIDENCE_TEMPLATE.format(topic=topic) if topic else NO_EVIDENCE_UNKNOWN_TOPIC
