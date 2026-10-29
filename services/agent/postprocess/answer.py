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

from typing import Any

from langchain_core.messages import AIMessage

from services.agent.postprocess import citations


NO_EVIDENCE_TEMPLATE = (
    'There is no information in the patient\'s chart regarding "{topic}".'
)
NO_EVIDENCE_UNKNOWN_TOPIC = (
    "There is no information in the patient's chart regarding that question."
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


def _asked_about(question: str) -> str:
    """The caller's own question, without the admission the contract appends.

    Quoted rather than paraphrased: a sentence assembled in code must not invent a subject
    for what was asked, and the caller's words are already the subject. The admission
    suffix is dropped because "regarding \\"... For admission 90000023.\\"" names a record
    number in the middle of a clinical sentence.
    """
    head, marker, _ = (question or "").rpartition(" For admission ")
    return (head if marker else (question or "")).strip()


def sanitize_answer_markers(state: dict) -> None:
    """Apply the citation bound to the answer this turn will serve.

    Applied at settlement — the one point both routes pass through — rather than in the
    presentation layer, because the trace, the judge and the browser read the answer from
    three different places and only the state is common to all three.

    Both answer channels are cleaned: the model's prose, and a command's
    `deterministic_answer` (which for `/meds` is written in code and carries its own
    `^[1]`).

    The bound and the enforcement are `citations`'. This function only decides *which
    string* is the turn's answer; it does not decide what a citation is, and it holds no
    regex.
    """
    bound = citations.citation_bound(
        state.get("tool_calls") or [], state.get("replayed_tool_calls") or [],
    )
    pmids = citations.literature_pmids(state.get("tool_calls") or [])
    message = _final_message(state)
    if message is not None and isinstance(message.content, str):
        cleaned, _ = citations.enforce_citations(message.content, bound, pmids)
        if cleaned != message.content:
            message.content = cleaned
    deterministic = state.get("deterministic_answer")
    if isinstance(deterministic, str) and deterministic:
        cleaned, _ = citations.enforce_citations(deterministic, bound, pmids)
        if cleaned != deterministic:
            state["deterministic_answer"] = cleaned


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
    if "search_literature" in names:
        return None
    if "rag_search" not in names:
        return None

    found_passages = False
    for name in ("rag_search", "rag_search_sections"):
        for call in tool_calls:
            if call.get("name") != name:
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
