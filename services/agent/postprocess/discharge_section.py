"""The `/ehr` discharge section, held to the sections that can state a discharge plan.

The defect this closes is one sentence, written four times across two days:

    **Discharge plan & medications.** The patient was prescribed Plavix to be taken daily
    for one month ^[3].

`^[3]` is `brief_hospital_course`, and that passage does say the patient was told to take
the drug. What the sentence does not say is WHICH section it is speaking for. Under a
heading that promises the discharge regimen, "was prescribed" asserts a home prescription
the record never made: the discharge instructions for that admission list no medications at
all. The chart is right, the marker is right, the sentence is true of the course and false
of the heading, and a clinician reading the heading has no way to tell.

The prompt was told not to do this three times, in three spellings, and stage 6H's version
was the most explicit of them:

    a drug named inside `brief_hospital_course` is a hospital-course fact and is reported
    under **Hospital course & procedures.**, cited to that passage, never here.
    ... no sentence under it opens "the patient was prescribed".

The next run did it again. A formatting turn is a model turn, and an instruction is a
request rather than a guarantee — the same lesson the citation bound learned in Phase 1.
So this is the guarantee: after the markers are bound and before the page is numbered, a
sentence under that heading either rests on a section that can state a discharge plan or it
is not served.

Two deliberate limits.

It does nothing unless the turn actually holds a discharge section to compare against. A
sparse note whose only text is the hospital course has no discharge medication section, and
silently deleting the one sentence that mentions a medication would leave the reader with
less than the record says. The hospital course's own medication instruction is legitimate
content; it is only the CLAIM OF PLACEMENT that is wrong, and with nothing to place it
against there is no placement to correct.

And it removes a sentence rather than rewriting one. Rewriting "was prescribed Plavix" into
"the hospital course records an instruction to take Plavix" would be this layer composing
clinical prose, which is the engine's job and the model's. What is left is what the section
can support; the fact itself is reported under **Hospital course & procedures.**, where the
model already writes it — every one of the four measured answers had it there or in turn 3.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from services.agent.postprocess import citations

#: The label the `/ehr` instruction requires, and the paragraph this module owns.
DISCHARGE_LABEL = "**Discharge plan & medications.**"

#: Sections a sentence under that label may rest on. Each states something the discharge
#: plan is made of: the regimen, the instructions, the follow-up plan, where the patient
#: went, the condition they left in, the plan, or a note that has no headings at all.
DISCHARGE_SECTIONS: frozenset[str] = frozenset({
    "discharge_medications",
    "discharge_instructions",
    "followup_instructions",
    "discharge_disposition",
    "discharge_condition",
    "assessment_and_plan",
    "discharge_summary",
})

#: A medication list that opens this way is the list the patient ARRIVED on. In a
#: transitional-care note the heading "Medications on transfer" carries the acute
#: hospital's list, and a drug on it may have been stopped before discharge — so a
#: sentence citing it cannot state a discharge regimen however the section is named.
ARRIVAL_OPENER = re.compile(
    r"^\s*(?:medications?\s+)?on\s+(?:transfer|admission)\b", re.IGNORECASE)

#: The next section of the answer, which ends this module's paragraph.
_NEXT_HEADING = re.compile(r"\*\*[A-Z][^*\n]*\*\*")

#: Endings that look like a sentence end and are not one. Lower-cased, with the full stop.
_ABBREVIATIONS: frozenset[str] = frozenset({
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.",
    "p.o.", "p.r.n.", "b.i.d.", "t.i.d.", "q.i.d.", "q.d.", "i.v.", "i.m.", "s.c.",
    "q.h.", "q.12h.", "q.8h.", "q.6h.", "q.4h.", "q.24h.", "q.48h.",
    "e.g.", "i.e.", "vs.", "approx.", "no.", "fig.", "ref.",
})


def _looks_like_abbreviation(text: str, stop: int) -> bool:
    """Whether the token ending at `stop` is an abbreviation rather than a sentence end."""
    tail = text[:stop + 1]
    match = re.search(r"([A-Za-z](?:[A-Za-z.]*[A-Za-z])?)\.$", tail)
    if not match:
        return False
    token = match.group(1)
    # A single capital letter is an initial — "Dr. X" — and never ends a sentence.
    if len(token) == 1 and token.isalpha():
        return True
    return f"{token.lower()}." in _ABBREVIATIONS


def split_sentences(text: str) -> list[str]:
    """Sentences, without splitting inside a decimal, an initial or a clinical shorthand.

    The clinical vocabulary is why this exists rather than a regex on `[.!?]\\s+`: a
    discharge paragraph is dense with `p.o.`, `b.i.d.`, `q.12h.`, `0.05%`, `10.6%` and
    honorifics, and a splitter that breaks on any of them hands this module half a sentence
    and then deletes it.
    """
    sentences: list[str] = []
    start = 0
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char not in ".!?":
            index += 1
            continue
        if char == "." and 0 < index < length - 1 \
                and text[index - 1].isdigit() and text[index + 1].isdigit():
            index += 1
            continue
        following = index + 1
        while following < length and text[following] in " \t\n":
            following += 1
        if following >= length:
            break
        if not (text[following].isupper() or text[following] in "\"*'"):
            index += 1
            continue
        if char == "." and _looks_like_abbreviation(text, index):
            index += 1
            continue
        sentences.append(text[start:index + 1].strip())
        start = following
        index = following
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return [sentence for sentence in sentences if sentence]


def _paragraph(prose: str) -> tuple[int, int, str] | None:
    """`(start, end, body)` of the discharge paragraph, or None when there is not one.

    `start` is the label's first character and `end` is the first character after the
    paragraph, so a caller can replace the range without touching what follows.
    """
    label_at = prose.find(DISCHARGE_LABEL)
    if label_at < 0:
        return None
    body_at = label_at + len(DISCHARGE_LABEL)
    following = _NEXT_HEADING.search(prose, body_at)
    end = following.start() if following else len(prose)
    return label_at, end, prose[body_at:end]


def _sections_by_position(units: Iterable[Mapping[str, Any]]) -> dict[int, str]:
    """Marker number -> section name, for the array a pre-compaction marker indexes."""
    return {position: str(unit.get("section") or "")
            for position, unit in enumerate(units, start=1)
            if isinstance(unit, Mapping)}


def _arrival_positions(units: Iterable[Mapping[str, Any]] | None) -> set[int]:
    """Positions of `discharge_medications` units whose text opens with an arrival header."""
    found: set[int] = set()
    for position, unit in enumerate(units or [], start=1):
        if not isinstance(unit, Mapping):
            continue
        if str(unit.get("section") or "") != "discharge_medications":
            continue
        if ARRIVAL_OPENER.match(str(unit.get("text") or "")):
            found.add(position)
    return found


def has_a_discharge_section(passages: Iterable[Mapping[str, Any]]) -> bool:
    """Whether this turn holds a section that can state a discharge plan.

    `discharge_instructions` always can: it is the section a note writes the home regimen
    in. A `discharge_medications` passage can, unless it opens with an arrival header, in
    which case it is the list the patient came in on and states nothing about going home.
    """
    for passage in passages:
        section = str(passage.get("section") or "")
        if section == "discharge_instructions":
            return True
        if section == "discharge_medications" \
                and not ARRIVAL_OPENER.match(str(passage.get("text") or "")):
            return True
    return False


def filter_discharge_section(
    prose: str,
    *,
    passages: Iterable[Mapping[str, Any]] | None = None,
    units: Iterable[Mapping[str, Any]] | None = None,
    discharge_location: str | None = None,
) -> tuple[str, tuple[str, ...]]:
    """The answer with unsupportable sentences removed from its discharge section.

    `passages` are this turn's retrieved passages — what decides whether the filter applies
    at all. `units` are the citable units a marker indexes, position `n` at `[n - 1]`.

    Returns the text and the flags naming what was removed, so the removal is visible in
    telemetry rather than silent: a sentence the reader never sees is a judgement about the
    answer, and the turn should say it made one.
    """
    passages = list(passages or [])
    if not has_a_discharge_section(passages):
        return prose, ()

    found = _paragraph(prose or "")
    if found is None:
        return prose, ()
    start, end, body = found

    by_position = _sections_by_position(units or [])
    # The arrival list cannot state a discharge regimen, so a sentence citing it is treated
    # exactly as one citing the hospital course: the section name is right and the claim of
    # placement is not.
    unplaceable = _arrival_positions(units)

    kept: list[str] = []
    dropped: list[str] = []
    for sentence in split_sentences(body):
        # What counts as a marker is Law 1's question and it is answered in `citations`.
        # A second regex here would be a second definition, and the two would drift on
        # the first form one of them learned that the other had not.
        positions = citations.cited_numbers(sentence)
        if not positions:
            # Unmarked: the disposition sentence, which the block states and which carries
            # no marker by design.
            kept.append(sentence)
            continue
        sections = [by_position.get(position, "") for position in positions]
        if all(section in DISCHARGE_SECTIONS and position not in unplaceable
               for section, position in zip(sections, positions)):
            kept.append(sentence)
            continue
        dropped.append(sentence)

    if not dropped:
        return prose, ()

    flags = tuple(f"discharge_section_dropped:{by_position.get(position, 'unresolved')}"
                  for sentence in dropped
                  for position in citations.cited_numbers(sentence)[:1])

    new_body = " ".join(kept).strip()
    if not new_body:
        # Nothing under the label can be said. The label goes with the sentences: an
        # empty heading promises a section and delivers a gap.
        tail = prose[end:]
        head = prose[:start]
        rejoined = (head.rstrip() + ("\n\n" + tail.lstrip() if tail.strip() else ""))
        return rejoined, flags + ("discharge_section_withheld",)

    if discharge_location and "discharg" not in new_body.lower() \
            and any("discharg" in sentence.lower() for sentence in dropped):
        # A dropped sentence carried the disposition with it, and the block states where
        # the patient went. One unmarked sentence restores the fact a reader needs.
        new_body = f"{new_body} The patient was discharged {discharge_location}."

    # The paragraph break belongs to the boundary, not to any sentence, so rebuilding the
    # body from sentences loses it: `split_sentences` returns no trailing whitespace and
    # `end` stops at the next heading's first character. `dev_t4_02` turn 1 was served
    # `... ^[5].**Readmission risk & trajectory.**` — the heading running into the full
    # stop of the sentence before it — on every turn where a sentence was kept. Restored
    # here, from what follows the paragraph rather than from what was in it.
    tail = prose[end:]
    if not tail.strip():
        return f"{prose[:start]}{DISCHARGE_LABEL} {new_body}", flags
    return f"{prose[:start]}{DISCHARGE_LABEL} {new_body}\n\n{tail.lstrip()}", flags
