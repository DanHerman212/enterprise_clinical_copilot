"""Citation reconciliation and discharge-note section resolution.

This module owns the deterministic half of the citation contract. The model's
`^[n]` numbers are advisory — a meds answer cites `^[3]` because
discharge_medications is the third section in `rag_search_sections` order,
which reads illogically to a user and mis-identifies the supporting passage.
The fixes are deterministic and live HERE, in the agent, because this is the
only layer where the answer, the guardrails, and the retrieved evidence meet:

  1. `renumber_citations` rewrites markers to first-appearance order and
     collapses stacked citations (the model dumping every passage onto one
     claim) to a single marker.
  2. `citation_remap` emits {renumbered: original} so the browser can resolve
     a clicked footnote back to the original passage.
  3. `dedupe_list_citations` hoists one repeated marker off a list and onto the
     line that introduces it, because a badge on every bullet of a list drawn
     from one passage is noise that reads as eleven sources.
  4. `intent_sections` maps a question to the note section(s) it clearly
     targets, so the canvas shows THAT section regardless of the model's
     numbering.
  5. `extract_section` pulls a section body out of a (possibly whole-note)
     passage, so a citation click shows the cited section, not the whole note.

The alias vocabulary is derived from the canonical `KNOWN_HEADINGS` used by
index build and serving (services/mcp/retrieval/sections.py), with a small set
of presentation-only extras (e.g. "MRN", "Activity") that the demo notes use
as bounding headers. One vocabulary, one drift surface (S7-02 / ECC-33).
"""

import re

from services.mcp.retrieval.sections import KNOWN_HEADINGS

# Presentation-only aliases on top of the canonical KNOWN_HEADINGS. These
# appear in MTSamples-derived demo notes as section boundaries but are not
# part of the index/serving vocabulary.
_EXTRA_HEADINGS: dict[str, tuple[str, ...]] = {
    "unit_no": ("Unit No.", "Medical Record Number", "MRN"),
    "admission_date": ("Date of Admission",),
    "discharge_date": ("Date of Discharge",),
    "date_of_birth": ("DOB",),
    "allergies": ("Allergy", "Allergies to"),
    # Not a KNOWN_HEADINGS section at all; needed so a trailing "Activity:"
    # line bounds the preceding section instead of being swallowed by it.
    "activity": ("Activity",),
}

SECTION_ALIASES: dict[str, tuple[str, ...]] = {}
for _section, _headings in KNOWN_HEADINGS.items():
    SECTION_ALIASES[_section] = tuple(_headings) + tuple(
        h for h in _EXTRA_HEADINGS.get(_section, ()) if h not in _headings
    )
for _section, _headings in _EXTRA_HEADINGS.items():
    SECTION_ALIASES.setdefault(_section, tuple(_headings))
del _section, _headings

# Every known header, flattened — bounds the end of an extracted section.
_ALL_HEADERS = tuple(h for aliases in SECTION_ALIASES.values() for h in aliases)


def extract_section(note_text: str, section: str) -> str | None:
    """Extract a named section's body from a full note.

    Alias-aware: matches the section's header variants (e.g. "Hospital
    Course:" for brief_hospital_course), then bounds the body at the next
    known header. Returns None only when the section is genuinely absent.
    """
    if not note_text:
        return None
    aliases = SECTION_ALIASES.get(section) or (str(section or "").replace("_", " "),)
    if not aliases[0]:
        return None
    # Locate the section start via the first alias that appears in the note.
    start = None
    matched = None
    for alias in aliases:
        # Anchor to a line start (^ or \n), consistent with the end bound, so
        # generic aliases (History, Condition, Medications…) can't match
        # mid-sentence and truncate the wrong body (S7-12).
        m = re.search(
            rf"(^|\n)\s*{re.escape(alias)}\b\s*:", note_text, re.IGNORECASE
        )
        if m:
            anchor = m.group(1)
            rest = m.group(0)[len(anchor):]
            ws = len(rest) - len(rest.lstrip())
            start = m.start() + len(anchor) + ws
            matched = rest[ws:]
            break
    if start is None or matched is None:
        return None
    # Bound the end at the next known header after the section title.
    end = len(note_text)
    for header in _ALL_HEADERS:
        hm = re.search(
            rf"\n\s*{re.escape(header)}\s*:",
            note_text[start + len(matched):],
            re.IGNORECASE,
        )
        if hm:
            candidate = start + len(matched) + hm.start()
            if candidate < end:
                end = candidate
    return note_text[start:end].strip()


def first_citation(answer: str) -> int:
    """The first citation number in the answer prose (^[n]), or 1 if none.

    The canvas's SourceCard mirrors the agent's own citation: whichever passage
    the answer cites first is the one the canvas shows.
    """
    m = re.search(r"\^\[(\d+)", answer or "")
    return int(m.group(1)) if m else 1


_CITATION_RE = re.compile(r"\^\[(\d+(?:\s*,\s*\d+)*|\d+\s*-\s*\d+)\]")


def _expand_citations(inner: str) -> list[int]:
    """Expand a ^[n] marker body ("3" / "1, 2" / "1-3") to a number list."""
    if "-" in inner:
        a, b = (int(x.strip()) for x in inner.split("-", 1))
        return list(range(a, b + 1))
    return [int(x.strip()) for x in inner.split(",") if x.strip()]


def _renumber_citations(answer: str) -> tuple[str, dict[int, int]]:
    """Renumber ^[n] markers to first-appearance order.

    Returns (renumbered_answer, old_to_new) where old_to_new maps the model's
    original citation number (the tool's array position) to its renumbered
    number (first-appearance order, 1-based).
    """
    if not answer:
        return answer, {}
    markers = list(_CITATION_RE.finditer(answer))
    if not markers:
        return answer, {}
    # Group markers separated only by whitespace into one cluster.
    clusters: list[list[re.Match]] = [[markers[0]]]
    for m in markers[1:]:
        gap = answer[clusters[-1][-1].end():m.start()]
        if gap.strip():
            clusters.append([m])
        else:
            clusters[-1].append(m)
    # Renumber by first appearance, cluster by cluster.
    old_to_new: dict[int, int] = {}
    nxt = 1
    for cl in clusters:
        for m in cl:
            for num in _expand_citations(m.group(1)):
                if num not in old_to_new:
                    old_to_new[num] = nxt
                    nxt += 1
    # Rebuild: one marker per cluster, carrying the cluster's first number.
    out: list[str] = []
    last = 0
    for cl in clusters:
        out.append(answer[last:cl[0].start()])
        first = _expand_citations(cl[0].group(1))[0]
        out.append(f"^[{old_to_new[first]}]")
        last = cl[-1].end()
    out.append(answer[last:])
    return "".join(out), old_to_new


#: Any `^[...]` marker at all, valid or not. Used to find the ones that are not
#: citations so they can be removed rather than printed.
_ANY_MARKER_RE = re.compile(r"\^\[[^\]]*\]")

#: What a marker must look like to be KEPT: a note-passage number or set of them,
#: or a literature PMID badge. Everything else is a model invention and is dropped.
#: The PMID form is here deliberately — it is a citation the interface draws as its
#: own badge, and a filter that only knew about numbers would delete it.
_KNOWN_MARKER_RE = re.compile(
    r"\^\[\s*(?:\d+(?:\s*,\s*\d+)*|\d+\s*-\s*\d+|PMID:\s*\d+(?:\s*,\s*\d+)*)\s*\]",
    re.IGNORECASE,
)


def _drop_unknown_markers(answer: str) -> str:
    """Remove `^[...]` markers that are neither a passage number nor a PMID badge.

    The model sometimes writes the passage's identity instead of its number —
    `^[brief_hospital_course]` rather than `^[2]` — and `_CITATION_RE` does not
    match it, so the renumbering pass skipped it and the reader was shown the raw
    braces in the middle of a clinical sentence (seen 2026-10-17, from a turn whose
    note section was cited by name). Raw markup in the prose is worse than a
    missing citation: it reads as corruption, and the claim it was attached to is
    still checked by the guardrails either way.
    """
    if "^[" not in (answer or ""):
        return answer
    return _ANY_MARKER_RE.sub(
        lambda m: m.group(0) if _KNOWN_MARKER_RE.fullmatch(m.group(0)) else "",
        answer,
    )


#: A markdown list item — a bullet or an ordinal, its indent, and its marker.
_LIST_ITEM_RE = re.compile(r"^(\s*)(?:[-*+]|\d+[.)])\s+")

#: A citation marker closing a line, with whatever punctuation followed it.
_TRAILING_MARKER_RE = re.compile(r"\s*\^\[\s*(\d+)\s*\](?P<tail>[.,;:]?)\s*$")


def dedupe_list_citations(answer: str) -> str:
    """Take one repeated marker off a list, and put it where it belongs.

    A list drawn from a single passage arrived with the same badge on every item —
    ten discharge diagnoses, ten `^[3]`s. The claim the marker supports is the LIST,
    not each of its items, and eleven identical badges across eleven lines do not read
    as one source cited eleven times: they read as eleven sources, which is the
    misreading the whole citation contract exists to prevent. A reader also cannot
    tell a list where every item genuinely rests on the same passage from one where
    the model stopped thinking.

    Deterministic rather than asked for in the prompt, because it is a shape and not
    a judgement: a contiguous block of list items whose last elements are all the
    SAME marker is a list the model attributed to one passage, and the move is the
    same every time.

    The marker is hoisted onto the line that introduces the list, before its colon —
    "The discharge diagnoses recorded are ^[3]:" — and dropped from the items. With
    no such introducing line it is left on the final item only, so the list is still
    attributed to something and the reader can still open it.

    Lists whose items cite DIFFERENT passages are untouched: that is a list making a
    per-item claim, and hoisting one of them would leave the others unattributed.
    """
    if not answer or "^[" not in answer:
        return answer

    lines = answer.split("\n")
    out = list(lines)
    index = 0
    while index < len(lines):
        if not _LIST_ITEM_RE.match(lines[index]):
            index += 1
            continue
        start = index
        while index < len(lines) and _LIST_ITEM_RE.match(lines[index]):
            index += 1
        block = list(range(start, index))
        if len(block) < 2:
            continue
        marks = {line: _TRAILING_MARKER_RE.search(lines[line]) for line in block}
        if any(marks[line] is None for line in block):
            continue
        if len({marks[line].group(1) for line in block}) != 1:
            continue

        marker = f"^[{marks[block[0]].group(1)}]"
        stripped = {}
        for line in block:
            mark = marks[line]
            text = (lines[line][: mark.start()] + mark.group("tail")).rstrip()
            # An item that was nothing but its marker keeps it: silently emptying a
            # bullet would change what the list says, which is not this pass's job.
            if not _LIST_ITEM_RE.sub("", text, count=1).strip():
                stripped = None
                break
            stripped[line] = text
        if not stripped:
            continue

        lead_in = start - 1
        introduces = lead_in >= 0 and lines[lead_in].rstrip().endswith(":")
        keep_on = None if introduces else block[-1]
        for line in block:
            if line != keep_on:
                out[line] = stripped[line]
        if introduces and marker not in lines[lead_in]:
            opening = lines[lead_in].rstrip()
            out[lead_in] = f"{opening[:-1].rstrip()} {marker}:"
    return "\n".join(out)


def renumber_citations(answer: str) -> str:
    """Renumber ^[n] markers to first-appearance order (1, 2, 3, ...).
    The model numbers citations by the tool's array position: a meds-only
    answer cites ^[3] because discharge_medications is the THIRD section in
    rag_search_sections order. In a meds-only turn that reads illogically —
    the first (and only) citation should be ^[1]. Renumber deterministically
    so citations always start at 1 in order of appearance; the canvas's
    section-intent resolution keeps the passage mapping correct regardless.

    Also collapses STACKED citations (^[1]^[2]^[3]... with only whitespace
    between them — the model dumping every returned passage onto one claim) to
    a single marker, since a row of citations on one claim has no per-claim
    granularity to preserve.

    Finally, any marker left over that is not a number is dropped rather than
    printed: a `^[section_name]` the model invented is not a citation, and letting
    it through puts raw markup in front of a clinician.

    A repeated marker on a list is hoisted onto the line that introduces it first
    (see `dedupe_list_citations`), so what is renumbered is what will be read.
    """
    return _drop_unknown_markers(_renumber_citations(dedupe_list_citations(answer))[0])


def cited_numbers(answer: str) -> list[int]:
    """Distinct citation numbers in the answer prose, ascending."""
    return sorted({n for m in _CITATION_RE.finditer(answer or "")
                   for n in _expand_citations(m.group(1))})


def citation_remap(answer: str) -> dict[str, int]:
    """{renumbered citation number: original passage number} for the client.

    `renumber_citations` rewrites the model's ^[n] markers to first-appearance
    order, so a clicked (renumbered) number no longer matches the passage
    array index when the model cited passages out of array order. The client
    uses this map to translate a clicked ^[n] back to the original passage.
    Keys are strings for JSON.

    Read off the same text `renumber_citations` renumbered — through the same list
    deduplication — because a marker this removes from a list is a marker that no
    longer takes a number, and a remap built from the un-deduped answer would shift
    every later citation by one.
    """
    _, old_to_new = _renumber_citations(dedupe_list_citations(answer))
    return {str(new): old for old, new in old_to_new.items()}


# Discharge-note sections a question can target, in preference order, with the
# words that reveal the intent. A question maps to a SECTION SET because the
# content can live in different sections across MTSamples notes: the meds
# claim's supporting text is usually discharge_medications, but many notes
# carry it in discharge_instructions instead. Resolving by section (not the
# model's ^[n] numbers) is what keeps the canvas honest.
_INTENT_SECTIONS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("discharge_medications", "discharge_instructions"),
     ("medication", "medications", "meds", "discharged on")),
    (("discharge_instructions",),
     ("instruction", "instructions")),
    (("discharge_diagnosis",),
     ("diagnosis", "diagnoses")),
    (("brief_hospital_course",),
     ("hospital course", "admission course")),
)


def intent_sections(question: str | None) -> tuple[str, ...]:
    """The discharge-note sections a question clearly targets, in preference
    order — or an empty tuple when the question has no single-section intent.

    Summarize/risk questions map to () — their citation-by-number behavior is
    left untouched.

    A COMPOSITE question maps to () as well (Phase 01, 2026-10-22). Section intent
    exists because the model mis-numbers an extraction: asked for the medications, it
    cites ^[1] while the list sits in the third passage. But that override is only
    safe when the question asks for ONE section. "What was the disposition and
    follow-up plan?" mentions two things, and a single intent then remaps the model's
    correct citation onto a chunk that answers only half of it.
    """
    if not question:
        return ()
    q = question.lower()
    matched = [sections for sections, needles in _INTENT_SECTIONS
               if any(n in q for n in needles)]
    if len(matched) != 1:
        return ()
    return matched[0]
