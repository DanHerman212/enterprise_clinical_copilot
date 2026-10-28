"""Citation reconciliation and discharge-note section resolution.

This module owns the deterministic half of the citation contract. The model's
`^[n]` numbers are advisory — a meds answer cites `^[3]` because
discharge_medications is the third section in `rag_search_sections` order,
which reads illogically to a user and mis-identifies the supporting passage.
The fixes are deterministic and live HERE, in the agent, because this is the
only layer where the answer, the guardrails, and the retrieved evidence meet:

  1. `renumber_citations` keeps every marker's canonical number and expands a
     citation cluster to one single-number marker per source, so a claim that
     rests on six passages carries six pills rather than the first one only.
  2. `citation_remap` emits {marker: passage}; the map is the identity now that
     numbers are preserved, and it is still published so the browser keeps one
     resolution path.
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


#: What may sit between two markers and still leave them ONE citation cluster: nothing,
#: horizontal whitespace, and the separator a model writes between adjacent citations
#: (`^[3], ^[4]`). A newline or a full stop ends the cluster.
_CLUSTER_GAP_RE = re.compile(r"[ \t]*[,;]?[ \t]*")


def _renumber_citations(answer: str) -> tuple[str, dict[int, int]]:
    """Expand each citation cluster to one single-number marker per source.

    The NUMBERS are not rewritten. A marker's number is the position of its unit in the
    array the conversation handed the model (`citable_passages`), and that is the number
    every other layer resolves against: `citation_map`, `sources`, the guardrail and the
    browser. Reassigning them to first-appearance order broke that agreement — on Gate 3
    (`00e`) `holdout_t1_03` cited sodium first and the reassignment moved it to `^[1]`,
    displacing four of nine markers one position onto the wrong analyte.

    The SHAPE is rewritten. A cluster — `^[4, 5, 6]`, `^[3], ^[4]`, `^[3] ^[4]` —
    becomes one marker per number, in order of appearance within the cluster and
    deduplicated: `^[4] ^[5] ^[6]`. The client's marker regex reads a single number
    (`static/js/a2ui_clinical_components.js`, `MARKER`), so a comma-separated marker was
    unrenderable there, and this function used to solve that by keeping only the first
    number of a cluster — silently dropping every other source from the page. Expanding
    keeps them all and gives the client pills it can draw.

    Returns (expanded_answer, {n: n}). The map is the identity, and is still returned so
    `citation_remap` and its callers keep one contract.
    """
    if not answer:
        return answer, {}
    markers = list(_CITATION_RE.finditer(answer))
    if not markers:
        return answer, {}
    # Group markers separated only by whitespace or a citation separator.
    clusters: list[list[re.Match]] = [[markers[0]]]
    for m in markers[1:]:
        gap = answer[clusters[-1][-1].end():m.start()]
        if _CLUSTER_GAP_RE.fullmatch(gap):
            clusters[-1].append(m)
        else:
            clusters.append([m])
    # Rebuild: one marker per distinct number, numbers untouched.
    numbers: dict[int, int] = {}
    out: list[str] = []
    last = 0
    for cl in clusters:
        out.append(answer[last:cl[0].start()])
        seen: list[int] = []
        for m in cl:
            for num in _expand_citations(m.group(1)):
                if num not in seen:
                    seen.append(num)
                    numbers[num] = num
        out.append(" ".join(f"^[{num}]" for num in seen))
        last = cl[-1].end()
    out.append(answer[last:])
    return "".join(out), numbers


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


#: A citation marker with whatever space precedes it, so removing the marker does not
#: leave a gap before the punctuation that followed it.
_SPACED_MARKER_RE = re.compile(r"([ \t]*)\^\[([^\]]*)\]")


def sanitize_citation_range(answer: str, source_total: int) -> str:
    """Remove citation markers that address no citable unit this conversation showed.

    Numbering is 1-based and bounded: a thread that was shown `source_total` citable
    units has numbers `1..source_total` and nothing else. A marker outside that range is
    not a citation in any sense — it is a number the model wrote, and it reaches the
    reader as a footnote that opens nothing. On Gate 3 (`00d2`) a synthesis turn whose
    previous `/meds` turn had retrieved no passages wrote "the discharge medications
    section is empty ^[0]": zero is not a passage position at all, and the sentence was
    doubly wrong about the record (see the prompt rule on empty versus not-retrieved).

    Bounded rather than merely positive because the same class of invention produces
    `^[7]` on a thread with three units, which resolves to nothing and scores as a
    fabricated source.

    A multi-number marker keeps its valid numbers and loses the rest, so `^[1, 9]` on a
    three-unit thread becomes `^[1]` rather than disappearing: the supported half of the
    the claim still points at its evidence. The kept numbers are emitted as one
    single-number marker each (`^[1] ^[2] ^[9]`), because that is the only shape the
    browser's marker regex can turn into a control. A marker with no valid number left
    is removed whole, together with the space before it, so the sentence closes up
    around the punctuation it had.

    PMID badges are left alone — they address a literature record, not a passage, and
    this function knows nothing about how many of those the turn returned.

    Deliberately not folded into `renumber_citations`: renumbering decides ORDER, and
    this decides EXISTENCE. Keeping them apart is what lets the renumbering be tested on
    its own contract, and what makes this rule auditable as the single place a marker can
    be deleted.
    """
    if "^[" not in (answer or ""):
        return answer
    limit = max(int(source_total or 0), 0)

    def _rebuild(match: re.Match) -> str:
        lead, inner = match.group(1), match.group(2).strip()
        if inner.upper().startswith("PMID"):
            return match.group(0)
        if not re.fullmatch(r"\d+(?:\s*,\s*\d+)*|\d+\s*-\s*\d+", inner):
            # Not a number and not a PMID: `_drop_unknown_markers` owns this case.
            return match.group(0)
        numbers = _expand_citations(inner)
        valid = [n for n in numbers if 1 <= n <= limit]
        seen: list[int] = []
        for n in valid:
            if n not in seen:
                seen.append(n)
        if not seen:
            return ""
        if len(seen) == 1 and len(numbers) == 1:
            return match.group(0)
        return f"{lead}" + " ".join(f"^[{n}]" for n in seen)

    return _SPACED_MARKER_RE.sub(_rebuild, answer)


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
    """Expand every citation marker to one single-number marker per source.

    The numbers are preserved. A marker's number is the position of its unit in
    the array the conversation handed the model, and `sources`, `citation_map`
    and the browser all resolve against exactly that; reassigning them to
    first-appearance order made the reader's number mean something other than
    the number the model wrote (`holdout_t1_03`, Gate 3 `00e`: a sodium marker
    written first displaced four of nine analytes onto their neighbours).

    What changes is the shape. A contiguous run of markers — `^[4, 5, 6]`,
    `^[3], ^[4]`, `^[1]^[2]^[3]` — becomes one marker per number, in order and
    deduplicated: `^[4] ^[5] ^[6]`. The browser's marker regex matches a single
    number, so a comma-separated marker drew nothing at all; the previous
    behaviour of keeping only the first number made the page render but dropped
    every other source the claim rested on.

    Finally, any marker left over that is not a number is dropped rather than
    printed: a `^[section_name]` the model invented is not a citation, and letting
    it through puts raw markup in front of a clinician.

    A repeated marker on a list is hoisted onto the line that introduces it first
    (see `dedupe_list_citations`), so what is expanded is what will be read.
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


# ---------------------------------------------------------------------------
# What in a tool result is citable, and what number it carries
# ---------------------------------------------------------------------------
#
# A marker addresses one array, and that array is every citable thing the conversation
# has put in front of the model — counted CONTINUOUSLY across tool calls and across
# replayed turns. Two layers have to agree on it: `graph` numbers what it shows the model,
# and the citation layer resolves the number the model wrote. They agreed by both counting
# `len(payload["passages"])`, which was the whole story while note passages were the only
# citable evidence.
#
# They are not any more. `get_lab_trends` returns a series rather than a passage array, so
# it counted as nothing: the model was handed a lab trajectory with no number on it, and
# `TF-10` requires every clinical answer to carry a `^[n]`. A question about a red-cell
# count therefore had no citable answer in front of it, and the loop went and searched the
# discharge note — for a citable passage, not for a fact — then appended a second
# paragraph drawn from the notes to have something to cite (90000023, 2026-10-26). One
# analyte is one unit of evidence; it gets one number, and the two layers count it the
# same way because they call this function.

#: The section a laboratory trajectory is filed under. Not a note heading — it is the
#: other channel a chart answer can rest on, and it is named here so the label map in
#: `note_presentation` has one key to translate rather than a second vocabulary.
LAB_SECTION = "laboratory_trends"

#: What the marker says for a laboratory citation, before the analyte is named.
LAB_SOURCE = "Laboratory trends"


def _format_reading(value: object, decimals: object) -> str:
    """One measured value, at the precision the panel reports it to."""
    try:
        places = int(decimals) if decimals is not None else 2
    except (TypeError, ValueError):
        places = 2
    try:
        return f"{float(value):.{places}f}"
    except (TypeError, ValueError):
        return str(value)


def lab_passage(trend: dict) -> dict:
    """One analyte's trajectory, as a passage a citation can resolve to.

    The text is what a reader sees when they open the marker: the values in order with
    the days they were drawn, the flags the panel raised, and the interval they were read
    against. A marker that opened onto "Creatinine" and no numbers would be a citation to
    a heading.
    """
    label = str(trend.get("label") or trend.get("key") or "Laboratory value").strip()
    unit = str(trend.get("unit") or "").strip()
    readings: list[str] = []
    for point in trend.get("series") or []:
        if not isinstance(point, dict):
            continue
        value = _format_reading(point.get("value"), trend.get("decimals"))
        day = point.get("day")
        flag = str(point.get("flag") or "").strip()
        when = f"day {day}" if isinstance(day, (int, float)) else "day unknown"
        # The flag is what made the value interesting, so it rides with the value rather
        # than in a separate list the reader has to line up by position.
        readings.append(f"{value} ({when}{', ' + flag if flag and flag != 'normal' else ''})")
    low, high = trend.get("reference_low"), trend.get("reference_high")
    interval = ""
    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
        interval = (
            f" Reference interval {_format_reading(low, trend.get('decimals'))}"
            f"-{_format_reading(high, trend.get('decimals'))}"
            f"{' ' + unit if unit else ''}."
        )
    trajectory = "; ".join(readings) if readings else "no values recorded"
    return {
        "section": LAB_SECTION,
        "text": (
            f"{label}{' (' + unit + ')' if unit else ''} across this admission: "
            f"{trajectory}.{interval}"
        ).strip(),
        # What a citation for this passage calls itself. Read by `note_citations`, so the
        # marker and the drawer row say "Creatinine trajectory" rather than naming the
        # whole panel, which is what a reader clicked on.
        "cite_source": LAB_SOURCE,
        "cite_title": f"{label} trajectory",
    }


def citable_passages(payload: object) -> list[dict]:
    """The citable units one tool result contributes, in the order it gave them.

    `passages` when the tool returns note passages — the retrieval tools — and one entry
    per analyte when it returns a laboratory panel. A payload with neither contributes
    nothing, which is correct: a prediction, a search result set and a medication list are
    answers rather than evidence, and none of them is citable as a passage.
    """
    if not isinstance(payload, dict):
        return []
    passages = payload.get("passages")
    if isinstance(passages, list):
        return [passage for passage in passages if isinstance(passage, dict)]
    trends = payload.get("trends")
    if isinstance(trends, list):
        return [lab_passage(trend) for trend in trends if isinstance(trend, dict)]
    return []


def passage_total(payload: object) -> int:
    """How many citation numbers one tool result consumes.

    The single definition of that count. `graph` advances its running number by this and
    the citation layer concatenates the same array, so a marker the model was given and
    the record it opens are the same position by construction.
    """
    return len(citable_passages(payload))
