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
  6. `strip_printed_constructs` removes, from the served prose, the constructs
     this package PRINTS into a tool result for the model to read. The model copies
     what it is shown and generalises what it can copy, so the guarantee is "no string
     we print reaches the reader undrawn" and it is kept beside the printer.

The alias vocabulary is derived from the canonical `KNOWN_HEADINGS` used by
index build and serving (services/mcp/retrieval/sections.py), with a small set
of presentation-only extras (e.g. "MRN", "Activity") that the demo notes use
as bounding headers. One vocabulary, one drift surface (S7-02 / ECC-33).
"""

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from services.agent.routing import routing
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


#: The token a citation marker opens with, and the one that closes it.
#:
#: Named once. The marker language is the thing that drifts when it grows a form — the
#: PMID badge did, and a cluster of records after it — and every pattern below is built
#: from these two strings rather than restating the escape. `re.compile` is never handed
#: a marker literal: the opening token has exactly one spelling in this repository, and
#: the harness reads it from here too (Charter 13 Law 1; audit L1-4).
MARKER_OPEN = r"\^\["
MARKER_CLOSE = r"\]"


def _marker_re(body: str, flags: int = 0) -> "re.Pattern[str]":
    """Compile a marker pattern from its BODY: everything between the two tokens."""
    return re.compile(MARKER_OPEN + body + MARKER_CLOSE, flags)


#: A marker's number, or the first number of a cluster or a range. Read without the
#: closing token on purpose: this is the "did the answer cite anything at all, and with
#: what" question, and a marker whose bracket was lost in transit still names a unit.
_FIRST_CITATION_RE = re.compile(MARKER_OPEN + r"(\d+)")


def first_citation(answer: str) -> int:
    """The first citation number in the answer prose (^[n]), or 1 if none.

    The canvas's SourceCard mirrors the agent's own citation: whichever passage
    the answer cites first is the one the canvas shows.
    """
    m = _FIRST_CITATION_RE.search(answer or "")
    return int(m.group(1)) if m else 1


_CITATION_RE = _marker_re(r"(\d+(?:\s*,\s*\d+)*|\d+\s*-\s*\d+)")


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
    browser. Reassigning them to first-appearance order broke that agreement — an answer
    that names its abnormal value first and the rest of the panel afterwards had every
    marker displaced by one analyte.

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


#: The retrieval tools, re-exported for the presentation layer.
#:
#: The grouping is `routing.PASSAGE_TOOLS`'s and stays there (Charter 13 Law 4). It is
#: re-exported here because `presentation/` may import this module and may not import
#: `routing/`, and a second literal in `a2ui.rag_payload` was the fifth copy of a
#: tool regrouping nobody could find.
RETRIEVAL_TOOLS: frozenset[str] = routing.PASSAGE_TOOLS


#: Any `^[...]` marker at all, valid or not, with its body captured. The one regex every
#: pass in this module scans with: `_drop_unknown_markers` decides which non-numeric forms
#: survive, and `enforce_citations` decides which numeric and PMID forms do.
_ANY_MARKER_RE = _marker_re(r"([^\]]*)")

#: The same pattern for readers outside this module — the presentation layer and the
#: evaluation harness both scan prose for markers, and both used to spell the token out.
MARKER_PATTERN = _ANY_MARKER_RE

#: What separates one paragraph from the next. Used by `collapse_repeated_citations`,
#: which compares a marker only with the marker before it IN THE SAME PARAGRAPH: a section
#: opened two paragraphs later is a new citation, however recently the same source was
#: cited.
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")

#: A sentence break between two markers, so a run cannot cross one.
#:
#: The unit a citation belongs to is the SENTENCE, not the paragraph. A run that crossed
#: a boundary turned a paragraph of four sentences about one passage into one badge on the
#: last of them, which is a valid marker deleted from three claims: measured 2026-11-06,
#: `citation_repeat_collapsed:^4` on both admissions, `:^7` and `:^9` on admission B. The
#: rule this restores is the one the module already states — a marker is a claim about the
#: sentence it is attached to — and the collapse stays for what it was built for: the same
#: marker repeated inside one sentence, or a paragraph that stamped the same pill on every
#: sentence of one continuous thought.
_SENTENCE_BREAK = re.compile(r"[.!?][\"')\]]*\s+")

#: What a marker must look like to be KEPT: a note-passage number or set of them,
#: or a literature PMID badge. Everything else is a model invention and is dropped.
#: The PMID form is here deliberately — it is a citation the interface draws as its
#: own badge, and a filter that only knew about numbers would delete it.
_KNOWN_MARKER_RE = _marker_re(
    r"\s*(?:\d+(?:\s*,\s*\d+)*|\d+\s*-\s*\d+|PMID:\s*\d+(?:\s*[,;]\s*(?:PMID\s*:?\s*)?\d+)*)\s*",
    re.IGNORECASE,
)

#: The numeric forms a marker body may take: `3`, `1, 2`, `1-3`.
_NUMERIC_BODY_RE = re.compile(r"\d+(?:\s*,\s*\d+)*|\d+\s*-\s*\d+")

#: A marker that names a literature record rather than a position.
_PMID_BODY_RE = re.compile(r"PMID:?\s*(\d+)", re.IGNORECASE)

#: A badge naming literature records, one or several: `PMID: 35186616`, or the grouped
#: `PMID: 39559360, 37028984, 36774746` the model writes when one sentence rests on
#: three papers. `_KNOWN_MARKER_RE` already accepts both forms, so a group reaches
#: enforcement; this is what recognises it once it is there.
_PMID_LIST_BODY_RE = re.compile(r"PMID:?\s*\d+(?:\s*,\s*\d+)*", re.IGNORECASE)


#: A badge naming several published records at once: `PMID: 1, PMID: 2`, `PMID: 1, 2`,
#: `PMID: 1; PMID: 2`. Every shape the model writes when one sentence rests on two cohorts.
#: Built through `_marker_re` like every other marker pattern here, because Law 1 forbids
#: compiling the marker language inline anywhere — this module included.
_GROUPED_PMID_RE = _marker_re(
    r"\s*PMID\s*:?\s*\d+(?:\s*[,;]\s*(?:PMID\s*:?\s*)?\d+)+\s*",
    re.IGNORECASE,
)


def normalise_grouped_pmid_markers(answer: str) -> str:
    """One badge per record, from a badge that named several.

    The model writes `^[PMID: 35186616, 37028984]` when a sentence rests on two cohorts,
    and the browser draws markers one at a time, so the grouped form was reaching
    settlement and dying there: `_KNOWN_MARKER_RE` accepts `PMID: 1, 2` but not
    `PMID: 1, PMID: 2`, and a marker it does not recognise is deleted — so on 2026-11-03 a
    risk paragraph naming two published cohorts lost both badges and the reader had two
    rates with nothing to open.

    Splitting is the repair rather than widening the pattern, because the grouped shape
    has no single valid rendering: one badge can only be one control. Two ids under one
    badge become two badges, which is what the sentence meant.

    A badge naming ONE record is left exactly as written — the shape is already correct and
    rewriting it would be churn on the common path. A bare `^[1, 2]` is untouched too:
    those are note positions, not identifiers, and this function reads the `PMID` label to
    tell them apart.
    """
    if not _GROUPED_PMID_RE.search(str(answer or "")):
        return answer

    def _split(match: "re.Match[str]") -> str:
        identifiers = re.findall(r"\d+", match.group(0))
        # One record, one badge: a group that names the same identifier twice is naming it
        # once, which is the rule `enforce_citations` applies to a single badge.
        ordered = list(dict.fromkeys(identifiers))
        return " ".join(f"^[PMID: {pmid}]" for pmid in ordered)

    return _GROUPED_PMID_RE.sub(_split, answer)


def enforce_citations(
    answer: str,
    slots: int,
    pmids: "Iterable[str]" = (),
) -> tuple[str, list[str]]:
    """Keep the markers that resolve; delete the ones that do not; report the latter.

    The single place a citation marker can be deleted, and the single definition of what
    "resolves" means. Two functions did this until 2026-10-29 — this one's predecessor
    (`sanitize_citation_range`) and `guardrail.check_citations` — with two different
    counts and two different deletion mechanisms. They disagreed about laboratory
    analytes, which is what failed four items on `pilot-verify-20261028-205050`: markers
    the model had written correctly were deleted as out of range.

    `slots` is the bound, and it is the length of the array the model was numbered
    against — one per note passage AND one per laboratory analyte, across the replayed
    turns and this one (`citation_bound`). `pmids` are the records a literature turn
    actually retrieved; a `^[PMID: ...]` badge naming anything else is an invented
    identifier and is removed rather than printed.

    Three outcomes per marker:

      * **resolvable** — left exactly as written;
      * **partly resolvable** — a multi-unit marker (`^[1, 9]`, `^[2-6]`,
        `^[PMID: 39559360, 37028984]`) keeps its valid units and loses the rest, emitted
        as one single-unit marker each, because a single unit is the only shape the
        browser's marker regex can turn into a control;
      * **unresolvable** — removed whole.

    A removal takes the run of spaces in front of the marker with it, so no gap is left
    behind. What it cannot take is the separator that FOLLOWED it: a sentence ending
    `discharge ^[4], ^[5], ^[6].` loses all three markers and is left holding the commas
    that sat between them. `_tidy_removed_markers` then closes those runs up, which is
    why the output reads `discharge.` rather than `discharge,,,,,,.` — the artefact four
    answers carried when the deletion and the tidy lived in different modules.

    A marker that is neither numeric nor a PMID (`^[brief_hospital_course]`) is left
    alone here: `_drop_unknown_markers`, reached through `renumber_citations`, owns that
    vocabulary. One regex, two rules about what a marker may say, no third opinion.
    """
    if "^[" not in (answer or ""):
        return answer, []
    # A badge that named two records is two badges before anything reads it. Done first
    # because every step below classifies a marker by its body, and the grouped shape
    # classifies as nothing — which is how a real citation came to be deleted.
    answer = normalise_grouped_pmid_markers(answer)
    limit = max(int(slots or 0), 0)
    allowed = {str(pmid) for pmid in pmids}
    flags: list[str] = []
    out: list[str] = []
    last = 0

    for match in _ANY_MARKER_RE.finditer(answer):
        body = match.group(1).strip()
        conclusion = _conclude(body, limit, allowed)
        if conclusion is None:
            continue
        # Taken from the start of the whitespace in front of the marker, so a deletion
        # does not leave a gap and a rewrite keeps exactly one separating space.
        start = match.start()
        while start > last and answer[start - 1] in " \t":
            start -= 1
        lead = answer[start:match.start()]
        out.append(answer[last:start])
        kept, found = conclusion
        flags.extend(found)
        if kept:
            # A body rather than a bare number: a note position is `1`, a literature
            # badge is `PMID: 35186616`, and both are written the same way here so one
            # marker per unit reaches the browser in the one shape it can draw.
            out.append(lead + " ".join(f"^[{marker}]" for marker in kept))
        last = match.end()
    out.append(answer[last:])
    kept, collapse_flags = collapse_repeated_citations(
        _tidy_removed_markers("".join(out)))
    return kept, flags + collapse_flags


def _marker_key(body: str) -> str:
    """One marker's identity, compared rather than printed: `1, 2` and `1,2` agree."""
    return ",".join(part.strip() for part in str(body or "").split(","))


def collapse_repeated_citations(answer: str) -> tuple[str, list[str]]:
    """Keep one marker out of a run of the same citation in one sentence.

    Reported from a printed `/ehr` (2026-11-02): one paragraph carried the same section's
    pill on every sentence. The first rule collapsed a run of immediate repeats; the
    revision of 2026-11-03 made the run cross sentence boundaries and keep only its last
    marker, which fixed the wall of pills and broke the other direction — four sentences
    citing one passage became one badge on the fourth, so three claims were served uncited
    (`citation_repeat_collapsed:^4` on `90000069`, `:^7` and `:^9` on `90000015`, measured
    2026-11-06).

    The run is scoped to the sentence now, which is the unit the marker belongs to: what
    collapses is the accidental double stamp (`^[4] ^[4]`) and a sentence that repeats a
    marker within itself, and what survives is the badge on every later sentence that makes
    its own claim. A paragraph of four sentences on one passage still reads as four badges
    rather than one; that is the cost, and it is the right side of the trade — a reader can
    ignore a repeated pill, and cannot reopen a citation that was deleted. The rule lives here, not in the renderer, because this
    module is the single owner of what a marker is and of whether it survives settlement —
    a browser that decides not to draw one has taken a citation decision and left the
    prose and the evidence disagreeing on the server.

    A RUN, not a pair, and the LAST marker of it survives. That is the change of
    2026-11-03: the earlier rule dropped a marker only when the one immediately before it
    named the same source, and it kept the FIRST of the pair — so three consecutive
    sentences on one passage lost two markers and the paragraph still opened with a badge,
    and four sentences lost three. Keeping the last also subsumes the within-sentence case:
    a sentence that wrote `^[4] ^[4]` keeps the one at its end.

    A run is broken by a different marker, by a paragraph break, or by a marker that is not
    a note passage at all. So `^[3]` then `^[4]` keeps both, `^[4]`, `^[3]`, `^[4]` keeps
    all three — the run was interrupted, and the return to a source is a claim about the
    sentence it is attached to — and a published record's `^[PMID: …]` is never collapsed,
    because two sentences citing the same cohort study are two claims about that study
    rather than one marker repeated.

    The reference list is unaffected — it is built from `sources`, not from the prose — so
    the row is still there to be opened. Each removal is flagged, because a silent deletion
    of a citation is the kind of change this layer exists to make visible.
    """
    if "^[" not in (answer or ""):
        return answer, []
    markers = [
        (match.start(), match.end(), _marker_key(match.group(1)))
        for match in _ANY_MARKER_RE.finditer(answer)
    ]
    if len(markers) < 2:
        return answer, []

    removals: list[tuple[int, int]] = []
    flags: list[str] = []
    run: list[tuple[int, int, str]] = []

    def close_run() -> None:
        """Every marker of a run but the last one goes."""
        for start, end, body in run[:-1]:
            removals.append((start, end))
            flags.append(f"citation_repeat_collapsed:^{body}")

    previous_end = 0
    for start, end, body in markers:
        # A published record's badge is not a section marker: a paragraph naming the same
        # cohort twice is making two claims, so it breaks the run and joins none.
        if not _NUMERIC_BODY_RE.fullmatch(body or ""):
            close_run()
            run = []
        elif (_PARAGRAPH_BREAK.search(answer, previous_end, start)
              or _SENTENCE_BREAK.search(answer, previous_end, start)
              or (run and run[-1][2] != body)):
            close_run()
            run = [(start, end, body)]
        else:
            run.append((start, end, body))
        previous_end = end
    close_run()

    if not removals:
        return answer, []
    out: list[str] = []
    last = 0
    for start, end in removals:
        # The space in front of a removed marker goes with it, so removing the marker
        # before a full stop does not leave "morphine ." behind.
        cut = start
        while cut > last and answer[cut - 1] in " \t":
            cut -= 1
        out.append(answer[last:cut])
        last = end
    out.append(answer[last:])
    return _tidy_removed_markers("".join(out)), flags



def _conclude(
    body: str, limit: int, allowed: "set[str]",
) -> "tuple[list[int | str], list[str]] | None":
    """What to do with one marker body. None means leave the marker exactly as written.

    Returns the marker bodies to emit and the flags to report; an empty body list means
    the marker is unresolvable and goes. A body is a bare number (`1`) or a labelled
    literature badge (`PMID: 35186616`); the caller writes both into a marker.
    """
    if not body:
        return [], ["citation_out_of_range:^[]"]
    # A record identifier, in either form the model writes it: the labelled
    # `^[PMID: 35186616]`, or the bare `^[35186616]` it produces when it has been told
    # twice not to label one. Both name the same record, so both resolve the same way.
    if _PMID_LIST_BODY_RE.fullmatch(body):
        return _conclude_pmids(body, allowed)
    if not _NUMERIC_BODY_RE.fullmatch(body):
        # A section name or other invention: `_drop_unknown_markers` decides its fate.
        return None

    def resolves(number: int) -> bool:
        return 1 <= number <= limit or str(number) in allowed

    numbers = _expand_citations(body)
    valid: list[int] = []
    for number in numbers:
        if resolves(number) and number not in valid:
            valid.append(number)
    missing = [n for n in dict.fromkeys(numbers) if not resolves(n)]
    flags = [f"citation_out_of_range:^{n}" for n in missing]
    if len(numbers) == 1:
        # One number, and it resolves: the marker is already correct, so it is left
        # byte-for-byte. Anything else here would rewrite a good citation.
        return (None if valid else ([], flags))
    # A cluster is expanded whether or not any of it is out of range: one number per
    # marker is the only shape the browser can turn into a control.
    return valid, flags


def _conclude_pmids(body: str, allowed: "set[str]") -> "tuple[list[str], list[str]] | None":
    """Resolve a badge naming literature records, one or several.

    The grouped form is one marker standing for several records, and it was the one
    marker shape this layer neither resolved, split nor deleted. `_PMID_BODY_RE` is a
    `fullmatch` on a single identifier, so `PMID: 39559360, 37028984, 36774746` failed
    here, failed the numeric test as well, and fell through to "leave the marker exactly
    as written" — unresolved, unverified and unflagged. The browser could not draw it
    either (`MARKER` is one number and one closing bracket), so the reader got the raw
    badge in the middle of a clinical sentence while every machine check reported the
    turn as clean. Seen live on 2026-10-30, on a declared literature follow-up that cited
    three records from the turn before.

    Split here rather than in the browser, for the reason the numeric branch splits
    `^[1, 9]` here: one marker per unit is the only shape the console can turn into a
    control, and only this layer knows which identifiers name records the turn may cite.
    """
    # Every digit run in the body is an identifier. `_PMID_BODY_RE` cannot be used for
    # this: it requires the `PMID:` prefix, which only the first number in a group
    # carries, so `findall` over a group returns one identifier and the other two vanish
    # — which is how a split that looked right still left the badge whole.
    found = re.findall(r"\d+", body)
    if len(found) == 1:
        # A single badge is left byte-for-byte when it resolves. Anything else would
        # rewrite a correct citation's own spacing, and a passing answer is the thing
        # a rewrite damages first.
        if found[0] in allowed:
            return None
        return [], [f"citation_out_of_range:^{found[0]}"]
    unique = list(dict.fromkeys(found))
    kept = [f"PMID: {number}" for number in unique if number in allowed]
    flags = [
        f"citation_out_of_range:^{number}"
        for number in unique
        if number not in allowed
    ]
    return kept, flags


def _tidy_removed_markers(answer: str) -> str:
    """Close up the punctuation a deleted marker left stranded.

    The cases measured on 2026-10-28, all caused by removing a marker that sat between
    two separators:

      * `discharge ^[4], ^[5], ^[6].` -> `discharge,,,,,,,,.`
      * `strictures, ^[11] ^[12].` -> `strictures,.`
      * `creatinine ^[6] and eGFR` -> a double space where the space before the marker
        and the space after it met

    Each rule is a no-op on text that had no marker removed, so a clean answer passes
    through unchanged; that is asserted, because a whitespace rule that rewrites passing
    prose is how the meds guard once damaged a correct answer.
    """
    if not answer:
        return answer
    cleaned = re.sub(r"[ \t]{2,}", " ", answer)
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)
    # A separator left in front of another separator, or in front of the sentence's own
    # terminator, says nothing: `,,` and `,.` are debris, not punctuation.
    for _ in range(3):
        collapsed = re.sub(r"([,;:])\s*(?=[.,;:])", "", cleaned)
        if collapsed == cleaned:
            break
        cleaned = collapsed
    return cleaned


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


def strip_markers(answer: str) -> str:
    """Every citation marker removed, for a wire that publishes no citations at all.

    `/labs` is that wire (2026-11-03). Its evidence is a trajectory table drawn directly
    under the sentence, row by row, with each reading's reference interval and flag; a
    marker in the prose addresses a card that repeats the row it sits beside, and the
    drawer beneath it listed one synthetic record per analyte. So the instruction asks for
    no markers, and this removes any the model wrote anyway — a `^[4]` with no citation
    behind it renders as a dead badge, which is worse than the pill it was imitating.

    The whitespace in front of a removed marker goes with it, so a sentence that ended
    `… 7.1 g/dL ^[2].` still ends `… 7.1 g/dL.` rather than gaining a stray space before
    its full stop.
    """
    if "^[" not in (answer or ""):
        return answer
    stripped = _ANY_MARKER_RE.sub("", answer)
    return _tidy_removed_markers(stripped)


def restore_global_markers(
    answer: str, citation_map: dict[str, int] | None, cite_offset: int = 0
) -> str:
    """A replayed answer's markers put back into the conversation's own numbering.

    `compact_display_numbers` renumbers the served page to `1..K` per turn, so the
    numbers a stored turn carries are the reader's, not the array's. `strip_numeric_markers`
    resolved that by deleting them; deleting them also deleted the pattern a follow-up
    imitates, and a turn that answered without citing because the earlier turns had no
    markers in them is the cost (`dev_t2_02` turn 3, 2026-11-05).

    The turn's own `citation_map` — `{display: position in the conversation's array}`,
    shipped with the answer and handed back on replay — is what makes the deletion
    unnecessary. This restores each number to the position the served answer actually
    addressed.

    `cite_offset` is where the turn's own units begin in the array the replay is
    numbering against, so a map carrying turn-local positions lands on the same integers
    the replayed tool results stamp. With the map shipped absolute and `cite_offset = 0`
    the arithmetic is the identity, which is why the two conventions cannot drift apart.

    A number the map does not hold — a marker the map's own answer never carried, a map
    from a caller that predates this field — is DROPPED rather than kept: it addresses a
    position nobody can resolve, and a marker that resolves to the wrong passage is the
    failure this whole mechanism exists to prevent. When nothing in the answer can be
    restored, the answer is `strip_numeric_markers`' output unchanged, which is what
    every replay did before this field existed.

    `^[PMID: ...]` badges are untouched: they name a record rather than a position, so
    they mean the same thing to whoever reads them.
    """
    if "^[" not in (answer or ""):
        return answer
    mapping: dict[str, int] = {}
    for key, position in (citation_map or {}).items():
        if isinstance(position, int) and not isinstance(position, bool):
            mapping[str(key)] = position
    if not mapping:
        return strip_numeric_markers(answer)

    out: list[str] = []
    last = 0
    restored_any = False
    for match in _ANY_MARKER_RE.finditer(answer):
        body = match.group(1).strip()
        if not _NUMERIC_BODY_RE.fullmatch(body):
            continue
        restored: list[str] = []
        for number in _expand_citations(body):
            position = mapping.get(str(number))
            if position is None:
                continue
            absolute = cite_offset + position
            if absolute >= 1 and str(absolute) not in restored:
                restored.append(str(absolute))
        start = match.start()
        if not restored:
            # The marker is being removed, so the space in front of it goes too — the
            # same rule `strip_numeric_markers` applies, so a half-restored answer does
            # not end a sentence with a trailing space before its full stop.
            while start > last and answer[start - 1] in " \t":
                start -= 1
        out.append(answer[last:start])
        if restored:
            out.append(" ".join(f"^[{number}]" for number in restored))
            restored_any = True
        last = match.end()
    out.append(answer[last:])
    if not restored_any:
        return strip_numeric_markers(answer)
    return _tidy_removed_markers("".join(out))


def strip_numeric_markers(answer: str) -> str:
    """Numeric markers removed; `^[PMID: ...]` badges left exactly as written.

    The numbering a served answer shows is NOT the numbering the evidence carries.
    `compact_display_numbers` renumbers the page to `1..K` per turn, while the tool
    results the model is shown keep their positions in the conversation's array. Both
    were replayed into one context, so a prior turn's `^[2]` (the page's second cited
    passage) sat beside a replayed passage stamped `cite: 2` that was a different
    section — and the model, told to copy the `cite` of the unit it means, had two
    incompatible answers to "what is 2" (`dev_t4_02` turn 3, 2026-11-05).

    On replay the prose keeps every clinical word and loses only the numbers that can no
    longer be true. A `^[PMID: ...]` badge is a different fact: it names a record rather
    than a position, so it means the same thing to whoever reads it and survives — which
    is what lets a follow-up keep its published comparators (`dev_t2_03`).

    Nothing is re-pointed and nothing is renumbered. A sentence that loses its marker
    reads as a claim whose evidence sits in the replayed tool results beside it, which is
    the truth of the situation: those results are the only numbered source in context.
    """
    if "^[" not in (answer or ""):
        return answer
    out: list[str] = []
    last = 0
    removed = False
    for match in _ANY_MARKER_RE.finditer(answer):
        body = match.group(1).strip()
        if not _NUMERIC_BODY_RE.fullmatch(body):
            # A literature badge, or a section name the model invented. Neither is a
            # position in the array, and `_drop_unknown_markers` owns the second.
            continue
        start = match.start()
        while start > last and answer[start - 1] in " \t":
            start -= 1
        out.append(answer[last:start])
        last = match.end()
        removed = True
    if not removed:
        return answer
    out.append(answer[last:])
    return _tidy_removed_markers("".join(out))


def drop_unknown_markers(answer: str) -> str:
    """The vocabulary rule, for the settlement pass.
    Named publicly because `answer.sanitize_answer_markers` applies it at settlement —
    the one point the trace, the judge and the browser all read the answer from — and
    that pass may not reach for a private name. It lived only inside
    `renumber_citations`, which the presentation layer called; that layer publishes the
    settled prose verbatim now (Charter 13 Law 5), so the sweep has to happen before the
    prose is published rather than on the way to the browser.
    """
    return _drop_unknown_markers(answer)


#: A markdown list item — a bullet or an ordinal, its indent, and its marker.
_LIST_ITEM_RE = re.compile(r"^(\s*)(?:[-*+]|\d+[.)])\s+")

#: A citation marker closing a line, with whatever punctuation followed it.
_TRAILING_MARKER_RE = re.compile(
    r"\s*" + MARKER_OPEN + r"\s*(\d+)\s*" + MARKER_CLOSE + r"(?P<tail>[.,;:]?)\s*$"
)


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
    the number the model wrote — a marker written first was renumbered to `^[1]`
    and the panel after it slid one place onto its neighbours.

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


def compact_display_numbers(answer: str) -> tuple[str, dict[str, int]]:
    """Renumber the answer's markers to the order they first appear: `1..K`.

    Returns the prose and the map the client needs to resolve what it now shows:
    `{display_number: position in the conversation's array}`.

    Two numberings, one owner. A marker's position in the conversation's evidence array
    is stable and cumulative by design — that is what lets a follow-up cite a passage an
    earlier turn retrieved (`test_cite_offset`, TF-10) — but it is not what a reader
    should be asked to read. Turn 2 of a real thread showed `^[5]` and `^[8]` for its own
    two passages and Turn 3 showed `^[14]` and `^[15]`, because four note sections and
    nine passages came before them. The gaps are the array working correctly; the reader
    sees a document with holes in it.

    So the array keeps its numbering and the PAGE gets 1..K. This runs at settlement,
    inside the citation owner, before the trace is written, so the served prose, the
    trace and the judge read one string (Charter 13 Law 5) and `sources[].cite` — which
    `resolve_sources` derives from `cited_numbers(answer)` — follows the same numbers
    automatically.

    Only numeric markers are rewritten. A `^[PMID: …]` badge names a record rather than a
    position, so it addresses the same thing whoever reads it and is left alone.

    A marker the enforcement kept is a marker whose unit exists, so a body that is not a
    number after `enforce_citations` has already been expanded to one marker per unit; a
    cluster arriving here is compacted number by number in place.
    """
    if not answer:
        return answer, {}
    display: dict[str, int] = {}
    out: list[str] = []
    last = 0
    for match in _ANY_MARKER_RE.finditer(answer):
        body = match.group(1).strip()
        if not _NUMERIC_BODY_RE.fullmatch(body):
            continue
        numbers = _expand_citations(body)
        shown: list[str] = []
        for number in numbers:
            key = next((k for k, original in display.items() if original == number), None)
            if key is None:
                key = str(len(display) + 1)
                display[key] = number
            if key not in shown:
                shown.append(key)
        out.append(answer[last:match.start()])
        # Written exactly as `_renumber_citations` writes one: `MARKER_OPEN` is a pattern
        # fragment (it carries the backslashes), not the token a reader sees.
        out.append(" ".join(f"^[{number}]" for number in shown))
        last = match.end()
    if not display:
        return answer, {}
    out.append(answer[last:])
    return "".join(out), display


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

    The single definition of that count. The model's numbering and every enforcement
    pass advance by this and resolve against it, so a marker the model was given and the
    record it opens are the same position by construction.
    """
    return len(citable_passages(payload))


@dataclass(frozen=True)
class EvidenceLedger:
    """One turn's citable units, in the order the model was shown them.

    The audit's §2.3: four modules resolved the same evidence and each counted it its own
    way — this module counted it, `engine` numbered it for the model, `presentation`
    re-counted it, and `evaluation` re-derived it. A number that means one thing in the
    transcript and another in the drawer is a reader clicking a citation and landing on
    an unrelated passage.

    So the array is built once, here, and everything positional is asked of it:

      * `units` — every citable unit the conversation produced, note passages and
        laboratory analytes, in rendering order. That is the array a marker's number
        indexes, so a marker and the record it opens are the same position by
        construction.
      * `passage_count` — its length, and the ONLY count of it. `citation_bound` and
        the presentation layer both read this one.
      * `literature_start` — where the published records begin: after every citable
        unit, not after the units the answer happened to cite. A turn that cited
        passages 4 and 7 has consumed up to 7 while only two rows resolve, and numbering
        the records from 3 gave a published record the same number as note passage 4
        (2026-10-30).

    Immutable and cheap: it holds references to the tool calls, not to copied prose.
    """

    #: The calls whose results it was built from, in the order the model received them:
    #: the replayed turns first, then this one.
    calls: tuple[dict, ...] = ()
    #: Every citable unit those calls contributed, concatenated in the same order.
    units: tuple[dict, ...] = ()

    @classmethod
    def of(cls, *call_groups: "list[dict] | None",
           evidence_state: Any = None) -> "EvidenceLedger":
        """Build a ledger from call groups, in the order the model was shown them.

        The order is the caller's, because two callers ask two different questions.
        `citation_bound` is asked what the answer may address and passes the replayed
        turns first, then this one — the numbering the model was given. The presentation
        layer is asked the same and passes the same.

        `evidence_state` replaces that concatenation with the session's own array
        (`SessionEvidenceState.ordered_citable_units`). The two disagree the moment a
        passage is retrieved twice: concatenating gives it two positions, the record gives
        it the position it already had. Only one of them can be the numbering a marker
        was written against, and it is the record's — a marker written in turn 1 still
        addresses the passage it was written for after turn 2 re-retrieves it. Passing the
        state is therefore not an optimisation: it is the difference between one array and
        two.

        Duck-typed rather than annotated with the class: `evidence_state` imports this
        module, so naming it here would close a cycle. `ordered_citable_units` is the
        whole contract between the two.
        """
        calls = tuple(
            call for group in call_groups for call in (group or [])
            if isinstance(call, dict)
        )
        if evidence_state is not None:
            return cls(calls=calls,
                       units=tuple(evidence_state.ordered_citable_units()))
        return cls(
            calls=calls,
            units=tuple(
                unit for call in calls
                for unit in citable_passages(call.get("response"))
            ),
        )

    @classmethod
    def from_state(cls, evidence_state: Any,
                   *call_groups: "list[dict] | None") -> "EvidenceLedger":
        """The ledger of a session's record, with the calls it was built from.

        `of(..., evidence_state=...)` is the same construction; this spelling exists so a
        reader who has the record does not have to pass it as a keyword beside the calls
        it already holds.
        """
        return cls.of(*call_groups, evidence_state=evidence_state)

    @property
    def passage_count(self) -> int:
        """How many numbered positions the tool results consumed."""
        return len(self.units)

    @property
    def literature_start(self) -> int:
        """The number the first published record takes, after every citable unit."""
        return self.passage_count + 1

    def unit(self, number: int) -> dict | None:
        """The citable unit at a 1-based marker number, or None when it addresses nothing."""
        if 1 <= number <= len(self.units):
            return self.units[number - 1]
        return None


#: A sentence that says it is reporting the note. The complement is what the laboratory
#: check below is for: a marker into the series beside a claim about the chart.
_NOTE_CLAIM_RE = re.compile(
    r"\b(?:notes?|chart|record|documented|documentation|discharge summary)\b",
    re.IGNORECASE,
)

#: Sentences end at punctuation the prose actually uses. Deliberately crude: this splits
#: to attribute markers to a sentence, and an over-long sentence only widens the window a
#: flag is raised over — it cannot invent a mismatch.
_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")


def evidence_kind_mismatches(
    answer: str,
    tool_calls: list[dict] | None,
    conversation_calls: list[dict] | None = None,
    *,
    evidence_state: Any = None,
) -> list[str]:
    """Markers that cite laboratory series beside a claim about the note.

    `enforce_citations` checks that a marker resolves; it cannot check that it resolves
    to the right KIND of evidence, because it never sees what the unit is. The prompt
    states the rule the check is about — "never attach a laboratory-trend index to a
    narrative claim and never attach a note-section index to a laboratory value"
    (`engine/prompts.py`) — and on 2026-10-31 turn 3 of the 90000023 thread attached two
    laboratory series (`^[14]`, `^[15]`) to "The discharge notes for this admission do
    not provide a clinical prioritization", and nothing objected: both numbers were in
    range, so the range check passed a sentence whose evidence is the wrong kind.

    Reported, not rewritten. Deleting or replacing the sentence needs a claim classifier
    this system does not have, and guessing at one is how a working answer gets damaged;
    a flag is what makes the defect visible to the reader and countable by the harness.

    `evidence_state` is read for the same reason `enforce_citations` reads it: a number is
    a position in ONE array, and this check must classify the unit at the position the
    enforcement kept rather than at a position the payload concatenation happens to give
    the same integer.
    """
    if not answer:
        return []
    ledger = EvidenceLedger.of(conversation_calls, tool_calls,
                               evidence_state=evidence_state)
    if not ledger.units:
        return []
    flags: list[str] = []
    for sentence in _SENTENCE_RE.findall(answer):
        cited: list[int] = []
        for match in _ANY_MARKER_RE.finditer(sentence):
            body = match.group(1).strip()
            if _NUMERIC_BODY_RE.fullmatch(body):
                cited.extend(_expand_citations(body))
        if not cited:
            continue
        units = [ledger.unit(number) for number in cited]
        # Every marker in the sentence must land on a laboratory series. A sentence
        # resting on both kinds is the mixed case the prompt allows, and flagging it
        # would flag correct prose.
        if not all(unit is not None and unit.get("section") == LAB_SECTION
                   for unit in units):
            continue
        if not _NOTE_CLAIM_RE.search(sentence):
            continue
        flags.extend(f"citation_kind_mismatch:^{number}" for number in cited)
    return flags


def citation_bound(
    tool_calls: list[dict] | None,
    conversation_calls: list[dict] | None = None,
    *,
    evidence_state: Any = None,
) -> int:
    """How many numbered citations this conversation's answer may address.

    The length of the array the model was numbered against: one unit per note passage AND
    one per laboratory analyte, over the replayed turns first and this turn second
    (`replay_messages` and `tool_node` advance the same sum). A literature turn has no
    such units, so its article count stands in — a `^[n]` on a `/pubmed` answer addresses
    one of the records the search returned.

    The one counter. Two existed until 2026-10-29 — this sum and a passages-only count in
    `guardrail` — and their disagreement deleted nine correct analyte markers from a
    first-turn `/labs` answer (`pilot-verify-20261028-205050`). It is now asked of the
    ledger rather than summed here, so there is one array and one length (Phase 4;
    audit §2.3).

    With `evidence_state`, the length is the record's: a passage retrieved twice is one
    position, not two.
    """
    ledger = EvidenceLedger.of(conversation_calls, tool_calls,
                               evidence_state=evidence_state)
    return ledger.passage_count or literature_slots(tool_calls)


#: The scope a `search_literature` call must declare to be the ANSWER's evidence.
_EVIDENCE_SCOPE = "general"


def _searched_for_evidence(call: dict) -> bool:
    """Whether this search was made to answer the question rather than to serve another part of the
    pipeline.

    `search_literature` has two callers with different questions. The model asks what the
    literature says, and those records are the answer's evidence: its pills, its drawer
    rows, the PMIDs enforcement accepts. The readmission retrieval asks for the candidate
    pool an extraction turn verifies rates out of, and its survivors are published by the
    comparison table instead — as cohort rows carrying their own PMIDs.

    Both arrive in `tool_calls` under one name, so the call's own declared `scope` is what
    separates them. Without this the `/ehr` turn that gate 2 wired would have drawn itself
    as a literature turn: the risk widget suppressed, a caption about PubMed records, and
    twenty-five candidates published as if the answer had cited them.

    A call that declares no scope searched for evidence — `general` is the tool's own
    default, and every caller before the comparator work left it unset.
    """
    scope = str((call.get("args") or {}).get("scope") or _EVIDENCE_SCOPE).strip().lower()
    return scope == _EVIDENCE_SCOPE


def literature_calls(
    tool_calls: list[dict] | None,
    conversation_calls: list[dict] | None = None,
) -> list[dict]:
    """Every `search_literature` call this turn's answer may rest on, in call order.

    ALL of this turn's searches, not the first. A model that rephrases its query
    searched four times in one turn on 2026-10-30 and wrote its answer against the
    records the LAST of them returned, while every reader here took the first — so the
    two records it cited were not in the allowance, the composer had no articles, and
    the answer came back with a stripped citation and an empty references drawer.

    Failing that, the newest search in the replayed turns, for the reason
    `literature_call` gives: a declared follow-up binds no literature tool because the
    earlier result is already in the transcript. The replay is a fallback, never an
    addition — a turn with its own searches answers from those.

    A search made for another consumer of the pipeline is not one of these; see
    `_searched_for_evidence`.
    """
    found = [
        call for call in (tool_calls or [])
        if isinstance(call, dict) and call.get("name") in routing.LITERATURE_TOOLS
        and _searched_for_evidence(call)
    ]
    if found:
        return found
    replayed = [
        call for call in reversed(list(conversation_calls or []))
        if isinstance(call, dict) and call.get("name") in routing.LITERATURE_TOOLS
        and _searched_for_evidence(call)
    ]
    return replayed[:1]


def literature_call(
    tool_calls: list[dict] | None,
    conversation_calls: list[dict] | None = None,
) -> dict | None:
    """The first call `literature_calls` returns, or None.

    Kept for the callers that want one call rather than all of them — the provenance a
    caption names, the arguments a record was fetched with. Anything deciding what the
    answer may CITE has to use `literature_articles` over `literature_calls`, or it
    reads one search and calls it the turn's literature.
    """
    calls = literature_calls(tool_calls, conversation_calls)
    return calls[0] if calls else None


def literature_articles(calls: list[dict] | None) -> list[dict]:
    """Every record those searches returned, once each, in the order first seen.

    Deduplicated by pmid and keyed by it: two searches for the same condition usually
    overlap, and a record listed twice would be two drawer rows and two numbers for one
    paper. A record with no pmid is dropped, which is the rule everywhere downstream —
    a record that cannot be linked or cited cannot be shown (`a2ui._usable_articles`).
    """
    seen: set[str] = set()
    articles: list[dict] = []
    for call in calls or []:
        response = (call or {}).get("response")
        if not isinstance(response, dict):
            continue
        for article in response.get("articles") or []:
            if not isinstance(article, dict):
                continue
            pmid = str(article.get("pmid") or "").strip()
            if not pmid or pmid in seen:
                continue
            seen.add(pmid)
            articles.append(article)
    return articles


def literature_slots(tool_calls: list[dict] | None) -> int:
    """How many articles a literature turn can cite: the records holding a PMID.

    Matched to the canvas, which drops a record with no pmid because it cannot be cited
    or linked (`a2ui._usable_articles`), so the numbered citations a reader can open are
    exactly these records.

    Every search of this turn counts, because the prompt numbers the records the model
    was shown and the model was shown all of them. Read from this turn's own calls only:
    the numbered form addresses positions in the array the model was numbered against,
    and a turn that ran no search was handed no such array.
    """
    return len(literature_articles(literature_calls(tool_calls)))


def literature_pmids(
    tool_calls: list[dict] | None,
    conversation_calls: list[dict] | None = None,
) -> frozenset[str]:
    """The PMIDs a literature turn may name instead of a position.

    The companion to `literature_slots`, read from the same records, so the identifiers
    `enforce_citations` accepts and the records the canvas draws are the same set.

    Every search of the turn contributes, not the first: a model that refined its query
    cited a record from its third search, and an allowance built from the first deleted
    the citation from a sentence that was correct.

    `conversation_calls` carries the replayed turns' calls, and is passed only by a turn
    whose question is about published records — the same `wants_literature` that let the
    replay keep the earlier search in the first place (`graph.replay_messages`). Read
    against this turn alone, a declared follow-up that cites the records it was shown has
    an empty allowance, so every PMID in it is an identifier nobody retrieved. That is
    the live defect of 2026-10-30: a chip answer citing three PMIDs from the turn before
    carried no resolvable citation and an empty references drawer, while the model was
    doing exactly what the replay invited.

    Not widened for the numbered form: see `literature_slots`. A badge names a record,
    and a record may be cited from the transcript; a position names a slot in this turn's
    own array, and there is no such array when this turn ran no search.
    """
    return frozenset(
        str(article["pmid"]) for article in literature_articles(
            literature_calls(tool_calls, conversation_calls))
    )


def record_pmids(rows: "Iterable[dict] | None") -> frozenset[str]:
    """The PMIDs of a list of verified records — published cohorts or articles.

    One rule for every channel that may widen the badge allowance: a record is citable
    because something verified it, and what is read is its identifier. The comparison
    table's cohorts and the records an earlier turn of the thread retrieved are verified
    by different routes and allowed by the same rule (2026-11-04).
    """
    return frozenset(
        str(row.get("pmid") or "").strip()
        for row in rows or []
        if isinstance(row, dict) and str(row.get("pmid") or "").strip()
    )


def comparator_pmids(comparators: "list[dict] | None") -> frozenset[str]:
    """The PMIDs of the published cohorts this turn verified and drew.

    A third channel beside the note passages and the answer's own literature, because a
    cohort row is neither: it is a record retrieval offered, an extraction turn checked
    against the abstract, and the comparison table published. Its PMID is on screen in the
    table and in the references list, so the narrative that explains the comparison has to
    be able to name it — and a `^[PMID: …]` badge naming a record the turn did not verify
    is exactly the invented identifier `enforce_citations` exists to delete.

    Read from the VERIFIED rows rather than from the retrieval pool. Twenty-five
    candidates come back and at most two survive; allowing the pool would let the answer
    cite a cohort the table refused, and the pill would resolve to nothing because
    `register_comparator_sources` registers only the rows that were drawn.

    Empty for every turn that is not `/ehr`, and for `/ehr` on an admission whose
    condition the literature could not answer for: no rows, nothing citable, and the prose
    says so in its own words rather than in a marker.
    """
    return record_pmids(comparators)


def cited_pmids(answer: str) -> list[str]:
    """The PMIDs the ANSWER names, in the order it first names them.

    The complement of `literature_pmids`, which says what the turn may cite. This says
    what it did, and the two are not the same question: a search returns three records and
    an answer rests on one, which is what the reader's reference list should reflect.

    Read off the badges, because a published record is addressed by identifier rather than
    by position (`citation_header` asks the model to copy the badge). A turn that cites no
    record returns an empty list, and the caller must decide what that means for its
    reference list — it is not the same statement as "this turn cited none of the
    retrieved records", which is what this function can tell apart.
    """
    found: list[str] = []
    for match in _ANY_MARKER_RE.finditer(answer or ""):
        body = match.group(1).strip()
        if not _PMID_LIST_BODY_RE.fullmatch(body):
            continue
        for number in re.findall(r"\d+", body):
            if number not in found:
                found.append(number)
    return found


def is_evidence_literature_call(call: dict[str, Any]) -> bool:
    """Whether one `search_literature` call searched for the answer's evidence.

    The canonical spelling of `_searched_for_evidence`, exposed because three separate
    stages now have to ask the same question and only two of them had a name for it:
    the settlement allowance, the record's ingest, and `cited_articles`. The third
    asked nothing at all — it takes a payload and an answer, never the call — so a
    comparator pool reached whatever called it.

    See `_searched_for_evidence` for what the rule is and why the scope decides it.
    """
    return _searched_for_evidence(call)


def cited_articles(payload: object, answer: str, *,
                   call: dict[str, Any] | None = None) -> list[dict]:
    """The records one search returned that that turn's OWN answer addressed by PMID.

    A published record is thread evidence only when the turn that retrieved it also rested
    on it. A comparator search returns twenty-five candidates and at most two survive
    extraction; carrying the pool into a later turn would let a follow-up cite a cohort no
    earlier turn used, which is the invented identifier `enforce_citations` exists to
    delete (the same argument `comparator_pmids` makes about the pool it reads).

    Read off the badges in the answer, because an identifier is how prose addresses a
    published record rather than a position in the array. An answer that cites no record
    contributes none, and the turn it came from is then not evidence for a later one.

    `call` is the search this payload came from, and passing it is what applies the scope
    rule: a call that declared `scope="readmission"` was made for the comparison table,
    not for the answer, and contributes nothing here however many of its records the prose
    names. Omitting it is the pre-2026-11-05 behaviour and is still correct for a caller
    that has already established the call is an evidence search — `graph.replay_messages`
    is the one such caller, and it must keep the cited cohorts, because the badge a
    follow-up repeats has to resolve against the record the thread actually rested on
    (`test_the_follow_up_sees_the_prediction_and_the_comparator_records`).
    """
    if call is not None and not is_evidence_literature_call(call):
        return []
    if not isinstance(payload, dict):
        return []
    named = set(cited_pmids(answer))
    if not named:
        return []
    return [
        article for article in payload.get("articles") or []
        if isinstance(article, dict) and str(article.get("pmid") or "") in named
    ]


def citation_header(number: int, modality: str) -> str:
    """The token the model is asked to copy, printed beside the evidence: `^[6]`.

    `cite` alone says which number a unit carries, and that was not enough — six
    multi-turn items scored `citation <= 1` — so a copyable token was printed beside the
    evidence. The token was `[CITATION ^[6] | CLINICAL NOTE SECTION — report]`, and that
    was the defect: on 2026-11-01 the served answer of an `/ehr` chip follow-up ended
    with the bracket verbatim, drawn around a resolved pill, and a risk-chip follow-up
    carried two of them with labels no code path prints (`ADMISSION METADATA`,
    `PREDICT_READMISSION`). A grammar is a thing that can be generalised; a token cannot.

    So the printed artifact is now the marker and NOTHING else — the one string the
    browser can draw. Whatever the model copies is drawable by construction, there is no
    bracket to over-copy, and no per-modality syntax to extend onto evidence that owns no
    citation at all. `modality` stays in the signature because the label still travels in
    `cite_modality`, which the prompt reads and `evidence_kind_mismatches` checks; it is
    deliberately not interpolated into the token.

    A copy of the OLD form still reaching an answer is handled by
    `strip_printed_constructs`, which is the half of this contract that does not depend on
    the model complying.
    """
    del modality  # carried by `cite_modality`; never interpolated into the token
    return f"^[{number}]"


def citation_fields(number: int, modality: str) -> dict[str, Any]:
    """The three fields every citable unit carries, built in one place.

    `cite` is the integer position, `cite_modality` the kind of evidence, and
    `cite_header` the marker printed ready to copy. One function because they must agree:
    a header naming `^[6]` beside a `cite` of 3 would be a worse defect than no header.
    """
    return {
        "cite": number,
        "cite_modality": modality,
        "cite_header": citation_header(number, modality),
    }


#: What a zero-unit payload says about itself. Printed where the model composes the
#: sentence, not only in the system prompt: on `pilot-verify-20261029-170609`,
#: `holdout_t2_02` wrote `^[13]` on "the discharge medication section is not documented"
#: — a sentence about a section that returned `"passages": []` and therefore owned no
#: number. The rule against it was already in `SYSTEM_PROMPT`; what the model lacked was
#: any local signal that this result had no slot to reach for, so it took the next number
#: it could see, which was the brief hospital course.
ZERO_UNIT_NOTE = (
    "this result publishes no citable unit; no sentence about it may carry a ^[n]"
)


def zero_unit_fields() -> dict[str, Any]:
    """The annotation a citation-bearing payload carries when it contributes no number.

    A retrieval result with no unit is not inert: it is the difference between "the
    section lists nothing" and "the section was never fetched", and the model has to know
    that about the result in front of it.

    Only a payload that carries `passages` or `trends` can wear these fields. A
    `predict_readmission` result and a `search_literature` result are answers, not
    evidence: the first owns no citation surface at all, and the second is cited by
    `^[PMID: ...]`, so stamping either one `citable: false` would contradict a marker that
    is valid on that turn.

    These fields are description only. `citable_passages` and `passage_total` key on
    `passages` and `trends` and never read `cite`, so a payload wearing them still
    counts zero — the numbering of every later unit is unchanged.
    """
    return {"cite": None, "citable": False, "cite_note": ZERO_UNIT_NOTE}


#: Every construct this package prints FOR the model, by name. The normaliser below
#: handles each one and the flag it raises names the form, so an answer that served one
#: records WHICH. A form added to a printer without an entry here is a leak no test can
#: see, which is why `tests/agent/test_printed_constructs.py` walks this tuple and fails
#: until the form has a sample the normaliser removes.
PRINTED_FORMS = ("cite_header", "admission_metadata", "zero_unit_note")

#: Spans this package prints FOR the model, recognised by the word they open with and
#: never by their contents. The contents are the model's to fill in — it invented
#: `ADMISSION METADATA` and `PREDICT_READMISSION` on 2026-11-01 — so a rule written
#: against the labels would need one entry per label the model has not invented yet.
_PRINTED_SPAN_TAGS = ("citation", "cite")

_PRINTED_SPAN_RE = re.compile(
    r"\s*(?:" + "|".join(_PRINTED_SPAN_TAGS) + r")\b", re.IGNORECASE
)

#: The annotated block the admission row arrives in
#: (`routing.admission_context.render`). Printed into the prompt, so it is copyable.
_ADMISSION_BLOCK_RE = re.compile(
    r"<admission_metadata\b[^>]*>.*?</admission_metadata>|<admission_metadata\b[^>]*/?>",
    re.IGNORECASE | re.DOTALL,
)


def _marker_mask(text: str) -> list[bool]:
    """Which characters of `text` belong to a citation marker.

    A marker is itself bracketed — `^[1]` — so a scan that pairs brackets has to step
    over the marker WHOLE, or the marker's own bracket is read as the wrapper it is
    nested inside and every marker in the answer looks like machine syntax.
    """
    mask = [False] * len(text)
    for match in _ANY_MARKER_RE.finditer(text):
        for index in range(match.start(), match.end()):
            mask[index] = True
    return mask


def _span_end(text: str, start: int, mask: list[bool]) -> int | None:
    """The index just past the `]` closing the bracket opened at `start`, or None."""
    depth = 0
    cursor = start
    while cursor < len(text):
        if mask[cursor]:
            cursor += 1
            continue
        if text[cursor] == "[":
            depth += 1
        elif text[cursor] == "]":
            depth -= 1
            if depth == 0:
                return cursor + 1
        cursor += 1
    return None


def _tidy_printed_constructs(answer: str) -> str:
    """Close up what a removed construct left behind.

    A construct that stood on its own line leaves a paragraph break, and — when it carried
    a marker the answer had not already used — a marker starting a line. That marker
    belongs to the sentence above it, because that is the claim the construct was
    annotating, so it is pulled up onto that line and the break is closed.

    Only reached when something was removed, so a clean answer never passes through it.
    """
    text = re.sub(r"[ \t]*(?:\n[ \t]*)+(?=" + MARKER_OPEN + r")", " ", answer)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+(?=\n)", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_printed_constructs(answer: str) -> tuple[str, list[str]]:
    """Remove the constructs this package prints for the model from the served prose.

    The served answer is a rendered document, and the only machine syntax it may carry is
    syntax the renderer can draw: `^[n]`, `^[PMID: …]`. Everything else this package
    prints is a fact about the prompt, and the reader must never see it.

    Two mechanisms produced the defect this closes. A model told to "copy that marker"
    copied the whole bracket (`[CITATION ^[1] | CLINICAL NOTE SECTION —
    discharge_instructions]`, trailing the answer, 2026-11-01); and a model shown a
    grammar wrote new instances of it, labelling a note passage `PREDICT_READMISSION` and
    the admission row `ADMISSION METADATA` — strings no code path prints. `citation_header`
    no longer prints a bracket, so the first cannot recur by compliance; this pass is what
    does not depend on compliance at all.

    Three rules, and they are the print vocabulary rather than a list of sightings:

      * a span that OPENS with one of `_PRINTED_SPAN_TAGS` is machine syntax, whatever
        label follows — so an invented label costs nothing to handle;
      * a bracketed span that CONTAINS a marker is machine syntax, because a marker is a
        pivot that stands alone in a sentence and never sits inside a bracketed phrase;
      * the two annotated blocks — `<admission_metadata>` and `ZERO_UNIT_NOTE` — are
        removed by name, from the same module that prints them.

    A removed span keeps the markers it carried, unless the answer already cited that
    passage: the prompt allows one citation per passage, so a second copy of the same
    number is annotation rather than support. This is why the trailing
    `[CITATION ^[1] | …]` of a fully-cited answer disappears instead of leaving a stray
    pill on its own line.

    Returns (prose, flags). The flags reach `guardrail_flags` through
    `answer.settle_answer`, because the defect this repairs was invisible for a day: a
    model that ignored its instructions produced a turn every check called clean.
    """
    if not answer:
        return answer, []

    flags: list[str] = []
    text = answer

    if _ADMISSION_BLOCK_RE.search(text):
        text = _ADMISSION_BLOCK_RE.sub("", text)
        flags.append("citation_printed_construct:admission_metadata")
    if ZERO_UNIT_NOTE in text:
        text = text.replace(ZERO_UNIT_NOTE, "")
        flags.append("citation_printed_construct:zero_unit_note")

    mask = _marker_mask(text)
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        if not mask[index] and text[index] == "[":
            end = _span_end(text, index, mask)
            if end is not None:
                inner = text[index + 1:end - 1]
                if _PRINTED_SPAN_RE.match(inner) or any(mask[index:end]):
                    spans.append((index, end))
                    index = end
                    continue
        index += 1

    if not spans:
        if not flags:
            return answer, []
        return _tidy_printed_constructs(text), list(dict.fromkeys(flags))

    out: list[str] = []
    seen: set[str] = set()
    last = 0
    for start, end in spans:
        plain = text[last:start]
        out.append(plain)
        for match in _ANY_MARKER_RE.finditer(plain):
            seen.add(match.group(1).strip())
        kept: list[str] = []
        for match in _ANY_MARKER_RE.finditer(text[start:end]):
            body = match.group(1).strip()
            if not body or body in seen:
                continue
            seen.add(body)
            kept.append(match.group(0))
        if kept:
            out.append(" ".join(kept))
        flags.append("citation_printed_construct:cite_header")
        last = end
    out.append(text[last:])

    return _tidy_printed_constructs("".join(out)), list(dict.fromkeys(flags))


def note_modality(section: Any) -> str:
    """The modality label for one note passage, as the model reads it.

    The two labels are the vocabulary the prompt's citation rule is written in, so they
    are built here and nowhere else: a second spelling of either would be a rule the
    model cannot match against what it was shown.
    """
    name = str(section or "").strip() or "unrecorded section"
    return f"CLINICAL NOTE SECTION — {name}"


def lab_modality(trend: dict[str, Any]) -> str:
    """The modality label for one analyte's series."""
    name = str(trend.get("label") or trend.get("key") or "").strip() or "laboratory value"
    return f"STRUCTURED LAB TREND — {name}"


def literature_modality(article: dict[str, Any]) -> str:
    """The modality label for one published record.

    The identifying field and nothing else: the prompt reads `cite_modality` to check that
    a marker points at the KIND of evidence its sentence rests on, so this says what the
    unit is. A title or a journal here would be a second thing to keep in step with the
    payload, and the record already carries both.
    """
    return "PUBLISHED RECORD"


def render_numbered(
    payload: Any,
    offset: int = 0,
    dress: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    *,
    ref_ids: Sequence[int] | None = None,
) -> Any:
    """Stamp the citation numbers onto the copy of a tool result the model reads.

    One unit, one number, continuing from `offset` — how many units the conversation had
    already shown. Which payloads carry units is `citable_passages`' answer, and this
    function follows it exactly. A payload that carries `passages` or `trends` and
    contributes no unit gains `zero_unit_fields()`; anything else is returned unchanged.

    `ref_ids` is those numbers when the caller already knows them: the session record
    assigned a position to every unit it ingested and says which, in payload order
    (`SessionEvidenceState` → `IngestResult.slot_ids`). Given them, the `i`-th unit is
    stamped with `ref_ids[i]` and `offset` is not consulted. That is the difference
    between a re-retrieved passage keeping the position it was first given and being
    handed a second one, which would make a marker written in the earlier turn address
    the wrong evidence in this one.

    `dress` is the caller's own per-unit shaping (dropping machine ids, refusing an
    over-long passage). It runs before the numbers are attached, so the numbers describe
    the units the model is about to read rather than the ones the tool returned.

    This belongs here, not in the tool client: a number the model writes is a number this
    module will later have to resolve or delete, and the two must not be able to disagree.
    """
    if not isinstance(payload, dict):
        return payload

    def number_of(index: int) -> int:
        """The position of the `index`-th (1-based) unit of this payload."""
        if ref_ids is not None and 0 < index <= len(ref_ids):
            return int(ref_ids[index - 1])
        return offset + index

    def unit(item: dict[str, Any], number: int, modality: str) -> dict[str, Any]:
        shaped = dress(item) if dress is not None else dict(item)
        return {**shaped, **citation_fields(number, modality)}

    passages = payload.get("passages")
    if isinstance(passages, list):
        if not any(isinstance(passage, dict) for passage in passages):
            return {**payload, **zero_unit_fields()}
        return dict(payload, passages=[
            unit(passage, number_of(index), note_modality(passage.get("section")))
            if isinstance(passage, dict) else passage
            for index, passage in enumerate(passages, start=1)
        ])
    trends = payload.get("trends")
    if isinstance(trends, list):
        if not any(isinstance(trend, dict) for trend in trends):
            return {**payload, **zero_unit_fields()}
        return dict(payload, trends=[
            unit(trend, number_of(index), lab_modality(trend))
            if isinstance(trend, dict) else trend
            for index, trend in enumerate(trends, start=1)
        ])
    articles = payload.get("articles")
    if isinstance(articles, list) and any(isinstance(article, dict)
                                         for article in articles):
        # A published record is not a citable UNIT — `citable_passages` and
        # `passage_total` key on `passages` and `trends` and must keep counting this
        # payload as zero, because a marker into a note passage must not shift by three
        # when a search happens to sit beside it. What it does need is a number to write.
        #
        # `LITERATURE_FORMATTING` tells the model that the FIRST record is `^[1]`, and
        # until now the payload carried no `cite` at all: the only position available to
        # the model was the order of its own sentences, so a summary that opened on the
        # third record's finding wrote `^[1]` and the reader opened the first (2026-11-04).
        # The number is stamped here, on the copy the model reads, so the record it names
        # and the record it was given are the same position by construction.
        return dict(payload, articles=[
            {**article, **citation_fields(index, literature_modality(article))}
            if isinstance(article, dict) else article
            for index, article in enumerate(articles, start=1)
        ])
    if "passages" in payload or "trends" in payload:
        # The key is present but is not a usable array: still a retrieval result that
        # published nothing, and the one case the model must not try to cite.
        return {**payload, **zero_unit_fields()}
    # Neither array: a prediction, a medication row list, a literature search. Returned
    # unchanged — `search_literature` is cited by `^[PMID: ...]`, so marking it
    # `citable: false` would deny a marker that is valid on that turn.
    return payload
