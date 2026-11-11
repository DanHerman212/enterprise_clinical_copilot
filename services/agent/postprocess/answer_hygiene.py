"""Three checks on what a settled answer asserts about the evidence it cites.

Each of these closes a defect that was measured, and each is written for the CLASS of
defect rather than for the sentence that exposed it. All three run after the markers are
resolved and before the page is numbered, for the same reason the discharge filter and the
laboratory-attribution guard run there: a marker has to open a source before what the
sentence claims can be compared with it.

1. **An acronym the chart used.** A discharge summary is written for colleagues, and the
   record's own shorthand — `CHF`, `CKD`, `CABG` — is shorter and more precise for that
   reader than the words it stands for. A model turn tends to expand what the note
   abbreviated, and the expansion is not an error in itself; losing the acronym is,
   because the reader then has to map the words back to the shorthand the rest of the
   chart uses. Where a passage defines the shorthand in parentheses, the definition is
   inserted on first mention and nowhere else.

2. **A marker on a claim that something is absent.** "The record contains no dedicated
   discharge medication section" is a statement about the shape of the record. A marker on
   that sentence opens a passage, and the passage does not state an absence — it states a
   medication, in a different section. The badge promises the reader a source for a claim
   the source does not make. The markers that resolve to the section the sentence is
   talking about survive; the rest go.

3. **A run of markers on a sentence that reads nothing.** When every value in the panel
   stayed inside its interval, the honest answer says so once, and the sentence that says
   it rests on the whole panel at once — which is what a run of nine markers looks like,
   and what it means to a reader is that each of nine passages was checked against a claim
   none of them makes. A panel-wide statement carries no marker; the movements that are
   worth naming carry one each.

None of the three edits clinical prose. Each removes an attribution, or adds the chart's
own abbreviation, and each records a flag so the removal is countable rather than silent.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from services.agent.postprocess import citations
from services.agent.postprocess.discharge_section import split_sentences

#: Flags. One spelling each, so a counter in the harness reads them.
FLAG_ACRONYM_RESTORED = "chart_acronym_restored"
FLAG_ABSENCE_CITATION = "absence_claim_citation_stripped"
FLAG_BLANKET_LAB_RUN = "blanket_lab_marker_run_stripped"

#: `(ACRONYM)` next to the words it abbreviates. The phrase itself is found backwards from
#: the parenthetical: a leftmost-match regex would anchor on the earliest word it could
#: reach and compare `man with congestive heart failure` against `CHF`.
_DEFINITION_RE = re.compile(r"\(([A-Z]{2,6})\)")

#: The words an acronym can be built from, counted back from the parenthetical.
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]*")

#: How many words a definition may span. Six is the longest clinical term this pass is
#: meant for; longer than that is a clause, and a clause's initials spell nothing.
_MAX_DEFINITION_WORDS = 6

#: Words that carry no initial of their own in an acronym: "congestive heart failure" gives
#: CHF, and "history of present illness" gives HPI only if `of` is skipped.
_STOP_WORDS = frozenset({"of", "and", "the", "with", "in", "on", "for", "to", "a", "an",
                         "or", "by", "at", "from"})

#: A marker of the section a sentence says is absent: the noun it names after "no
#: dedicated", "no separate", "no distinct". Read for the SECTION vocabulary below rather
#: than parsed for grammar.
_ABSENCE_RE = re.compile(
    r"\bno\s+(?:dedicated|separate|distinct|discrete|standalone)\s+"
    r"(?P<noun>[a-z][a-z\s/,&\-]{0,60})",
    re.IGNORECASE,
)

#: The sections a "no dedicated X section" sentence can be about, keyed by the words such a
#: sentence uses. Matched against the sentence's own wording, so the sentence keeps its
#: markers when it names the section it is talking about. Singular and plural both, because
#: a note writes "no dedicated discharge medication section" as often as "... medications".
_ABSENCE_SECTION_WORDS: dict[str, tuple[str, ...]] = {
    "discharge_medications": ("discharge medication", "discharge medications",
                              "discharge meds"),
    "discharge_instructions": ("discharge instruction", "discharge instructions"),
    "brief_hospital_course": ("hospital course",),
    "history_of_present_illness": ("history of present illness", "hpi"),
    "discharge_diagnosis": ("discharge diagnos",),
    "chief_complaint": ("chief complaint",),
    "medications": ("medication list", "medication section", "medication table"),
}

#: A number as prose writes it. Not a citation pattern — Law 1's marker language is
#: `citations`' and this module never spells it — but "does this sentence quote a reading"
#: needs the numerals a sentence carries.
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

#: A unit that carries a digit of its own: `x10^3/uL` holds `10` and `3`, and those are a
#: scale rather than a reading. Removed before the scan, so only quoted VALUES survive it.
_UNIT_TOKEN_RE = re.compile(
    r"x\s*10\s*\^\s*\d+"
    r"|\b10\s*\^\s*[36]\b"
    r"|\b1\.73\s*m2\b",
    re.IGNORECASE,
)

#: How many markers make a run. Three analytes named in one sentence is a sentence doing
#: its job; four is a statement resting on a panel.
_BLANKET_RUN_MIN = 4


def _initials(phrase: str) -> str:
    """The acronym a phrase spells, skipping the words an acronym skips."""
    letters = [
        word[0].upper() for word in phrase.split()
        if word and word.lower() not in _STOP_WORDS
    ]
    return "".join(letters)


def _passage_texts(passages: Iterable[Mapping[str, Any]] | None) -> list[str]:
    texts: list[str] = []
    for passage in passages or ():
        if not isinstance(passage, Mapping):
            continue
        text = " ".join(str(passage.get("text") or "").split())
        if text:
            texts.append(text)
    return texts


def _definitions(passage: str) -> list[tuple[str, str]]:
    """Every `Expanded Phrase (ACRONYM)` this passage writes, phrase and acronym.

    The phrase is taken from the words immediately before the parenthetical, trying the
    LONGEST run first and shortening until the initials match. Trying only the leftmost
    run found "man with congestive heart failure" for `CHF` and rejected it.
    """
    found: list[tuple[str, str]] = []
    for match in _DEFINITION_RE.finditer(passage):
        acronym = match.group(1)
        before = passage[:match.start()]
        words = _WORD_RE.findall(before)
        for length in range(min(_MAX_DEFINITION_WORDS, len(words)), 1, -1):
            run = words[-length:]
            # The run that spells the acronym usually starts on a preposition — "with
            # congestive heart failure (CHF)" — and the phrase a reader's prose contains
            # is the term alone, so the connective is trimmed before the comparison.
            while run and run[0].lower() in _STOP_WORDS:
                run = run[1:]
            while run and run[-1].lower() in _STOP_WORDS:
                run = run[:-1]
            if len(run) < 2:
                continue
            phrase = " ".join(run)
            if _initials(phrase) == acronym:
                found.append((phrase, acronym))
                break
    return found


def restore_chart_acronyms(
    prose: str,
    passages: Iterable[Mapping[str, Any]] | None = None,
) -> tuple[str, tuple[str, ...]]:
    """The prose with the chart's own acronyms put back on first mention.

    Only a definition the PASSAGE writes is used: `<Expanded Phrase> (<ACRONYM>)` where the
    acronym is two to six capitals and spells the initials of the phrase's content words.
    Nothing is inferred from a drug name or a diagnosis list, because an acronym this
    module invented would be a piece of clinical shorthand the record never used.

    The parenthetical goes after the first occurrence of the phrase, which is where a
    definition belongs, and only when the acronym appears nowhere in the prose already —
    a sentence that writes `CHF` unexpanded elsewhere has the acronym, and a second
    definition in the middle of an answer is noise.

    Returns `(text, flags)`.
    """
    if not prose:
        return prose, ()
    text = prose
    restored: list[str] = []
    for passage in _passage_texts(passages):
        for phrase, acronym in _definitions(passage):
            if re.search(rf"\b{re.escape(acronym)}\b", text):
                continue
            found = re.search(rf"\b{re.escape(phrase)}\b", text, re.IGNORECASE)
            if not found:
                continue
            # A phrase already followed by its own parenthetical needs nothing, and the
            # `\b` search above would otherwise insert a second one inside it.
            tail = text[found.end():found.end() + len(acronym) + 4]
            if re.match(rf"\s*\(\s*{re.escape(acronym)}\b", tail):
                continue
            text = f"{text[:found.end()]} ({acronym}){text[found.end():]}"
            restored.append(f"{FLAG_ACRONYM_RESTORED}:{acronym}")
    return text, tuple(restored)


def _units_by_position(
    tool_calls: Iterable[Mapping[str, Any]] | None,
    replayed_tool_calls: Iterable[Mapping[str, Any]] | None,
    evidence_state: Any,
) -> dict[int, dict]:
    """Marker number -> the citable unit it opens, from the one ledger that owns the order."""
    ledger = citations.EvidenceLedger.of(
        list(replayed_tool_calls or ()), list(tool_calls or ()),
        evidence_state=evidence_state,
    )
    return {
        position: unit
        for position, unit in enumerate(ledger.units, start=1)
        if isinstance(unit, dict)
    }


def _named_absence_sections(sentence: str) -> set[str]:
    """The sections a "no dedicated X" sentence is talking about, from its own words."""
    text = sentence.lower()
    return {
        section for section, forms in _ABSENCE_SECTION_WORDS.items()
        if any(form in text for form in forms)
    }


def strip_absence_claim_citations(
    prose: str,
    *,
    tool_calls: Iterable[Mapping[str, Any]] | None = None,
    replayed_tool_calls: Iterable[Mapping[str, Any]] | None = None,
    evidence_state: Any = None,
) -> tuple[str, tuple[str, ...]]:
    """Markers taken off sentences that assert a section is missing.

    A sentence qualifies when it says the record has NO DEDICATED section of a named kind.
    That is a claim about the record's shape, and a passage does not evidence it by being
    cited — the marker opens a passage that states something else, which is how a
    medication sentence came to be cited to two passages under a sentence saying no
    medication section exists.

    The markers that resolve to the section the sentence names are kept: a sentence saying
    the discharge instructions document no medications may cite the discharge instructions,
    because that passage is the one making the claim. Everything else goes.

    A sentence that names no section at all loses every numeric marker, since there is then
    no section whose passage could support it.
    """
    if not prose:
        return prose, ()
    units = _units_by_position(tool_calls, replayed_tool_calls, evidence_state)
    text = prose
    cursor = 0
    stripped: list[int] = []
    for sentence in split_sentences(prose):
        at = text.find(sentence, cursor)
        if at < 0:
            continue
        cited = citations.cited_numbers(sentence)
        if not cited or not _ABSENCE_RE.search(sentence):
            cursor = at + len(sentence)
            continue
        named = _named_absence_sections(sentence)
        offending = [
            number for number in cited
            if str((units.get(number) or {}).get("section") or "") not in named
        ]
        if not offending:
            cursor = at + len(sentence)
            continue
        replacement = citations.remove_citations(sentence, offending)
        text = f"{text[:at]}{replacement}{text[at + len(sentence):]}"
        stripped.extend(offending)
        cursor = at + len(replacement)
    if not stripped:
        return prose, ()
    return text, tuple(f"{FLAG_ABSENCE_CITATION}:^{n}" for n in sorted(set(stripped)))


def _quoted_numbers(sentence: str) -> set[float]:
    """The numerals a sentence quotes as values, with markers and unit exponents removed."""
    scrubbed = _UNIT_TOKEN_RE.sub(" ", citations.strip_markers(sentence or ""))
    found: set[float] = set()
    for token in _NUMBER_RE.findall(scrubbed):
        try:
            found.add(float(token))
        except ValueError:
            continue
    return found


def strip_blanket_lab_marker_runs(
    prose: str,
    *,
    tool_calls: Iterable[Mapping[str, Any]] | None = None,
    replayed_tool_calls: Iterable[Mapping[str, Any]] | None = None,
    evidence_state: Any = None,
) -> tuple[str, tuple[str, ...]]:
    """Markers taken off a panel-wide statement that quotes no reading.

    Four or more markers on one sentence, every one of them opening a laboratory series,
    and not a single value written in the sentence: that is a blanket statement wearing the
    whole panel as attribution. It says "nothing crossed its reference interval" and cites
    nine passages, and no one of those passages says that — the claim is the interpretation
    of all of them together.

    The answer keeps at least one citation elsewhere, so the page still has a badge a reader
    can open. Where the blanket sentence is the only one, it keeps its markers: a served
    answer with no citation at all is a worse failure than a wide one.
    """
    if not prose or len(citations.cited_numbers(prose)) < _BLANKET_RUN_MIN:
        return prose, ()
    units = _units_by_position(tool_calls, replayed_tool_calls, evidence_state)
    lab_section = citations.LAB_SECTION

    def _is_lab(number: int) -> bool:
        return str((units.get(number) or {}).get("section") or "") == lab_section

    sentences = split_sentences(prose)
    candidates: list[tuple[str, list[int]]] = []
    for sentence in sentences:
        cited = citations.cited_numbers(sentence)
        if len(cited) < _BLANKET_RUN_MIN:
            continue
        if not all(_is_lab(number) for number in cited):
            continue
        if _quoted_numbers(sentence):
            continue
        candidates.append((sentence, cited))
    if not candidates:
        return prose, ()

    # Some other sentence has to keep a marker, or the page is served with no badge at
    # all. Counted per SENTENCE rather than per number, because the same series can be
    # cited on both sentences: the panel statement and the movement beside it may both
    # point at `^[1]`, and stripping one of them leaves the other's marker intact.
    edited = {sentence for sentence, _ in candidates}
    remaining = [citations.cited_numbers(sentence) for sentence in sentences
                 if sentence not in edited]
    if not any(remaining):
        return prose, ()

    text = prose
    cursor = 0
    stripped: list[int] = []
    for sentence in sentences:
        at = text.find(sentence, cursor)
        if at < 0:
            continue
        if sentence not in edited:
            cursor = at + len(sentence)
            continue
        cited = citations.cited_numbers(sentence)
        replacement = citations.remove_citations(sentence, cited)
        text = f"{text[:at]}{replacement}{text[at + len(sentence):]}"
        stripped.extend(cited)
        cursor = at + len(replacement)
    if not stripped:
        return prose, ()
    return text, tuple(f"{FLAG_BLANKET_LAB_RUN}:^{n}" for n in sorted(set(stripped)))
