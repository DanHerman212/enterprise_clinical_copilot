"""Chunking of discharge notes into citable passages.

Pure standard library, no cloud calls (same constraint as rag/sections.py).

A chunk is the unit of retrieval: one Vector Search datapoint, one citable
passage. Rules, in order:

  1. Split on section boundaries (parse_note does the splitting).
  2. Text before the first recognised heading is the history of a discharge
     summary: it is chunked as `history_of_present_illness`, and a note with no
     recognised heading at all is chunked whole as `discharge_summary`.
  3. A section at or under max_chars is one chunk.
  4. A longer section splits on paragraph (blank line), then sentence.
  5. Chunks that are entirely redaction or empty are dropped - they cost money
     to embed and can never support a citation.

chunk_id is deterministic: "{note_id}:{section}:{ordinal}", so re-running the
chunker produces identical IDs and a re-index never orphans chunks. Offsets are
within the section body, so a citation reconstructs as "note {note_id},
{section}, chars X-Y".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from services.mcp.retrieval.sections import (
    ADMISSION_DIAGNOSIS,
    DISCHARGE_DIAGNOSIS,
    DISCHARGE_SUMMARY,
    HISTORY_OF_PRESENT_ILLNESS,
    KNOWN_HEADINGS,
    SUMMARY_SECTIONS,
    Section,
    parse_note,
    redaction_profile,
)

# D2 default from the build guide; sections under this are single chunks.
DEFAULT_MAX_CHARS = 1500

# Build-time packing size used by the ingest pipeline. Serving re-chunks notes
# to resolve a citation's exact text, so it must pack with the SAME value or
# chunk ids drift and every packed section silently degrades to whole-note
# text. Change this only together with an index rebuild.
DEFAULT_PACK_TO = 700

# Narrative/assessment sections worth indexing. Metadata (name, dates, sex) and
# lab-line noise (pertinent_results) add embedding cost without retrieval value.
# Single source of truth: the ingest pipeline whitelist, the build scripts, and the
# serving-side datapoint-id parser all import this tuple.
#
# The clinical narrative a free-text question actually asks about lives in the sections
# added on 2026-11-03 — the operation, the complications, the consults, the plan, the
# follow-up and the complaints that opened the admission. Without them the only route to
# a procedure was whatever the hospital course happened to repeat: 39 of 87 cohort notes
# carry a `major_procedure` section the index could not return, and 10 carry an
# `assessment_and_plan` (measured 2026-11-03).
#
# `pertinent_results` stays out. It is a line-per-analyte block whose embedding buys
# nothing the structured lab tool does not answer better.
INDEX_SECTIONS = (
    "history_of_present_illness",
    "past_medical_history",
    "family_history",
    "social_history",
    "review_of_systems",
    "chief_complaint",
    "allergies",
    "physical_exam",
    "brief_hospital_course",
    "major_procedure",
    "complications",
    "consults",
    "assessment_and_plan",
    "discharge_condition",
    "admission_diagnosis",
    "discharge_diagnosis",
    "discharge_medications",
    "medications_on_admission",
    "discharge_disposition",
    "discharge_instructions",
    "followup_instructions",
    "discharge_summary",
)

_unknown = set(INDEX_SECTIONS) - set(KNOWN_HEADINGS)
if _unknown:  # a typo here would silently index nothing for that section
    raise ValueError(
        f"INDEX_SECTIONS entries not in rag.sections.KNOWN_HEADINGS: "
        f"{sorted(_unknown)}"
    )
del _unknown

# SUMMARY_SECTIONS lives in `sections.py` now, with the rest of the vocabulary. This
# module is the chunker, and the agent's image ships the vocabulary but not the
# chunker — so an agent that imported the constant from here failed to start, and the
# deploy was rolled back to the previous revision (live defect, 2026-10-13). It is
# still importable from this module: the import above re-exports it.

_summary_unknown = set(SUMMARY_SECTIONS) - set(INDEX_SECTIONS)
if _summary_unknown:  # a summary can only cite sections the index holds
    raise ValueError(
        f"SUMMARY_SECTIONS entries not in INDEX_SECTIONS: {sorted(_summary_unknown)}"
    )
del _summary_unknown

_PARAGRAPH_RE = re.compile(r"\n\s*\n")
# Sentence boundary: punctuation + whitespace + capital letter. Deliberately
# conservative; a run-on that defeats it hits the fixed-width fallback.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    hadm_id: int
    note_id: int
    section: str
    char_start: int  # within the section body
    char_end: int
    text: str


def chunk_note(note: dict, max_chars: int = DEFAULT_MAX_CHARS,
               pack_to: int | None = None) -> list[Chunk]:
    """Chunk one note dict {"hadm_id", "note_id", "text"} into passages.

    `pack_to`, when set, greedily merges adjacent pieces (paragraphs/sentences)
    into spans up to that size. Line-oriented sections like Pertinent Results
    would otherwise yield a 40-char chunk per line; packing gives retrieval
    units of usable size. Defaults to no packing (each piece is its own chunk).

    Lossless at the document's edges (2026-11-03). `parse_note` only names text that
    follows a RECOGNISED heading, so two kinds of note text produced no chunk at all and
    could not be retrieved by any query:

      * the narrative before a note's first heading, which is the history in a discharge
        summary — emitted as `history_of_present_illness`, the same name and the same
        text `_search_sections` gives it;
      * a note written without headings, which parsed to no sections at all — emitted as
        one `discharge_summary` passage rather than as nothing. Three of eighty-seven
        cohort admissions were invisible to `rag_search` for exactly this reason.

    Both were already handled on the `/ehr` path; this is the index agreeing with it.

    chunk_ids are unique per note: ordinal continues across repeated sections of
    the same name (a note can carry both "Physical Exam" and "Discharge Exam",
    or "Past Medical History" and its "PMH" alias), so the second instance does
    not collide with the first.
    """
    normalised = str(note.get("text") or "").replace("\r\n", "\n").replace("\r", "\n")
    parsed = parse_note(normalised)

    pieces: list[Section] = []
    if parsed.sections:
        preamble = normalised[:parsed.sections[0].start].strip()
        if preamble and not redaction_profile(preamble).is_placeholder_only:
            pieces.append(Section(name=HISTORY_OF_PRESENT_ILLNESS, heading="",
                                  body=preamble, start=0,
                                  end=parsed.sections[0].start))
        pieces.extend(parsed.sections)
    else:
        body = normalised.strip()
        if body and not redaction_profile(body).is_placeholder_only:
            pieces.append(Section(name=DISCHARGE_SUMMARY, heading="", body=body,
                                  start=0, end=len(normalised)))

    chunks: list[Chunk] = []
    name_ordinals: dict[str, int] = {}
    for section in _drop_duplicate_diagnoses(pieces):
        base = name_ordinals.get(section.name, 0)
        section_chunks = _chunk_section(note, section, max_chars, pack_to, base)
        chunks.extend(section_chunks)
        name_ordinals[section.name] = base + len(section_chunks)
    return chunks


def _norm_body(text: str) -> str:
    """A section body folded to the form two bodies are compared in.

    Whitespace collapsed and case removed, so a note that prints one block under two
    headings with different line wrapping is recognised as one block. Deliberately the
    same rule `retrieval._norm_body` applies on the serving side: the two layers have to
    agree about when two bodies are one, or the index holds two copies and the passage
    hides one of them.
    """
    return " ".join(str(text or "").split()).lower()


def _drop_duplicate_diagnoses(sections: list[Section]) -> list[Section]:
    """One passage per diagnosis list, labelled the discharge way.

    A note that prints the same list under `ADMISSION DIAGNOSES` and again under
    `DISCHARGE DIAGNOSES` is stating one fact twice. `_search_sections` keeps the second
    and tags it as the discharge diagnosis (`_keep_final_diagnosis`), so the reading a
    summary answers with is the final one; an index that embedded both copies would let a
    retrieval cite the admission heading for a discharge fact, and would spend two
    vectors on one block.

    Only that pair, and only when the bodies agree once folded. A note whose diagnoses
    were revised between admission and discharge holds two different lists, and both are
    indexed: the difference between them is a finding, not a duplicate.
    """
    final = {_norm_body(section.body) for section in sections
             if section.name == DISCHARGE_DIAGNOSIS}
    if not final:
        return sections
    return [
        section for section in sections
        if not (section.name == ADMISSION_DIAGNOSIS
                and _norm_body(section.body) in final)
    ]


def _chunk_section(note: dict, section, max_chars: int,
                   pack_to: int | None, base: int) -> list[Chunk]:
    body = section.body
    if not body or redaction_profile(body).is_placeholder_only:
        return []

    if len(body) <= max_chars:
        return [_make_chunk(note, section, body, 0, len(body), base + 1)]

    pieces = [
        (text, start, end)
        for text, start, end in _split_long_body(body, max_chars)
        if text.strip() and not redaction_profile(text.strip()).is_placeholder_only
    ]
    spans = _pack(pieces, pack_to, body)

    chunks: list[Chunk] = []
    for ordinal, (start, end) in enumerate(spans, start=base + 1):
        stripped = body[start:end].strip()
        if not stripped or redaction_profile(stripped).is_placeholder_only:
            continue
        leading = end - start - len(body[start:end].lstrip())
        chunks.append(
            _make_chunk(note, section, stripped,
                        start + leading, start + leading + len(stripped), ordinal)
        )
    return chunks


def _pack(pieces: list[tuple[str, int, int]], pack_to: int | None,
          body: str) -> list[tuple[int, int]]:
    """Greedily merge adjacent pieces into spans up to `pack_to` chars.

    A merged span runs from the first piece's start to the last piece's end;
    the whitespace between them stays in the span, which keeps the invariant
    body[start:end] == chunk text true. Pieces merge only when the body
    between them is pure whitespace: a non-blank gap is a piece the caller
    filtered out (redaction-only), and spanning it would re-include the very
    text the filter dropped.
    """
    if pack_to is None:
        return [(start, end) for _, start, end in pieces]
    spans: list[tuple[int, int]] = []
    cur_start: int | None = None
    cur_end = 0
    for _, start, end in pieces:
        if cur_start is None:
            cur_start, cur_end = start, end
        elif end - cur_start <= pack_to and not body[cur_end:start].strip():
            cur_end = end
        else:
            spans.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    if cur_start is not None:
        spans.append((cur_start, cur_end))
    return spans


def _make_chunk(note: dict, section, text: str, start: int, end: int,
                ordinal: int) -> Chunk:
    return Chunk(
        chunk_id=f"{note['note_id']}:{section.name}:{ordinal}",
        hadm_id=note["hadm_id"],
        note_id=note["note_id"],
        section=section.name,
        char_start=start,
        char_end=end,
        text=text,
    )


def _split_long_body(body: str, max_chars: int) -> list[tuple[str, int, int]]:
    """Paragraph-then-sentence splits as (text, start, end) within the body."""
    out: list[tuple[str, int, int]] = []
    cursor = 0
    for match in _PARAGRAPH_RE.finditer(body):
        _emit_paragraph(out, body[cursor:match.start()], cursor, max_chars)
        cursor = match.end()
    _emit_paragraph(out, body[cursor:], cursor, max_chars)
    return out


def _emit_paragraph(out: list, para: str, base: int, max_chars: int) -> None:
    if not para.strip():
        return
    if len(para) <= max_chars:
        out.append((para, base, base + len(para)))
        return
    cursor = 0
    for match in _SENTENCE_RE.finditer(para):
        _emit_fragment(out, para[cursor:match.start()], base + cursor, max_chars)
        cursor = match.end()
    _emit_fragment(out, para[cursor:], base + cursor, max_chars)


def _emit_fragment(out: list, frag: str, base: int, max_chars: int) -> None:
    if not frag.strip():
        return
    if len(frag) <= max_chars:
        out.append((frag, base, base + len(frag)))
        return
    # Fixed-width fallback for run-ons; keeps every chunk bounded. Prefer the
    # last whitespace in the window so pieces break at word boundaries — a
    # mid-word cut corrupts the citation text AND its embedding. A window with
    # no whitespace (one giant token) still hard-cuts at max_chars.
    start = 0
    while start < len(frag):
        if len(frag) - start <= max_chars:
            out.append((frag[start:], base + start, base + len(frag)))
            break
        window = frag[start:start + max_chars]
        cut = max(window.rfind(" "), window.rfind("\n"), window.rfind("\t"))
        if cut <= 0:
            cut = max_chars
        out.append((frag[start:start + cut], base + start, base + start + cut))
        start += cut
