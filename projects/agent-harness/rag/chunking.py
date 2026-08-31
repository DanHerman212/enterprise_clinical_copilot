"""Chunking of discharge notes into citable passages.

Pure standard library, no cloud calls (same constraint as rag/sections.py).

A chunk is the unit of retrieval: one Vector Search datapoint, one citable
passage. Rules, in order:

  1. Split on section boundaries (parse_note does the splitting).
  2. A section at or under max_chars is one chunk.
  3. A longer section splits on paragraph (blank line), then sentence.
  4. Chunks that are entirely redaction or empty are dropped - they cost money
     to embed and can never support a citation.

chunk_id is deterministic: "{note_id}:{section}:{ordinal}", so re-running the
chunker produces identical IDs and a re-index never orphans chunks. Offsets are
within the section body, so a citation reconstructs as "note {note_id},
{section}, chars X-Y".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.sections import KNOWN_HEADINGS, parse_note, redaction_profile

# D2 default from the build guide; sections under this are single chunks.
DEFAULT_MAX_CHARS = 1500

# Build-time packing size used by the ingest pipeline. Serving re-chunks notes
# to resolve a citation's exact text, so it must pack with the SAME value or
# chunk ids drift and every packed section silently degrades to whole-note
# text. Change this only together with an index rebuild.
DEFAULT_PACK_TO = 700

# Narrative/assessment sections worth indexing. Metadata (name, dates, sex) and
# lab-line noise (pertinent_results) add embedding cost without retrieval
# value. Single source of truth: the ingest pipeline whitelist, the build
# scripts, and the serving-side datapoint-id parser all import this tuple.
INDEX_SECTIONS = (
    "history_of_present_illness",
    "past_medical_history",
    "family_history",
    "social_history",
    "physical_exam",
    "brief_hospital_course",
    "discharge_condition",
    "discharge_diagnosis",
    "discharge_medications",
    "medications_on_admission",
    "discharge_disposition",
    "discharge_instructions",
    "discharge_summary",
)

_unknown = set(INDEX_SECTIONS) - set(KNOWN_HEADINGS)
if _unknown:  # a typo here would silently index nothing for that section
    raise ValueError(
        f"INDEX_SECTIONS entries not in rag.sections.KNOWN_HEADINGS: "
        f"{sorted(_unknown)}"
    )
del _unknown

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

    chunk_ids are unique per note: ordinal continues across repeated sections of
    the same name (a note can carry both "Physical Exam" and "Discharge Exam",
    or "Past Medical History" and its "PMH" alias), so the second instance does
    not collide with the first.
    """
    chunks: list[Chunk] = []
    name_ordinals: dict[str, int] = {}
    for section in parse_note(note["text"]).sections:
        base = name_ordinals.get(section.name, 0)
        section_chunks = _chunk_section(note, section, max_chars, pack_to, base)
        chunks.extend(section_chunks)
        name_ordinals[section.name] = base + len(section_chunks)
    return chunks


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
