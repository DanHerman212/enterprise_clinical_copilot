"""Tests for discharge-note chunking (rag/chunking.py).

These are pure-function tests on synthetic text - no cloud, no medspaCy, so they
run under the harness venv. They lock in the §4 rules: section boundaries,
paragraph-then-sentence for long sections, redaction-only chunks dropped,
deterministic chunk_ids, and offsets that always point at real text.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.chunking import (  # noqa: E402
    DEFAULT_MAX_CHARS,
    INDEX_SECTIONS,
    Chunk,
    _pack,
    chunk_note,
)

# HPI: two paragraphs, each ~950 chars (so with max_chars=1000 they stay whole
# paragraphs but the total forces a split). BHC: one paragraph ~1050 chars (so
# with max_chars=300 it sentence-splits). Social History is pure redaction.
LONG_HPI = """
History of Present Illness:
Mr. ___ is a ___ year old man with a past medical history of congestive heart
failure and type 2 diabetes mellitus who presents with worsening shortness of
breath over the past week. He reports progressive dyspnea on exertion,
orthopnea, and paroxysmal nocturnal dyspnea. He stopped taking his furosemide
several days ago because he ran out of medications and could not get a refill
in time. On presentation he was tachycardic with a heart rate of 112 and
hypoxic, requiring two liters of supplemental oxygen by nasal cannula. Physical
exam revealed jugular venous distension, pulmonary crackles to the midfields,
and two plus pitting lower extremity edema. He has had two prior admissions in
the past year for the same problem, each followed by a brief period of
stability.

His family is concerned about his ability to care for himself at home once
discharged. His daughter reports that he has been noncompliant with his
medications and has missed several outpatient appointments with his
cardiologist. He lives alone in a second floor walkup apartment and has no
caregiver available during the day. Social work was consulted to assess his
living situation and to arrange for visiting nurse services after discharge.
He was engaged and able to participate in goals of care discussions with his
family during the admission. Discharge planning will depend on securing
appropriate home support.

Social History:
___

Brief Hospital Course:
Patient was diuresed aggressively with intravenous furosemide and improved over
the course of the admission. He tolerated the regimen well without electrolyte
derangements. He was transitioned to oral diuretics on hospital day three with
close monitoring. He missed several outpatient appointments prior to this
admission which raises concern about follow up after discharge. A family
meeting was held to discuss goals of care and the importance of medication
adherence. I am concerned about his ability to manage at home given his
functional decline and lack of caregiver support. Nephrology was consulted for
the acute on chronic kidney disease noted on admission labs. Repeat labs on
hospital day two showed improvement in creatinine from baseline. Echocardiogram
performed during the admission demonstrated an ejection fraction of 35 percent
with moderate mitral regurgitation. He was stable on discharge to home with
home health services and a scheduled follow up with cardiology within one week.

Discharge Condition:
Mental Status: Clear and coherent. Level of Consciousness: Alert.
"""

NOTE = {"hadm_id": 20924467, "note_id": 12345, "text": LONG_HPI}


def test_short_section_is_one_chunk():
    chunks = chunk_note(NOTE)
    discharge_condition = [c for c in chunks if c.section == "discharge_condition"]
    assert len(discharge_condition) == 1
    assert discharge_condition[0].text == (
        "Mental Status: Clear and coherent. Level of Consciousness: Alert."
    )


def test_placeholder_only_section_is_dropped():
    chunks = chunk_note(NOTE)
    assert all(c.section != "social_history" for c in chunks)


def test_chunks_never_exceed_max_chars():
    for chunk in chunk_note(NOTE):
        assert len(chunk.text) <= DEFAULT_MAX_CHARS


def test_long_section_splits_on_paragraph_boundary():
    """HPI (~1900 chars, two ~950-char paragraphs) splits at the blank line."""
    chunks = chunk_note(NOTE, max_chars=1000)
    hpi = [c for c in chunks if c.section == "history_of_present_illness"]
    assert len(hpi) == 2
    assert "On presentation he was tachycardic" in hpi[0].text
    assert "His family is concerned" in hpi[1].text


def test_long_single_paragraph_splits_on_sentence():
    """BHC is one ~1050-char paragraph; at max 300 it becomes sentence chunks."""
    chunks = chunk_note(NOTE, max_chars=300)
    bhc = [c for c in chunks if c.section == "brief_hospital_course"]
    assert len(bhc) >= 4
    for chunk in bhc:
        assert len(chunk.text) <= 300


def test_chunk_offsets_point_at_real_text():
    for chunk in chunk_note(NOTE):
        # Rebuild the section body, then check the offset slice matches.
        from rag.sections import parse_note
        section = parse_note(NOTE["text"]).get(chunk.section)
        body = section.body
        assert chunk.text == body[chunk.char_start:chunk.char_end]


def test_chunk_ids_are_deterministic_and_unique():
    a = chunk_note(NOTE)
    b = chunk_note(NOTE)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)


def test_repeated_sections_get_unique_ids():
    """'Physical Exam' + 'Discharge Exam' both map to physical_exam; the second
    instance must not collide with the first. This is the 2026-08-06 bug: 11,451
    duplicate chunk_ids shipped to the index and were silently dropped."""
    note = {"hadm_id": 1, "note_id": 99, "text": """
Physical Exam:
General: Well appearing
Vitals: T 98.6

Discharge Exam:
Vitals: T 98.4, BP 118/72
Activity: Ambulatory
"""}
    chunks = chunk_note(note)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), f"duplicate chunk ids: {ids}"
    pe = [c for c in chunks if c.section == "physical_exam"]
    assert len(pe) == 2
    assert pe[0].chunk_id == "99:physical_exam:1"
    assert pe[1].chunk_id == "99:physical_exam:2"


def test_alias_sections_get_unique_ids():
    """'Past Medical History' + 'PMH' alias in one note must not collide."""
    note = {"hadm_id": 1, "note_id": 7, "text": """
Past Medical History:
Hypertension

PMH:
Diabetes
"""}
    chunks = chunk_note(note)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    pmh = sorted(c.chunk_id for c in chunks if c.section == "past_medical_history")
    assert pmh == ["7:past_medical_history:1", "7:past_medical_history:2"]


def test_chunk_id_shape():
    chunk = chunk_note(NOTE)[0]
    assert chunk.chunk_id.startswith("12345:")


def test_empty_note_yields_no_chunks():
    assert chunk_note({"hadm_id": 1, "note_id": 2, "text": ""}) == []


def test_redaction_only_note_yields_no_chunks():
    note = {"hadm_id": 1, "note_id": 2, "text": "Social History:\n___\nFamily History:\n___"}
    assert chunk_note(note) == []


def test_hard_fallback_bounds_a_run_on():
    """A single sentence longer than max_chars must still be bounded."""
    sentence = "x" * 4000
    note = {"hadm_id": 1, "note_id": 2, "text": f"Brief Hospital Course:\n{sentence}\n"}
    chunks = chunk_note(note, max_chars=500)
    assert len(chunks) == 8
    assert all(len(c.text) <= 500 for c in chunks)


def test_packing_merges_line_oriented_sections():
    """Lab lines (blank-line paragraphs) merge into one packed chunk."""
    labs = """
Pertinent Results:
07:15AM BLOOD WBC-8.0

07:16AM BLOOD HGB-11.2

07:17AM BLOOD PLT-220

07:18AM CHEM NA-138 K-4.1

07:19AM CHEM BUN-28 CREAT-1.6

07:20AM CHEM GLU-118

07:21AM CHEM GLU-121

07:22AM CHEM GLU-109

07:23AM CHEM GLU-114
"""
    note = {"hadm_id": 1, "note_id": 7, "text": labs}
    # Body ~200 chars: over max 160, so paragraphs split; each line ≤ 160.
    unpacked = chunk_note(note, max_chars=160)
    packed = chunk_note(note, max_chars=160, pack_to=300)

    assert len(unpacked) == 9  # one chunk per line
    assert len(packed) == 1  # all lines packed into one span
    assert "WBC" in packed[0].text and "GLU" in packed[0].text


def test_packing_respects_its_cap():
    body = "b" * 200 + " " + "c" * 200 + " " + "d" * 200
    pieces = [
        ("b" * 200, 0, 200),
        ("c" * 200, 201, 401),
        ("d" * 200, 402, 602),
    ]
    assert len(_pack(pieces, 500, body)) == 2  # 200+200 fits, third starts new span


def test_packing_does_not_reinclude_filtered_pieces():
    """A redaction-only paragraph dropped by the chunk filter must not be
    resurrected by packing the pieces on either side of it into one span."""
    body = ("Alert and oriented on arrival to the floor today.\n\n"
            "______\n\n"
            "Tolerated the diuresis regimen well without derangements.")
    note = {"hadm_id": 1, "note_id": 3,
            "text": "Brief Hospital Course:\n" + body}
    # max_chars below body length forces paragraph splitting; pack_to large
    # enough to invite merging across the dropped placeholder paragraph.
    chunks = chunk_note(note, max_chars=80, pack_to=400)
    assert chunks, "expected chunks for real narrative text"
    for chunk in chunks:
        assert "______" not in chunk.text


def test_fixed_width_fallback_cuts_at_word_boundaries():
    """A run-on sentence that defeats the sentence splitter must not be cut
    mid-word: a truncated token corrupts the citation and its embedding."""
    words = "lisinopril metoprolol furosemide spironolactone atorvastatin " * 30
    note = {"hadm_id": 1, "note_id": 4,
            "text": "Brief Hospital Course:\n" + words.strip() + "\n"}
    chunks = chunk_note(note, max_chars=100)
    assert len(chunks) > 1
    vocabulary = set(words.split())
    for chunk in chunks:
        assert len(chunk.text) <= 100
        for token in chunk.text.split():
            assert token in vocabulary, f"mid-word cut produced {token!r}"


def test_index_sections_are_canonical_section_names():
    """Every indexed section must be a canonical parser name (ECC-33): a typo
    here would silently index nothing for that section."""
    from rag.sections import KNOWN_HEADINGS
    assert set(INDEX_SECTIONS) <= set(KNOWN_HEADINGS)


def test_chunks_are_frozen_and_typed():
    chunk = chunk_note(NOTE)[0]
    assert isinstance(chunk, Chunk)
    with pytest.raises(Exception):
        chunk.text = "mutated"
