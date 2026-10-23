"""The judge grades a marker against the numbering the reader clicked.

`/ehr` renumbers an answer's markers to first-appearance order and publishes `sources`
in exactly that order, so `^[3]` in the answer resolves to entry 3 of the source list a
clinician sees. The judge was handed the raw tool payloads instead, whose passages keep
the order the tools returned them in — and with the discharge instructions third in one
array and fourth in the other, a correct answer scored `citation: 1` on three
consecutive Gate 1 canaries (2026-10-23).

These tests pin the alignment: the published numbering travels with the turn, the prompt
states it beside the answer, and the raw evidence is labelled as a different order.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evaluation.platform.judge_call import build_user_prompt  # noqa: E402

SOURCES = [
    {"cite": 1, "section": "history_of_present_illness", "text": "A 61-year-old woman."},
    {"cite": 2, "section": "brief_hospital_course", "text": "Admitted for rehab."},
    {"cite": 3, "section": "discharge_instructions", "text": "Continue Coumadin."},
    {"cite": 4, "section": "discharge_diagnosis", "text": "Status post knee replacement."},
]

INPUT = {
    "turns": ["/ehr", "What is the taper?"],
    "patient_manifest": {"encounter": "inpatient"},
}

OUTPUT = {
    "turn_responses": ["Summary with ^[3] on instructions.", "Taper: ^[1]."],
    "turn_sources": [SOURCES, [{"cite": 1, "section": "discharge_medications",
                                "text": "oxycodone SR 20 mg q12h for five days."}]],
    "tool_evidence": [
        {"tool": "rag_search_sections", "result": {"passages": [
            {"section": "history_of_present_illness", "text": "A 61-year-old woman."},
            {"section": "brief_hospital_course", "text": "Admitted for rehab."},
            {"section": "discharge_diagnosis", "text": "Status post knee replacement."},
            {"section": "discharge_instructions", "text": "Continue Coumadin."},
        ]}},
    ],
}


def test_the_published_numbering_travels_with_each_turn():
    prompt = build_user_prompt(INPUT, OUTPUT)

    assert "THE NUMBERING THIS ANSWER USES" in prompt
    assert "[3] discharge_instructions: Continue Coumadin." in prompt
    assert "[4] discharge_diagnosis: Status post knee replacement." in prompt


def test_a_marker_is_graded_against_the_published_list_not_the_raw_array():
    """The regression: the raw array puts the diagnosis third, the answer's list fourth.

    Entry [3] of the published list is the instructions passage while the third raw
    passage is the diagnosis — so a judge reading only the raw evidence sees a swap that
    is not there.
    """
    prompt = build_user_prompt(INPUT, OUTPUT)

    published = prompt.split("TURN 1 — THE NUMBERING THIS ANSWER USES")[1]
    published = published.split("TURN 2")[0]
    raw = prompt.split("RAW TOOL EVIDENCE")[1]

    assert "[3] discharge_instructions" in published
    assert raw.index("discharge_diagnosis") < raw.index("discharge_instructions"), (
        "the raw array still lists the diagnosis before the instructions")
    assert "numbering the answer cites" in prompt


def test_the_second_turn_gets_its_own_numbering():
    prompt = build_user_prompt(INPUT, OUTPUT)
    second = prompt.split("TURN 2 — THE NUMBERING THIS ANSWER USES")[1]

    assert "[1] discharge_medications: oxycodone SR" in second


def test_a_trace_without_sources_falls_back_to_the_raw_evidence():
    """An item from before the field existed must still be gradeable."""
    prompt = build_user_prompt(INPUT, {"turn_responses": ["A.", "B."],
                                       "tool_evidence": OUTPUT["tool_evidence"]})

    assert "THE NUMBERING THIS ANSWER USES" not in prompt
    assert "RAW TOOL EVIDENCE" in prompt


def test_the_raw_evidence_block_is_still_present_for_grounding():
    """Citation numbering was the problem; the passages themselves are what fidelity and
    grounding are checked against, so they stay."""
    prompt = build_user_prompt(INPUT, OUTPUT)

    assert "Continue Coumadin" in prompt
    assert "Status post knee replacement" in prompt
