"""The judge grades a marker against the numbering the reader clicked.

`/ehr` publishes `sources` carrying the same number the answer's marker uses, and since
2026-10-28 that number is never rewritten (`citations._renumber_citations`). Both halves
travel with the turn: the published list states the numbering beside the answer, and the
raw tool evidence carries a `cite` marker on every passage and analyte it shows.

The history this replaced: markers used to be renumbered to first-appearance order while
the raw payloads kept the order the tools returned, so a correct answer scored
`citation: 1` on three consecutive Gate 1 canaries (2026-10-23).
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
    # What `task._evidence` now builds: every unit carries the marker the answer must
    # write, numbered across the whole thread.
    "tool_evidence": [
        {"tool": "rag_search_sections", "result": {"passages": [
            {"cite": "^[1]", "section": "history_of_present_illness",
             "text": "A 61-year-old woman."},
            {"cite": "^[2]", "section": "brief_hospital_course",
             "text": "Admitted for rehab."},
            {"cite": "^[3]", "section": "discharge_diagnosis",
             "text": "Status post knee replacement."},
            {"cite": "^[4]", "section": "discharge_instructions",
             "text": "Continue Coumadin."},
        ]}},
    ],
}


def test_the_published_numbering_travels_with_each_turn():
    prompt = build_user_prompt(INPUT, OUTPUT)

    assert "THE NUMBERING THIS ANSWER USES" in prompt
    assert "[3] discharge_instructions: Continue Coumadin." in prompt
    assert "[4] discharge_diagnosis: Status post knee replacement." in prompt


def test_every_raw_unit_carries_the_marker_the_answer_must_write():
    """The raw block is now directly checkable: `"cite": "^[4]"` IS the instructions.

    A judge no longer has to count array positions, and a marker can be resolved to a
    named passage in one read.
    """
    prompt = build_user_prompt(INPUT, OUTPUT)
    raw = prompt.split("RAW TOOL EVIDENCE")[1]

    assert '"cite": "^[3]"' in raw
    assert '"cite": "^[4]"' in raw
    # The fourth raw passage is the instructions, and it says so.
    instructions = raw.split('"cite": "^[4]"')[1]
    assert "discharge_instructions" in instructions
    # The prompt tells the judge what the marker means, in the same words the payload
    # uses, so the two cannot drift into different conventions.
    assert '"cite": "^[10]"' in prompt


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
