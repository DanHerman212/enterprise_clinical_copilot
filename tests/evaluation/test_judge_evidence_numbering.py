"""The judge grades a marker against the numbering the reader clicked.

`/ehr` publishes `sources` carrying the same number the answer's marker uses, and that
number is never rewritten to first-appearance order (`citations._renumber_citations`).
The published list travels with the turn it belongs to, so the numbering stated beside an
answer is the numbering that answer was written in.

Until 2026-11-04 every raw payload ALSO carried a `cite` badge numbered cumulatively
across the thread. Two schemes for one passage reached the judge together, and it resolved
turn 2's `^[1]` against the cumulative block and scored a correct citation as pointing at
the wrong section (`dev_t3_02`, `dev_t3_01`). `task._evidence` no longer stamps those
badges: the per-turn `sources` list is the single place a number is stated.
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
    # What `task._evidence` builds: the payloads as the tools returned them, with no
    # citation numbering on them. The per-turn `sources` list is the only numbering.
    "tool_evidence": [
        {"tool": "rag_search_sections", "result": {"passages": [
            {"section": "history_of_present_illness",
             "text": "A 61-year-old woman."},
            {"section": "brief_hospital_course",
             "text": "Admitted for rehab."},
            {"section": "discharge_diagnosis",
             "text": "Status post knee replacement."},
            {"section": "discharge_instructions",
             "text": "Continue Coumadin."},
        ]}},
    ],
}


def test_the_published_numbering_travels_with_each_turn():
    prompt = build_user_prompt(INPUT, OUTPUT)

    assert "THE NUMBERING THIS ANSWER USES" in prompt
    assert "[3] discharge_instructions: Continue Coumadin." in prompt
    assert "[4] discharge_diagnosis: Status post knee replacement." in prompt


def test_the_raw_evidence_carries_no_second_numbering():
    """One passage must not arrive with two numbers.

    The per-turn `sources` list states what `^[k]` means. A `cite` badge on the payload
    stated a different number for the same passage, and the judge used the wrong one.
    """
    prompt = build_user_prompt(INPUT, OUTPUT)
    raw = prompt.split("RAW TOOL EVIDENCE")[1]

    assert '"cite": "^[' not in raw
    assert "Continue Coumadin" in raw
    assert "Status post knee replacement" in raw


def test_the_prompt_says_a_turns_list_starts_at_one():
    """Restarting at [1] is the display contract, not a renumbering defect."""
    prompt = build_user_prompt(INPUT, OUTPUT)

    assert "numbered from [1] independently of every other turn" in prompt


def test_each_turns_passage_text_reaches_the_judge_whole():
    """No character cap: a fact at character 1,600 is still visible."""
    long_text = "A" * 1600 + " Lovenox was discontinued on 08/30/2007."
    output = {"turn_responses": ["Answer ^[1]."],
              "turn_sources": [[{"cite": 1, "section": "brief_hospital_course",
                                 "text": long_text}]],
              "tool_evidence": []}
    prompt = build_user_prompt({"turns": ["/ehr"]}, output)

    assert "Lovenox was discontinued on 08/30/2007." in prompt


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
