"""The chip a reader clicks is consumed, not offered back.

§5.1 of `docs/ux_workshop/strategy_and_tactics/05-a1-multi_turn_followup_audit.md`
recorded the production symptom: clicking a follow-up chip left the row as it was,
with the same question still on it. Measured against `agent-00175-ttr` on 2026-10-24,
the cause was not the selector — it was the string it was given. `suggest` excludes any
chip matching the question already asked, and the question it received was the
*composed prompt*:

    What is the dosing and taper schedule for oxycodone SR at discharge? For
    admission 90000015.

No chip carries that sentence, so nothing matched and the chip came back. The request
now carries `own_words` beside `question`, and `http._asked_questions` reads the
current turn the same way it reads the replayed ones.

These tests pin both halves: the request keeps the reader's words, and the selector
excludes the chip when it is given them.
"""

from __future__ import annotations

from services.agent.contracts import parse_agent_request
from services.agent.followups import _fingerprint, suggest
from services.agent.http import _asked_questions

CHIP = "What is the dosing and taper schedule for oxycodone SR at discharge?"

#: Enough of a discharge note for the taper template to fire: the cue followed by
#: the drug, which is how the cohort's notes word it.
TAPER_NOTE = (
    "Discharge medications: oxycodone SR 20 mg p.o. q.12h for five days, then "
    "decrease to oxycodone SR 10 mg p.o. q.12h for five days. "
    "Coumadin 7 mg daily; INR every Monday and Thursday."
)


def _state() -> dict:
    return {
        "tool_calls": [{
            "name": "rag_search_sections",
            "args": {"hadm_id": 90000015},
            "response": {"returned": 1, "query": "discharge medications",
                         "passages": [{"section": "discharge_medications",
                                       "text": TAPER_NOTE}]},
        }],
        "replayed_tool_calls": [],
        "command": None,
        "command_payload": None,
    }


def test_the_taper_chip_is_offered_when_nothing_has_been_asked():
    """The fixture is real: without it the exclusion below proves nothing."""
    assert CHIP in suggest("/ehr", "answer text", state=_state())


def test_the_chip_the_reader_clicked_is_not_offered_back():
    parsed = parse_agent_request(
        {"question": CHIP, "hadm_id": 90000015}, max_question_chars=500,
    )

    chips = suggest(parsed["question"], "answer text", state=_state(),
                    asked=_asked_questions(parsed))

    assert CHIP not in chips


def test_the_composed_prompt_alone_would_not_have_matched():
    """The defect, stated as the reason `own_words` exists.

    `_fingerprint` is what the selector compares; the composed prompt and the chip
    reduce to different strings, so a selector given only the prompt cannot know the
    chip was just clicked.
    """
    parsed = parse_agent_request(
        {"question": CHIP, "hadm_id": 90000015}, max_question_chars=500,
    )

    assert _fingerprint(parsed["own_words"]) == _fingerprint(CHIP)
    assert _fingerprint(parsed["question"]) != _fingerprint(CHIP)


def test_a_replayed_turn_and_the_current_turn_are_read_the_same_way():
    """Both ends of the comparison come from the reader, not from the prompt."""
    parsed = parse_agent_request(
        {
            "question": "and the potassium?",
            "hadm_id": 90000015,
            "turns": [{"question": CHIP, "answer": "20 mg then 10 mg."}],
        },
        max_question_chars=500,
    )

    assert _asked_questions(parsed) == [CHIP, "and the potassium?"]
