"""Where a marker goes, and that the number the model is shown is the one it opens.

Three synthesis defects from the 2026-11-04 forensic run, and the numbering invariant that
made one of them unreadable to the judge.

`dev_t7_02` left a documented fact uncited and attached both its markers to the sentence
reporting an absence — an uncited claim and a misplaced citation in one answer. The system
prompt forbade markers on absences but never said the documenting sentence must carry one.

`dev_t2_02` summarised the discharge instructions while citing the history of present
illness and the hospital course. `dev_t3_01` reported "dose adjusted by INR" and dropped the
7 mg daily the same passage states. Both are prompt-level omissions, so both rules are now
stated in the prompt rather than left implied.

The last test is the invariant behind `dev_t3_02`: the number `render_numbered` stamps onto
the payload the model reads must be the number `resolve_sources` resolves back to the same
passage. When those two disagreed, a correct citation was scored as pointing at the wrong
section.
"""

from __future__ import annotations

from services.agent.engine.prompts import SYSTEM_PROMPT
from services.agent.postprocess import citations
from services.agent.presentation import a2ui

HADM = 90000015


def test_a_positive_sentence_must_carry_its_own_marker():
    assert "Every sentence that states a positive clinical fact from the chart carries" in (
        SYSTEM_PROMPT)
    assert "The marker names the\n  section that CONTAINS the fact" in SYSTEM_PROMPT or (
        "section that CONTAINS the fact" in SYSTEM_PROMPT)


def test_the_documented_sentence_is_the_one_that_gets_the_marker():
    assert "The sentence\n  stating the DOCUMENTED fact carries the marker" in SYSTEM_PROMPT or (
        "stating the DOCUMENTED fact carries the marker" in SYSTEM_PROMPT)


def test_a_number_and_the_rule_that_changes_it_are_both_stated():
    assert "state BOTH" in SYSTEM_PROMPT
    assert "the dose will be adjusted according to the INRs" in SYSTEM_PROMPT


def _passage(section: str, text: str) -> dict:
    return {"id": f"MT-X-DS_{section}_1", "section": section, "text": text}


def test_the_number_shown_to_the_model_is_the_number_resolved_back():
    """A follow-up's first new passage continues the thread's numbering, both ways.

    `render_numbered` decides what the model reads; `resolve_sources` decides what a marker
    opens. Both read the same ledger, and this pins that they agree across a replayed turn.
    """
    replayed = [
        {"name": "rag_search_sections", "args": {"hadm_id": HADM},
         "response": {"passages": [
             _passage("history_of_present_illness", "A 61-year-old woman."),
             _passage("brief_hospital_course", "Admitted for rehabilitation."),
         ]}},
    ]
    new_payload = {"passages": [
        _passage("discharge_instructions", "Continue Coumadin, dose adjusted by INR.")]}

    offset = sum(citations.passage_total(call.get("response")) for call in replayed)
    shown = citations.render_numbered(new_payload, offset)
    numbered = shown["passages"][0]
    assert numbered["cite"] == offset + 1
    assert numbered["cite_header"] == f"^[{offset + 1}]"

    call = {"name": "rag_search", "args": {"hadm_id": HADM}, "response": new_payload}
    # The answer as the reader receives it: `compact_display_numbers` renumbers the prose
    # to 1..k for this turn and hands back the map the payload resolves through.
    prose, display = citations.compact_display_numbers(
        f"Continue Coumadin, dose adjusted by INR ^[{offset + 1}].")
    assert prose == "Continue Coumadin, dose adjusted by INR ^[1]."

    out = a2ui.compose_presentation(
        "What do the discharge instructions say?",
        prose, [call], evidence=replayed + [call], citation_map=display,
    )

    assert [source["section"] for source in out["sources"]] == ["discharge_instructions"]
    assert out["sources"][0]["cite"] == 1
    assert "Continue Coumadin" in out["sources"][0]["text"]


def test_a_valid_number_is_never_overridden_by_the_question_s_intent():
    """`^[2]` means the second passage, whatever section the question asked about.

    `resolve_source` used to run the intent loop first and, on a match, never consult
    `passage_n` — so a citation to the third hospital-course chunk was served as the first
    one. The docstring has claimed "a PREFERENCE and never a veto" since the intent was
    introduced; the code now does what it says (2026-11-04).
    """
    passages = [
        {"section": "discharge_medications", "text": "On transfer, Celebrex."},
        {"section": "brief_hospital_course", "text": "Admitted for rehabilitation."},
        {"section": "brief_hospital_course", "text": "Discontinued the prophylaxis."},
    ]

    resolved = a2ui.resolve_source(
        passages, 2, ("discharge_medications",))

    assert resolved["section"] == "brief_hospital_course"
    assert resolved["text"] == "Admitted for rehabilitation."


def test_the_intent_still_recovers_an_out_of_range_number():
    passages = [
        {"section": "history_of_present_illness", "text": "A 61-year-old woman."},
        {"section": "discharge_medications", "text": "Celebrex 200 mg daily."},
    ]

    resolved = a2ui.resolve_source(passages, 99, ("discharge_medications",))

    assert resolved["section"] == "discharge_medications"
    assert "Celebrex" in resolved["text"]


def test_the_intent_may_choose_the_body_inside_the_cited_passage():
    """A whole-note chunk labelled with a coarser heading can contain the asked-for one."""
    passages = [
        {"section": "discharge_summary",
         "text": "Discharge medications: Celebrex 200 mg daily.\nFollow-up: clinic in 2 weeks."},
    ]

    resolved = a2ui.resolve_source(passages, 1, ("discharge_medications",))

    assert resolved["section"] == "discharge_medications"
    assert "Celebrex" in resolved["text"]
    assert "Follow-up" not in resolved["text"]
