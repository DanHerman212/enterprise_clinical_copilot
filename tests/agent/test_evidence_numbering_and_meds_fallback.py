"""Two seams where the model was given evidence it could not address, and one where it was asked the wrong question.

`render_numbered` gave a note passage and a laboratory analyte a `cite` number and a
`cite_header` to copy, and gave a published record neither. `LITERATURE_FORMATTING` told the
model that the first record was `^[1]`, so the only number available to it for a record was
the position of its own sentence — a summary that opened on the third record's finding wrote
`^[1]` and the reader opened the first.

`MEDS_FORMATTING` asks what the patient was sent home on, which is the right question for a
record with a discharge medication section and the wrong one for a record without: the
discharge-only rule discarded most of the drugs the fallback narrative named, and the
sentence then reported the survivors as what the section "mentions".

Both are decided by the evidence in front of the turn, so both are testable without a model.
"""

from __future__ import annotations

import pytest

from services.agent.engine import deterministic
from services.agent.engine.prompts import (
    MEDS_FALLBACK_FORMATTING,
    MEDS_FORMATTING,
)
from services.agent.postprocess import answer as answer_text
from services.agent.postprocess import citations

ARTICLES = {
    "query": "antiplatelet therapy after coronary stenting",
    "articles": [
        {"pmid": "35186616", "title": "Dual antiplatelet therapy after PCI"},
        {"pmid": "37028984", "title": "Duration of DAPT in stable coronary disease"},
        {"pmid": "24108298", "title": "Stent thrombosis and adherence"},
    ],
    "returned": 3,
}


def _rows(*names: str) -> list[dict[str, str]]:
    return [{"medication": name, "dosage": "", "frequency": ""} for name in names]


# ------------------------------------------------------- published-record numbering


def test_a_published_record_is_given_a_number_to_copy():
    """The record the model names and the record it was given must be one position."""
    rendered = citations.render_numbered(ARTICLES)

    assert [article["cite"] for article in rendered["articles"]] == [1, 2, 3]
    assert [article["cite_header"] for article in rendered["articles"]] == [
        "^[1]", "^[2]", "^[3]"]
    assert all(article["cite_modality"] == "PUBLISHED RECORD"
               for article in rendered["articles"])


def test_numbering_a_record_does_not_make_it_a_citable_unit():
    """`passage_total` counts note passages and analytes; an article is addressed by position.

    If a search beside three note passages started contributing units, every marker into a
    note would shift by three and resolve to the wrong passage.
    """
    rendered = citations.render_numbered(ARTICLES)

    assert citations.passage_total(rendered) == 0
    assert citations.citable_passages(rendered) == []


def test_the_payload_the_tool_returned_is_not_rewritten():
    """The numbering is stamped on the copy the model reads, never on the record."""
    citations.render_numbered(ARTICLES)

    assert all("cite" not in article for article in ARTICLES["articles"])


def test_a_note_passage_and_a_record_are_numbered_independently():
    """A payload with passages is still numbered by offset; only articles restart at 1."""
    payload = {
        "passages": [{"section": "brief_hospital_course", "text": "A."},
                     {"section": "discharge_diagnosis", "text": "B."}],
    }
    rendered = citations.render_numbered(payload, 7)
    assert [passage["cite"] for passage in rendered["passages"]] == [8, 9]

    records = citations.render_numbered(ARTICLES, 7)
    assert [article["cite"] for article in records["articles"]] == [1, 2, 3]


# ------------------------------------------------------------ the medication question


def test_a_dedicated_section_is_asked_for_the_discharge_regimen():
    assert deterministic.meds_formatting_for(("discharge_medications",)) is MEDS_FORMATTING
    assert deterministic.meds_formatting_for(
        ("discharge_medications", "discharge_instructions")) is MEDS_FORMATTING


def test_a_fallback_retrieval_is_asked_what_the_sections_name():
    """A record with no discharge list cannot answer a discharge-list question."""
    assert deterministic.meds_formatting_for(
        ("brief_hospital_course",)) is MEDS_FALLBACK_FORMATTING
    assert deterministic.meds_formatting_for(
        ("discharge_instructions",)) is MEDS_FALLBACK_FORMATTING
    assert deterministic.meds_formatting_for(()) is MEDS_FALLBACK_FORMATTING


def test_the_fallback_instruction_asks_for_every_named_drug():
    """The instruction that replaced "exclude what was given in hospital" says so."""
    lowered = MEDS_FALLBACK_FORMATTING.lower()
    assert "every medication" in lowered
    assert "whether it was given during the stay" in lowered
    assert "exclude" not in lowered


# ---------------------------------------------------------------- the narrative's wording


def test_a_fallback_with_drugs_says_what_the_section_is():
    text = answer_text.medication_narrative(
        _rows("Plavix", "aspirin"), 1, sections=("brief_hospital_course",))

    assert "does not contain a dedicated discharge medications section" in text
    assert "rather than a formal discharge medication list" in text
    assert "brief hospital course" in text
    assert "Plavix" in text and "aspirin" in text
    assert text.rstrip().endswith("^[1].")


def test_a_fallback_with_no_drugs_does_not_speak_for_the_whole_record():
    """`dev_t4_02`: one section holding no list is not the record holding no medication."""
    text = answer_text.medication_narrative(
        [], 1, sections=("discharge_instructions",))

    assert "no medication list is recorded in the discharge instructions" in text
    assert "documented in the clinical course narrative" in text
    assert "in the discharge instructions during this stay" not in text


def test_a_dedicated_section_with_no_rows_still_blames_the_section():
    text = answer_text.medication_narrative(
        [], 1, sections=("discharge_medications",))

    assert text == ("The discharge medications section was retrieved, but no medication "
                    "could be read from it ^[1].")


# ------------------------------------------------- replay: which numbering survives


def test_numeric_markers_are_stripped_and_pmid_badges_survive():
    """The page's numbers are not the array's, so the page's numbers do not travel."""
    text = ("The course states this ^[1], ^[2]. A rate held at 4.1% ^[PMID: 33534292].")

    stripped = citations.strip_numeric_markers(text)

    assert "^[1]" not in stripped and "^[2]" not in stripped
    assert "^[PMID: 33534292]" in stripped, "a record badge means the same thing anywhere"
    assert stripped == ("The course states this. A rate held at 4.1% ^[PMID: 33534292].")


def test_a_text_with_no_numeric_markers_is_returned_unchanged():
    assert citations.strip_numeric_markers("No markers here.") == "No markers here."
    assert citations.strip_numeric_markers("Only a badge ^[PMID: 33534292].") == (
        "Only a badge ^[PMID: 33534292].")


@pytest.mark.anyio
async def test_replayed_prose_loses_its_page_numbers_and_the_evidence_keeps_its_cites():
    """`dev_t4_02`: a prior `^[2]` meant one section while `cite: 2` was another."""
    from langchain_core.messages import AIMessage, ToolMessage

    from services.agent.engine import graph

    turns = [{
        "question": "/ehr",
        "answer": "The diagnosis is stated ^[1], and the cohort rate is ^[PMID: 33534292].",
        "tool_calls": [{
            "name": "rag_search_sections",
            "args": {"hadm_id": 1},
            "payload": {"passages": [
                {"section": "history_of_present_illness", "text": "A."},
                {"section": "brief_hospital_course", "text": "B."},
            ]},
        }],
    }]

    messages, _calls, _articles = await graph.replay_messages(None, turns)
    replayed_answer = [m for m in messages if isinstance(m, AIMessage) and m.content]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]

    assert replayed_answer, "the prior answer must be replayed"
    assert "^[1]" not in replayed_answer[0].content
    assert "^[PMID: 33534292]" in replayed_answer[0].content
    assert tool_messages, "the replayed evidence must be replayed"
    assert '"cite": 1' in tool_messages[0].content
    assert '"cite": 2' in tool_messages[0].content
