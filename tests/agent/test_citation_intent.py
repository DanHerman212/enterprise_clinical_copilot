"""Citation intent: a composite question must not inherit a single-section override.

`intent_sections` exists because the model mis-numbers an extraction — asked for the
medications, it cites ^[1] while the list sits in a later passage. The override is only
safe when the question asks for ONE section. On 2026-10-22 a Turn 2 asked for the
discharge disposition AND the follow-up plan, matched a section needle, and the
resolver replaced the model's citation with a chunk that answered half the question.
"""

from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pytest  # noqa: E402

from services.agent.presentation import a2ui
from services.agent.postprocess import citations  # noqa: E402


@pytest.mark.parametrize("question,expected", [
    # One section clearly targeted: the override still applies.
    ("What are the discharge medications?", ("discharge_medications",
                                             "discharge_instructions")),
    ("What were the discharge diagnoses?", ("discharge_diagnosis",)),
    ("Summarise the hospital course", ("brief_hospital_course",)),
    ("What were the discharge instructions?", ("discharge_instructions",)),
])
def test_a_single_section_question_keeps_its_intent(question, expected):
    assert citations.intent_sections(question) == expected


@pytest.mark.parametrize("question", [
    # Two sections named: the answer cites whichever passage supports each half, so
    # there is no single intent to override the model's numbering with.
    "What was the discharge disposition and follow-up plan?",
    "What medications were given and what were the discharge instructions?",
    "Summarise the hospital course and the discharge diagnosis",
    # No section named at all.
    "Summarise the note",
    "",
    None,
])
def test_a_composite_or_open_question_has_no_single_intent(question):
    assert citations.intent_sections(question) == ()


def test_a_composite_answer_resolves_by_the_models_own_number():
    """The resolver, not just the predicate: the citation must land where it was put."""
    passages = [
        {"section": "brief_hospital_course", "id": "p1",
         "text": "Brief hospital course: discharged home with home health."},
        {"section": "discharge_instructions", "id": "p2",
         "text": "Discharge instructions: follow up with cardiology in one week."},
    ]
    answer = (
        'The patient was discharged home with home health ^[1]. '
        'Follow-up is with cardiology in one week ^[2].'
    )
    question = "What was the discharge disposition and follow-up plan?"

    sources = a2ui.resolve_sources(
        a2ui.renumber_citations(answer), a2ui.citation_remap(answer),
        {"passages": passages, "query": "discharge disposition follow-up"},
        citations.intent_sections(question),
    )

    by_cite = {s["cite"]: s for s in sources}
    assert "home health" in by_cite[1]["text"]
    assert "cardiology" in by_cite[2]["text"]

def test_two_retrieval_calls_number_globally_across_both():
    """A turn that searches twice cites ^[3] for the first passage of the second call.

    The prompt promises global numbering ("the second call's passages continue after
    them"), so the array the marker indexes must be the concatenation. The resolver
    used to read only the first call's payload, which made every marker past the first
    call resolve to the wrong passage or clamp to the end of it. `/ehr` is the shape:
    a sectioned call plus the record-wide fallback.
    """
    tool_calls = [
        {"name": "rag_search_sections", "response": {"query": "discharge sections",
                                                     "passages": [
            {"section": "brief_hospital_course", "text": "Course: improved."},
            {"section": "discharge_diagnosis", "text": "Diagnosis: pneumonia."},
        ]}},
        {"name": "rag_search", "response": {"query": "follow-up",
                                            "passages": [
            {"section": "discharge_instructions", "text": "Follow up in one week."},
        ]}},
    ]

    merged = a2ui.rag_payload(tool_calls)
    assert len(merged["passages"]) == 3
    assert merged["returned"] == 3

    answer = "Follow-up is in one week ^[3]."
    sources = a2ui.resolve_sources(
        a2ui.renumber_citations(answer), a2ui.citation_remap(answer), merged, ())

    assert len(sources) == 1
    assert "one week" in sources[0]["text"], sources[0]
