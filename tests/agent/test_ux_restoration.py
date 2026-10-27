"""Phase 02.5: the agent-side rules the live UX defects were traced to.

Six behaviours, each one a defect that a green backend suite did not catch:

* a lab drill-down announced itself as "Consulting a Tool" (label table gaps);
* `search_literature` was bound on every free-text turn, so a chart follow-up came
  back with an empty PubMed card and dead note citations;
* `discharge_medications` returns several chunks while `/meds` cites with one marker,
  so `^[1]` pointed at a fragment of the list the grid had just shown in full;
* `/ehr` instructed markdown `##` headings, which streamed as literal characters;
* there was no `verified` field at all, so the console's badge could not exist;
* and an empty literature payload was treated as a literature turn.

These are contract tests: they assert what the wire owes the browser.
"""

from __future__ import annotations

import pytest

from services.agent.engine import deterministic
from services.agent.routing import routing
from services.agent.presentation.a2ui import compose_presentation
from services.agent.engine.prompts import SYSTEM_PROMPT
from services.agent.telemetry.stages import RESULT_LABELS, RESULT_NOUNS, TOOL_LABELS


def _by_id(canvas):
    components = canvas["messages"][1]["updateComponents"]["components"]
    return {c["id"]: c for c in components}


def _types(out):
    return {c["component"] for c in _by_id(out["a2ui"]).values()}


RAG = {
    "hadm_id": 90000015,
    "query": "discharge notes",
    "returned": 1,
    "passages": [
        {"id": "n_1", "section": "discharge_diagnosis",
         "text": "Sepsis with acute kidney injury."},
    ],
}

#: A search that ran and matched nothing — the payload that used to clobber the notes.
EMPTY_LITERATURE = {"query": "sepsis readmission", "returned": 0, "articles": []}

LITERATURE = {
    "query": "sepsis readmission",
    "returned": 1,
    "articles": [{
        "pmid": "37495280",
        "title": "Readmission after sepsis",
        "journal": "JAMA",
        "year": "2023",
        "abstract": "Cohort study.",
    }],
}

LABS = {"trends": [{"analyte": "Creatinine", "points": [{"t": 0, "value": 1.2}]}]}
MEDS_PAYLOAD = {"medications": [{"medication": "warfarin", "dose": "4 mg"}]}


# --- Defect 2: a lab drill-down has a name -------------------------------------

def test_lab_trends_has_a_label_a_result_and_a_noun():
    """`get_lab_trends` was missing from all three tables until 2026-10-22.

    The fallback is true ("Consulting a Tool") and useless: a reader watching a
    lab-value question was told a tool was being consulted, on every step.
    """
    assert TOOL_LABELS["get_lab_trends"] == "Reading the Lab Trajectories"
    assert RESULT_LABELS["get_lab_trends"] == "Read the Lab Trajectories"
    assert RESULT_NOUNS["get_lab_trends"] == "Lab Trajectory"


# --- Defect 6: literature only when asked -------------------------------------

@pytest.mark.parametrize("question", [
    "What does the literature say about sepsis readmission?",
    "Are there any trials of early antibiotics?",
    "What are the guidelines for anticoagulation?",
    "Show me a systematic review of this",
    "Is there meta-analysis evidence on this?",
    "Search pubmed for this",
    "What is the evidence for this practice?",
])
def test_a_question_about_published_evidence_is_recognised(question):
    assert routing.asks_for_literature(question) is True


@pytest.mark.parametrize("question", [
    "Was she on anticoagulants during the stay?",
    "Why did her creatinine rise?",
    "What is the readmission risk for admission 90000015?",
    "What did the discharge note say about her anxiety?",
])
def test_a_chart_question_is_not_a_literature_question(question):
    assert routing.asks_for_literature(question) is False


def test_the_literature_tool_is_not_bound_by_default():
    assert "search_literature" not in routing.FREETEXT_TOOLS
    assert routing.LITERATURE_TOOLS == frozenset({"search_literature"})


def test_an_empty_literature_result_does_not_take_the_note_citations():
    """The clobber: a dict arrived, so the turn was treated as a literature turn.

    The note passages had resolved, `^[1]` addressed them, and every marker came back
    dead beside a card saying no literature was found about a question nobody asked.
    """
    out = compose_presentation(
        "What did the discharge note say about her diagnosis?",
        "She was admitted with sepsis^[1].",
        [{"name": "rag_search", "response": RAG},
         {"name": "search_literature", "response": EMPTY_LITERATURE}],
    )

    assert [row["cite"] for row in out["sources"]] == [1]
    assert out["literature"] is None
    assert "LiteratureList" not in _types(out)


def test_a_real_literature_result_still_takes_the_canvas():
    """The swap survives Phase F; the records list does not.

    A literature turn still suppresses the risk widgets — the question was about published
    evidence, so a widget would be the wrong thing to be looking at. What changed is where
    the records live: in the citation channel, so they draw as numbered pills and
    reference rows rather than as a second list under the prose.
    """
    out = compose_presentation(
        "What does the literature say about sepsis readmission?",
        "Readmission is common^[1].",
        [{"name": "search_literature", "response": LITERATURE}],
    )

    assert out["literature"] is not None
    assert "LiteratureList" not in _types(out)
    assert "RiskInfluencers" not in _types(out)


# --- Defect 4: /meds cites the whole list -------------------------------------

def _meds_recorded():
    return [{
        "name": "discharge_medications",
        "args": {"hadm_id": 90000015},
        "response": {
            "query": "discharge medications",
            "returned": 3,
            "passages": [
                {"id": "m1", "section": "discharge_medications", "text": "warfarin"},
                {"id": "m2", "section": "discharge_medications", "text": "metoprolol"},
                {"id": "i1", "section": "discharge_instructions",
                 "text": "warfarin 4 mg daily, metoprolol 50 mg daily"},
            ],
        },
    }]


def test_the_medication_passages_are_consolidated_into_the_first_one():
    recorded = _meds_recorded()

    deterministic._consolidate_medication_passages(recorded)

    passages = recorded[0]["response"]["passages"]
    assert len(passages) == 1
    assert recorded[0]["response"]["returned"] == 1
    # The section KEY is unchanged: citation resolution matches on it.
    assert passages[0]["section"] == "discharge_medications"
    # Every chunk is in the cited text, including the instructions that carry the doses.
    for fragment in ("warfarin", "metoprolol", "50 mg daily"):
        assert fragment in passages[0]["text"]


def test_a_single_passage_is_left_alone():
    recorded = [{"name": "discharge_medications", "args": {}, "response": {
        "returned": 1,
        "passages": [{"id": "m1", "section": "discharge_medications", "text": "warfarin"}],
    }}]

    deterministic._consolidate_medication_passages(recorded)

    assert len(recorded[0]["response"]["passages"]) == 1


def test_the_meds_marker_resolves_to_the_consolidated_text():
    """The wiring, not just the helper: `^[1]` must address the whole list."""
    recorded = _meds_recorded()
    deterministic._consolidate_medication_passages(recorded)
    answer = deterministic.medication_narrative(
        [{"medication": "warfarin", "dose": "4 mg"}], 1)

    out = compose_presentation(
        "What medications is she going home on? ",
        answer, recorded, MEDS_PAYLOAD, "meds",
    )

    assert out["sources"], "the /meds answer cites nothing"
    assert "metoprolol" in out["sources"][0]["text"]
    assert "50 mg daily" in out["sources"][0]["text"]


# --- Defect 8: no markdown headings in /ehr -----------------------------------

def test_the_ehr_instruction_no_longer_asks_for_markdown_headings():
    """`##` streamed to the reader as literal characters until the answer settled."""
    assert "## " not in routing.EHR_FORMATTING
    for label in ("**Hospital course.**", "**Clinical course.**",
                  "**Discharge diagnosis.**", "**Discharge instructions.**"):
        assert label in routing.EHR_FORMATTING


def test_the_ehr_instruction_matches_the_summary_style():
    """One style for a section label, whether the summary came from /ehr or /pubmed."""
    assert "**Clinical course.**" in SYSTEM_PROMPT
    assert "**Discharge diagnosis.**" in SYSTEM_PROMPT


# --- Defect 7: verified ------------------------------------------------------

def test_a_chart_answer_is_verified():
    out = compose_presentation(
        "What did the discharge note say about her diagnosis?",
        "She was admitted with sepsis^[1].",
        [{"name": "rag_search", "response": RAG}],
    )
    assert out["verified"] is True


def test_a_lab_answer_is_verified_without_a_citation():
    """`/labs` rests on a trajectory; requiring a `^[n]` would withhold the badge."""
    out = compose_presentation(
        "What did her creatinine do?", "It rose.", 
        [{"name": "get_lab_trends", "args": {"hadm_id": 90000015},
          "response": LABS}],
    )
    assert out["sources"] == []
    assert out["verified"] is True


def test_a_medication_command_is_verified():
    out = compose_presentation(
        "What medications is she going home on?",
        "2 discharge medications: warfarin, metoprolol.",
        [{"name": "discharge_medications", "args": {"hadm_id": 90000015},
          "response": _meds_recorded()[0]["response"]}],
        MEDS_PAYLOAD, "meds",
    )
    assert out["verified"] is True


def test_a_pure_literature_turn_is_not_verified():
    """Those are other people's patients. The badge says "against the chart"."""
    out = compose_presentation(
        "What does the literature say about sepsis readmission?",
        "Readmission is common^[1].",
        [{"name": "search_literature", "response": LITERATURE}],
    )
    assert out["verified"] is False


def test_a_turn_that_rests_on_nothing_is_not_verified():
    out = compose_presentation(
        "Tell me something.", "I am not able to answer that.", [],
    )
    assert out["verified"] is False


def test_replayed_chart_calls_count_as_grounding():
    """A follow-up answered from the earlier turn's score is still chart-grounded."""
    out = compose_presentation(
        "Why is that?", "Prior admissions contributed most of it.", [],
        context_grounded=True,
    )
    assert out["verified"] is True


def test_a_widget_only_turn_can_be_verified():
    """`/risk` has no prose; the widget IS the answer, and it rests on the chart."""
    out = compose_presentation(
        "What is her readmission risk?",
        "",
        [{"name": "predict_readmission", "args": {"hadm_id": 90000015},
          "response": {
              "hadm_id": 90000015, "probability": 0.62, "threshold": 0.5,
              "decision": 1,
              "top_factors": [{"feature": "prior_inpatient_visits",
                               "contribution": 0.11, "direction": "increases"}]}}],
        None, "risk",
    )
    assert out["verified"] is True
