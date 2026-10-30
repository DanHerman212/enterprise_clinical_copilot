"""A turn that answers from the chart AND the literature cites both, with one numbering.

The live case (2026-10-30, trace `079be88f3daa505b2a80491f8759539d`): paragraph one
summarised a PubMed record, paragraph two summarised the patient's own admission. The
narrative carried an article-only citation list, so the note markers `^[4]` and `^[7]`
had nothing to open — grey circles a reader could not click — and the references drawer
listed the note row and none of the records.

Three things had to be true together:

  * `sources` holds the note passages and the records in one array, numbered in that
    order, so a marker addresses one number however many channels it reaches;
  * the narrative cites both channels with those same numbers;
  * the record numbers start after the note numbers, so a published record never
    shares a number with a note passage.

The last one was wrong independently of the narrative: `literature_sources(articles)`
numbered from 1 while `register_benchmark_sources` beside it continued from
`len(sources) + 1`.
"""

from __future__ import annotations

from services.agent.postprocess import citations
from services.agent.presentation.a2ui import compose_presentation

#: Seven note passages, so `^[4]` and `^[7]` are both inside the citation bound.
PASSAGES = [
    {"section": section, "text": f"passage {index}"}
    for index, section in enumerate(
        ["history_of_present_illness", "brief_hospital_course", "discharge_diagnosis",
         "brief_hospital_course", "discharge_medications", "discharge_instructions",
         "discharge_disposition"],
        start=1,
    )
]

CHART = {
    "name": "rag_search",
    "args": {"hadm_id": 90000059, "query": "hypothermia admission"},
    "response": {"query": "hypothermia admission", "returned": len(PASSAGES),
                 "passages": PASSAGES},
}

ARTICLES = [
    {"pmid": "35010760", "title": "Accidental hypothermia", "journal": "N Engl J Med",
     "pub_date": "2022 Jan", "abstract": "Rewarming.", "url": "u1"},
    {"pmid": "37797337", "title": "Rewarming strategies", "journal": "Resuscitation",
     "pub_date": "2023 Sep", "abstract": "Methods.", "url": "u2"},
]

SEARCH = {
    "name": "search_literature",
    "args": {"query": "accidental hypothermia rewarming management"},
    "response": {"query": "accidental hypothermia rewarming management",
                 "returned": len(ARTICLES), "articles": ARTICLES},
}

#: The live answer, in shape: an article paragraph and a chart paragraph.
ANSWER = (
    "The literature emphasises a structured chain of survival.^[PMID: 35010760] "
    "The admission documents a hospital course of progressive hypothermia.^[4] "
    "The patient was discharged home in stable condition.^[7]"
)


def _presented():
    return compose_presentation(
        "can you search pubmed for hypothermia?", ANSWER, [CHART, SEARCH])


def _narrative(presentation):
    return next(
        component for message in presentation["a2ui"]["messages"]
        for component in message.get("updateComponents", {}).get("components", [])
        if component["component"] == "CitedNarrative"
    )


def _indexes(presentation):
    return sorted(citation["index"] for citation in _narrative(presentation)["citations"])


class TestTheNumbering:
    def test_the_records_are_numbered_after_the_note_passages(self):
        """The live drawer listed `4.` and no record. Numbering records from 1 would
        have given two different sources the same number and one of them no row."""
        presentation = _presented()
        published = [row for row in presentation["references"] if row.get("pmid")]

        assert [row["index"] for row in published] == [8, 9]

    def test_a_record_never_shares_a_number_with_a_note_passage(self):
        presentation = _presented()
        indexes = [row["index"] for row in presentation["references"]]

        assert len(indexes) == len(set(indexes)), f'duplicate reference numbers: {indexes}'

    def test_the_drawer_lists_the_records_and_the_note_passages_together(self):
        """Both channels, in one list: a reader checking a claim finds the row whichever
        kind of source it was."""
        presentation = _presented()
        titles = [row["title"] for row in presentation["references"]]

        assert "Accidental hypothermia" in titles
        assert "Rewarming strategies" in titles
        assert "Brief hospital course" in titles


class TestTheNarrative:
    def test_a_mixed_answer_cites_both_channels(self):
        """The defect: an article-only citation list turned every note marker into a
        disabled control."""
        assert _indexes(_presented()) == [4, 7, 8, 9]

    def test_the_clinicians_note_markers_resolve(self):
        citations = {citation["index"]: citation for citation in _narrative(_presented())["citations"]}

        assert citations[4]["title"] == "Brief hospital course"
        assert citations[7]["title"] == "Discharge disposition"
        assert citations[8]["pmid"] == "35010760"

    def test_a_literature_only_turn_numbers_its_records_from_one(self):
        """The 2026-10-22 rule is not weakened, and with no citable note units the
        records start at 1 as they always did."""
        presentation = compose_presentation(
            "search pubmed for hypothermia", ANSWER, [SEARCH])

        assert _indexes(presentation) == [1, 2]
        assert all(citation.get("pmid") or citation["index"] in (1, 2)
                   for citation in _narrative(presentation)["citations"])

    def test_a_chart_only_turn_is_unchanged(self):
        chart_only = compose_presentation(
            "what happened during the admission?", "Seen.^[4]", [CHART])

        assert _indexes(chart_only) == [4]
        assert all("pmid" not in row for row in chart_only["references"])


class TestTheDeadBadge:
    def test_a_marker_inside_the_bound_always_produces_a_row(self):
        """The checked question: `^[7]` drew as a grey circle with no `7.` row beneath it.

        Both sides of that must agree — `citation_bound` counts `citable_passages` over
        the conversation's evidence and `resolve_sources` resolves against the same
        array — so a marker the enforcement keeps has a row. This asserts the agreement
        on the shape the live turn had, seven passages and two records.
        """
        presentation = _presented()
        published = {row["index"] for row in presentation["references"]}

        for marker in (4, 7):
            assert marker in published, f'^{marker} survived enforcement with no row'

    def test_a_marker_beyond_the_bound_is_removed_rather_than_drawn_dead(self):
        """The other half: a position nothing can resolve must not reach the page.

        Settlement removes it — the composer is pure and draws what it is given — so this
        drives both passes in order, which is the sequence a turn goes through.
        """
        settled, flags = citations.enforce_citations(
            "Seen.^[9]", len(PASSAGES))
        presentation = compose_presentation(
            "what happened during the admission?", settled, [CHART])

        assert flags == ["citation_out_of_range:^9"]
        assert _indexes(presentation) == []
        assert presentation["references"] == []
