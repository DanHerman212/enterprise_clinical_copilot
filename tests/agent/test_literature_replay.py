"""A declared follow-up answers from the search already in its transcript.

`graph._freetext_allowed` withholds `search_literature` from a declared follow-up, and it
is right to: the chip was generated from a tool result in the conversation, so the result
is in the replay before the chip is clicked. What was missing was every reader AFTER the
model. The replay put the records in front of the model; the badge allowance and the
composer read this turn's `tool_calls` alone, so a turn that cited the records it was
shown had no allowance to cite them with and no records to draw.

Live on 2026-10-30: `/ehr` → `/pubmed bladder cancer` → the literature chip. The third
turn ran no search, wrote `^[PMID: 39559360, 37028984, 36774746]`, and came back with
the badge as raw text, no references drawer and chart follow-up chips.

The gate is `wants_literature`, the same flag that let the replay keep the search, stamped
on the state by `graph.ask`. The tests below pin both halves: a turn about published
records may read the replayed search, and a chart turn may not — 90000058, where a search
from three turns earlier bound `^[1]` under a medication answer and badged it General
Medical Evidence.
"""

from services.agent.engine.graph import turn_is_about_literature
from services.agent.postprocess import citations
from services.agent.postprocess.followups import suggest
from services.agent.presentation.a2ui import compose_presentation

ARTICLES = [
    {
        "pmid": "39559360",
        "title": "Bladder cancer immunotherapy in 2024",
        "journal": "Nat Rev Urol",
        "pub_date": "2024 Oct",
        "abstract": "BCG remains first line for high-risk non-muscle-invasive disease.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/39559360/",
    },
    {
        "pmid": "37028984",
        "title": "Neoadjuvant chemotherapy for muscle-invasive disease",
        "journal": "Eur Urol",
        "pub_date": "2023 May",
        "abstract": "Cisplatin-based neoadjuvant chemotherapy improves survival.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/37028984/",
    },
    {
        "pmid": "36774746",
        "title": "Cystectomy outcomes in an ageing cohort",
        "journal": "J Urol",
        "pub_date": "2023 Feb",
        "abstract": "Robotic cystectomy had comparable complication rates.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/36774746/",
    },
]

PMIDS = frozenset({"39559360", "37028984", "36774746"})

#: The recorded call as the store keeps it — name, arguments, response.
SEARCH = {
    "name": "search_literature",
    "args": {"query": "bladder cancer"},
    "response": {"query": "bladder cancer", "returned": 3, "articles": ARTICLES},
}

#: A chart read the same conversation made, so a chart turn has a scope to keep.
CHART = {
    "name": "rag_search",
    "args": {"hadm_id": 90000001, "query": "discharge medications"},
    "response": {
        "passages": [{"section": "discharge_medications", "text": "Aspirin 81 mg daily."}],
        "query": "discharge medications",
        "returned": 1,
    },
}

CHIP = "What else does the published literature report on bladder cancer?"
ANSWER = "Two further trials report benefit.^[PMID: 39559360, 37028984, 36774746]"


class TestTheBadgeAllowance:
    def test_the_replayed_search_supplies_the_identifiers_the_turn_may_cite(self):
        """The turn ran no search, and the records it cites are the replayed ones."""
        assert citations.literature_pmids([], [SEARCH]) == PMIDS

    def test_without_the_replay_the_allowance_is_empty(self):
        """The default is the old behaviour, so a caller that says nothing gets nothing.

        This is the state every chart turn is in: no allowance, so a badge naming a
        record from an earlier search is an invented identifier and is removed.
        """
        assert citations.literature_pmids([]) == frozenset()

    def test_a_live_search_is_read_instead_of_the_replayed_one(self):
        """A turn that looked something up answers from what it looked up.

        The replay is the fallback, never a second opinion: a turn with its own search
        must not silently cite the earlier turn's records instead.
        """
        other = {
            "name": "search_literature",
            "args": {"query": "cystectomy"},
            "response": {"query": "cystectomy", "returned": 1, "articles": [
                {"pmid": "36774746", "title": "Cystectomy outcomes",
                 "journal": "J Urol", "pub_date": "2023 Feb", "abstract": "x", "url": ""},
            ]},
        }

        assert citations.literature_pmids([other], [SEARCH]) == frozenset({"36774746"})

    def test_the_newest_replayed_search_is_the_one_that_counts(self):
        """Two searches in the transcript: the last one the model was shown wins.

        `search_literature` calls accumulate in the replay oldest-first, and the answer
        was written against the most recent of them.
        """
        older = {
            "name": "search_literature",
            "args": {"query": "old"},
            "response": {"query": "old", "returned": 1, "articles": [
                {"pmid": "11111111", "title": "Old", "journal": "J",
                 "pub_date": "2020", "abstract": "x", "url": ""},
            ]},
        }

        assert citations.literature_pmids([], [older, SEARCH]) == PMIDS


class TestTheDrawer:
    def test_the_replayed_records_reach_the_references_and_the_scope(self):
        """The whole visible repair: three rows, and the badge that goes with them.

        `references` is what the console's drawer lists, and `scope` is what the pill
        says. Both were empty/`patient_chart` while the answer cited three published
        records — the one combination that teaches a reader to distrust the badge.
        """
        presentation = compose_presentation(
            CHIP, ANSWER, [], evidence=[SEARCH], literature_replay=True)

        assert presentation["scope"] == "general_evidence"
        assert [row.get("pmid") for row in presentation["references"]] == [
            "39559360", "37028984", "36774746"]

    def test_a_chart_turn_does_not_inherit_the_replayed_search(self):
        """90000058, as a test: the same evidence, the flag off, nothing drawn.

        The search is in the conversation and the chart read is this answer's evidence.
        Without the flag the turn is what it was before the fix — patient chart, no
        literature rows — because a search belongs to the turn that made it until that
        turn's own question says otherwise. The note's own citation still draws its row:
        what is withheld is the published record, not the drawer.
        """
        presentation = compose_presentation(
            "What are the discharge medications?", "Aspirin 81 mg daily.^[1]",
            [], evidence=[CHART, SEARCH])

        assert presentation["scope"] == "patient_chart"
        assert [row["index"] for row in presentation["references"]] == [1]
        assert all("pmid" not in row for row in presentation["references"])
        assert presentation["literature"] is None

    def test_the_badge_resolves_against_the_replayed_records_end_to_end(self):
        """Settlement and presentation agree: the split markers name the drawn rows.

        The two halves of the fix meet here. The allowance comes from the replay, so the
        grouped badge expands instead of being deleted, and the records the markers name
        are the rows the drawer lists.
        """
        allowed = citations.literature_pmids([], [SEARCH])
        settled, flags = citations.enforce_citations(ANSWER, 0, allowed)
        presentation = compose_presentation(
            CHIP, settled, [], evidence=[SEARCH], literature_replay=True)

        assert flags == []
        assert settled.count("^[PMID: ") == 3
        assert {row["pmid"] for row in presentation["references"]} == PMIDS


class TestTheChips:
    def test_the_literature_chip_names_the_topic_the_search_ran(self):
        """The topic comes from the call, not from the question that cited it.

        A declared follow-up holds no search of its own, so without the replayed call the
        topic fell back to the raw question and the chip read "What else does the
        published literature report on Does the literature support neoadjuvant
        chemotherapy?" — a control that repeats the question instead of advancing it.
        Here the question and the search are about different things, so the chip shows
        which one it read.
        """
        chips = suggest(
            "Does the literature support neoadjuvant chemotherapy?", ANSWER,
            scope="general_evidence", literature=True,
            state={"tool_calls": [], "replayed_tool_calls": [SEARCH]},
        )

        assert chips == [
            "What else does the published literature report on bladder cancer?"]

    def test_the_question_just_clicked_is_not_offered_back(self):
        """The live turn 3, followed to its end: the chip is consumed, not repeated.

        With the topic read from the search, the chip is the question that was just asked
        — so the row is empty rather than inviting the same click twice. Asserted because
        it is what the repaired turn will actually show, and an empty row read as a
        regression would send someone looking for it here.
        """
        assert suggest(
            CHIP, ANSWER, scope="general_evidence", literature=True,
            state={"tool_calls": [], "replayed_tool_calls": [SEARCH]},
        ) == []

    def test_a_chart_turn_after_a_literature_turn_offers_no_literature_chip(self):
        """The replay holding a search is not a reason to ask about one.

        Same state, flag off: every chip must be about the record. This is the isolation
        that keeps a chart answer from being followed by a literature question, and its
        mirror — a chart answer that draws literature — is 90000058.
        """
        chips = suggest(
            "What are the discharge medications?",
            "Aspirin 81 mg daily, with no anticoagulant on the list.",
            scope="patient_chart", literature=False,
            state={"tool_calls": [CHART], "replayed_tool_calls": [SEARCH]},
        )

        assert all("PubMed" not in chip for chip in chips)


class TestTheGate:
    def test_a_pubmed_command_declares_itself_through_its_intent(self):
        """`/pubmed bladder cancer` carries no literature cue in the typed words."""
        assert turn_is_about_literature("bladder cancer", "literature") is True

    def test_a_question_about_the_published_evidence_is_a_literature_turn(self):
        """The chip's own wording, which is what the live turn 3 was."""
        assert turn_is_about_literature(CHIP, "follow_up") is True

    def test_a_chart_question_is_not_a_literature_turn(self):
        """A declared follow-up about the record declares nothing about the literature."""
        assert turn_is_about_literature(
            "What are the discharge medications?", "follow_up") is False
