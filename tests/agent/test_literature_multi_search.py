"""A turn that searches four times is one turn, and every search is part of its evidence.

On 2026-10-30 a free-text question searched PubMed four times — the model refining its
query — and wrote its answer against the records of the later searches. Every reader on
this side took the FIRST call: `literature_pmids` had an allowance holding none of the
records the answer cited, `literature_payload` had no articles, `sources` published no
records, and `_literature_topic` named a query from a search whose results were not the
ones on the screen.

The reader saw a sentence ending `"dead" .` where a `^[PMID: …]` badge had been removed,
and a references drawer with one note row in it.

One definition now: `citations.literature_calls` returns every search of the turn,
`literature_articles` unions their records by pmid in call order, and the allowance, the
count, the payload and the chip topic all read that one list.
"""

from __future__ import annotations

from services.agent.postprocess import citations
from services.agent.postprocess.followups import suggest
from services.agent.presentation.a2ui import compose_presentation, literature_payload


def _search(pmids, query, **overrides):
    articles = [
        {"pmid": pmid, "title": f"T{pmid}", "journal": "J", "pub_date": "2023",
         "abstract": f"A{pmid}", "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"}
        for pmid in pmids
    ]
    payload = {"query": query, "returned": len(articles), "articles": articles}
    payload.update(overrides)
    return {"name": "search_literature", "args": {"query": query}, "response": payload}


#: The live turn, in order: an empty search, then three that found records.
EMPTY = _search([], "hypothermia")
FIRST = _search(["37797337"], "hypothermia rewarming")
SECOND = _search(["35010760"], "accidental hypothermia")
LAST = _search(["35010760", "33485655"], "accidental hypothermia rewarming management")

FOUR_SEARCHES = [EMPTY, FIRST, SECOND, LAST]


class TestTheUnion:
    def test_every_search_contributes_its_records(self):
        assert citations.literature_articles(
            citations.literature_calls(FOUR_SEARCHES)) == [
                {"pmid": "37797337", "title": "T37797337", "journal": "J",
                 "pub_date": "2023", "abstract": "A37797337",
                 "url": "https://pubmed.ncbi.nlm.nih.gov/37797337/"},
                {"pmid": "35010760", "title": "T35010760", "journal": "J",
                 "pub_date": "2023", "abstract": "A35010760",
                 "url": "https://pubmed.ncbi.nlm.nih.gov/35010760/"},
                {"pmid": "33485655", "title": "T33485655", "journal": "J",
                 "pub_date": "2023", "abstract": "A33485655",
                 "url": "https://pubmed.ncbi.nlm.nih.gov/33485655/"},
            ]

    def test_a_record_two_searches_both_returned_appears_once(self):
        """Two searches for one condition overlap. A record listed twice is two drawer
        rows and two numbers for one paper."""
        articles = citations.literature_articles(
            citations.literature_calls([SECOND, LAST]))

        assert [article["pmid"] for article in articles] == ["35010760", "33485655"]

    def test_the_allowance_holds_the_records_the_last_search_returned(self):
        """The defect: the answer cited 35010760 and the allowance was built from a
        search that returned nothing."""
        assert citations.literature_pmids(FOUR_SEARCHES) == frozenset(
            {"37797337", "35010760", "33485655"})

    def test_a_turn_with_one_search_is_unchanged(self):
        assert citations.literature_pmids([LAST]) == frozenset(
            {"35010760", "33485655"})
        assert citations.literature_slots([LAST]) == 2

    def test_the_numbered_count_is_the_union_too(self):
        """`literature_slots` bounds a `^[n]` on a literature turn, and the model was
        shown every search's records — so the bound is the union, not one search's."""
        assert citations.literature_slots(FOUR_SEARCHES) == 3

    def test_a_single_call_still_reads_as_one_call(self):
        assert citations.literature_call([LAST]) is LAST


class TestThePayload:
    def test_the_payload_carries_every_record_the_turn_retrieved(self):
        merged = literature_payload(FOUR_SEARCHES)

        assert [article["pmid"] for article in merged["articles"]] == [
            "37797337", "35010760", "33485655"]
        assert merged["returned"] == 3

    def test_the_payload_names_the_query_the_answer_was_written_against(self):
        """The caption says what was searched. The first query would describe a search
        whose records may not be on the screen at all."""
        assert literature_payload(FOUR_SEARCHES)["query"] == (
            "accidental hypothermia rewarming management")

    def test_a_failed_search_alone_is_still_reported_as_a_failure(self):
        """No records means the payload is returned as it stands, because `error` is a
        different statement from "nothing was found"."""
        failed = {"name": "search_literature", "args": {"query": "x"},
                  "response": {"error": "search_failed", "query": "x"}}

        assert literature_payload([failed]) == {"error": "search_failed", "query": "x"}

    def test_a_failure_beside_successes_is_not_called_a_failure(self):
        """One call failing while another returned records is a turn that has records.
        Reporting it as failed would draw the unavailable caption over a full list."""
        failed = {"name": "search_literature", "args": {"query": "x"},
                  "response": {"error": "search_failed", "query": "x"}}

        merged = literature_payload([failed, LAST])

        assert "error" not in merged
        assert merged["returned"] == 2

    def test_no_literature_call_is_still_none(self):
        assert literature_payload([]) is None


class TestTheChipTopic:
    def test_the_topic_comes_from_the_last_query(self):
        """The chip asks what to read next about the records the answer rests on."""
        chips = suggest(
            "can you search pubmed for hypothermia?",
            "Two trials report benefit.^[PMID: 35010760]",
            scope="general_evidence", literature=True,
            state={"tool_calls": FOUR_SEARCHES, "replayed_tool_calls": []},
        )

        assert chips == ["What else does the published literature report on "
                         "accidental hypothermia rewarming management?"]


class TestTheRecordedTurn:
    def test_the_answer_cites_a_record_the_turn_retrieved(self):
        """End to end on the live shape: the badge the model wrote survives settlement,
        the record reaches the drawer, and the two name the same paper."""
        answer = 'Rewarming is the priority.^[PMID: 35010760]'

        allowed = citations.literature_pmids(FOUR_SEARCHES)
        settled, flags = citations.enforce_citations(answer, 0, allowed)
        out = compose_presentation(
            "can you search pubmed for hypothermia?", settled, FOUR_SEARCHES)

        assert flags == []
        assert settled.endswith("^[PMID: 35010760]")
        published = [row for row in out["references"] if row.get("pmid")]
        # Union order — the first search's record, then the later ones — because that is
        # the order the model was shown them in.
        assert [row["pmid"] for row in published] == [
            "37797337", "35010760", "33485655"]
