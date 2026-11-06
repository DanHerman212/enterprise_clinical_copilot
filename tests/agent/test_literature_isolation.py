"""A search is an event, not a state — and the event stays in the transcript.

Reported on 90000058 (2026-10-26): a `/pubmed` turn was followed by a `/meds` chip, and
the medication answer came back with `^[1]` bound to a *World Neurosurgery* article, the
badge flipped to General Medical Evidence, and the old `LiteratureList` re-rendered under
the medication instructions.

The first fix removed the earlier turn's `search_literature` call from the replay, and
the composer was taught to read the literature payload from THIS turn's calls alone.

That first half was withdrawn on 2026-11-04, and this file pins both halves of why:

  * the transcript is the thread's evidence, and hiding a call the thread made is how a
    follow-up came to answer "the discharge notes do not document peer-reviewed research"
    about a benchmark the thread had already retrieved and compared against;
  * the composer still reads this turn's calls for the records it DRAWS, so a chart turn
    never acquires a record list it did not ask for.

What travels instead of the pool is the thread's verified evidence: the records a search
returned that the same turn's answer then cited by PMID. Offline — no model, no MCP.
"""

from __future__ import annotations

import asyncio

from services.agent.engine import graph
from services.agent.presentation.a2ui import compose_presentation

ARTICLE = {
    "pmid": "29950074",
    "title": "Readmission after lumbar spine surgery.",
    "journal": "World Neurosurgery",
    "pub_date": "2018 Jul 01",
    "abstract": "The 30-day readmission rate was 4.1%.",
    "url": "https://pubmed.ncbi.nlm.nih.gov/29950074/",
}

SEARCH_CALL = {
    "name": "search_literature",
    "args": {"query": "lumbar stenosis readmission"},
    "response": {"query": "lumbar stenosis readmission", "returned": 1,
                 "articles": [ARTICLE]},
}

MEDS_CALL = {
    "name": "discharge_medications",
    "args": {"hadm_id": 90000058},
    "response": {
        "hadm_id": 90000058,
        "query": "discharge medications",
        "returned": 1,
        "passages": [{
            "id": "meds-1", "section": "discharge_medications",
            "text": "Discharge Medications:\nwarfarin 4 mg QD",
        }],
    },
}

PREDICT_CALL = {
    "name": "predict_readmission",
    "args": {"hadm_id": 90000058},
    "response": {"hadm_id": 90000058, "probability": 0.2, "threshold": 0.11,
                 "decision": 1, "model_version": "test-v1"},
}


class _MustNotBeCalled:
    """The stored payload is the payload; the replay must not go and ask again."""

    async def call(self, *_args, **_kwargs):  # pragma: no cover - must not run
        raise AssertionError("the replay re-resolved a stored literature result")


def _types(out: dict) -> list[str]:
    return [c["component"]
            for c in out["a2ui"]["messages"][1]["updateComponents"]["components"]]


class TestTheReplayCarriesEveryCall:
    def test_a_chart_turn_is_handed_the_earlier_search(self):
        """The transcript is the thread's evidence, whatever the current question is.

        It used to be dropped here, to stop a stale result re-entering a chart answer
        (90000058). What that cost was a follow-up that could not see the benchmark the
        thread had already compared against, could not search for it again, and answered
        from the discharge note instead (2026-11-04). The composer is where an unwanted
        record is kept out of an answer, and it still is.
        """
        turns = [{
            "question": "/pubmed lumbar stenosis readmission",
            "answer": "The 30-day rate after lumbar surgery is 4.1%.^[PMID: 29950074]",
            "tool_calls": [{"name": "search_literature",
                            "args": {"query": "lumbar stenosis readmission"},
                            "payload": SEARCH_CALL["response"]}],
        }]

        messages, calls, articles = asyncio.run(graph.replay_messages(
            _MustNotBeCalled(), turns))

        assert [call["name"] for call in calls] == ["search_literature"]
        # The records travel as EVIDENCE, not as a replayed tool body: `calls` hands the
        # payload forward for the guardrail and the composer, and the session record
        # renders it under `<literature>` for the model. Nothing is synthesised into the
        # transcript any more (Phase 1 Sprint 1.4).
        assert calls[0]["response"]["articles"][0]["pmid"] == ARTICLE["pmid"]
        assert not any("World Neurosurgery" in str(message.content)
                       for message in messages), (
            "a replayed turn contributes its question and its answer, nothing else")
        # And the record the turn rested on is thread evidence, addressed by identifier.
        assert [article["pmid"] for article in articles] == [ARTICLE["pmid"]]

    def test_the_pool_a_search_returned_is_not_thread_evidence(self):
        """Twenty-five candidates come back and one is cited; the rest stay behind.

        Carrying the whole pool would let a later turn cite a record no turn of this
        thread ever used — which is the invented identifier `enforce_citations` exists to
        delete, wearing the thread's authority.
        """
        uncited = dict(ARTICLE, pmid="11111111",
                       title="A record this turn never cited.")
        turns = [{
            "question": "/pubmed lumbar stenosis readmission",
            "answer": "The rate is 4.1%.^[PMID: 29950074]",
            "tool_calls": [{"name": "search_literature",
                            "args": {"query": "lumbar stenosis readmission"},
                            "payload": {"query": "q", "returned": 2,
                                        "articles": [ARTICLE, uncited]}}],
        }]

        _, _, articles = asyncio.run(graph.replay_messages(_MustNotBeCalled(), turns))

        assert [article["pmid"] for article in articles] == [ARTICLE["pmid"]]

    def test_a_turn_that_cited_nothing_contributes_nothing(self):
        """A search its own answer never used is not replayed at all.

        The turn contributed no evidence to that answer, so a later turn inherits nothing
        from it — and the twenty-five abstracts it returned never reach a prompt.
        """
        turns = [{
            "question": "/pubmed lumbar stenosis readmission",
            "answer": "Two records were returned.",
            "tool_calls": [{"name": "search_literature",
                            "args": {"query": "lumbar stenosis readmission"},
                            "payload": SEARCH_CALL["response"]}],
        }]

        messages, calls, articles = asyncio.run(graph.replay_messages(
            _MustNotBeCalled(), turns))

        assert calls == []
        assert articles == []
        assert not any("World Neurosurgery" in str(message.content)
                       for message in messages)

    def test_a_chart_turn_still_replays_its_own_chart_calls(self):
        """Narrowness check: nothing was dropped, and nothing was added."""
        turns = [{
            "question": "/meds",
            "answer": "Warfarin 4 mg daily.",
            "tool_calls": [{"name": "discharge_medications",
                            "args": {"hadm_id": 90000058},
                            "payload": MEDS_CALL["response"]}],
        }]

        _, calls, articles = asyncio.run(graph.replay_messages(
            _MustNotBeCalled(), turns))

        assert [call["name"] for call in calls] == ["discharge_medications"]
        assert articles == []


class TestTheSurfaceReadsThisTurnOnly:
    def test_a_medication_follow_up_stays_a_chart_answer(self):
        """The reported turn, composed. A search in the replay must not reach the badge,
        the record list or the citation the answer writes."""
        out = compose_presentation(
            "Are any of these anticoagulants?",
            "Warfarin is an anticoagulant.^[1]",
            [MEDS_CALL],
            evidence=[SEARCH_CALL, MEDS_CALL, PREDICT_CALL],
        )

        assert out["scope"] == "patient_chart"
        assert out["literature"] is None
        assert "LiteratureList" not in _types(out)

        # The marker resolves to the medication passage, not to the article.
        assert out["sources"][0]["section"] == "discharge_medications"
        assert out["references"][0]["source"] == "Discharge note"
        # And nothing anywhere in the turn points at the record.
        assert "pmid" not in out["sources"][0]
        assert "World Neurosurgery" not in str(out["references"])

    def test_the_same_turn_that_does_search_draws_its_records(self):
        """The gate is the turn's own call, in both directions: on a turn that searched,
        the records are the evidence — as citations, since Phase F."""
        out = compose_presentation(
            "What does the literature say about readmission after spine surgery?",
            "The 30-day rate was 4.1%.^[1]",
            [SEARCH_CALL, PREDICT_CALL],
        )

        assert out["literature"] is not None
        assert [s["kind"] for s in out["sources"]] == ["Literature"]
        assert out["references"][0]["pmid"] == ARTICLE["pmid"]
        assert out["scope"] == "general_evidence"
