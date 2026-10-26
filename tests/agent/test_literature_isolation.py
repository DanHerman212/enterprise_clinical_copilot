"""A search is an event, not a state.

Reported on 90000058 (2026-10-26): a `/pubmed` turn was followed by a `/meds` chip, and
the medication answer came back with `^[1]` bound to a *World Neurosurgery* article, the
badge flipped to General Medical Evidence, and the old `LiteratureList` re-rendered under
the medication instructions.

Two causes, and both had to go:

  1. The replay carried the earlier turn's `search_literature` call, so the loop was
     handed somebody else's patients as context for a question about this one.
  2. The composer read the literature payload from the whole conversation's evidence, so
     even a turn that searched nothing found records to draw.

The rule is the one the rest of the citation layer already follows: what a turn answers
from is what that turn looked up. Offline — no model, no MCP.
"""

from __future__ import annotations

import asyncio

from services.agent import graph
from services.agent.a2ui import compose_presentation

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


class TestTheReplayKeepsSearchesOut:
    def test_a_chart_turn_is_not_handed_an_earlier_search(self):
        """The loop cannot repeat a mistake it is not offered: the record was retrieved for
        a question that was asked and answered."""
        turns = [{
            "question": "/pubmed lumbar stenosis readmission",
            "answer": "The 30-day rate after lumbar surgery is 4.1%.^[PMID: 29950074]",
            "tool_calls": [{"name": "search_literature",
                            "args": {"query": "lumbar stenosis readmission"},
                            "payload": SEARCH_CALL["response"]}],
        }]

        messages, calls = asyncio.run(graph.replay_messages(
            _MustNotBeCalled(), turns, keep_literature=False))

        assert calls == []
        assert not any("World Neurosurgery" in str(message.content)
                       for message in messages)

    def test_a_literature_turn_keeps_them(self):
        """The other direction, which is why the flag is a parameter and not a rule: a
        turn about published records may cite the ones already retrieved, and discarding
        them would be the same mistake pointing the other way."""
        turns = [{
            "question": "/pubmed lumbar stenosis readmission",
            "answer": "The rate is 4.1%.^[PMID: 29950074]",
            "tool_calls": [{"name": "search_literature",
                            "args": {"query": "lumbar stenosis readmission"},
                            "payload": SEARCH_CALL["response"]}],
        }]

        _, calls = asyncio.run(graph.replay_messages(
            _MustNotBeCalled(), turns, keep_literature=True))

        assert [call["name"] for call in calls] == ["search_literature"]

    def test_a_chart_turn_still_replays_its_own_chart_calls(self):
        """Narrowness check: the filter removes one tool, not the transcript."""
        turns = [{
            "question": "/meds",
            "answer": "Warfarin 4 mg daily.",
            "tool_calls": [{"name": "discharge_medications",
                            "args": {"hadm_id": 90000058},
                            "payload": MEDS_CALL["response"]}],
        }]

        _, calls = asyncio.run(graph.replay_messages(
            _MustNotBeCalled(), turns, keep_literature=False))

        assert [call["name"] for call in calls] == ["discharge_medications"]


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
