"""Seam 1 — turn-2 literature continuity.

Turn 1 is `/ehr`: its comparison table is built from a PubMed search that declares
`scope="readmission"`, and its prose cites one or two of those cohorts by PMID. Turn 2 is
the clinician asking how this patient compares to *those* cohorts.

The defect this module pins: `replay_messages` pruned the search to the cited records and
then handed the record the ORIGINAL arguments, which still declared the candidate pool. The
record refuses a candidate pool — correctly, a pool is not an answer's evidence — so the
pruned, cited, verified records were refused too. `<literature>` rendered empty on turn 2,
`_session_holds_evidence` said the session held no published evidence, the model was sent
back to PubMed, and the turn ended by reporting that no indexed literature matched a query
for a cohort it had already been shown.

Each test is one link of that chain, so a regression names its own cause.
"""

from __future__ import annotations

import asyncio

from services.agent.engine import graph
from services.agent.engine.mcp_client import _presented_call
from services.agent.postprocess import citations
from services.agent.postprocess.evidence_state import SessionEvidenceState

HADM = 90000069

#: The two cohorts a 41-year-old's cervical-fusion briefing cites: one it rests on, and one
#: it refuses as a population mismatch (an elderly-only subgroup). Both are in the pool;
#: only the first is named by the answer.
CITED_PMID = "36198705"
POOL_ONLY_PMID = "35120405"


def _article(pmid: str, title: str, abstract: str) -> dict:
    return {
        "pmid": pmid,
        "title": title,
        "journal": "Spine",
        "pub_date": "2022 Apr",
        "authors": "Okafor N, Lindqvist E.",
        "abstract": abstract,
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    }


POOL = {
    "query": "anterior cervical discectomy and fusion readmission",
    "returned": 2,
    "articles": [
        _article(CITED_PMID,
                 "Thirty-day readmission after anterior cervical discectomy and fusion.",
                 "The 30-day readmission rate was 4.1% in 8,412 patients."),
        _article(POOL_ONLY_PMID,
                 "Readmission after cervical fusion in patients over 80.",
                 "The 30-day readmission rate was 9.2% in octogenarians."),
    ],
}

#: Turn 1 exactly as the site replays it: a comparator search, and an answer that names one
#: of its records. `citation_map` is empty because this answer's only marker is a PMID
#: badge, which is an identifier rather than a position.
COMPARATOR_TURN = {
    "question": "/ehr For admission 90000069.",
    "answer": (
        "Her estimate is above the 4.1% published for this procedure "
        f"^[PMID: {CITED_PMID}], which is a population benchmark rather than a "
        "prediction about her."
    ),
    "tool_calls": [{
        "name": "search_literature",
        "args": {"query": "anterior cervical discectomy and fusion readmission",
                 "scope": "readmission"},
        "payload": POOL,
    }],
    "citation_map": {},
}


class _NoReResolve:
    """A toolbox that fails the test if a stored payload is fetched again."""

    async def call(self, name, arguments):  # pragma: no cover - asserted absence
        raise AssertionError(f"re-executed {name!r} although its payload was stored")


def _replayed():
    return asyncio.run(graph.replay_messages(
        _NoReResolve(), [COMPARATOR_TURN], hadm_id=HADM))


def test_the_pruned_cited_records_enter_the_session_record():
    """The cited cohort is thread evidence; the pool it came from is not."""
    _messages, calls, thread_articles = _replayed()
    session = SessionEvidenceState.hydrate_from_calls(
        HADM, [_presented_call(call) for call in calls])

    assert set(session.literature) == {CITED_PMID}
    assert POOL_ONLY_PMID not in session.literature
    assert [str(article["pmid"]) for article in thread_articles] == [CITED_PMID]


def test_the_candidate_pool_scope_does_not_survive_the_prune():
    """The scope described the search; what survives it is the answer's own evidence.

    Asserted on the replayed call rather than only on the outcome, because this is the
    line the fix is: the record's refusal is correct behaviour, and the bug was feeding it
    a scope that no longer described the payload.
    """
    _messages, calls, _articles = _replayed()

    assert calls[0]["args"].get("scope") is None
    assert calls[0]["args"]["query"] == POOL["query"], (
        "the search's own arguments are kept; only the candidate-pool scope is dropped")


def test_the_record_renders_the_cited_abstract_for_the_next_turn():
    _messages, calls, _articles = _replayed()
    session = SessionEvidenceState.hydrate_from_calls(
        HADM, [_presented_call(call) for call in calls])

    rendered = session.render_for_prompt()

    assert "<literature>" in rendered
    assert f"PMID: {CITED_PMID}" in rendered
    assert "The 30-day readmission rate was 4.1%" in rendered, (
        "the abstract travels, because a rate quoted on turn 3 has to be readable")
    assert POOL_ONLY_PMID not in rendered


def test_a_literature_question_over_that_thread_needs_no_new_search():
    """The routing consequence: the record answers the modality, so `route` ends the turn."""
    _messages, calls, _articles = _replayed()
    session = SessionEvidenceState.hydrate_from_calls(
        HADM, [_presented_call(call) for call in calls])

    assert graph._session_holds_evidence(
        {"evidence_state": session, "thread_literature": []},
        wants_literature=True,
    ) is True


def test_an_unpruned_pool_is_still_refused():
    """The rule the fix must not weaken: turn 1's pool is not the answer's evidence."""
    session = SessionEvidenceState(hadm_id=HADM)

    session.ingest_tool_output(
        "search_literature",
        {"query": POOL["query"], "scope": "readmission"},
        POOL,
    )

    assert session.literature == {}


def test_a_search_the_answer_cited_nothing_from_is_not_replayed():
    turn = {
        "question": "/ehr For admission 90000069.",
        "answer": "Her estimate is 23.4%, above the operating threshold.",
        "tool_calls": list(COMPARATOR_TURN["tool_calls"]),
        "citation_map": {},
    }

    messages, calls, thread_articles = asyncio.run(graph.replay_messages(
        _NoReResolve(), [turn], hadm_id=HADM))

    assert calls == []
    assert thread_articles == []
    assert not any("literature" in str(message.content) for message in messages)


def test_the_stored_answer_keeps_its_badges_through_the_replay():
    """A PMID badge names a record, so it survives with or without a `citation_map`."""
    messages, _calls, _articles = _replayed()

    assistant = [m for m in messages if type(m).__name__ == "AIMessage"][-1]
    assert f"^[PMID: {CITED_PMID}]" in assistant.content
    assert citations.cited_pmids(assistant.content) == [CITED_PMID]
