"""A citation removed at settlement is a fact about the turn, and it reached no telemetry.

`answer.sanitize_answer_markers` is the last pass over the served text: it applies the
citation bound and the PMID allowance to the answer the caller will read. It calls the
same `enforce_citations` the guardrails do, and it threw the verdict away —
`cleaned, _ = ...` — so a marker deleted here left no `citation_out_of_range` flag
anywhere. Turn 3 of the 2026-10-30 session reported `guardrail_flags: []` while a
`^[PMID: …]` badge had been removed from a sentence that was correct, which is how a
mangled citation reaches a reader looking like a clean turn.

The guardrails run AFTER settlement (`turn` guards the settled text), so the one place
the fleet's two passes meet is the state. Flags are recorded there and merged into the
payload by `turn`.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from services.agent.postprocess.answer import sanitize_answer_markers


def _state(text: str, *, tool_calls=None, pmids_from=None):
    return {
        "messages": [AIMessage(content=text)],
        "tool_calls": tool_calls or [],
        "replayed_tool_calls": pmids_from or [],
        "wants_literature": bool(pmids_from),
    }


def test_a_marker_removed_at_settlement_is_reported():
    """No literature in the turn, so a PMID badge names a record nobody retrieved."""
    state = _state("Warming is the priority.^[PMID: 35010760]")

    sanitize_answer_markers(state)

    assert state["messages"][-1].content == "Warming is the priority."
    assert state["guardrail_flags"] == ["citation_out_of_range:^35010760"]


def test_a_marker_the_allowance_keeps_is_not_reported():
    search = {
        "name": "search_literature",
        "args": {"query": "hypothermia"},
        "response": {"query": "hypothermia", "returned": 1, "articles": [
            {"pmid": "35010760", "title": "T", "journal": "J", "pub_date": "2022",
             "abstract": "A", "url": ""},
        ]},
    }
    state = _state("Warming is the priority.^[PMID: 35010760]", tool_calls=[search])

    sanitize_answer_markers(state)

    assert state["messages"][-1].content == "Warming is the priority.^[PMID: 35010760]"
    assert "guardrail_flags" not in state


def test_a_number_beyond_the_bound_is_reported_too():
    state = _state("The course was uneventful.^[9]")

    sanitize_answer_markers(state)

    assert state["messages"][-1].content == "The course was uneventful."
    assert state["guardrail_flags"] == ["citation_out_of_range:^9"]


def test_the_deterministic_answer_is_covered_by_the_same_record():
    """`/meds` writes its answer in code, and that text carries the same markers."""
    state = _state("nothing")
    state["deterministic_answer"] = "Two discharge medications.^[9]"

    sanitize_answer_markers(state)

    assert state["deterministic_answer"] == "Two discharge medications."
    assert state["guardrail_flags"] == ["citation_out_of_range:^9"]


def test_flags_already_on_the_state_are_kept():
    """The state is shared with the graph, which may have recorded a verdict first."""
    state = _state("The course was uneventful.^[9]")
    state["guardrail_flags"] = ["citation_out_of_range:^3"]

    sanitize_answer_markers(state)

    assert state["guardrail_flags"] == [
        "citation_out_of_range:^3", "citation_out_of_range:^9"]
