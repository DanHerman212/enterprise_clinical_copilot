"""The free-text boundary: which tools a question may use, and what an empty
search is allowed to become.

UX Phase 1B. Two rules, both enforced in code rather than asked for in a prompt:

  - a question that is not a command is answered with retrieval and nothing else,
    so the two tools that own a deterministic answer are absent from the loop
    rather than forbidden inside it;
  - a search that comes back with nothing becomes a fixed sentence decided from
    the tool records, so an absence of evidence can never be served as a
    synthesised answer.

Offline: a stub toolbox, a stub graph, and the real prompt.
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage  # noqa: E402

from services.agent import graph, routing  # noqa: E402
from services.agent.mcp_client import (  # noqa: E402
    MCPToolbox,
    _external_argument_refusal,
)
from services.mcp.tools import __all__ as REGISTERED_TOOLS  # noqa: E402


class _FakeTool:
    description = "does something clinical"
    input_schema = {"type": "object", "properties": {}}


def _box() -> MCPToolbox:
    """A real toolbox carrying a fake tool for everything the server advertises.

    Real rather than stubbed because the tool wrapper validates its toolbox: the
    thing under test is which entries a filter lets through, and a stand-in that
    could not be wrapped at all would test the stand-in.
    """
    box = MCPToolbox(session=object())
    box._tools = {name: _FakeTool() for name in REGISTERED_TOOLS}
    return box


# --- which tools the loop is given -------------------------------------------

def test_the_reasoning_loop_is_given_only_the_retrieval_tools():
    """The two command-owned tools are ABSENT, not merely refused.

    A tool the model cannot call and a tool the model is told not to call are
    different systems: the second relies on the model agreeing, and the failure —
    a risk score nobody asked a model to estimate — is indistinguishable from a
    real one in the answer.
    """
    given = {tool.name for tool in
             graph._tools(_box(), allowed=routing.FREETEXT_TOOLS)}

    assert given == set(routing.FREETEXT_TOOLS)
    assert "predict_readmission" not in given
    assert "discharge_medications" not in given


def test_every_registered_tool_is_reachable_from_some_path():
    """A tool that no command owns and that free text may not use is a tool
    nothing can reach — dead weight in the server, and a sign that the two lists
    have drifted apart.

    Checked as the union rather than as two literals, so adding a tool to the
    server forces a decision about which path owns it.
    """
    command_tools = {command.tool for command in routing.COMMANDS}

    assert command_tools | routing.FREETEXT_TOOLS == set(REGISTERED_TOOLS)


def test_a_command_tool_is_the_reason_text_tool_is_hidden():
    """The two lists are complements by intent, not by accident: a tool is hidden
    from free text precisely because a command answers it deterministically."""
    hidden = {command.tool for command in routing.COMMANDS} - routing.FREETEXT_TOOLS

    assert hidden == {"predict_readmission", "discharge_medications"}


def test_the_loop_is_bound_with_the_allow_list():
    """The filter has to be applied at the binding, not merely available."""
    source = Path(graph.__file__).read_text(encoding="utf-8")

    assert "bind_tools(\n        _tools(toolbox, allowed=routing.FREETEXT_TOOLS)\n    )" in source


# --- what an empty search becomes --------------------------------------------

def _call(name, response):
    return {"name": name, "args": {}, "response": response}


EMPTY = _call("rag_search", {"returned": 0, "passages": []})
WITH_PASSAGES = _call("rag_search", {"returned": 2, "passages": [
    {"id": "n_discharge_diagnosis_1", "text": "Pneumonia."}]})
SECTIONS_EMPTY = _call("rag_search_sections", {"returned": 0, "passages": []})
FAILED = _call("rag_search", {"error": "search_failed"})
LITERATURE = _call("search_literature", {"returned": 2, "articles": []})

QUESTION = "Was there any concern about a blood clot in her leg?"


def test_an_empty_search_becomes_the_fixed_sentence():
    answer = graph.no_evidence_answer(QUESTION, [EMPTY])

    assert answer == (
        'There is no information in the patient\'s chart regarding '
        '"Was there any concern about a blood clot in her leg?".'
    )


def test_the_admission_suffix_is_not_quoted_into_the_sentence():
    """The contract appends the admission to the question the model is asked; a
    record number inside a clinical sentence reads as a reference to nothing."""
    answer = graph.no_evidence_answer(f"{QUESTION} For admission 90000023.", [EMPTY])

    assert "90000023" not in answer
    assert "For admission" not in answer


def test_a_search_that_returned_passages_is_left_to_the_model():
    assert graph.no_evidence_answer(QUESTION, [WITH_PASSAGES]) is None


def test_a_turn_that_searched_nothing_is_not_an_absence():
    """Saying "the chart does not mention it" about a search nobody ran invents a
    fact about the record."""
    assert graph.no_evidence_answer(QUESTION, []) is None
    assert graph.no_evidence_answer(QUESTION, [SECTIONS_EMPTY]) is None


def test_a_failed_search_is_not_an_empty_one():
    """An outage says nothing about whether the chart holds the answer, and
    reporting one as an absence turns infrastructure into a clinical statement."""
    assert graph.no_evidence_answer(QUESTION, [FAILED]) is None


def test_an_answer_built_on_the_literature_is_an_answer():
    """The fallback is about this patient's notes. A question answered from
    published evidence was answered."""
    assert graph.no_evidence_answer(QUESTION, [EMPTY, LITERATURE]) is None


def test_passages_from_the_summary_tools_are_evidence_too():
    sections = _call("rag_search_sections", {"returned": 1, "passages": [
        {"id": "n_brief_hospital_course_1", "text": "..."}]})

    assert graph.no_evidence_answer(QUESTION, [EMPTY, sections]) is None


# --- the model's own verdict on what it was given (UX Phase 1B) ----------------
#
# The count is not the only way to know the notes did not answer. Vector search
# returns its nearest neighbours whatever it is asked, so a question the notes
# cannot answer arrives looking like one they can — live on 2026-10-12, "was the
# patient intubated during this admission?" retrieved five passages, none of them
# about intubation, and the model answered from them anyway. The model is the one
# component that can read what came back, so it is asked for a verdict; the
# sentence is still ours, so the clinician hears fixed words either way.

DECLINE_OBJECT = '{"evidence_found": false}'

FIXED_SENTENCE = (
    'There is no information in the patient\'s chart regarding '
    '"Was there any concern about a blood clot in her leg?".'
)


def test_prose_is_an_answer():
    reply = graph.read_evidence_reply("The notes record a low-grade fever.")

    assert reply.declined is False
    assert reply.text == "The notes record a low-grade fever."


def test_the_decline_object_is_read_as_a_decline():
    assert graph.read_evidence_reply(DECLINE_OBJECT).declined is True
    assert graph.read_evidence_reply(DECLINE_OBJECT).text == ""


def test_a_reply_with_nothing_in_it_is_a_decline():
    assert graph.read_evidence_reply("  \n ").declined is True


def test_a_wrapped_answer_is_served_as_its_prose():
    """The object is a contract, not a payload: an answer inside it is still an
    answer, and a clinician must never be shown the wrapper."""
    reply = graph.read_evidence_reply(
        '{"evidence_found": true, "narrative_text": "Temperature was 97.7."}')

    assert reply.declined is False
    assert reply.text == "Temperature was 97.7."


def test_json_that_is_not_the_contract_is_left_as_it_arrived():
    """Second-guessing the model's text is how an answer gets edited by a parser."""
    text = '{"pmid": "31234567"}'

    assert graph.read_evidence_reply(text).declined is False
    assert graph.read_evidence_reply(text).text == text


def test_an_unclosed_brace_is_prose_not_a_parse_target():
    text = "{the notes record a DVT"

    assert graph.read_evidence_reply(text).text == text


def test_the_verdict_is_what_the_count_cannot_see():
    """Five passages, none of them an answer: the count is 5 and the answer is
    absent, which is the pair a count cannot distinguish."""
    assert graph.no_evidence_answer(
        QUESTION, [WITH_PASSAGES], declined=True) == FIXED_SENTENCE


def test_a_verdict_cannot_make_an_absence_out_of_a_failed_search():
    """The verdict is the model's; the sentence is ours. An outage still says
    nothing about whether the chart holds the answer."""
    assert graph.no_evidence_answer(QUESTION, [FAILED], declined=True) is None


def test_a_verdict_without_a_search_is_not_an_absence():
    assert graph.no_evidence_answer(QUESTION, [], declined=True) is None


def test_a_verdict_on_a_literature_turn_is_not_an_absence():
    assert graph.no_evidence_answer(QUESTION, [EMPTY, LITERATURE], declined=True) is None


# --- and that it is the answer that reaches the caller ------------------------

class _StubGraph:
    def __init__(self, state):
        self._state = state

    async def ainvoke(self, *_args, **_kwargs):
        return self._state


def test_the_model_s_own_text_is_discarded_when_the_search_was_empty(monkeypatch):
    """Wired, not merely available: the sentence is set on the state the routes
    compose from, so the model's plausible answer to an unaddressed question never
    leaves the service."""
    state = {
        "messages": [AIMessage(content="The notes document a deep vein thrombosis.")],
        "tool_calls": [EMPTY],
        "replayed_tool_calls": [],
    }
    monkeypatch.setattr(graph, "build_graph", lambda *_a, **_k: _StubGraph(state))

    out = asyncio.run(graph.ask(object(), f"{QUESTION} For admission 90000023."))

    assert out["deterministic_answer"] == (
        'There is no information in the patient\'s chart regarding '
        '"Was there any concern about a blood clot in her leg?".'
    )


def test_a_turn_with_evidence_is_not_replaced(monkeypatch):
    state = {
        "messages": [AIMessage(content="The notes record a DVT.")],
        "tool_calls": [WITH_PASSAGES],
        "replayed_tool_calls": [],
    }
    monkeypatch.setattr(graph, "build_graph", lambda *_a, **_k: _StubGraph(state))

    out = asyncio.run(graph.ask(object(), QUESTION))

    assert "deterministic_answer" not in out


def test_a_declined_turn_is_served_the_fixed_sentence(monkeypatch):
    """Wired, not merely available. The model answered out of passages that did not
    answer the question; the reply says so, and the reply decides."""
    state = {
        "messages": [AIMessage(content=DECLINE_OBJECT)],
        "tool_calls": [WITH_PASSAGES],
        "replayed_tool_calls": [],
    }
    monkeypatch.setattr(graph, "build_graph", lambda *_a, **_k: _StubGraph(state))

    out = asyncio.run(graph.ask(object(), QUESTION))

    assert out["deterministic_answer"] == FIXED_SENTENCE


def test_a_declined_turn_with_nothing_to_report_carries_no_answer(monkeypatch):
    """Declined where no absence may be claimed: no retrieval ran. The flag is not an
    answer and the fixed sentence would be a claim the records do not support, so the
    turn carries no text and the routes report an unavailable answer rather than a
    fabricated one."""
    state = {
        "messages": [AIMessage(content=DECLINE_OBJECT)],
        "tool_calls": [],
        "replayed_tool_calls": [],
    }
    monkeypatch.setattr(graph, "build_graph", lambda *_a, **_k: _StubGraph(state))

    out = asyncio.run(graph.ask(object(), QUESTION))

    assert "deterministic_answer" not in out
    assert graph.final_text(out) == ""


def test_a_wrapped_answer_is_unwrapped_before_it_leaves(monkeypatch):
    """The guardrails, the citation layer and the canvas all read prose, so the
    envelope comes off here rather than at each of them."""
    state = {
        "messages": [AIMessage(
            content='{"evidence_found": true, '
                    '"narrative_text": "The notes record a DVT."}')],
        "tool_calls": [WITH_PASSAGES],
        "replayed_tool_calls": [],
    }
    monkeypatch.setattr(graph, "build_graph", lambda *_a, **_k: _StubGraph(state))

    out = asyncio.run(graph.ask(object(), QUESTION))

    assert graph.final_text(out) == "The notes record a DVT."


def test_prose_reaches_the_caller_untouched(monkeypatch):
    """Nothing about an ordinary answer changed, down to its whitespace."""
    state = {
        "messages": [AIMessage(content="  The notes record a DVT.  ")],
        "tool_calls": [WITH_PASSAGES],
        "replayed_tool_calls": [],
    }
    monkeypatch.setattr(graph, "build_graph", lambda *_a, **_k: _StubGraph(state))

    out = asyncio.run(graph.ask(object(), QUESTION))

    assert graph.final_text(out) == "The notes record a DVT."


# --- what may leave the service ----------------------------------------------

class _RecordingBox:
    def __init__(self):
        self.calls = []

    async def call(self, name, args):
        self.calls.append((name, dict(args)))
        return {"returned": 0, "articles": []}


def _literature_call(query, hadm_id=90000023):
    events = []
    box = _RecordingBox()
    guard = lambda name, args: graph._outbound_patient_identifier(  # noqa: E731
        name, args, hadm_id)
    messages, recorded = asyncio.run(graph._execute_tool_calls(
        box, [{"name": "search_literature", "args": {"query": query}, "id": "1"}],
        events.append, guard=guard,
    ))
    return box, events, messages, recorded


def test_a_literature_search_carrying_the_admission_id_is_not_dispatched():
    """The one disclosure that cannot be taken back, so it is checked rather than
    asked for: the prompt says not to put patient text in the query, and the agent
    is the layer that knows what this patient's identifier looks like."""
    box, events, messages, recorded = _literature_call("stent outcomes 90000023")

    assert box.calls == [], "the call reached the tool"
    assert "patient_identifier_refused" in messages[0].content
    # Work that did not happen is not narrated, same rule as a refused budget call.
    assert events == []
    assert recorded[0]["response"]["error"] == "patient_identifier_refused"


def test_a_clean_literature_query_is_dispatched():
    box, events, _, _ = _literature_call("dual antiplatelet therapy duration")

    assert box.calls == [("search_literature", {"query": "dual antiplatelet therapy duration"})]
    assert [event["stage"] for event in events] == ["tool", "result"]


def test_the_identifier_check_is_scoped_to_what_leaves_the_service():
    """A note search is internal: an admission id in ITS query is how retrieval
    works, and refusing that would break the thing being protected."""
    guard = lambda name, args: graph._outbound_patient_identifier(  # noqa: E731
        name, args, 90000023)

    assert guard("rag_search", {"hadm_id": 90000023, "query": "90000023"}) is None


def test_the_tool_boundary_refuses_an_argument_it_does_not_declare():
    """The SDK drops an undeclared argument in transit rather than rejecting it —
    verified live, which is how a patient identifier would have travelled with a
    search and left no trace of the fact."""
    class _Tool:
        input_schema = {"properties": {"query": {}, "max_results": {}}}

    refusal = _external_argument_refusal(
        "search_literature", {"query": "x", "hadm_id": 90000023}, _Tool())

    assert refusal is not None
    assert "hadm_id" in refusal["message"]
    # Only the names, never the values: the value here is the thing that must not
    # be written down.
    assert "90000023" not in refusal["message"]


def test_the_tool_boundary_leaves_declared_arguments_and_local_tools_alone():
    class _Tool:
        input_schema = {"properties": {"query": {}, "max_results": {}}}

    assert _external_argument_refusal(
        "search_literature", {"query": "x", "max_results": 3}, _Tool()) is None
    # A harmless extra key on a local tool is not worth failing an answer over.
    assert _external_argument_refusal("rag_search", {"query": "x", "top_k": 5}, _Tool()) is None


def test_the_literature_tool_cannot_be_given_a_patient_identifier_at_all():
    """The structural half: the tool's own signature has no parameter that could
    carry an identifier, so the check above is a second line rather than the only
    one. Adding one would need to get past this test first."""
    import inspect

    from services.mcp.tools import search_literature

    assert set(inspect.signature(search_literature).parameters) == {
        "query", "max_results"}


def test_the_graph_is_built_with_the_identifier_guard():
    """Wired, not merely available: the guard has to reach the dispatch."""
    source = Path(graph.__file__).read_text(encoding="utf-8")

    assert "guard=lambda name, args: _outbound_patient_identifier(name, args, hadm_id)" in source
    assert "toolbox, calls, on_event, guard=guard" in source

