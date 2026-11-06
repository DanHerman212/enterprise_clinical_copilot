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
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage  # noqa: E402

from services.agent.engine import graph
from services.agent.postprocess import answer as answer_text  # noqa: E402
from services.agent.routing import routing  # noqa: E402
from services.agent.routing.admission_context import PREFLIGHT_TOOLS  # noqa: E402
from services.agent.engine.mcp_client import (  # noqa: E402
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
    # The score stays a command's answer. A question ABOUT the drivers is answered by
    # the loop, but only on the turn that asks for the model's reasoning — see
    # `_freetext_allowed`, asserted below.
    assert "predict_readmission" not in given
    # Phase 01: the medication extraction is the loop's too. A question about the
    # medications is answered, not redirected, and the guardrail verifies doses against
    # the note text either way.
    assert "discharge_medications" in given


def test_every_registered_tool_is_reachable_from_some_path():
    """A tool that no command owns and that free text may not use is a tool
    nothing can reach — dead weight in the server, and a sign that the two lists
    have drifted apart.

    Checked as the union rather than as two literals, so adding a tool to the
    server forces a decision about which path owns it. A command OWNS its primary
    tool and any supplementary call it declares: `/ehr` is what reaches
    `predict_readmission` now that the command that named it alone was retired.
    """
    command_tools = {command.tool for command in routing.COMMANDS}
    command_tools |= {spec.tool for command in routing.COMMANDS
                      for spec in command.supplementary}

    # The pre-flight tools are reachable by a third path — the caller's own call,
    # before the model runs — so the union has three terms, and a new tool still has
    # to belong to exactly one of them.
    assert (command_tools | routing.FREETEXT_TOOLS
            | set(PREFLIGHT_TOOLS)) == set(REGISTERED_TOOLS)


def test_a_command_tool_is_the_reason_text_tool_is_hidden():
    """Which tools a typed question cannot reach, and why.

    Phase 01 admitted `discharge_medications` to the loop, so the complement narrowed;
    after `/risk` was retired the predictor is nobody's primary tool — it is `/ehr`'s
    supplementary call, so it is still absent from the loop's standing set and a model
    asked for a score still has no way to invent one. It is bound per turn when a
    question asks WHY a factor is a driver, which is the one shape that cannot be
    answered without the model's own attribution.

    `search_literature` is not in the standing allow-list either; it is bound per turn
    by `asks_for_literature`.
    """
    tool_of_command = {command.tool for command in routing.COMMANDS}
    supplementary = {spec.tool for command in routing.COMMANDS
                     for spec in command.supplementary}

    # The predictor is reached only as a supplementary call, and never by default.
    assert "predict_readmission" in supplementary
    assert "predict_readmission" not in tool_of_command
    assert "predict_readmission" not in routing.FREETEXT_TOOLS
    # Every primary tool of a command is off the standing allow-list, and the only one
    # that the loop can be granted per turn is the literature search.
    assert tool_of_command - routing.FREETEXT_TOOLS - routing.LITERATURE_TOOLS == set()
    assert routing.LITERATURE_TOOLS <= tool_of_command - routing.FREETEXT_TOOLS


def test_the_loop_is_bound_with_the_allow_list():
    """The filter has to be applied at the binding, not merely available."""
    source = Path(graph.__file__).read_text(encoding="utf-8")

    # Two facts, asserted as two facts rather than as one formatting of the call: the
    # allow-list has to be what is built, and every binding has to be given it. Pinning
    # the whitespace of a single call made a refactor that preserved both facts fail,
    # which is a test reporting on its own brittleness rather than on the boundary.
    #
    # The allow-list is now a function of the turn, because one question shape — "why is
    # this a risk driver?" — cannot be answered without the model's own attribution
    # (`_freetext_allowed`). The boundary is unchanged for every other turn.
    #
    # It is also computed once and passed by name (`allowed=allowed`), because the bound
    # set now SHRINKS as a turn spends it: `bound_for` re-binds from what
    # `_still_bound` leaves, and a second call to `_freetext_allowed` there would
    # re-derive the standing set instead of the remainder. What this test is for is
    # unchanged — the allow-list is what is built, and no binding is made without it.
    assert "allowed = _freetext_allowed(" in source
    assert "freetext_tools = _tools(toolbox, allowed=allowed)" in source
    assert source.count("bind_tools(freetext_tools") == 2
    # Every CALL to `_tools` in the graph names an allow-list (the definition itself is
    # excluded). A call without one would hand the model every tool the server advertises.
    calls = re.findall(r"(?<!def )_tools\(toolbox[^)]*\)", source)
    assert calls
    assert all("allowed=" in call for call in calls), calls

    assert graph._freetext_allowed(False, False) == routing.FREETEXT_TOOLS
    # A driver question adds the attribution and NARROWS the standing set: the lab
    # series documents nothing a question about a count or a disposition asked, and
    # calling it anyway put a lab chip under a utilization answer (agent-00176-ngw,
    # 2026-10-24). A driver that IS an analyte keeps it — see
    # `tests/agent/test_risk_driver_turn.py`.
    assert graph._freetext_allowed(True, False) == frozenset(
        ({*routing.FREETEXT_TOOLS, "predict_readmission"} - {"get_lab_trends"}))
    assert graph._freetext_allowed(True, False, True) == frozenset(
        {*routing.FREETEXT_TOOLS, "predict_readmission"})
    # Published evidence is bound only when the question asked for it (Phase 02.5), and it
    # is bound WITH the record retrieval the question is answered against and nothing else
    # (Phase 4; audit §2.1). The union with the whole standing set is what the audit
    # measured: one literature question kept `get_lab_trends` and `discharge_medications`
    # bound and spent its five calls on a chart read and four searches, then drew a lab
    # sparkline under a literature answer. `routing` owns the set.
    assert graph._freetext_allowed(False, True) == routing.LITERATURE_QUESTION_TOOLS
    assert graph._freetext_allowed(True, True) == frozenset(
        {*routing.LITERATURE_QUESTION_TOOLS, "predict_readmission"})


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
    answer = answer_text.no_evidence_answer(QUESTION, [EMPTY])

    assert answer == (
        'There is no information in the patient\'s chart regarding '
        '"Was there any concern about a blood clot in her leg?".'
    )


def test_the_admission_suffix_is_not_quoted_into_the_sentence():
    """The contract appends the admission to the question the model is asked; a
    record number inside a clinical sentence reads as a reference to nothing."""
    answer = answer_text.no_evidence_answer(f"{QUESTION} For admission 90000023.", [EMPTY])

    assert "90000023" not in answer
    assert "For admission" not in answer


def test_a_search_that_returned_passages_is_left_to_the_model():
    assert answer_text.no_evidence_answer(QUESTION, [WITH_PASSAGES]) is None


def test_a_turn_that_searched_nothing_is_not_an_absence():
    """Saying "the chart does not mention it" about a search nobody ran invents a
    fact about the record."""
    assert answer_text.no_evidence_answer(QUESTION, []) is None
    assert answer_text.no_evidence_answer(QUESTION, [SECTIONS_EMPTY]) is None


def test_a_failed_search_is_not_an_empty_one():
    """An outage says nothing about whether the chart holds the answer, and
    reporting one as an absence turns infrastructure into a clinical statement."""
    assert answer_text.no_evidence_answer(QUESTION, [FAILED]) is None


def test_an_answer_built_on_the_literature_is_an_answer():
    """The fallback is about this patient's notes. A question answered from
    published evidence was answered."""
    assert answer_text.no_evidence_answer(QUESTION, [EMPTY, LITERATURE]) is None


def test_passages_from_the_summary_tools_are_evidence_too():
    sections = _call("rag_search_sections", {"returned": 1, "passages": [
        {"id": "n_brief_hospital_course_1", "text": "..."}]})

    assert answer_text.no_evidence_answer(QUESTION, [EMPTY, sections]) is None


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
    assert answer_text.no_evidence_answer(
        QUESTION, [WITH_PASSAGES], declined=True) == FIXED_SENTENCE


def test_a_verdict_cannot_make_an_absence_out_of_a_failed_search():
    """The verdict is the model's; the sentence is ours. An outage still says
    nothing about whether the chart holds the answer."""
    assert answer_text.no_evidence_answer(QUESTION, [FAILED], declined=True) is None


def test_a_verdict_without_a_search_is_not_an_absence():
    assert answer_text.no_evidence_answer(QUESTION, [], declined=True) is None


def test_a_verdict_on_a_literature_turn_is_not_an_absence():
    assert answer_text.no_evidence_answer(QUESTION, [EMPTY, LITERATURE], declined=True) is None


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
    # And it is not EVIDENCE either: the tool never ran (Sprint 2.2).
    assert recorded == []


def test_a_clean_literature_query_is_dispatched():
    box, events, _, _ = _literature_call("dual antiplatelet therapy duration")

    assert box.calls == [("search_literature", {"query": "dual antiplatelet therapy duration"})]
    assert [event["stage"] for event in events] == ["tool", "result"]


def _scoped_call(tool, args, hadm_id):
    box = _RecordingBox()
    guard = lambda name, a: graph._outbound_patient_identifier(name, a, hadm_id)  # noqa: E731
    messages, recorded = asyncio.run(graph._execute_tool_calls(
        box, [{"name": tool, "args": args, "id": "1"}], [].append, guard=guard,
    ))
    return box, messages, recorded


def test_a_retrieval_naming_another_admission_is_not_dispatched():
    """The turn's admission is fixed by the request, not by the model.

    `rag_search` resolves whatever id it is given, so a model that names a different
    admission reads another patient's chart during this patient's turn. Verified
    against the live tool: hadm_id 90000015 returned five passages of another
    patient's discharge summary in a turn about 90000002. Under injection the probe
    does not have to invent an id — naming a real one is enough.
    """
    box, messages, recorded = _scoped_call(
        "rag_search", {"hadm_id": 90000015, "query": "discharge diagnosis"}, 90000002,
    )

    assert box.calls == [], "another patient's record was read"
    assert "patient_scope_refused" in messages[0].content
    # The refusal reaches the model and stops there. Recording it made the turn look as
    # though it had read another patient's chart, which is the opposite of what happened.
    assert recorded == []


def test_an_invented_admission_on_a_turn_about_no_patient_is_not_dispatched():
    """A turn with no patient has no admission to read, invented or otherwise."""
    box, messages, _ = _scoped_call(
        "rag_search", {"hadm_id": 10000002, "query": "notes"}, None,
    )

    assert box.calls == []
    assert "patient_scope_refused" in messages[0].content


def test_the_turns_own_admission_is_dispatched_either_way_round():
    """A model that writes the id as a string is asking for the same patient.

    Comparing without normalising would refuse a correct call for a type, which is a
    guard that costs the answer it exists to protect.
    """
    box, _, _ = _scoped_call(
        "rag_search", {"hadm_id": "90000002", "query": "notes"}, 90000002,
    )

    assert box.calls == [("rag_search", {"hadm_id": "90000002", "query": "notes"})]


def test_a_tool_scoped_to_the_turns_own_admission_is_dispatched():
    """The guard checks ids the model chose, not tools that take none."""
    box, _, _ = _scoped_call("get_admission_profile", {"hadm_id": 90000002}, 90000002)

    assert box.calls == [("get_admission_profile", {"hadm_id": 90000002})]


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
    one. Adding one would need to get past this test first.

    Stage Gate 2 added `scope` and `synonyms`, both query-shaped: a scope names which
    retrieval to run and a synonym list carries more terms for the same search. Neither
    can name a patient, which is what this test is for, so the assertion is now that the
    parameters remain a subset of the search vocabulary rather than an exact two.
    """
    import inspect

    from services.mcp.tools import search_literature

    parameters = set(inspect.signature(search_literature).parameters)

    assert parameters == {"query", "max_results", "scope", "synonyms"}
    assert not parameters & {"hadm_id", "subject_id", "patient_id", "mrn"}


def test_the_graph_is_built_with_the_identifier_guard():
    """Wired, not merely available: the guard has to reach the dispatch."""
    source = Path(graph.__file__).read_text(encoding="utf-8")

    assert "guard=lambda name, args: _outbound_patient_identifier(name, args, hadm_id)" in source
    assert "toolbox, calls, on_event, guard=guard" in source


# --- Sprint 2.1: the admission is the session's, not the model's -------------
#
# A tool that reads one admission describes its SUBJECT, and the subject of a turn is
# fixed by the request. Advertising `hadm_id` asked the model to generate a number it
# could not know, and the only defence was a refusal of the attempt after it had been
# made. The argument is now absent from the copy the model reads and supplied at
# execution time; the guard stays as defence in depth for a model that writes one anyway.


class _SchemaTool:
    """An advertised tool carrying a real input schema."""

    description = "reads one admission's record"

    def __init__(self, schema, name="rag_search"):
        self.input_schema = schema
        self.name = name


#: The three shapes that matter: a tool with other arguments, a tool whose ONLY argument
#: is the admission, and a tool that takes none.
_SCHEMAS = {
    "rag_search": {
        "type": "object",
        "properties": {"hadm_id": {"type": "integer"},
                       "query": {"type": "string"},
                       "top_k": {"type": "integer"}},
        "required": ["hadm_id", "query"],
    },
    "rag_search_sections": {
        "type": "object",
        "properties": {"hadm_id": {"type": "integer"}},
        "required": ["hadm_id"],
    },
    "search_literature": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}


def _schema_box():
    box = MCPToolbox(session=object())
    box._tools = {name: _SchemaTool(schema, name) for name, schema in _SCHEMAS.items()}
    return box


def test_the_model_is_never_shown_an_admission_argument():
    """Both schema consumers: the LangChain binding and the Gemini declaration."""
    box = _schema_box()

    for tool in graph._tools(box):
        schema = tool.args_schema or {}
        assert "hadm_id" not in (schema.get("properties") or {}), tool.name
        assert "hadm_id" not in (schema.get("required") or []), tool.name

    declarations = {decl.name: decl for decl in box.gemini_tool.function_declarations}
    for name in ("rag_search", "rag_search_sections"):
        assert "hadm_id" not in dict(declarations[name].parameters.properties or {})
    # What the tool DOES ask the model for is untouched: 2026-10-12 added the refusal of
    # an argument the external tool does not declare, and this sprint must not have made
    # the model unable to write the ones it does.
    assert "query" in dict(declarations["search_literature"].parameters.properties or {})


def test_a_tool_with_nothing_left_to_ask_for_becomes_a_zero_argument_schema():
    """`rag_search_sections` takes one admission and nothing else.

    Stripping it must leave a schema a zero-argument call is valid against — not an object
    with an empty `properties` and a `required` list naming the argument that was removed.
    """
    box = MCPToolbox(session=object())
    box._tools = {"rag_search_sections": _SchemaTool(
        _SCHEMAS["rag_search_sections"], "rag_search_sections")}

    (tool,) = graph._tools(box)

    assert tool.args_schema == {"type": "object", "properties": {}}


def _injected_call(tool, args, hadm_id):
    """One call through the real executor, with the session's admission supplied."""
    box = _RecordingBox()
    guard = lambda name, a: graph._outbound_patient_identifier(name, a, hadm_id)  # noqa: E731
    messages, recorded = asyncio.run(graph._execute_tool_calls(
        box, [{"name": tool, "args": args, "id": "1"}],
        [].append, guard=guard, hadm_id=hadm_id,
    ))
    return box, messages, recorded


def test_a_call_without_an_admission_is_completed_by_the_session():
    """The model writes `{"query": "chf"}`; the tool receives an admission too."""
    box, _messages, recorded = _injected_call("rag_search", {"query": "chf"}, 90000010)

    assert box.calls == [("rag_search", {"query": "chf", "hadm_id": 90000010})]
    # The complete arguments, because this is the record of what the tool was asked.
    assert recorded[0]["args"] == {"query": "chf", "hadm_id": 90000010}
    assert not (recorded[0]["response"] or {}).get("error")


def test_a_call_taking_no_arguments_at_all_is_completed_by_the_session():
    box, _messages, recorded = _injected_call("rag_search_sections", {}, 90000010)

    assert box.calls == [("rag_search_sections", {"hadm_id": 90000010})]
    assert recorded[0]["args"]["hadm_id"] == 90000010


def test_a_patient_call_on_a_turn_with_no_admission_is_refused():
    """Audit Finding 2: an OMITTED admission used to be dispatched.

    The mismatch check passes when the call carries no admission at all, so a
    patient-scoped call on a turn about nobody reached the tool. There is no admission to
    read, so there is no call to make.
    """
    box, messages, recorded = _injected_call("rag_search_sections", {}, None)

    assert box.calls == [], "a patient-scoped tool ran on a turn about no patient"
    assert "patient_scope_refused" in messages[0].content
    assert recorded == []


def test_a_literature_search_on_a_turn_with_no_admission_still_runs():
    """The Finding 2 fix must not over-refuse: a literature turn reads no chart.

    This is the case a blanket "no admission, no call" rule would have broken —
    `search_literature` is deliberately not told which patient is under discussion.
    """
    box, _messages, recorded = _injected_call("search_literature", {"query": "sepsis"}, None)

    assert box.calls == [("search_literature", {"query": "sepsis"})]
    assert "hadm_id" not in recorded[0]["args"]
    assert not (recorded[0]["response"] or {}).get("error")


def test_an_invented_admission_is_still_refused_rather_than_replaced():
    """Defence in depth: the guard sees what the model wrote, before any injection.

    Injection must not become a silent correction. A model that names another admission
    has attempted a cross-patient read, the attempt is what the guard is for, and the
    refusal is what the evaluator and the operator count.
    """
    box, messages, recorded = _injected_call(
        "rag_search", {"query": "notes", "hadm_id": 90000015}, 90000002)

    assert box.calls == []
    assert "patient_scope_refused" in messages[0].content
    # And the foreign id is nowhere in the turn's evidence, so it cannot reach the wire,
    # a replay, a scope badge or the evaluator's `banned_source_record` check.
    assert recorded == []
    assert "90000015" not in str(recorded)


# --- Sprint 2.2: a refusal is a message, not evidence ------------------------
#
# `recorded` is what the turn DID. A call the executor refused before dispatch never ran,
# so it has no result, no citation position and nothing for a later turn to replay. It
# stays in the transcript as the model's answer to its own request, and stops there.
#
# The distinction is not cosmetic. `a2ui.turn_scope` badges a turn from `args.hadm_id`
# alone, so a refused cross-patient attempt used to badge the turn `patient_chart`; the
# evaluator's `turn_tool_calls` check scored the same attempt as a disclosure that never
# happened (`dev_t7_03`).


def test_a_refused_call_does_not_badge_the_turn_as_a_chart_read():
    """The turn asked an unanswerable question and read nothing.

    A `Patient Chart Synthesis` badge over an answer that rested on no passage is a claim
    about evidence the turn does not have — the same defect the scope badge exists to
    prevent, arriving through the refusal path.
    """
    from services.agent.presentation.a2ui import turn_scope

    box, messages, recorded = _injected_call(
        "rag_search", {"query": "notes", "hadm_id": 90000015}, 90000002)

    assert box.calls == []
    assert "patient_scope_refused" in messages[0].content
    assert recorded == []
    assert turn_scope(recorded) is None, "no tool ran, so the turn rested on nothing"


def test_the_refused_call_carries_no_foreign_admission_into_the_turn_record():
    """What `banned_source_record` reads is `turn_tool_calls`, which is `recorded`.

    A cross-patient id that appears only inside a refusal message is not a disclosure: no
    tool was asked for that admission, and nothing was returned for it.
    """
    box, messages, recorded = _injected_call(
        "rag_search", {"query": "notes", "hadm_id": 90000015}, 90000002)

    assert box.calls == []
    assert "90000015" in messages[0].content, "the model is still told what it asked for"
    assert "90000015" not in str(recorded)


def test_an_executed_call_that_the_server_rejected_is_still_evidence():
    """The line the separation draws: DISPATCHED, not successful.

    The guard refuses before the tool is called; the tool's own `unknown_patient` arrives
    after a call was made. The second is a tool call the turn made, and `_tool_error_codes`
    exists to report it — so it stays in `recorded`, and only the pre-dispatch refusals
    leave.
    """
    box, _messages, recorded = _injected_call("rag_search", {"query": "notes"}, 90000010)

    assert box.calls == [("rag_search", {"query": "notes", "hadm_id": 90000010})]
    assert len(recorded) == 1
