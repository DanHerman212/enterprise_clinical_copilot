"""The note-reading tools are spent, not merely budgeted.

Measured on 2026-10-26 from the console: `/ehr` followed by a free-text question — "what
was the inpatient diagnosis?" — ran EIGHTEEN workflow steps. `rag_search_sections` was
called again for four canonical summary sections that the replayed turn already held, and
then `rag_search` ran four separate times looking for an "Inpatient diagnosis" heading
this cohort does not document. Four of those steps were the model being told, one call at
a time, to answer from what it had.

`MAX_TOOL_CALLS_PER_TURN` did not stop it because five calls is inside the budget. A
budget bounds spend; it does not bound a shape. What bounds this shape is the absence of
the tool: `rag_search_sections` is dropped once the transcript holds it, and `rag_search`
is dropped once the turn has used its one query. A tool that is not bound cannot be
called, and a refusal — the thing the budget relies on the model to read — is not needed.

The step count is asserted, not just the call count. A turn that no longer calls twice
but loops anyway would be the same defect with better manners.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from langchain_core.messages import AIMessage

from services.agent.engine import graph
from services.agent.engine import model as model_gateway
from services.agent.engine import model
from services.agent.telemetry import stages

HADM_ID = 90000015


class _FakeTool:
    """The two attributes `graph._tools` reads off an advertised tool."""

    description = "Read the patient's discharge note."
    input_schema = {"type": "object", "properties": {}}


class _ToolLover:
    """A model that keeps asking for the tools a ReAct turn should not re-use.

    It asks only for what it was actually bound with, which is the point: a real model
    cannot call a tool that is not in its request, and a douter that ignores its own
    binding would pass a cap test while proving nothing.
    """

    def __init__(self, wants: tuple[str, ...] = ("rag_search_sections", "rag_search")):
        self.wants = wants
        self.offered: list[set[str]] = []
        self.turns = 0
        self.unbound_turns = 0

    def bind_tools(self, tools, **kwargs):
        names = {tool.name for tool in tools}
        return _Bound(self, names)

    async def ainvoke(self, messages, **kwargs):
        """The unbound model: nothing left to call, so it answers."""
        self.unbound_turns += 1
        self.offered.append(set())
        return AIMessage(content="The record does not document a separate inpatient "
                                 "diagnosis heading; the diagnoses are under the "
                                 "discharge diagnosis.")


class _Bound:
    def __init__(self, owner: _ToolLover, names: set[str]):
        self._owner = owner
        self._names = names

    async def ainvoke(self, messages, **kwargs):
        # Recorded here rather than in `bind_tools`, because `build_graph` binds twice up
        # front — once for AUTO and once for the forced retrieval — before any turn runs.
        # What matters is what THIS turn had left when the model was asked, not what the
        # path could call in principle.
        self._owner.offered.append(set(self._names))
        self._owner.turns += 1
        for name in self._owner.wants:
            if name in self._names:
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "name": name,
                        # A different query every time, so the identical-arguments
                        # repeat guard is never what stops this.
                        "args": {"hadm_id": HADM_ID,
                                 "query": f"inpatient diagnosis {self._owner.turns}"},
                        "id": f"call-{self._owner.turns}",
                    }],
                )
        return AIMessage(content="The discharge diagnosis records the diagnoses.")


def _box(*names: str):
    """A real `MCPToolbox` with the named tools advertised and its transport stubbed."""
    from services.agent.engine.mcp_client import MCPToolbox

    box = MCPToolbox(session=None)
    box._tools = {name: _FakeTool() for name in names}
    called: list[str] = []

    async def call(name, arguments):
        called.append(name)
        return {"hadm_id": arguments.get("hadm_id"), "passages": []}

    box.call = call
    return box, called


#: What the replayed `/ehr` turn left in the transcript: the four summary sections, read
#: by `rag_search_sections`. The passages are not what is under test — the CALL is.
REPLAYED_SECTIONS = [{
    "name": "rag_search_sections",
    "args": {"hadm_id": HADM_ID},
    "response": {"hadm_id": HADM_ID, "passages": [
        {"cite": 1, "section": "brief_hospital_course", "text": "Admitted with sepsis."},
        {"cite": 2, "section": "discharge_diagnosis", "text": "Sepsis, resolved."},
    ]},
}]


def _run(*tools, replayed=None, wants=("rag_search_sections", "rag_search")):
    """One Turn-2 follow-up, with the replayed turns seeded rather than re-executed."""
    model = _ToolLover(wants=wants)
    box, called = _box(*tools)
    events: list[dict] = []

    async def replay(toolbox, turns, **_kwargs):
        return [], list(replayed or [])

    with patch.object(model_gateway, "build_llm", lambda name: model), \
            patch.object(graph, "replay_messages", replay):
        state = asyncio.run(graph.ask(
            box, "what was the inpatient diagnosis?", hadm_id=HADM_ID, model="test",
            on_event=events.append, turns=[{"role": "user", "content": "Summarize"}],
        ))
    steps = sum(
        1 for event in events
        if event["stage"] in (stages.STAGE_PLANNING, stages.STAGE_REVIEWING,
                              stages.STAGE_TOOL)
    )
    return state, model, called, steps


def test_a_section_read_in_the_replay_is_not_offered_again():
    state, model, called, _ = _run("rag_search", "rag_search_sections",
                                  replayed=REPLAYED_SECTIONS)

    # Never bound on any turn, so there was nothing for the model to call.
    assert all("rag_search_sections" not in names for names in model.offered)
    assert "rag_search_sections" not in called
    assert [call["name"] for call in state["tool_calls"]] == ["rag_search"]


def test_the_vector_search_is_spent_once():
    state, model, called, _ = _run("rag_search", "rag_search_sections",
                                  replayed=REPLAYED_SECTIONS)

    assert called.count("rag_search") == 1
    # The tool is gone from the last binding, which is what stops the loop without a
    # refusal the model could ignore.
    assert "rag_search" not in model.offered[-1]


def test_the_follow_up_settles_in_a_handful_of_steps():
    """The regression was eighteen steps. Four is the shape this turn should have."""
    _, _, _, steps = _run("rag_search", "rag_search_sections", replayed=REPLAYED_SECTIONS)

    assert steps <= 6, f"the follow-up ran {steps} workflow steps"


def test_a_turn_with_no_replay_may_still_read_the_sections_once():
    """The caps must not withdraw the tool from the turn that legitimately needs it."""
    state, _, called, _ = _run("rag_search", "rag_search_sections",
                               wants=("rag_search_sections",))

    assert called == ["rag_search_sections"]
    assert [call["name"] for call in state["tool_calls"]] == ["rag_search_sections"]
