"""The tool budget has to END the loop, not just refuse the call.

The failure this pins was live on 2026-10-17. A turn answered the question it was
asked, then asked for the same search three more times; each call came back with
"answer from the results you have", which the model ignored, and the run died on
LangGraph's recursion limit with a correct answer already on the reader's screen.

Raising the limit did not fix it — it moved the death from step 10 to step 25, and
the run failed the same way. A refusal the model can ignore is not a bound. What
bounds it is the absence of anything to call: once the budget is spent the agent
turn binds no tools at all, so the only thing the model can do is answer. That is
the same rule the command path has always used.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from langchain_core.messages import AIMessage

from services.agent import graph


class _ToolHungryModel:
    """Asks for a tool on every bound turn; answers when nothing is bound.

    Two objects rather than one flag, because that is the real contract: `bind_tools`
    returns a *different* runnable, and the point of the fix is that the unbound one
    is used once the budget is gone.
    """

    def __init__(self):
        self.bound_turns = 0
        self.unbound_turns = 0

    def bind_tools(self, tools, **kwargs):
        return _Bound(self)

    async def ainvoke(self, messages, **kwargs):
        self.unbound_turns += 1
        return AIMessage(content="Answered from the results already retrieved.")


class _Bound:
    def __init__(self, owner: _ToolHungryModel):
        self._owner = owner

    async def ainvoke(self, messages, **kwargs):
        self._owner.bound_turns += 1
        # `search_literature` rather than a chart tool: since Phase E (2026-10-26) every
        # chart tool — the note reads, `get_lab_trends`, `discharge_medications` — is
        # capped at ONE call per turn (`graph.ONCE_PER_TURN`), so a hunger for those is
        # stopped by the absence of the tool rather than by the budget. This test is about
        # the budget, so it spends the one tool the budget alone bounds.
        #
        # A different argument each time, so the repeat guard is not what stops this:
        # the budget has to be, even for calls it has never seen.
        return AIMessage(
            content="",
            tool_calls=[{
                "name": "search_literature",
                "args": {"query": f"topic {self._owner.bound_turns}"},
                "id": f"call-{self._owner.bound_turns}",
            }],
        )


class _FakeTool:
    """The two attributes `graph._tools` reads off an advertised tool."""

    description = "Retrieve passages from the discharge note."
    input_schema = {"type": "object", "properties": {}}


def _box():
    """A real `MCPToolbox` with a stubbed session.

    A stand-in object would not do: the graph wraps each advertised tool in a
    pydantic `_MCPTool` whose `box` field is validated against the `MCPToolbox`
    dataclass, so anything else fails before a single turn runs. Stubbing the
    instance keeps the double honest about the type without opening a transport.
    """
    from services.agent.mcp_client import MCPToolbox

    box = MCPToolbox(session=None)
    box._tools = {"search_literature": _FakeTool()}

    async def call(name, arguments):
        return {"query": arguments.get("query"), "returned": 0, "articles": []}

    box.call = call
    return box


def _run(model):
    with patch.object(graph, "_build_llm", lambda name: model):
        return asyncio.run(
            graph.ask(_box(), "What does the note say?", hadm_id=90000015, model="test")
        )


def test_the_budget_ends_the_loop_and_the_answer_survives():
    model = _ToolHungryModel()

    state = _run(model)

    # It ended, it answered, and it spent exactly the budget — not one call more.
    assert graph.final_text(state) == "Answered from the results already retrieved."
    assert len(state["tool_calls"]) == graph.MAX_TOOL_CALLS_PER_TURN
    assert model.bound_turns == graph.MAX_TOOL_CALLS_PER_TURN
    assert model.unbound_turns == 1


def test_the_turn_does_not_need_the_recursion_limit_to_stop_it():
    """The breaker is a backstop; a well-behaved turn must never reach it.

    Steps are bounded by the budget, so the graph's own limit stays a catastrophic
    circuit breaker rather than the thing that ends an ordinary turn.
    """
    model = _ToolHungryModel()

    state = _run(model)

    steps = model.bound_turns + model.unbound_turns
    assert steps <= graph.MAX_TOOL_CALLS_PER_TURN + 1
    assert steps < graph.RECURSION_LIMIT
    assert state.get("command") is None
