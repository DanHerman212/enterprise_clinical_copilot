"""Explicit LangGraph StateGraph: START -> agent -> tools -> agent -> END.

Written as an explicit graph rather than `create_react_agent` so the control
flow is visible: the agent turns once to emit a tool call, the tool node runs
it, and the agent turns *again* to narrate the result. That second turn is the
one the guardrails in prompts.py apply to.

Messages are `google.genai` Content objects. LangGraph does not care what the
state holds, and keeping Gemini's own types avoids a translation layer whose
only job would be to convert them back.
"""

import operator
import sys
from pathlib import Path
from typing import Annotated, Any, TypedDict

from google import genai
from google.genai import types
from langgraph.graph import END, START, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.mcp_client import MCPToolbox  # noqa: E402
from agent.prompts import SYSTEM_PROMPT  # noqa: E402
from mcp_server.config import (  # noqa: E402
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MODEL,
    LOCATION,
    PROJECT,
)


class AgentState(TypedDict):
    """`operator.add` makes each node append to the transcript."""

    messages: Annotated[list[types.Content], operator.add]
    tool_calls: Annotated[list[dict[str, Any]], operator.add]


def _function_calls(content: types.Content | None) -> list[types.FunctionCall]:
    if content is None or not content.parts:
        return []
    return [part.function_call for part in content.parts if part.function_call]


def build_graph(toolbox: MCPToolbox, model: str = GEMINI_MODEL):
    client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[toolbox.gemini_tool],
        temperature=0,
        # Budgets thinking AND the answer. Too small and the model spends it all
        # on thoughts, returns finish_reason=MAX_TOKENS with empty text, and
        # raises nothing — which in a graph looks like a silently skipped tool
        # call. See §9.
        max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
    )

    async def agent_node(state: AgentState) -> dict:
        response = await client.aio.models.generate_content(
            model=model, contents=state["messages"], config=config
        )

        if not response.candidates:
            raise RuntimeError(
                f"Gemini returned no candidates. "
                f"prompt_feedback={response.prompt_feedback}"
            )

        candidate = response.candidates[0]
        finish = candidate.finish_reason

        # Fail loudly rather than let an empty turn look like a considered one.
        if finish == types.FinishReason.MAX_TOKENS and not (response.text or "").strip():
            raise RuntimeError(
                "Gemini hit MAX_TOKENS with no output text: the thinking budget "
                f"consumed the whole allowance ({GEMINI_MAX_OUTPUT_TOKENS} tokens). "
                "Raise GEMINI_MAX_OUTPUT_TOKENS."
            )

        return {"messages": [candidate.content], "tool_calls": []}

    async def tool_node(state: AgentState) -> dict:
        calls = _function_calls(state["messages"][-1])
        parts: list[types.Part] = []
        recorded: list[dict[str, Any]] = []

        for call in calls:
            arguments = dict(call.args or {})
            payload = await toolbox.call(call.name, arguments)
            recorded.append({"name": call.name, "args": arguments, "response": payload})
            parts.append(
                types.Part.from_function_response(name=call.name, response=payload)
            )

        return {
            "messages": [types.Content(role="user", parts=parts)],
            "tool_calls": recorded,
        }

    def route(state: AgentState) -> str:
        return "tools" if _function_calls(state["messages"][-1]) else END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


async def ask(toolbox: MCPToolbox, question: str, model: str = GEMINI_MODEL) -> dict:
    """Run one question to completion. Returns the final state."""
    graph = build_graph(toolbox, model=model)
    return await graph.ainvoke(
        {
            "messages": [types.Content(role="user", parts=[types.Part(text=question)])],
            "tool_calls": [],
        }
    )


def final_text(state: dict) -> str:
    for content in reversed(state["messages"]):
        if content.role == "model" and content.parts:
            text = "".join(part.text or "" for part in content.parts).strip()
            if text:
                return text
    return ""
