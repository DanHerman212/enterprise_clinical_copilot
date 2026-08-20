"""Explicit LangGraph StateGraph: START -> agent -> tools -> agent -> END.

Written as an explicit graph rather than `create_react_agent` so the control
flow is visible: the agent turns once to emit a tool call, the tool node runs
it, and the agent turns *again* to narrate the result. That second turn is the
one the guardrails in prompts.py apply to.

Messages are `google.genai` Content objects. LangGraph does not care what the
state holds, and keeping Gemini's own types avoids a translation layer whose
only job would be to convert them back.
"""

import json
import operator
import os
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

# --- Langfuse observability (optional, no-op without keys) ---
# Enabled only when all three env vars are present, so local/dev runs without
# them never import langfuse and never log. When enabled, every /ask becomes a
# Langfuse trace with a generation span for the Gemini call and a span per MCP
# tool call, so the golden-eval fix-and-retest loop can open a failing case and
# see exactly which passages went in and what the model said.
LANGFUSE_ENABLED = bool(
    os.environ.get("LANGFUSE_PUBLIC_KEY")
    and os.environ.get("LANGFUSE_SECRET_KEY")
    and os.environ.get("LANGFUSE_HOST")
)

if LANGFUSE_ENABLED:
    from langfuse.decorators import (  # noqa: E402
        langfuse_context,
        observe as _langfuse_observe,
    )

    langfuse_context.configure(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.environ["LANGFUSE_HOST"],
    )

    def observe(*args: Any, **kwargs: Any):
        return _langfuse_observe(*args, **kwargs)

else:

    def observe(*args: Any, **kwargs: Any):
        def decorator(fn):
            return fn

        return decorator


class AgentState(TypedDict):
    """`operator.add` makes each node append to the transcript."""

    messages: Annotated[list[types.Content], operator.add]
    tool_calls: Annotated[list[dict[str, Any]], operator.add]


def _function_calls(content: types.Content | None) -> list[types.FunctionCall]:
    if content is None or not content.parts:
        return []
    return [part.function_call for part in content.parts if part.function_call]


def _serialize_contents(contents: list[types.Content]) -> list[dict[str, Any]]:
    """Collapse genai Content objects into JSON-safe shapes for Langfuse."""
    out: list[dict[str, Any]] = []
    for content in contents:
        parts: list[dict[str, Any]] = []
        for part in content.parts or []:
            if part.text:
                parts.append({"text": part.text})
            elif part.function_call:
                parts.append(
                    {
                        "function_call": {
                            "name": part.function_call.name,
                            "args": dict(part.function_call.args or {}),
                        }
                    }
                )
            elif part.function_response:
                parts.append(
                    {
                        "function_response": {
                            "name": part.function_response.name,
                            "response": part.function_response.response,
                        }
                    }
                )
        out.append({"role": content.role, "parts": parts})
    return out


def _jsonable(value: Any) -> Any:
    """Best-effort JSON-safe value; fall back to repr for exotic leaves."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def _system_instruction(instruction: Any) -> Any:
    """Serialize GenerateContentConfig.system_instruction into a JSON-safe
    shape so the Langfuse input shows the FULL prompt — system prompt +
    conversation — not just the conversation messages. The system prompt is
    the thing that tells the model *how* to behave; without it in the trace,
    you cannot see what the model was actually instructed to do."""
    if instruction is None:
        return None
    if isinstance(instruction, str):
        return instruction
    # genai Content (may have multiple parts) — robust fallback.
    if isinstance(instruction, types.Content):
        return [
            part.text if part.text else _jsonable(part)
            for part in instruction.parts or []
        ]
    return _jsonable(instruction)


@observe(as_type="generation", name="gemini.generate")
async def _generate(
    client: genai.Client,
    model: str,
    contents: list[types.Content],
    config: types.GenerateContentConfig,
):
    """One Gemini call, captured as a Langfuse generation span."""
    response = await client.aio.models.generate_content(
        model=model, contents=contents, config=config
    )
    if LANGFUSE_ENABLED:
        langfuse_context.update_current_observation(
            model=model,
            input={
                "system_instruction": _system_instruction(config.system_instruction),
                "messages": _serialize_contents(contents),
            },
            output=response.text,
            metadata={
                "finish_reason": (
                    str(response.candidates[0].finish_reason)
                    if response.candidates
                    else None
                ),
            },
        )
    return response


@observe(as_type="span", name="mcp.tool")
async def _call_tool(box: MCPToolbox, name: str, arguments: dict) -> dict:
    """One MCP tool call, captured as a Langfuse span."""
    payload = await box.call(name, arguments)
    if LANGFUSE_ENABLED:
        langfuse_context.update_current_observation(
            name=f"tool.{name}",
            input=_jsonable(arguments),
            output=_jsonable(payload),
        )
    return payload


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
        response = await _generate(client, model, state["messages"], config)

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
            payload = await _call_tool(toolbox, call.name, arguments)
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


@observe(name="agent.ask")
async def ask(toolbox: MCPToolbox, question: str, model: str = GEMINI_MODEL) -> dict:
    """Run one question to completion. Returns the final state."""
    graph = build_graph(toolbox, model=model)
    state = await graph.ainvoke(
        {
            "messages": [types.Content(role="user", parts=[types.Part(text=question)])],
            "tool_calls": [],
        }
    )
    if LANGFUSE_ENABLED:
        langfuse_context.update_current_trace(
            input=question,
            output={
                "answer": final_text(state),
                "tool_calls": _jsonable(state["tool_calls"]),
            },
            metadata={"model": model, "project": PROJECT},
        )
        # Publish the Langfuse trace id on the returned state so the eval loop
        # (collect -> judge) can attach rubric scores to the right trace.
        state["langfuse_trace_id"] = langfuse_context.get_current_trace_id()
    return state


def final_text(state: dict) -> str:
    for content in reversed(state["messages"]):
        if content.role == "model" and content.parts:
            text = "".join(part.text or "" for part in content.parts).strip()
            if text:
                return text
    return ""
