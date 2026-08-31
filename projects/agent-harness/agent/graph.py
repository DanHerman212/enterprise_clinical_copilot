"""LangChain-native agent: START -> agent -> tools -> agent -> END.

Written as an explicit `StateGraph` (rather than `create_react_agent`) so the
control flow stays visible: the agent turns once to emit a tool call, the tool
node runs it, and the agent turns *again* to narrate the result. That second
turn is the one the guardrails in prompts.py apply to.

Messages are LangChain `BaseMessage` objects (`SystemMessage` for the prompt,
`HumanMessage` for the question, `AIMessage` for model turns including tool
calls, `ToolMessage` for tool results). LangGraph is LangChain-native, so the
graph consumes these directly with no translation layer.

Observability is a side effect of the architecture, not hand-rolled code:

  - The model is `ChatGoogleGenerativeAI` (langchain-google-genai, Vertex
    backend + ADC), so every model call is a LangChain LLM run — the official
    Langfuse `CallbackHandler` captures it as a generation span with the full
    message list, the response, and usage.
  - MCP tools are wrapped as LangChain `BaseTool` subclasses that call through
    to `MCPToolbox` — every tool call is a LangChain tool run, captured as a
    tool span by the same callback.
  - The graph is invoked with `config={"callbacks": [handler]}`; the handler
    builds the native LangGraph trace (nodes/edges) and exposes
    `last_trace_id`, which we publish on the state for the eval loop
    (collect.py -> judge.py).

No `@observe`, no `langfuse_context`, no manual serialization. When the
LANGFUSE_* env vars are absent the handler is a no-op, preserving the
previous `LANGFUSE_ENABLED` gate behavior.
"""

import json
import operator
import os
from typing import Annotated, Any, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

from mcp_server.config import (
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MODEL,
    LOCATION,
    PROJECT,
)

from agent.mcp_client import MCPToolbox, _clean_schema
from agent.prompts import SYSTEM_PROMPT

# --- Langfuse observability (optional, no-op without keys) ---
# Enabled only when all three env vars are present, so local/dev runs without
# them never create a handler. When enabled, every /ask becomes a native
# Langfuse trace (built by the official LangGraph CallbackHandler) with the
# graph nodes/edges, a generation span for the Gemini call, and a tool span
# per MCP call — so the golden-eval fix-and-retest loop can open a failing
# case and see exactly which passages went in and what the model said.
LANGFUSE_ENABLED = bool(
    os.environ.get("LANGFUSE_PUBLIC_KEY")
    and os.environ.get("LANGFUSE_SECRET_KEY")
    and os.environ.get("LANGFUSE_HOST")
)


def _make_handler():
    """Return the official Langfuse CallbackHandler, or a no-op stand-in.

    The stand-in mimics `last_trace_id` (starts None) and is inert when
    LANGFUSE_* is not set, so the graph code is identical in both modes.
    """
    if not LANGFUSE_ENABLED:
        return _NoopHandler()
    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


class _NoopHandler:
    """Inert handler for when Langfuse is disabled (matches old gate)."""

    def __init__(self) -> None:
        self.last_trace_id = None


class AgentState(TypedDict):
    """`operator.add` makes each node append to the transcript."""

    messages: Annotated[list[BaseMessage], operator.add]
    tool_calls: Annotated[list[dict[str, Any]], operator.add]


# Spend bounds (ECC-02). One question is one tool call in every designed flow;
# these caps exist so a pathological model turn cannot buy unbounded Vertex/
# BigQuery spend. RECURSION_LIMIT counts LangGraph supersteps: agent -> tools
# -> agent is 3, so 10 allows ~4 tool rounds before the graph raises.
MAX_TOOL_CALLS_PER_TURN = 5
RECURSION_LIMIT = 10


async def _execute_tool_calls(
    toolbox: MCPToolbox, calls: list[dict],
) -> tuple[list[BaseMessage], list[dict[str, Any]]]:
    """Run one turn's tool calls, refusing those beyond the per-turn budget.

    Refused calls get a structured error payload (the same contract as tool
    failures) so the model sees WHY the result is missing instead of silently
    losing a call.
    """
    messages: list[BaseMessage] = []
    recorded: list[dict[str, Any]] = []
    for index, call in enumerate(calls):
        arguments = dict(call.get("args") or {})
        if index >= MAX_TOOL_CALLS_PER_TURN:
            payload: dict[str, Any] = {
                "error": "tool_call_limit",
                "message": (
                    f"Per-turn tool budget is {MAX_TOOL_CALLS_PER_TURN} calls; "
                    "this call was not executed."
                ),
            }
        else:
            payload = await toolbox.call(call["name"], arguments)
        recorded.append(
            {"name": call["name"], "args": arguments, "response": payload}
        )
        # JSON, not Python repr, so the model reads the exact contract the
        # prompt describes (ECC-13); the <tool_result> delimiter marks the
        # content — including retrieved note text — as data, paired with the
        # DATA VS INSTRUCTIONS system rule (ECC-05).
        body = payload if isinstance(payload, str) else json.dumps(
            payload, ensure_ascii=False
        )
        messages.append(
            ToolMessage(
                content=(
                    f'<tool_result name="{call["name"]}">\n{body}\n</tool_result>'
                ),
                tool_call_id=call.get("id", ""),
                name=call["name"],
            )
        )
    return messages, recorded


class _MCPTool(BaseTool):
    """One MCP tool, exposed to the model as a LangChain tool.

    `_arun` calls through to `MCPToolbox.call`, which returns a plain dict
    (already normalised / JSON-safe, and errors are structured `{"error": ...}`
    payloads — never exceptions). Returning the dict as the tool's content
    keeps that error contract intact for the agent's prompts.py guardrails.
    """

    name: str
    description: str
    box: MCPToolbox

    def _run(self, **kwargs: Any) -> Any:
        raise NotImplementedError("MCP tools are async-only")

    async def _arun(self, **kwargs: Any) -> Any:
        return await self.box.call(self.name, dict(kwargs))


def _build_llm(model: str) -> ChatGoogleGenerativeAI:
    """The chat model, bound to the Vertex backend with our generation config."""
    return ChatGoogleGenerativeAI(
        model=model,
        project=PROJECT,
        location=LOCATION,
        vertexai=True,
        temperature=0,
        # Budgets thinking AND the answer. Too small and the model spends it all
        # on thoughts, returns finish_reason=MAX_TOKENS with empty text, and
        # raises nothing — which in a graph looks like a silently skipped tool
        # call. See §9.
        max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
        max_retries=3,
    )


def _tools(toolbox: MCPToolbox) -> list[BaseTool]:
    """Build one LangChain tool per MCP tool, sharing the toolbox.

    Each MCP `input_schema` is declared to the model as the tool's
    `args_schema` (Gemini-safe subset), so `bind_tools` advertises the real
    parameter names and types instead of leaving the model to guess them from
    prose (ECC-10) — which is also what made the old kwargs-flatten heuristic
    necessary.
    """
    return [
        _MCPTool(
            name=name,
            description=(tool.description or "").strip(),
            args_schema=_clean_schema(tool.input_schema),
            box=toolbox,
        )
        for name, tool in toolbox._tools.items()
    ]


def build_graph(toolbox: MCPToolbox, model: str = GEMINI_MODEL):
    llm = _build_llm(model).bind_tools(_tools(toolbox))

    async def agent_node(state: AgentState) -> dict:
        response: AIMessage = await llm.ainvoke(state["messages"])
        return {"messages": [response], "tool_calls": []}

    async def tool_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        calls = last.tool_calls if isinstance(last, AIMessage) else []
        messages, recorded = await _execute_tool_calls(toolbox, calls)
        return {"messages": messages, "tool_calls": recorded}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        has_calls = isinstance(last, AIMessage) and bool(last.tool_calls)
        return "tools" if has_calls else END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


async def ask(
    toolbox: MCPToolbox,
    question: str,
    model: str = GEMINI_MODEL,
    name: str | None = None,
    tags: list[str] | None = None,
) -> dict:
    """Run one question to completion. Returns the final state.

    `name`/`tags` are forwarded to the LangGraph run so the Langfuse trace is
    cleanly identifiable (e.g. eval traces get name=`eval.risk` + tags).
    """
    graph = build_graph(toolbox, model=model)
    handler = _make_handler()

    config: dict = {"callbacks": [handler] if LANGFUSE_ENABLED else None,
                    "recursion_limit": RECURSION_LIMIT}
    if name is not None:
        config["run_name"] = name
    if tags:
        config["tags"] = list(tags)

    state = await graph.ainvoke(
        {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=question),
            ],
            "tool_calls": [],
        },
        config=config,
    )
    # Publish the Langfuse trace id on the returned state so the eval loop
    # (collect -> judge) can attach rubric scores to the right trace.
    if LANGFUSE_ENABLED:
        state["langfuse_trace_id"] = handler.last_trace_id
    return state


def final_text(state: dict) -> str:
    """Text of the FINAL assistant message — never an earlier one.

    Falling back to the last NON-EMPTY AI message served a stale pre-tool
    preamble with HTTP 200 whenever the final turn came back empty (the
    documented MAX_TOKENS failure that raises nothing). Empty means the answer
    is unavailable, and the server reports exactly that (ECC-12).
    """
    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    if not isinstance(last, AIMessage):
        return ""
    text = last.content
    if isinstance(text, list):
        # A list of content blocks — keep the text parts.
        text = "".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in text
        )
    return text.strip() if isinstance(text, str) else ""
