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

from agent.mcp_client import MCPToolbox
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
    """Build one LangChain tool per MCP tool, sharing the toolbox."""
    return [
        _MCPTool(
            name=name,
            description=(tool.description or "").strip(),
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
        messages: list[BaseMessage] = []
        recorded: list[dict[str, Any]] = []

        for call in calls:
            arguments = dict(call.get("args") or {})
            # langchain-google-genai (Vertex backend) can nest the real args
            # under a single "kwargs" key. Flatten it so the first call
            # succeeds instead of burning a turn on a pydantic validation
            # error (the model self-corrects, but why waste the call).
            if len(arguments) == 1 and isinstance(arguments.get("kwargs"), dict):
                arguments = arguments["kwargs"]
            payload = await toolbox.call(call["name"], arguments)
            recorded.append(
                {"name": call["name"], "args": arguments, "response": payload}
            )
            messages.append(
                ToolMessage(
                    content=payload if isinstance(payload, str) else str(payload),
                    tool_call_id=call.get("id", ""),
                    name=call["name"],
                )
            )

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

    config: dict = {"callbacks": [handler] if LANGFUSE_ENABLED else None}
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
    """The last non-empty assistant text in the transcript."""
    for content in reversed(state["messages"]):
        if isinstance(content, AIMessage) and content.content:
            text = content.content
            if isinstance(text, list):
                # A list of content blocks — keep the text parts.
                text = "".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in text
                )
            if text.strip():
                return text
    return ""
