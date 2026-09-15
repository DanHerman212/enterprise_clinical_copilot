"""LangChain-native agent: START -> agent -> tools -> agent -> END.

Written as an explicit `StateGraph` (rather than `create_react_agent`) so the
control flow stays visible: the agent turns once to emit a tool call, the tool
node runs it, and the agent turns *again* to narrate the result. That second
turn is the one the guardrails in prompts.py apply to.

Messages are LangChain `BaseMessage` objects (`SystemMessage` for the prompt,
`HumanMessage` for the question, `AIMessage` for model turns including tool
calls, `ToolMessage` for tool results). LangGraph is LangChain-native, so the
graph consumes these directly with no translation layer.

Tracing is not implemented here, and that is a decision rather than an omission.
The Langfuse stack was torn down on 2026-09-12 and its scaffolding removed on
2026-09-15. Observability is layer 10's deliverable and evaluation is layer 9's,
both to be rebuilt from a written design (00-reference-architecture.md, 6). What
this layer contributes in the meantime is the execution record in `chain.py`: one
structured line per execution carrying the revision, model, stages, tool names
and timings, which Cloud Logging already collects.

Two properties of this module make that replacement cheap when it comes, and they
are worth keeping: the model is `ChatGoogleGenerativeAI`, so every model call is
a LangChain LLM run; and MCP tools are wrapped as LangChain `BaseTool`
subclasses, so every tool call is a LangChain tool run. A callback handler would
therefore see the whole chain without any code here changing.
"""

import json
import logging
import operator
from typing import Annotated, Any, Callable, TypedDict

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

from services.mcp.config import (
    GEMINI_MAX_OUTPUT_TOKENS,
    LOCATION,
    PROJECT,
)

from services.agent.chain import MODEL_ID

from services.agent.mcp_client import MCPToolbox, _clean_schema
from services.agent.contracts import RecordedToolCall, validate_recorded_tool_call
from services.agent.prompts import SYSTEM_PROMPT
from services.agent import stages

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """`operator.add` makes each node append to the transcript."""

    messages: Annotated[list[BaseMessage], operator.add]
    tool_calls: Annotated[list[RecordedToolCall], operator.add]


# Spend bounds (ECC-02). One question is one tool call in every designed flow;
# these caps exist so a pathological model turn cannot buy unbounded Vertex/
# BigQuery spend. RECURSION_LIMIT counts LangGraph supersteps: agent -> tools
# -> agent is 3, so 10 allows ~4 tool rounds before the graph raises.
MAX_TOOL_CALLS_PER_TURN = 5
RECURSION_LIMIT = 10


def _emit(on_event: Callable[[dict[str, Any]], None] | None, event: dict[str, Any]) -> None:
    """Announce a stage, if anyone is listening.

    A progress callback is a courtesy to the caller, never a load-bearing part
    of the answer, so a listener that raises must not take the answer down with
    it. The event is dropped and the run continues — the worst case is a
    missing progress line, against the alternative of a 502 caused by the
    progress reporting itself.

    An unknown tool name is logged here rather than at the call site: this is
    the only place that sees every stage, so it is the only place that can
    notice `stages.TOOL_LABELS` has fallen behind the MCP server's tool list.
    """
    if on_event is None:
        return
    if event.get("stage") == stages.STAGE_TOOL and event.get("tool") not in stages.TOOL_LABELS:
        logger.warning(
            "no progress label for tool %r — add it to stages.TOOL_LABELS",
            event.get("tool"),
        )
    try:
        on_event(event)
    except Exception:
        logger.warning("progress callback failed; continuing", exc_info=True)


async def _execute_tool_calls(
    toolbox: MCPToolbox, calls: list[dict],
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[BaseMessage], list[RecordedToolCall]]:
    """Run one turn's tool calls, refusing those beyond the per-turn budget."""
    messages: list[BaseMessage] = []
    recorded: list[RecordedToolCall] = []
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
            # Emitted HERE, one line above the call it describes: the stage is
            # derived from an execution that is about to happen, not from a
            # script of what should happen. A refused call (the branch above)
            # announces nothing, because nothing was called.
            _emit(on_event, stages.tool_event(call["name"]))
            payload = await toolbox.call(call["name"], arguments)
            # What came back, in the call's own terms. Emitted here because this
            # is where the payload exists, and only for a call that actually
            # ran: a refused call announces nothing, result included (rule 3).
            _emit(on_event, stages.result_event(call["name"], payload))
        recorded.append(validate_recorded_tool_call({
            "name": call["name"], "args": arguments, "response": payload,
        }))
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


def build_graph(
    toolbox: MCPToolbox,
    model: str = MODEL_ID,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    question_kind: str | None = None,
):
    """Compile the graph. `on_event` receives progress stages; None disables them.

    Progress is opt-in per call, which is what keeps the single-response `/ask`
    path byte-for-byte what it was: with no listener, `_emit` returns
    immediately and the graph behaves exactly as before.
    """
    llm = _build_llm(model).bind_tools(_tools(toolbox))

    async def agent_node(state: AgentState) -> dict:
        # Which turn this is decides the label, and the transcript is the only
        # honest source for that. No prior AIMessage means the model has not
        # spoken yet, so the question is what it is reading; any later turn is
        # looking at evidence that came back from a tool. The label is chosen
        # before the call, and it stays true either way: this turn may answer
        # or may ask for another tool.
        seen_a_model_turn = any(
            isinstance(m, AIMessage) for m in state["messages"]
        )
        _emit(
            on_event,
            stages.reviewing_event()
            if seen_a_model_turn
            else stages.planning_event(question_kind),
        )
        response: AIMessage = await llm.ainvoke(state["messages"])
        return {"messages": [response], "tool_calls": []}

    async def tool_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        calls = last.tool_calls if isinstance(last, AIMessage) else []
        messages, recorded = await _execute_tool_calls(toolbox, calls, on_event)
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
    model: str = MODEL_ID,
    name: str | None = None,
    tags: list[str] | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    question_kind: str | None = None,
) -> dict:
    """Run one question to completion. Returns the final state.

    `name`/`tags` are forwarded to the LangGraph run as its `run_name` and tags,
    so a caller can label a run (the eval loop passes name=`eval.risk`). They
    label the run; nothing here produces or stores a trace id.

    `on_event` is called with a stage dict as the chain progresses (see
    `stages.py`). It is a plain synchronous callback — the streaming route
    hands it a queue's `put_nowait` — and it is optional, so the answer path
    and the progress path stay one implementation.
    """
    graph = build_graph(
        toolbox, model=model, on_event=on_event, question_kind=question_kind
    )

    config: dict = {"recursion_limit": RECURSION_LIMIT}
    if name is not None:
        config["run_name"] = name
    if tags:
        config["tags"] = list(tags)

    return await graph.ainvoke(
        {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=question),
            ],
            "tool_calls": [],
        },
        config=config,
    )


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
