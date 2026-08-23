"""Verify the rewritten LangChain-native graph.py without calling Vertex.

Checks: llm construction, BaseTool wrapping, tool _arun call-through, and the
no-op handler path. Run from projects/agent-harness with the harness venv.
"""
import asyncio
import sys

sys.path.insert(0, ".")

from agent.graph import (
    LANGFUSE_ENABLED,
    _MCPTool,
    _build_llm,
    _make_handler,
    _tools,
    final_text,
)
from agent.mcp_client import MCPToolbox


class FakeTool:
    name = "rag_search"
    description = "Search the patient's notes"
    input_schema = {"type": "object"}


async def main() -> None:
    # 1. LLM construction (no Vertex call yet)
    llm = _build_llm("gemini-2.0-flash")
    print("llm constructed:", type(llm).__name__, "| model:", llm.model)
    print("  project:", llm.project, "| location:", llm.location, "| temp:", llm.temperature)

    # 2. Tool wrapping
    box = MCPToolbox(session=object())
    box._tools = {"rag_search": FakeTool()}
    tools = _tools(box)
    print("wrapped tools:", [t.name for t in tools], "| type:", type(tools[0]).__name__)
    assert isinstance(tools[0], _MCPTool)

    # 3. Tool _arun calls through to box.call with dict kwargs
    async def fake_call(name, args):
        return {"name": name, "args": args, "ok": True}

    box.call = fake_call
    res = await tools[0]._arun(hadm_id=1, query="x")
    print("tool _arun result:", res)
    assert res["name"] == "rag_search" and res["args"] == {"hadm_id": 1, "query": "x"}

    # 4. No-op handler when Langfuse disabled
    print("LANGFUSE_ENABLED:", LANGFUSE_ENABLED)
    h = _make_handler()
    print("handler:", type(h).__name__, "| last_trace_id:", h.last_trace_id)
    assert h.last_trace_id is None

    # 5. final_text handles list-content AI messages
    from langchain_core.messages import AIMessage

    state = {"messages": [AIMessage(content="")]}
    print("final_text(empty content):", repr(final_text(state)))
    state = {"messages": [AIMessage(content="hello world")]}
    print("final_text:", repr(final_text(state)))

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
