"""Offline tests for agent answer integrity (Cluster B) — no cloud credentials.

Covers:
  - ECC-12: final_text accepts ONLY the final AI message (a stale pre-tool
    preamble must not ship when the last turn came back empty)
  - ECC-13/05: ToolMessages carry JSON wrapped in <tool_result> delimiters
  - ECC-10: tools declare a real args_schema built from the MCP input_schema,
    and the kwargs-flatten heuristic is gone
"""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from agent import graph as gr  # noqa: E402


# --- final_text (ECC-12) ------------------------------------------------------

def test_final_text_returns_the_last_ai_message():
    state = {"messages": [HumanMessage(content="q"),
                          AIMessage(content="the answer ^[1]")]}
    assert gr.final_text(state) == "the answer ^[1]"


def test_final_text_does_not_fall_back_to_an_earlier_ai_message():
    """The MAX_TOKENS failure: last turn empty, an earlier preamble exists."""
    state = {"messages": [
        HumanMessage(content="q"),
        AIMessage(content="Let me look that up."),
        ToolMessage(content="{}", tool_call_id="1", name="rag_search"),
        AIMessage(content=""),
    ]}
    assert gr.final_text(state) == ""


def test_final_text_is_empty_when_the_last_message_is_not_ai():
    state = {"messages": [
        AIMessage(content="preamble"),
        ToolMessage(content="{}", tool_call_id="1", name="rag_search"),
    ]}
    assert gr.final_text(state) == ""


def test_final_text_joins_content_blocks():
    state = {"messages": [AIMessage(content=[{"type": "text", "text": "a "},
                                             {"type": "text", "text": "b"}])]}
    assert gr.final_text(state) == "a b"


# --- ToolMessage formatting (ECC-13, ECC-05) -----------------------------------

class _StubToolbox:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def call(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        return {"returned": 1, "results": [{"text": "ok"}]}


def test_tool_messages_are_json_inside_tool_result_delimiters():
    box = _StubToolbox()
    calls = [{"name": "rag_search", "id": "c1",
              "args": {"hadm_id": 1, "query": "meds"}}]
    messages, recorded = asyncio.run(gr._execute_tool_calls(box, calls))
    assert len(messages) == 1
    content = messages[0].content
    assert content.startswith('<tool_result name="rag_search">\n')
    assert content.endswith("\n</tool_result>")
    inner = content.split("\n", 1)[1].rsplit("\n", 1)[0]
    assert json.loads(inner) == {"returned": 1, "results": [{"text": "ok"}]}
    assert recorded[0]["response"]["returned"] == 1


def test_arguments_are_passed_through_verbatim():
    """The kwargs-flatten heuristic is gone (ECC-10): a literal `kwargs` arg
    must reach the tool untouched, not be silently unwrapped."""
    box = _StubToolbox()
    calls = [{"name": "t", "id": "c1", "args": {"kwargs": {"hadm_id": 1}}}]
    asyncio.run(gr._execute_tool_calls(box, calls))
    assert box.calls == [("t", {"kwargs": {"hadm_id": 1}})]


# --- args_schema declaration (ECC-10) -------------------------------------------

def test_tools_declare_the_cleaned_mcp_input_schema():
    schema = {
        "type": "object",
        "properties": {
            "hadm_id": {"type": "integer", "description": "admission id"},
            "query": {"type": "string"},
        },
        "required": ["hadm_id", "query"],
        "additionalProperties": False,  # Gemini rejects this key
    }
    toolbox = gr.MCPToolbox(session=None)  # type: ignore[arg-type]
    toolbox._tools = {
        "rag_search": SimpleNamespace(description="Search notes.",
                                      input_schema=schema),
    }
    (tool,) = gr._tools(toolbox)
    assert tool.name == "rag_search"
    assert tool.args_schema["required"] == ["hadm_id", "query"]
    assert set(tool.args_schema["properties"]) == {"hadm_id", "query"}
    assert "additionalProperties" not in tool.args_schema
