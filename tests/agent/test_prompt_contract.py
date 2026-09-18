"""The prompt and the code it describes must agree.

The prompt is the least-covered artifact in this layer: every other piece has
tests, and the instructions the model is actually given had none. These tests
deliberately do *not* copy the prompt's sentences — that would fail on every
rewording and catch nothing. What they pin is the couplings, the places where the
prompt and the code describe the same thing and could drift apart, because that
is where a silent failure lives.

Two of them carry the weight.

The ``<tool_result>`` delimiter is a literal in two files: the graph wraps every
tool result in it, and the prompt tells the model that everything inside it is
patient data and never an instruction. Rename it on one side and the rule
describes a boundary that no longer exists — every retrieved note passage becomes
text the model was told it may obey. Nothing else in the suite would fail.

The tool names are written into the prompt ("you MUST call
``predict_readmission``") and defined again by the MCP server's registrations. If
they drift, the model is instructed to call a tool that is not there.
"""
import asyncio
import re

import pytest

from services.agent import graph
from services.agent.graph import _execute_tool_calls
from services.agent.mcp_client import MCPToolbox
from services.agent.prompts import SYSTEM_PROMPT


def _toolbox() -> MCPToolbox:
    """A toolbox whose calls return a payload and touch nothing."""
    box = MCPToolbox(session=object())

    async def call(name, args):  # noqa: ANN001 - mirrors MCPToolbox.call
        return {"name": name, "args": args}

    box.call = call
    return box


def _wrapped_tool_result() -> str:
    """The exact text the graph hands the model for one tool result.

    Driven through the real `_execute_tool_calls` rather than by formatting the
    tag here, so the test reads what the code emits instead of what this file
    believes it emits.
    """
    messages, _ = asyncio.run(
        _execute_tool_calls(
            _toolbox(),
            [{"name": "rag_search", "args": {"hadm_id": 1}, "id": "call-1"}],
        )
    )
    return messages[0].content


def test_the_prompt_names_a_literal_delimiter():
    """Without a literal tag the data-versus-instructions rule points at nothing."""
    assert re.findall(r"<([a-z_]+)>\.\.\.</\1>", SYSTEM_PROMPT), (
        "The prompt no longer names a literal delimiter, so its rule that note "
        "text is data and not instructions has nothing to refer to."
    )


def test_the_delimiter_the_wrapper_emits_is_the_one_the_prompt_names():
    tags = re.findall(r"<([a-z_]+)>\.\.\.</\1>", SYSTEM_PROMPT)
    wrapped = _wrapped_tool_result()
    for tag in tags:
        assert f"<{tag} name=" in wrapped, (
            f"The prompt calls the data boundary <{tag}> but the wrapper emits "
            f"{wrapped[:80]!r}. Renaming one side leaves the model trusting a "
            "boundary that does not exist."
        )


def test_the_prompt_names_exactly_the_tools_the_server_registers():
    """Every registered tool must be named, or the model is never told to call it.

    Only this direction is checked mechanically: the prompt backticks many
    identifiers that are not tools (`passages`, `top_factors`,
    `brief_hospital_course`), so the reverse check would have to guess which
    backticked words are tool names.
    """
    from services.mcp.tools import (
        predict_readmission,
        rag_search,
        rag_search_sections,
    )

    registered = {
        predict_readmission.__name__,
        rag_search.__name__,
        rag_search_sections.__name__,
    }
    unnamed = {name for name in registered if f"`{name}`" not in SYSTEM_PROMPT}
    assert not unnamed, f"The prompt never names these registered tools: {unnamed}"


# Each rule is checked by its operative phrase rather than by the sentence
# around it. A deletion fails; a rewrite has to keep the phrase, which is the
# same bargain the guardrail tests make — these are the rules the post-hoc
# guardrails exist to enforce, so losing one is a behaviour change.
RULE_MARKERS = (
    ("the data-versus-instructions rule", "never instructions"),
    ("the must-call-the-tool rule", "MUST call"),
    ("never invent a citation", "Never invent a citation"),
    ("the exact number, not a risk band", "above or below the operating threshold"),
    # A citation is resolved against the passages THIS turn retrieved, and the
    # post-hoc citation guard removes a marker that points anywhere else. The
    # rule and the guard have to say the same thing, or a follow-up answered from
    # an earlier turn loses a citation it was right to write (2026-09-18).
    ("citations are scoped to this turn", "only from a tool call you made in THIS turn"),
)


@pytest.mark.parametrize("rule,marker", RULE_MARKERS)
def test_a_load_bearing_rule_is_still_stated(rule, marker):
    assert marker in SYSTEM_PROMPT, (
        f"The prompt no longer states {rule}. If the rule moved, point this at "
        "its new wording; if it was deleted, that is a guardrail change and not "
        "a test fix."
    )


def test_the_graph_uses_the_prompt_this_suite_tests():
    """The artifact under test must be the shipped one.

    Checked at the import seam. Driving the graph node would assert more, but it
    needs a model stub and the node is nested inside `build_graph`; this is
    recorded as the cheaper seam rather than described as the stronger check.
    """
    assert graph.SYSTEM_PROMPT is SYSTEM_PROMPT
