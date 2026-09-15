"""What the agent tells the caller while a question is still in flight.

The answer itself is delivered complete, never token by token. The guardrails
in `guardrail.py` rewrite the model's text after it finishes — an unsupported
risk number is removed, an invented age is redacted, a medication that is not
in the retrieved notes is stripped — so a stream of tokens would be a draft the
user watches being corrected, and for a clinical answer the corrected part is
exactly the part that matters. What *can* be streamed honestly is progress:
which step of the chain is running.

This module exists to make that progress honest. Four rules:

1. **A stage is emitted where the work happens, not before it.** The tool node
   emits immediately before it calls the tool; the model node emits immediately
   before it calls the model. There is no timer, no scripted sequence, and no
   stage that is announced ahead of the work it describes.

2. **A stage describes an action being taken, never a result.** "Searching The
   Discharge Notes" is true whether the search returns passages or fails, so it
   cannot mislead. Whether the answer is good is the terminal event's business —
   a failure is reported there, not by quietly dropping a stage.

3. **A call that is refused is not announced.** The per-turn budget refusal in
   `graph._execute_tool_calls` returns a synthetic error without touching the
   tool. Announcing it would tell the user about work that never started, which
   is the specific dishonesty this design has to avoid.

4. **The vocabulary is closed.** `STAGES` is the complete set of values the
   stream can carry, so a client can switch on them exhaustively and a test can
   assert the set has not grown by accident.

The labels are display text: they are what the user reads, so they live here,
next to the prompt, rather than in the browser. The browser cannot know what
tools exist, and it must not be the place where wording about the agent's
internal steps is invented — same reasoning as the layer 1 decision that the
agent resolves citations and the browser renders them.
"""

from typing import Any

# --- The closed vocabulary -------------------------------------------------
# The first model turn reads the question; any later model turn is looking at
# evidence that came back from a tool, which is why the two labels differ.
STAGE_PLANNING = "planning"
STAGE_REVIEWING = "reviewing"
STAGE_TOOL = "tool"
STAGE_VERIFY = "verify"
STAGE_ANSWER = "answer"
STAGE_ERROR = "error"

STAGES = (
    STAGE_PLANNING,
    STAGE_REVIEWING,
    STAGE_TOOL,
    STAGE_VERIFY,
    STAGE_ANSWER,
    STAGE_ERROR,
)

# --- Labels ----------------------------------------------------------------
# Title Case, every word. A test enforces it (`test_progress_stream.py`), because
# these are the only strings a waiting user reads and the house style should not
# depend on whoever typed the last label remembering it.

LABEL_PLANNING = "Reading The Question"
LABEL_REVIEWING = "Reviewing The Evidence"
LABEL_VERIFYING = "Checking The Answer Against The Evidence"

# One label per tool the MCP server advertises (`services/mcp/server.py`).
# Each is phrased as the action the call performs, so it stays true even when
# the call fails. A test walks the live tool list and fails if a tool is added
# without a label here, which is what stops this table drifting into fiction.
TOOL_LABELS = {
    "predict_readmission": "Reading The Risk Model",
    "rag_search": "Searching The Discharge Notes",
    "rag_search_sections": "Reading The Note Sections",
}

# Used when the MCP server advertises a tool this table has not been taught.
# Deliberately vague AND still true: a tool is being consulted. The caller
# never sees a name it cannot interpret, and the omission is logged (see
# `graph._emit`) so it is fixed rather than silently rendered.
LABEL_UNKNOWN_TOOL = "Consulting A Tool"


def tool_label(name: str) -> str:
    """Display text for a tool call, or the documented fallback."""
    return TOOL_LABELS.get(name, LABEL_UNKNOWN_TOOL)


def planning_event() -> dict[str, Any]:
    """The first model turn: the question is being read."""
    return {"stage": STAGE_PLANNING, "label": LABEL_PLANNING}


def reviewing_event() -> dict[str, Any]:
    """A later model turn: tool evidence has come back and is being read."""
    return {"stage": STAGE_REVIEWING, "label": LABEL_REVIEWING}


def tool_event(name: str) -> dict[str, Any]:
    """A tool call, emitted immediately before the call is made.

    `tool` carries the real MCP tool name. It is not a new disclosure: the
    finished answer already lists every tool call by name in `tool_calls`
    (the browser renders them in its trace view), so a stage that names the
    tool tells the client nothing the terminal payload would not.
    """
    return {"stage": STAGE_TOOL, "label": tool_label(name), "tool": name}


def verify_event() -> dict[str, Any]:
    """The guardrails are about to check the answer against the evidence."""
    return {"stage": STAGE_VERIFY, "label": LABEL_VERIFYING}
