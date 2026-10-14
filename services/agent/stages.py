"""What the agent tells the caller while a question is still in flight.

The answer's prose is now announced as it is written (UX Phase 4, reversing the
Phase 2 policy this paragraph used to state). The reversal was made with its cost
named rather than argued away: the guardrails in `guardrail.py` rewrite the model's
text after it finishes — an unsupported risk number is removed, an invented age is
redacted, a medication that is not in the retrieved notes is stripped — so a caller
watching the prose arrive is watching a draft, and for a clinical answer the
corrected part is exactly the part that matters. A clinician can read a dose or a
probability and then watch the system withdraw it.

What the reversal buys is time-to-first-token on the paths a clinician waits on. What
is meant to carry the risk is no longer withholding the text, it is the swap: `http`
measures the announced prose against the prose finally served, and the terminal frame
tells the client whether the two agree. Where they do not, the client replaces what it
rendered and says so, and the caller is owed that notice rather than left to notice.
The measurement is therefore no longer a diagnostic — it was one while the policy was
"announce nothing that might move" — it is the trigger for the correction. The one
case where text is still withheld is the ReAct turn's opening chunks, because there
the text might not be the answer at all: see `PROSE_LOOKAHEAD_CHUNKS` in `graph`.

This module exists to keep the progress frames honest. Five rules:

1. **A stage is emitted where the work happens, not before it.** The tool node
   emits immediately before it calls the tool; the model node emits immediately
   before it calls the model. There is no timer, no scripted sequence, and no
   stage that is announced ahead of the work it describes.

2. **A tool stage describes an action being taken, never a result.** "Searching
   the Discharge Notes" is true whether the search returns passages or fails, so
   it cannot mislead. Whether the answer is good is the terminal event's
   business — a failure is reported there, not by quietly dropping a stage.
   What a call *returned* is a separate stage (rule 5), so the action label
   never has to be retracted.

3. **A call that is refused is not announced.** The per-turn budget refusal in
   `graph._execute_tool_calls` returns a synthetic error without touching the
   tool. Announcing it would tell the user about work that never started, which
   is the specific dishonesty this design has to avoid.

4. **The vocabulary is closed.** `STAGES` is the complete set of values the
   stream can carry, so a client can switch on them exhaustively and a test can
   assert the set has not grown by accident.

5. **A result stage reports the tool's own output, and nothing else.** It is
   emitted immediately after a call returns and carries what came back — a
   count, or the fact of a failure. Clinical values are still deliberately
   absent, and that rule survives the Phase 4 reversal for a reason of its own:
   a progress line is not where a number is stated, and a probability printed as
   a *stage* would be one the guardrails never read at all. The answer frame is
   the prose they read, which is why the number belongs there and nowhere else. A
   count is not a clinical value, and it is the difference between a progress
   line that repeats five fixed phrases and one that tells the user something.

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
STAGE_RESULT = "result"
STAGE_VERIFY = "verify"
STAGE_ANSWER = "answer"
#: One chunk of the answer's prose, announced as it is written. Distinct from
#: STAGE_ANSWER, which carries the settled answer: a client that did not ask for
#: prose streaming must be able to ignore this frame and still receive the answer.
#:
#: A client that does render these frames owes the caller the other half of the
#: bargain, and that half is not optional — what they add up to is a draft, and when
#: it differs from the answer frame's prose the difference is the guardrails' work.
#: STAGE_PROSE_SETTLED is how it finds out, and the caller is told either way.
STAGE_ANSWER_DELTA = "answer_delta"
#: Reported once, and only when the announced prose did not survive into the answer.
#: Absent means the caller was shown the answer, which is the common case and needs no
#: frame to say so. Carries one of two verdicts — see `http._prose_settled`, which also
#: says why the process that ran the guardrails is the only one that can decide between
#: them. A client that receives it replaces what it rendered; a client that never does
#: was shown the settled text all along.
STAGE_PROSE_SETTLED = "prose_settled"
STAGE_ERROR = "error"

STAGES = (
    STAGE_PLANNING,
    STAGE_REVIEWING,
    STAGE_TOOL,
    STAGE_RESULT,
    STAGE_VERIFY,
    STAGE_ANSWER,
    STAGE_ERROR,
)

# --- Labels ----------------------------------------------------------------
# Title Case, with the usual exception: an article, a short conjunction or a short
# preposition stays lowercase unless it opens the label — "Reading the Question",
# not "Reading The Question". `SMALL_WORDS` is that rule, and a test enforces the
# labels against it, because these are the only strings a waiting user reads and
# the house style should not depend on whoever typed the last label remembering
# it.

SMALL_WORDS = frozenset({
    "a", "an", "the",
    "and", "but", "or", "nor", "for",
    "of", "to", "in", "on", "at", "by", "with", "from", "as",
    "into", "onto", "over", "under", "via", "per",
})

LABEL_PLANNING = "Reading the Question"
LABEL_REVIEWING = "Reviewing the Evidence"
LABEL_VERIFYING = "Checking the Answer Against the Evidence"

# The planning label varies with what was asked, when the caller said. A chip
# name is not patient data and is not the question's wording, so it can name the
# progress line without putting anything clinical on screen early. Free text has
# no chip: its nature is only knowable by the model, and asking the model would
# be the narration this decision does not take, so it gets the generic label.
QUESTION_KIND_LABELS = {
    "risk": "Reading the Risk Question",
    "meds": "Reading the Medication Question",
    "summarize": "Reading the Summary Request",
    "literature": "Reading the Literature Request",
}

# One label per tool the MCP server advertises (`services/mcp/server.py`).
# Each is phrased as the action the call performs, so it stays true even when
# the call fails. A test walks the live tool list and fails if a tool is added
# without a label here, which is what stops this table drifting into fiction.
TOOL_LABELS = {
    "predict_readmission": "Reading the Risk Model",
    "rag_search": "Searching the Discharge Notes",
    "rag_search_sections": "Reading the Note Sections",
    "discharge_medications": "Reading the Discharge Medications",
    "search_literature": "Searching the Literature",
}

# Used when the MCP server advertises a tool this table has not been taught.
# Deliberately vague AND still true: a tool is being consulted. The caller
# never sees a name it cannot interpret, and the omission is logged (see
# `graph._emit`) so it is fixed rather than silently rendered.
LABEL_UNKNOWN_TOOL = "Consulting a Tool"

# Result labels. Each reports the call's own output; the failure wording exists
# so a tool that did not answer is never rendered as a count of zero, which
# would say "nothing was found" when the truth is "nothing was asked".
LABEL_RESULT_FAILED = "The Tool Did Not Respond"
LABEL_RESULT_RISK = "Read the Risk Score"
LABEL_RESULT_PASSAGES = "Read the Note Passages"
LABEL_RESULT_SECTIONS = "Read the Note Sections"
LABEL_RESULT_MEDICATIONS = "Read the Discharge Medications"
LABEL_RESULT_LITERATURE = "Read the Articles"

RESULT_LABELS = {
    "predict_readmission": LABEL_RESULT_RISK,
    "rag_search": LABEL_RESULT_PASSAGES,
    "rag_search_sections": LABEL_RESULT_SECTIONS,
    "discharge_medications": LABEL_RESULT_MEDICATIONS,
    "search_literature": LABEL_RESULT_LITERATURE,
}

# What a returned count counts. The noun travels with the tool because one number
# means different things to different tools, and a count is only informational if
# the noun is right: three passages of a discharge note, three published records.
# "Found 3 Note Passages" was served for a PubMed search until 2026-10-12 — the
# count was correct and the noun described a different system, which is the same
# defect as a label that names a tool the caller cannot see.
RESULT_NOUNS = {
    "rag_search": "Note Passage",
    "rag_search_sections": "Note Section",
    "search_literature": "Literature Record",
}

# For a tool this table has not been taught. Vague and true, like LABEL_UNKNOWN_TOOL.
RESULT_NOUN_FALLBACK = "Result"


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def tool_label(name: str) -> str:
    """Display text for a tool call, or the documented fallback."""
    return TOOL_LABELS.get(name, LABEL_UNKNOWN_TOOL)


def planning_label(kind: str | None) -> str:
    """Display text for the first model turn, which varies with what was asked."""
    return QUESTION_KIND_LABELS.get(kind or "", LABEL_PLANNING)


def planning_event(kind: str | None = None) -> dict[str, Any]:
    """The first model turn: the question is being read.

    `kind` is the chip the question came from, when it came from one, so the
    line can say which question is being read. It is not the question's text and
    not the patient's identity: the execution record keeps no patient-derived
    text, and the label is derived from the request rather than from the model,
    which is why it cannot be wrong.
    """
    return {"stage": STAGE_PLANNING, "label": planning_label(kind)}


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


def result_event(name: str, response: Any) -> dict[str, Any]:
    """What a tool call returned, emitted immediately after it returns.

    Reports the call's own output in the call's own terms: how many note passages,
    note sections or published records came back, or that the call failed. A count
    is a fact about the work; the passages themselves, and any probability, are not
    put on screen here — the answer is where a guarded clinical value appears
    (rule 5).
    """
    payload = response if isinstance(response, dict) else {}
    if payload.get("error"):
        return {
            "stage": STAGE_RESULT,
            "label": LABEL_RESULT_FAILED,
            "tool": name,
            "ok": False,
        }

    returned = payload.get("returned")
    if returned is None and isinstance(payload.get("passages"), list):
        returned = len(payload["passages"])
    if isinstance(returned, int):
        noun = RESULT_NOUNS.get(name, RESULT_NOUN_FALLBACK)
        return {
            "stage": STAGE_RESULT,
            "label": f"Found {_plural(returned, noun)}",
            "tool": name,
            "ok": True,
            "returned": returned,
        }

    return {
        "stage": STAGE_RESULT,
        "label": RESULT_LABELS.get(name, LABEL_RESULT_RISK),
        "tool": name,
        "ok": True,
    }


def verify_event() -> dict[str, Any]:
    """The guardrails are about to check the answer against the evidence."""
    return {"stage": STAGE_VERIFY, "label": LABEL_VERIFYING}


def answer_delta_event(text: str) -> dict[str, Any]:
    """A chunk of the answer's prose, in the order it was written.

    Carries no `label`, and that is the point: every other stage event is a progress
    line, and this is not one — it is the answer arriving early. A client that draws
    a progress row per event would draw one per token, so the frame is distinguished
    by its stage and by nothing else. Emitted only when the caller asked for it
    (`http._stream_chain`), and only on paths whose prose is the answer
    (`routing.Command.streams_prose`).
    """
    return {"stage": STAGE_ANSWER_DELTA, "text": text}


def prose_settled_event(swap: str) -> dict[str, Any]:
    """The verdict on the prose a caller watched arrive.

    One of `"visible"` or `"silent"`, and it carries no `label` for the same reason
    `answer_delta_event` does not: it is not a step of the work, it is a fact about the
    answer the caller is already holding. A client that drew a progress row per event
    would invent a step that never happened.
    """
    return {"stage": STAGE_PROSE_SETTLED, "swap": swap}
