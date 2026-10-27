"""The model client and the streaming turn that reads it back.

Layer 4. Everything that decides *how a message is obtained* lives here, so that
Layer 3 (`graph.py`) can decide *what to send* without also owning the client:
the Vertex-backed chat model, its generation parameters, the prompt's timeout, and
the lookahead that holds a ReAct turn's opening chunks until it is known whether
that turn is deliberation or the answer.

What does not live here: which tools are bound (`graph` binds them, because the
allowed set is a policy decision), what the prompt says (`prompts.py`), and how a
finished turn is read for its outcome (`model_turn.py`).

Two names are deliberately re-exported by `graph` under their old private
spellings — `_build_llm`, `_has_tool_call`, `_streamed_turn`, and
`MODEL_TIMEOUT_SECONDS`. Eleven tests patch `graph._build_llm` to substitute a
fake model, and `deterministic.py` imports it from `graph` at call time. Because
`graph` keeps its own module-level binding and resolves it at the call site, those
patches still intercept; a name imported *inside* this module would not be the
name they replace.
"""


from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_google_genai import ChatGoogleGenerativeAI

from services.agent.telemetry import stages
from services.agent.model_turn import text_of
from services.agent.telemetry.stages import emit
from services.mcp.config import (
    GEMINI_LOCATION,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_REASONING_EFFORT,
    GEMINI_SAFETY_THRESHOLDS,
    PROJECT,
)
from services.mcp.runtime import timeout_chain

# One model call, bounded. The bound is part of the chain `timeout_chain`
# enforces at startup — model < tool < ask < the proxy's own limit — and it is set
# on the client rather than left absent for two reasons: a call that stalls should
# fail as a model failure rather than spend the whole question's deadline looking
# like a slow answer, and with no timeout at all the SDK's retry-on-timeout can
# never fire, so the retry policy would be inert for the one failure it is most
# needed for.
MODEL_TIMEOUT_SECONDS = timeout_chain().model


def build_llm(model: str) -> ChatGoogleGenerativeAI:
    """The chat model, bound to the Vertex backend with our generation config."""
    return ChatGoogleGenerativeAI(
        model=model,
        project=PROJECT,
        # The model's endpoint, not the project's region: the pinned model has no
        # regional endpoint, and `LOCATION` also names the vector index and the
        # prediction endpoint, which do. See the note on `GEMINI_LOCATION`.
        location=GEMINI_LOCATION,
        vertexai=True,
        temperature=0,
        # The allowance covers thinking AND the answer. Too small and the model
        # spends it all on thoughts, returns finish_reason=MAX_TOKENS with empty
        # text, and raises nothing — which in a graph looks like a silently
        # skipped tool call. See §9.
        max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
        # ...so thinking is bounded separately rather than left to take what it
        # likes from that allowance. A level bounds effort rather than tokens, so
        # what it will spend is only known afterwards; `GEMINI_REASONING_EFFORT`
        # carries the measurements. The pin and this parameter move together: the
        # 2.5 family rejects a level outright, and the 3.x family deprecates the
        # budget, so there is no value both accept.
        reasoning_effort=GEMINI_REASONING_EFFORT,
        max_retries=3,
        timeout=MODEL_TIMEOUT_SECONDS,
        # The content filters are set explicitly, per category, rather than left to
        # the model's default — which differs between models, and is `OFF` on the
        # ones this pin is due to move to. What is set here is what applies; there
        # is no other place content filtering is configured.
        safety_settings=dict(GEMINI_SAFETY_THRESHOLDS),
    )


def has_tool_call(chunk: AIMessageChunk) -> bool:
    """Whether this chunk carries any part of a tool call.

    Read from `tool_call_chunks`, which arrives as fragments: a chunk can carry a
    name without arguments, or arguments without a name. Any fragment at all means
    the model is calling a tool, which is all this needs to know.

    Merged chunks answer this too, which is what makes it usable on a settled turn:
    `AIMessageChunk + AIMessageChunk` concatenates `tool_call_chunks`, so a turn that
    called at all reads as one that called.
    """
    return bool(getattr(chunk, "tool_call_chunks", None))


async def streamed_turn(llm, messages, on_event) -> AIMessage:
    """One model turn, announced once it has turned out to be the answer.

    Concatenates as it goes: `AIMessageChunk + AIMessageChunk` is how LangChain
    accumulates a streamed turn, and `AIMessageChunk` subclasses `AIMessage`, so the
    accumulator satisfies every `isinstance(m, AIMessage)` reading the graph already
    does — `route()` and `final_message()` included.

    The prose is held to the END of the turn, and announced then only if the turn called
    no tool. `agent_node` writes two different things through one call — the deliberation
    that picks a tool, and the answer the reader came for — and which one a turn is cannot
    be known while it is being written.

    This used to hold only the OPENING chunk, on the theory that a turn calling a tool
    writes the call first. It does not. The measured failure on 2026-10-26 was a follow-up
    after `/ehr` that opened with "I'll check the sections for an inpatient diagnosis",
    then called a tool four times: each of those paragraphs reached the client, was drawn,
    and was discarded when the run went back for evidence, so the reader watched the answer
    appear and vanish four times in one turn. A hold on the opening chunk cannot help with
    prose that PRECEDES the call.

    What it costs is time-to-first-token on free-text turns: the prose arrives when the turn
    ends rather than as it is written, and the client's reading cadence (`copilot_reveal.js`)
    paces whatever arrives either way. What it buys is that no reader is shown a sentence
    the turn then withdraws. The command path is untouched — `deterministic._format`
    announces on arrival, because no tool is bound there and a formatting turn cannot be
    anything but the answer.
    """
    pieces: list[str] = []
    turn: AIMessageChunk | None = None

    async for chunk in llm.astream(messages):
        turn = chunk if turn is None else turn + chunk
        # The chunk's own text, not its repr. Gemini answers in content blocks, and
        # `str(chunk.content)` announced the envelope and not the prose — measured live
        # on `/ehr`, 2026-10-14 (see `model_turn.text_of`).
        piece = text_of(chunk.content)
        if piece:
            pieces.append(piece)

    if turn is None:
        raise RuntimeError("the model returned no chunks")

    # Deliberation is never announced: not its held prefix, not the paragraph it wrote
    # first. The turn is a routing turn, so none of it is the answer.
    if not has_tool_call(turn):
        for piece in pieces:
            emit(on_event, stages.answer_delta_event(piece))

    return turn
