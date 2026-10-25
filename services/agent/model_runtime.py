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

from services.agent import stages
from services.agent.model_turn import text_of
from services.agent.stages import emit
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


#: How many opening chunks are held before a model turn is declared an answer.
#
# The hold exists because `agent_node` writes two different things through one call:
# the deliberation that picks a tool, and the answer the reader came for. Which one
# it is cannot be known until the turn ends, and announcing a routing thought and then
# withdrawing it is the mutation the streaming policy forbids (`stages.py`). A turn
# that calls a tool writes the call first, so holding the opening few chunks is enough
# to tell the two apart, and the latency it costs an answer is a fraction of a
# sentence.
#
# The residual assumption is named rather than hidden: a model that writes a paragraph
# and *then* calls a tool would have that paragraph announced before the call arrived.
# `http._prose_settled` measures announced prose against the prose finally served on
# every streamed turn, so that case is logged where it happens rather than assumed away
# here. Lowering this number widens that exposure, which is the trade below.
#
# Why it is 1 rather than 3 (2026-10-15), and the reason is measured. The client paces
# the draft to a reading cadence, but everything held here is released as ONE burst and
# painted at once — so this number sets how much of an answer is never paced at all. On
# a 593-character answer a three-chunk hold released 426 characters in the first paint:
# 72% of the text was on screen before the cadence had anything to do. At one chunk the
# first paint is a single chunk, and the cadence governs the rest.
#
# One chunk still keeps routing off the screen, because a turn that calls a tool opens
# with the call. It does not cover a model that opens with prose and calls the tool
# later; that is the case `_prose_settled` reports, and an unmasked deliberation would
# be worse than a first paint a beat larger.
PROSE_LOOKAHEAD_CHUNKS = 1


def has_tool_call(chunk: AIMessageChunk) -> bool:
    """Whether this chunk carries any part of a tool call.

    Read from `tool_call_chunks`, which arrives as fragments: a chunk can carry a
    name without arguments, or arguments without a name. Any fragment at all means
    the model is calling a tool, which is all this needs to know.
    """
    return bool(getattr(chunk, "tool_call_chunks", None))


async def streamed_turn(llm, messages, on_event) -> AIMessage:
    """One model turn announced as it is written, unless it turns out to be routing.

    Concatenates as it goes: `AIMessageChunk + AIMessageChunk` is how LangChain
    accumulates a streamed turn, and `AIMessageChunk` subclasses `AIMessage`, so the
    accumulator satisfies every `isinstance(m, AIMessage)` reading the graph already
    does — `route()` and `final_message()` included.
    """
    held: list[str] = []
    announcing = False
    routing = False
    turn: AIMessageChunk | None = None

    async for chunk in llm.astream(messages):
        turn = chunk if turn is None else turn + chunk
        if routing:
            # A tool call was seen: this turn is deliberation and none of it goes to
            # the caller. The held prefix is dropped unannounced.
            continue
        if has_tool_call(chunk):
            routing = True
            held = []
            continue
        # The chunk's own text, not its repr. Gemini answers in content blocks, and
        # `str(chunk.content)` announced the envelope and not the prose — measured live
        # on `/ehr`, 2026-10-14 (see `model_turn.text_of`).
        piece = text_of(chunk.content)
        if announcing:
            if piece:
                emit(on_event, stages.answer_delta_event(piece))
            continue
        held.append(piece)
        if len(held) >= PROSE_LOOKAHEAD_CHUNKS:
            announcing = True
            for piece in held:
                if piece:
                    emit(on_event, stages.answer_delta_event(piece))
            held = []

    if turn is None:
        raise RuntimeError("the model returned no chunks")

    if not routing and not announcing:
        # Shorter than the lookahead and no tool call: an answer that arrived in fewer
        # chunks than the hold. It is announced now rather than withheld — the hold is
        # a gate on ordering, not a reason to suppress a short answer.
        for piece in held:
            if piece:
                emit(on_event, stages.answer_delta_event(piece))

    return turn
