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

from services.agent.engine import gateway
from services.agent.postprocess import answer
from services.agent.telemetry import stages
from services.agent.telemetry.stages import emit
# Re-exported because the rest of the service reads it here and because it is the value
# `build_llm` produces for the served path; the adapter owns the choice.
from services.mcp.config import GEMINI_REASONING_EFFORT  # noqa: F401
from shared import model_outcomes

# One model call, bounded. The bound is part of the chain `timeout_chain`
# enforces at startup — model < tool < ask < the proxy's own limit — and it is set
# on the client rather than left absent for two reasons: a call that stalls should
# fail as a model failure rather than spend the whole question's deadline looking
# like a slow answer, and with no timeout at all the SDK's retry-on-timeout can
# never fire, so the retry policy would be inert for the one failure it is most
# needed for.
MODEL_TIMEOUT_SECONDS = gateway.MODEL_TIMEOUT_SECONDS

#: How many times the SDK retries one model call before the failure is ours.
#:
#: Three, which is what the client was already given — the number is now named so that
#: removing it is a visible edit rather than a deleted keyword. It is not larger because
#: the retries are bounded by `MODEL_TIMEOUT_SECONDS` on each attempt: a 429 burst that
#: outlives three attempts inside sixty seconds is an outage, and a fourth attempt buys
#: nothing but a longer wait before the caller is told.
#:
#: The policy itself belongs to the SDK (`langchain-google-genai` hands this to the
#: google-api-core retry layer, which backs off on 429 and 503 and not on 4xx). This
#: module's job is to state the bound and to make sure it is passed — see
#: `tests/agent/test_model_bounds.py`, which reads it back off the constructed client.
MODEL_MAX_RETRIES = gateway.MODEL_MAX_RETRIES


def build_llm(model: str):
    """The chat model for the served path, from the adapter.

    The adapter decides which provider serves an id, what that provider's endpoint and
    credentials are, and which parameters it accepts. This module keeps the name because
    eleven tests patch `graph._build_llm` and `deterministic.py` resolves it from there.
    """
    return gateway.chat_model(
        model, safety_profile=gateway.SAFETY_SERVED, effort=gateway.EFFORT_SERVED,
    )


#: Every configurable content filter off, for the extraction turns only.
#:
#: The served path keeps `GEMINI_SAFETY_THRESHOLDS` and should: that is a product control
#: on what the assistant says to a clinician. Extraction is not that job. It reads a
#: discharge note and returns a condition and a few phrases; its reply is never shown to
#: anyone, and its input is the note itself — which routinely describes overdoses, toxic
#: exposures, withdrawal, trauma and psychiatric illness, because a real hospital record
#: does. A consumer filter applied there does not withhold dangerous output, it withholds
#: an internal classification step and reads afterwards as "this chart names no condition".
#:
#: Measured 2026-11-02: `90000090`, a femoral-neck-fracture admission, returned
#: `finish_reason=SAFETY` with `HARM_CATEGORY_DANGEROUS_CONTENT` on three consecutive
#: calls. `services/mcp/config.py` already names dangerous content as the first category
#: due a review, "because a discharge note describes dangerous things".
#:
#: Two categories are not configurable and still apply, so this is not an off switch for
#: everything.
CLINICAL_SAFETY_SETTINGS = gateway.CLINICAL_SAFETY_SETTINGS


#: The reasoning level the structured EXTRACTION calls run at.
#:
#: `derive_target` and `_extract_once` return a small JSON object read out of text that is
#: already in front of them: a condition and a few anchor phrases, or a rate and the
#: sentence it was stated in. Nothing in that is open-ended clinical reasoning, and at the
#: synthesis level the model spends fifteen to thirty seconds thinking before it writes a
#: twenty-token object — on `/ehr` twice, before the answer's own formatting turn has
#: started.
#:
#: So the effort is a property of the JOB, passed at construction, and the served path
#: keeps `GEMINI_REASONING_EFFORT`: a clinician's answer is the one place the thinking is
#: doing work the reader will read.
EXTRACTION_REASONING_EFFORT = gateway.EXTRACTION_REASONING_EFFORT


def build_extraction_llm(model: str):
    """The chat model for structured extraction over clinical text.

    Same client, same generation config, different content-filter thresholds — the only
    difference that matters, and the reason it is a separate constructor rather than a
    parameter: a caller that must remember to pass the right thresholds is a caller that
    will eventually pass the wrong ones.

    The reasoning level is the second difference, and it is passed here rather than by the
    callers for the same reason: the two extraction steps are inside `deterministic.py`
    and neither should have to remember which effort its job wants.

    Which parameter expresses "low effort" is the adapter's answer, not this module's: one
    candidate rejects the level its siblings require, and the mapping from the job to the
    provider's spelling belongs in the one place that knows the provider.
    """
    return gateway.chat_model(
        model, safety_profile=gateway.SAFETY_EXTRACTION,
        effort=gateway.EFFORT_EXTRACTION,
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

    # The settled turn carries the neutral outcome before anyone reads it. A streamed turn
    # arrives as fragments and the metadata a provider reports the reason in is on the last
    # of them, so this is the one place the whole turn exists — stamping here means the
    # streaming path and the blocking path are read by the same rules.
    gateway.stamp(turn)

    # Deliberation is never announced: not its held prefix, not the paragraph it wrote
    # first. The turn is a routing turn, so none of it is the answer.
    if not has_tool_call(turn):
        for piece in pieces:
            emit(on_event, stages.answer_delta_event(piece))

    return turn


# ===========================================================================
# Consolidated from `model_turn.py` (Phase 7).
#
# How a model turn ended, read from the response instead of guessed at.
#
# Google's requirement is that a failure is reported for what it is. The response
# carries everything needed to do that, and until this module existed the chain read
# none of it: a truncated answer, a refusal by the model's content filters and a
# prompt that never reached the model all arrived at the caller as the same "please
# retry" — which is wrong advice for two of the three, because at temperature 0 a
# request that was refused reaches the same decision again.
#
# Three signals, one place:
#
#   - `finish_reason` on the message metadata — `STOP`, `MAX_TOKENS`, `SAFETY`,
#     `RECITATION`, `SPII`, `PROHIBITED_CONTENT` and so on.
#   - `safety_ratings`, per harm category, when the model scored the response.
#   - `prompt_feedback.block_reason`, when the prompt itself was blocked and no
#     candidate was ever produced. Google's own example of this is a prompt flagged
#     for prohibited content.
#
# The mapping to an outcome is deliberately small: "the allowance ran out" and "the
# model made a decision" are the two failures a caller can act on differently, and
# everything else stays unknown rather than being guessed into a category.
# ===========================================================================

from typing import Any

# What happened to the turn, in the terms a caller cares about. These four words are this
# module's public vocabulary and the execution record's; the neutral vocabulary they are
# derived from is `shared.model_outcomes`, which is what the code below actually reads.
OK = "ok"
TRUNCATED = "truncated"
REFUSED = "refused"
UNKNOWN = "unknown"

# The sentence a caller sees when nothing can be said about the turn.
UNKNOWN_SENTENCE = "The agent did not produce an answer. Please retry."
TRUNCATED_SENTENCE = (
    "The answer was cut off before it was complete. Please retry."
)


def metadata(message: Any) -> dict:
    """The response metadata of a message, or an empty mapping."""
    return dict(getattr(message, "response_metadata", None) or {})


def finish_reason(message: Any) -> str | None:
    """The model's own reason for stopping, upper-cased, or None if unreported.

    Which key holds it is the provider's business and is decided in
    `shared.model_outcomes`, which reads every shape a reason arrives in. This function is
    the name the rest of the service calls.
    """
    return model_outcomes.native_reason(message)


def blocked_reason(message: Any) -> str | None:
    """Why the prompt was blocked, or None when the prompt reached the model."""
    return model_outcomes.prompt_block_reason(message)


def safety_ratings(message: Any) -> list[dict]:
    """The per-category scores the provider returned, when it returned any."""
    return model_outcomes.safety_ratings(message)


def filtered_categories(message: Any) -> list[str] | None:
    """Which content-filter categories this response was flagged for.

    Read from the ratings rather than inferred from the finish reason, because a
    rating carries `blocked` per category and says *which* harm was involved. A
    prompt that never reached the model carries its ratings under `prompt_feedback`
    instead, so both places are read.

    Returns None when nothing was flagged, which is a different statement from an
    empty list: one means the response was scored and passed, the other that no
    scoring happened at all.
    """
    return list(model_outcomes.blocked_categories(message)) or None


def served_model(message: Any) -> str | None:
    """The model version that answered, as opposed to the one we asked for.

    The record's `model` field is the pin, which is what we intended; this is what
    served the request. They are the same string today and would not be during a
    migration, which is when the distinction matters for attribution.
    """
    name = metadata(message).get("model_name")
    return str(name) if name else None


def token_usage(state: dict) -> dict | None:
    """The tokens this execution was billed for, summed over its model turns.

    Summed rather than taken from the last turn, because each turn resends the
    conversation and is billed for it — so the sum is the cost of the question,
    not a count of unique text. Answering turns, tool-calling turns and thinking
    all count.

    `cached` is a subset of `input`, billed at a discount, so it is reported
    alongside rather than subtracted: what the totals would be after the discount
    is the billing system's answer, not ours.

    Returns None when no turn reported usage at all, rather than a dict of zeros.
    A zero is a measurement somebody made; absence is not.

    `thinking` and `cached` follow the same rule one level down: a provider that reports
    a reasoning breakdown is reporting a number, and zero is that number when it thought
    nothing. A provider with no such concept reports nothing, and the honest value there
    is None — a reported 0 would put a fabricated measurement into the cost table the
    programme is about to build.
    """
    totals = {"input": 0, "output": 0, "total": 0}
    reported = False
    thinking = 0
    cached = 0
    thinking_reported = False
    cached_reported = False
    for message in state.get("messages") or []:
        usage = getattr(message, "usage_metadata", None)
        if not usage:
            continue
        reported = True
        totals["input"] += usage.get("input_tokens") or 0
        totals["output"] += usage.get("output_tokens") or 0
        totals["total"] += usage.get("total_tokens") or 0
        output_details = usage.get("output_token_details")
        if isinstance(output_details, dict):
            thinking_reported = True
            thinking += output_details.get("reasoning") or 0
        input_details = usage.get("input_token_details")
        if isinstance(input_details, dict):
            cached_reported = True
            cached += input_details.get("cache_read") or 0
    if not reported:
        return None
    totals["thinking"] = thinking if thinking_reported else None
    totals["cached"] = cached if cached_reported else None
    return totals


def outcome(message: Any) -> str:
    """What happened to this turn: ok, truncated, refused or unknown.

    The four neutral outcomes are `shared.model_outcomes`'; these four words are the
    record's, and the mapping between them is the whole of this function. Reading the
    neutral outcome rather than the provider's token is what makes a refusal mean the same
    thing whichever model produced it.
    """
    result = model_outcomes.read(message)
    if result.refused:
        return REFUSED
    if result.truncated:
        return TRUNCATED
    if result.outcome == model_outcomes.STOP:
        return OK
    return UNKNOWN


def refusal_sentence(message: Any) -> str:
    """What to tell a caller whose request was refused.

    It deliberately does not suggest retrying. At temperature 0 the same question
    reaches the same decision, so "please retry" would be advice that cannot work —
    and in a clinical product, advice that cannot work is worse than none.
    """
    reason = blocked_reason(message) or finish_reason(message) or "unspecified"
    return (
        f"The model declined to answer this request ({reason}). "
        "Asking it again in the same way will not change the outcome."
    )


def silent_refusal(message: Any, *, outpatient: bool = False) -> str | None:
    """The sentence to serve when the model finished cleanly without speaking.

    Only an explicit `STOP` qualifies. A missing finish reason is unknown, and an
    unknown empty turn is still reported as unavailable: the difference between "the
    model decided to say nothing" and "the response did not arrive" is the whole
    reason this distinction exists, and guessing at it would hide a real fault behind
    a polite sentence.

    `outpatient` picks the wording that is true of the record. It is a property of the
    admission, not of the model's answer, which is why it is passed in rather than
    inferred from anything in this module.
    """
    if finish_reason(message) != "STOP":
        return None
    if blocked_reason(message) is not None:
        return None
    return answer.silent_turn_sentence(outpatient=outpatient)


def classify(message: Any) -> tuple[str, str, str | None]:
    """Describe a turn that produced no answer.

    Returns the error code, the sentence to show the caller, and the finish reason
    to record — the last of which is None when the response did not report one.
    Called only when there is no text to serve, so a turn with no signal at all is
    reported as unavailable rather than as an answer.
    """
    reason = finish_reason(message)
    kind = outcome(message)
    if kind == REFUSED:
        return "answer_refused", refusal_sentence(message), reason
    if kind == TRUNCATED:
        return "answer_truncated", TRUNCATED_SENTENCE, reason
    return "answer_unavailable", UNKNOWN_SENTENCE, reason


def text_of(content: Any) -> str:
    """The text in a message's `content`, whether it is a string or blocks.

    A model message's `content` is not always a string. Gemini answers with content
    blocks, so it arrives as a list shaped like `[{'type': 'text', 'text': '…'}]`,
    and a caller that assumes a string gets the repr of that list instead of the
    prose.

    This was `message_text.text_of`, a module with one function in it. The rule it
    enforces is the one `finish_reason` and `safety_ratings` enforce — read the
    response rather than guess at it — so it belongs beside them.

    Two callers are why it exists at all. `final_text` wore this unwrapping inline
    until the streaming paths needed it too: each chunk is shaped like the whole
    message, so the naive `str(chunk.content)` put `[{'type': 'text', 'text': …}]`
    on the wire. Measured live against `/ehr` on 2026-10-14 — 6,116 announced
    characters against a 1,021-character answer, every one of them envelope. The
    rule had two callers and had already drifted once, which is why it is a
    function rather than an inline copy.

    Blocks that carry no text — a function call, a thought signature, an image —
    contribute nothing. A list entry that is not a dict is stringified rather than
    dropped, which is what the original unwrapping did and is what keeps a plain
    list of strings working.

    A block whose type says it is deliberation contributes nothing either, and the type
    is checked before the `text` field is read. This matters now that more than one model
    can answer: a reasoning block that carries its prose under `text` would put the
    model's thinking on a clinician's screen, and a block that carries some other key
    would be stringified into the answer if the fallback went looking for one. Reading a
    block only when it is text, and never reading anything else out of it, is the rule
    that keeps the announcement equal to the answer.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for block in content:
            if isinstance(block, str):
                pieces.append(block)
            elif isinstance(block, dict):
                if _is_deliberation(block):
                    continue
                text = block.get("text")
                if isinstance(text, str):
                    pieces.append(text)
            else:
                pieces.append(str(block))
        return "".join(pieces)
    return ""


#: Block types that carry reasoning rather than answer, across the providers this
#: deployment can reach. A block of one of these types is never announced and never read
#: into the answer, whatever field it happens to carry its prose in.
_DELIBERATION_BLOCK_TYPES = frozenset({
    "thinking", "reasoning", "reasoning_content", "redacted_thinking", "thought",
    "thought_signature", "analysis",
})


def _is_deliberation(block: dict) -> bool:
    return str(block.get("type") or "").strip().lower() in _DELIBERATION_BLOCK_TYPES
