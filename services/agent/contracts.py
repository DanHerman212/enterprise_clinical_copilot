"""Validated request contract for the private agent HTTP service."""

from typing import Any, TypedDict

from services.agent.questions import UnknownChip, compose_question


class AgentRequest(TypedDict):
    question: str


class AgentToolCall(TypedDict):
    name: str
    args: dict[str, Any]
    response: dict[str, Any]
    derivable: bool


class RecordedToolCall(TypedDict):
    name: str
    args: dict[str, Any]
    response: dict[str, Any]


class ResolvedSource(TypedDict):
    cite: int
    section: str
    text: str
    query: str


# Whether a tool's result can be produced again by asking the same tool the same
# question, and therefore whether a caller that stores a turn needs to keep it.
#
# The site stores a turn so a later turn can be replayed, and what it keeps of a
# tool call is the name, the arguments, and the result only when the result
# cannot be obtained again: retrieval returns discharge-note text, and a store of
# clinical prose in the web application's database is exactly what the memory
# policy refuses. A prediction score is kept, because re-deriving it re-bills the
# endpoint and the score behind a published answer should be the score that was
# shown.
#
# The two sets are exhaustive over the tools the MCP server registers, and a test
# holds them to that: a tool that is added without being classified makes the
# suite fail rather than quietly deciding by default.
RE_DERIVABLE_TOOLS = frozenset({"rag_search", "rag_search_sections"})
RESULT_KEPT_TOOLS = frozenset({"predict_readmission"})


class AgentSuccess(TypedDict):
    question: str
    answer: str
    guardrail_flags: list[str]
    tool_calls: list[AgentToolCall]
    a2ui: dict[str, Any] | None
    sources: list[ResolvedSource]
    model: str
    code_revision: str
    mcp_transport: str


class AgentRequestError(ValueError):
    """A request failed the agent service's public input contract."""

    def __init__(self, code: str, message: str, status_code: int):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class AgentResponseError(ValueError):
    """The agent produced a payload outside its HTTP success contract."""


# The complete set of fields this request accepts. Closed deliberately: an
# unknown field used to be ignored, so a caller that sent `history` received a
# confident answer that had silently dropped it — a capability gap wearing the
# costume of a working feature. Refusing is the honest answer, and it is what
# keeps the shape of a request a decision rather than an accident.
#
# `turns` is the one accepted way to carry earlier turns, and it carries them as
# structure rather than as text: the chain replays each one as its own messages
# instead of pasting a transcript into the prompt. A caller that reaches for
# `history`, `messages`, `session_id` or any other name is refused by
# `CONVERSATION_FIELDS` below, which says why rather than leaving the caller to
# infer it from an answer that quietly lost its context.
ACCEPTED_REQUEST_FIELDS = frozenset({"question", "hadm_id", "chip", "turns"})

# Names a caller reaches for when it expects the agent to remember the
# conversation by itself. The agent holds nothing between requests, so a turn is
# only in the context window if the caller sent it in `turns`.
CONVERSATION_FIELDS = frozenset(
    {"history", "messages", "session_id", "conversation_id", "context"}
)

# What one replayed turn carries, and how much of the conversation may be
# replayed. Bounded because every replayed turn enters the context window: an
# unbounded transcript is a cost, and a long enough one pushes the current
# question out of the window it is being asked about.
#
# `tool_calls` is what makes a replay structurally faithful. A tool result has a
# place in the message sequence, and re-inserting one requires knowing the call
# that produced it. A call carries its name and arguments; its payload is carried
# only when it cannot be re-derived (a prediction score cannot), and is left out
# when it can (a retrieval result is resolved again from the corpus, which is
# what keeps clinical text out of the store).
REPLAYED_TURN_FIELDS = frozenset({"question", "answer", "tool_calls"})
REPLAYED_TOOL_CALL_FIELDS = frozenset({"name", "args", "payload"})
MAX_REPLAYED_TURNS = 6
MAX_REPLAYED_TOOL_CALLS = 5
MAX_TURN_ANSWER_CHARS = 8000
MAX_TOOL_NAME_CHARS = 64


def _parse_tool_calls(value: Any, turn_index: int) -> list[dict[str, Any]]:
    """Validate the tool calls a replayed turn says it made.

    Strict for the same reason the rest of the contract is: a field the chain
    cannot place in the message sequence is refused rather than carried, because
    a carried field is one the model never sees while the caller believes it was
    sent. Whether a name is a tool the server actually has is not decided here —
    that is answered by the call itself, and by a refusal at replay time.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        raise AgentRequestError(
            "invalid_request", f"'turns[{turn_index}].tool_calls' must be a list", 400
        )
    if len(value) > MAX_REPLAYED_TOOL_CALLS:
        raise AgentRequestError(
            "invalid_request",
            f"'turns[{turn_index}].tool_calls' holds {len(value)} calls; at most "
            f"{MAX_REPLAYED_TOOL_CALLS} are replayed.",
            413,
        )

    calls: list[dict[str, Any]] = []
    for call_index, call in enumerate(value):
        where = f"'turns[{turn_index}].tool_calls[{call_index}]'"
        if not isinstance(call, dict):
            raise AgentRequestError(
                "invalid_request", f"{where} must be an object", 400
            )
        unknown = sorted(set(call) - REPLAYED_TOOL_CALL_FIELDS)
        if unknown:
            named = ", ".join(f"'{name}'" for name in unknown)
            raise AgentRequestError(
                "unsupported_field",
                f"Unsupported field(s) in {where}: {named}. A replayed tool call "
                "carries 'name', 'args' and optionally 'payload'.",
                400,
            )
        name = call.get("name")
        if not isinstance(name, str) or not name.strip():
            raise AgentRequestError(
                "invalid_request", f"{where}.name must be a non-empty string", 400
            )
        if len(name) > MAX_TOOL_NAME_CHARS:
            raise AgentRequestError(
                "invalid_request", f"{where}.name is over "
                f"{MAX_TOOL_NAME_CHARS} characters",
                400,
            )
        args = call.get("args") or {}
        if not isinstance(args, dict):
            raise AgentRequestError(
                "invalid_request", f"{where}.args must be an object", 400
            )
        payload = call.get("payload")
        if payload is not None and not isinstance(payload, (dict, str)):
            raise AgentRequestError(
                "invalid_request",
                f"{where}.payload must be an object or a string when present",
                400,
            )
        calls.append({"name": name, "args": args, "payload": payload})
    return calls


def _parse_turns(value: Any, max_question_chars: int) -> list[dict[str, str]]:
    """Validate the earlier turns a caller wants this question considered with.

    Strict on purpose, and strict in the same way the rest of the contract is:
    a turn is a question and an answer and nothing else, each a bounded string,
    and the list is capped. A field the chain does not know how to replay is
    refused rather than carried, because a carried field is one the model never
    sees while the caller believes it was sent.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        raise AgentRequestError("invalid_request", "'turns' must be a list", 400)
    if len(value) > MAX_REPLAYED_TURNS:
        raise AgentRequestError(
            "too_many_turns",
            f"At most {MAX_REPLAYED_TURNS} earlier turns may be replayed.",
            413,
        )

    turns: list[dict[str, str]] = []
    for index, turn in enumerate(value):
        if not isinstance(turn, dict):
            raise AgentRequestError(
                "invalid_request", f"'turns[{index}]' must be an object", 400
            )
        unknown = sorted(set(turn) - REPLAYED_TURN_FIELDS)
        if unknown:
            named = ", ".join(f"'{name}'" for name in unknown)
            raise AgentRequestError(
                "unsupported_field",
                f"Unsupported field(s) in 'turns[{index}]': {named}. A replayed "
                "turn carries 'question' and 'answer'.",
                400,
            )
        question = turn.get("question")
        answer = turn.get("answer")
        if not isinstance(question, str) or not question.strip():
            raise AgentRequestError(
                "invalid_request",
                f"'turns[{index}].question' must be a non-empty string",
                400,
            )
        if len(question) > max_question_chars:
            raise AgentRequestError(
                "question_too_long",
                f"'turns[{index}].question' is over the "
                f"{max_question_chars} character limit.",
                413,
            )
        if not isinstance(answer, str) or not answer.strip():
            raise AgentRequestError(
                "invalid_request",
                f"'turns[{index}].answer' must be a non-empty string",
                400,
            )
        if len(answer) > MAX_TURN_ANSWER_CHARS:
            raise AgentRequestError(
                "answer_too_long",
                f"'turns[{index}].answer' is over the {MAX_TURN_ANSWER_CHARS} "
                "character limit.",
                413,
            )
        turns.append({
            "question": question.strip(),
            "answer": answer.strip(),
            "tool_calls": _parse_tool_calls(turn.get("tool_calls"), index),
        })
    return turns


def _unsupported_field_error(unknown: list[str]) -> AgentRequestError:
    """Refuse fields outside the contract, naming them.

    A conversation field gets its own wording because the refusal is the design
    decision made visible at the boundary: the product is single-turn, and
    conversation state belongs to the memory layer.
    """
    named = ", ".join(f"'{name}'" for name in unknown)
    if CONVERSATION_FIELDS.intersection(unknown):
        return AgentRequestError(
            "unsupported_field",
            f"Unsupported field(s): {named}. The agent is single-turn, so a "
            "conversation field is refused rather than silently ignored; "
            "conversation state belongs to the memory layer.",
            400,
        )
    return AgentRequestError(
        "unsupported_field",
        f"Unsupported field(s): {named}. Send 'question', or a 'chip' with a "
        "'hadm_id'.",
        400,
    )


def parse_agent_request(body: Any, *, max_question_chars: int) -> AgentRequest:
    """Validate JSON-decoded input and return the agent request contract.

    Three shapes, each of which the caller used to resolve itself before
    composing the question on its side:

        {"question": "..."}                  free text
        {"question": "...", "hadm_id": 123}  free text about one patient
        {"chip": "risk", "hadm_id": 123}     a starter chip

    The wording is composed here (`questions.compose_question`) rather than by the
    caller, so half the prompt no longer lives in another repository. The result
    still carries exactly one field — the question the model is asked — so
    everything downstream, including the response contract, is unchanged.

    The field set is closed: anything outside those three is refused with
    `unsupported_field` rather than ignored, and a field implying conversation
    state says why. There is no conversation state to send, and the contract is
    where that stops being implicit.

    The result carries one more field than it used to: `kind`, the chip the
    question came from, or None for free text. The progress stream labels its
    first stage from it (`stages.planning_label`). The question the model is
    asked is still a single string, so no downstream consumer changed.

    `turns` carries the earlier turns of the conversation the caller is having,
    when there is one: the caller owns that state, this service holds none, and
    the chain replays what it is given as messages rather than as text. An empty
    list and an absent field mean the same thing, which is that this question
    stands alone.
    """
    if not isinstance(body, dict):
        raise AgentRequestError(
            "invalid_request",
            "Send a 'question', or a 'chip' with a 'hadm_id'",
            400,
        )

    unknown = sorted(set(body) - ACCEPTED_REQUEST_FIELDS)
    if unknown:
        raise _unsupported_field_error(unknown)

    hadm_id = body.get("hadm_id")
    if hadm_id is not None:
        # bool is an int in Python: without the explicit check, hadm_id=True
        # would compose a question about admission 1.
        if not isinstance(hadm_id, int) or isinstance(hadm_id, bool) or hadm_id <= 0:
            raise AgentRequestError(
                "invalid_request", "'hadm_id' must be a positive integer", 400
            )

    chip = body.get("chip")
    if chip is not None and not isinstance(chip, str):
        raise AgentRequestError("invalid_request", "'chip' must be a string", 400)

    question = body.get("question")
    if question is not None and not isinstance(question, str):
        raise AgentRequestError("invalid_request", "'question' must be a string", 400)

    turns = _parse_turns(body.get("turns"), max_question_chars)

    try:
        composed = compose_question(chip=chip, question=question, hadm_id=hadm_id)
    except UnknownChip as exc:
        # Raised before UnknownChip's parent ValueError: it is the more specific
        # answer and the caller can act on it.
        raise AgentRequestError("unknown_chip", str(exc), 400) from exc
    except ValueError as exc:
        raise AgentRequestError(
            "invalid_request",
            "Send a 'question', or a 'chip' with a 'hadm_id'",
            400,
        ) from exc

    if len(composed) > max_question_chars:
        raise AgentRequestError(
            "question_too_long",
            f"Limit is {max_question_chars} characters.",
            413,
        )
    return {"question": composed, "kind": chip, "turns": turns}


def validate_agent_success(payload: Any) -> AgentSuccess:
    """Validate the complete payload emitted by the agent HTTP service."""
    if not isinstance(payload, dict):
        raise AgentResponseError("Agent produced a malformed response.")

    required_strings = ("question", "answer", "model", "mcp_transport")
    if any(not isinstance(payload.get(field), str) for field in required_strings):
        raise AgentResponseError("Agent produced an invalid response field.")
    if not payload["answer"].strip():
        raise AgentResponseError("Agent produced an empty answer.")

    flags = payload.get("guardrail_flags")
    if not isinstance(flags, list) or not all(
        isinstance(flag, str) for flag in flags
    ):
        raise AgentResponseError("Agent produced malformed guardrail flags.")

    calls = payload.get("tool_calls")
    if not isinstance(calls, list):
        raise AgentResponseError("Agent produced malformed tool calls.")
    for call in calls:
        if not isinstance(call, dict):
            raise AgentResponseError("Agent produced malformed tool calls.")
        if not isinstance(call.get("name"), str):
            raise AgentResponseError("Agent produced malformed tool calls.")
        if not isinstance(call.get("response"), dict):
            raise AgentResponseError("Agent produced malformed tool calls.")
        if not isinstance(call.get("args", {}), dict):
            raise AgentResponseError("Agent produced malformed tool calls.")
        if not isinstance(call.get("derivable"), bool):
            raise AgentResponseError("Agent produced malformed tool calls.")

    # The revision is what makes a stored turn explainable after a deploy. An
    # empty string is legitimate (a local run with no revision), a missing field
    # is not: it would leave the store unable to say which code answered.
    if not isinstance(payload.get("code_revision"), str):
        raise AgentResponseError("Agent produced no code revision.")

    # The same rule as the revision, with the same legitimacy for an empty
    # string: tracing is a sink, so a run with Langfuse unconfigured has no trace
    # to point at and says so by sending an empty id. What is not legitimate is
    # the field being absent or of another type — a caller stores this value, and
    # a caller that has to guess the type of a pointer guesses wrong.
    if not isinstance(payload.get("langfuse_trace_id"), str):
        raise AgentResponseError("Agent produced no langfuse trace id.")

    if payload.get("a2ui") is not None and not isinstance(payload["a2ui"], dict):
        raise AgentResponseError("Agent produced malformed A2UI data.")

    sources = payload.get("sources")
    if not isinstance(sources, list):
        raise AgentResponseError("Agent produced malformed sources.")
    for source in sources:
        if not isinstance(source, dict):
            raise AgentResponseError("Agent produced malformed sources.")
        if not isinstance(source.get("cite"), int) or isinstance(source.get("cite"), bool):
            raise AgentResponseError("Agent produced malformed sources.")
        if any(not isinstance(source.get(k), str) for k in ("section", "text", "query")):
            raise AgentResponseError("Agent produced malformed sources.")

    return payload


def validate_recorded_tool_call(payload: Any) -> RecordedToolCall:
    """Validate the internal trace record before it enters agent state."""
    if not isinstance(payload, dict):
        raise AgentResponseError("Recorded tool call is malformed.")
    if not isinstance(payload.get("name"), str):
        raise AgentResponseError("Recorded tool call name is invalid.")
    if not isinstance(payload.get("args"), dict):
        raise AgentResponseError("Recorded tool call arguments are invalid.")
    if not isinstance(payload.get("response"), dict):
        raise AgentResponseError("Recorded tool call response is invalid.")
    return payload