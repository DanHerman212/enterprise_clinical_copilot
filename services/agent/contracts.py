"""Validated request contract for the private agent HTTP service."""

from typing import Any, TypedDict

from services.agent.questions import UnknownChip, compose_question


class AgentRequest(TypedDict):
    question: str


class AgentToolCall(TypedDict):
    name: str
    response: dict[str, Any]


class RecordedToolCall(TypedDict):
    name: str
    args: dict[str, Any]
    response: dict[str, Any]


class ResolvedSource(TypedDict):
    cite: int
    section: str
    text: str
    query: str


class AgentSuccess(TypedDict):
    question: str
    answer: str
    guardrail_flags: list[str]
    tool_calls: list[AgentToolCall]
    a2ui: dict[str, Any] | None
    sources: list[ResolvedSource]
    model: str
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
# keeps single-turn a decision rather than an accident (layer 3, gap 4).
ACCEPTED_REQUEST_FIELDS = frozenset({"question", "hadm_id", "chip"})

# Names a caller reaches for when it expects the agent to remember the
# conversation. None of them do anything here, and the refusal says so rather
# than leaving the caller to infer it from an answer that quietly lost its
# context.
CONVERSATION_FIELDS = frozenset(
    {"history", "messages", "session_id", "conversation_id", "context", "turns"}
)


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
    return {"question": composed, "kind": chip}


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