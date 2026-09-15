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
    """
    if not isinstance(body, dict):
        raise AgentRequestError(
            "invalid_request",
            "Send a 'question', or a 'chip' with a 'hadm_id'",
            400,
        )

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
    return {"question": composed}


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