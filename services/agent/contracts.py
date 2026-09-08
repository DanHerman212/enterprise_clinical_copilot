"""Validated request contract for the private agent HTTP service."""

from typing import Any, TypedDict


class AgentRequest(TypedDict):
    question: str


class AgentToolCall(TypedDict):
    name: str
    response: dict[str, Any]


class RecordedToolCall(TypedDict):
    name: str
    args: dict[str, Any]
    response: dict[str, Any]


class AgentSuccess(TypedDict):
    question: str
    answer: str
    guardrail_flags: list[str]
    tool_calls: list[AgentToolCall]
    a2ui: dict[str, Any] | None
    citation_map: dict[str, int]
    intent_sections: list[str]
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
    """Validate JSON-decoded input and return the agent request contract."""
    if not isinstance(body, dict):
        raise AgentRequestError(
            "invalid_request",
            "'question' must be a non-empty string",
            400,
        )

    question = body.get("question")
    if not isinstance(question, str) or not question.strip():
        raise AgentRequestError(
            "invalid_request",
            "'question' must be a non-empty string",
            400,
        )
    if len(question) > max_question_chars:
        raise AgentRequestError(
            "question_too_long",
            f"Limit is {max_question_chars} characters.",
            413,
        )
    return {"question": question}


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

    citation_map = payload.get("citation_map")
    if not isinstance(citation_map, dict) or not all(
        isinstance(key, str) and isinstance(value, int)
        for key, value in citation_map.items()
    ):
        raise AgentResponseError("Agent produced a malformed citation map.")

    intent_sections = payload.get("intent_sections")
    if not isinstance(intent_sections, list) or not all(
        isinstance(section, str) for section in intent_sections
    ):
        raise AgentResponseError("Agent produced malformed intent sections.")

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