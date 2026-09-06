import pytest

from services.agent.contracts import (
    AgentRequestError,
    AgentResponseError,
    validate_recorded_tool_call,
    parse_agent_request,
    validate_agent_success,
)


def test_parse_agent_request_returns_the_validated_question():
    assert parse_agent_request(
        {"question": " assess risk "}, max_question_chars=20
    ) == {"question": " assess risk "}


@pytest.mark.parametrize("body", [None, [], "question", {"question": ""}])
def test_parse_agent_request_rejects_missing_or_non_string_question(body):
    with pytest.raises(AgentRequestError) as error:
        parse_agent_request(body, max_question_chars=20)

    assert error.value.code == "invalid_request"
    assert error.value.status_code == 400


def test_parse_agent_request_rejects_oversized_question():
    with pytest.raises(AgentRequestError) as error:
        parse_agent_request({"question": "12345"}, max_question_chars=4)

    assert error.value.code == "question_too_long"
    assert error.value.status_code == 413


def _success_payload():
    return {
        "question": "Assess risk.",
        "answer": "The estimate is below threshold.",
        "guardrail_flags": [],
        "tool_calls": [{"name": "predict_readmission", "response": {}}],
        "a2ui": None,
        "model": "gemini-2.5-flash",
        "mcp_transport": "http",
    }


def test_validate_agent_success_accepts_complete_payload():
    payload = _success_payload()

    assert validate_agent_success(payload) is payload


def test_validate_agent_success_rejects_empty_answer():
    payload = _success_payload()
    payload["answer"] = " "

    with pytest.raises(AgentResponseError):
        validate_agent_success(payload)


def test_validate_agent_success_rejects_malformed_tool_response():
    payload = _success_payload()
    payload["tool_calls"][0]["response"] = "not an object"

    with pytest.raises(AgentResponseError):
        validate_agent_success(payload)


def test_validate_recorded_tool_call_requires_internal_arguments():
    payload = {
        "name": "rag_search",
        "args": {"hadm_id": 90000009, "query": "medications"},
        "response": {"returned": 1},
    }

    assert validate_recorded_tool_call(payload) is payload


def test_validate_recorded_tool_call_rejects_missing_arguments():
    with pytest.raises(AgentResponseError):
        validate_recorded_tool_call({"name": "rag_search", "response": {}})