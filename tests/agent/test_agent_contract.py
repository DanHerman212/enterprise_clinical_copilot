import pytest

from services.agent.contracts import (
    AgentRequestError,
    AgentResponseError,
    validate_recorded_tool_call,
    parse_agent_request,
    validate_agent_success,
)


def test_parse_agent_request_canonicalises_the_question():
    """The question is composed here now, so surrounding whitespace is stripped.

    It used to pass through verbatim because the *caller* composed it and had
    already stripped it. Now that the agent owns the wording, the agent decides
    what the canonical question is — otherwise stray whitespace becomes part of
    the prompt.
    """
    assert parse_agent_request(
        {"question": " assess risk "}, max_question_chars=20
    ) == {"question": "assess risk", "kind": None}


# Single-turn is a decision, not an accident (layer 3, gap 4). These pin the
# refusal, because the failure they prevent is invisible: a dropped field
# produced a normal-looking answer, and the caller had no way to tell.

def test_a_conversation_field_is_refused_rather_than_ignored():
    with pytest.raises(AgentRequestError) as caught:
        parse_agent_request(
            {"question": "and his medications?", "history": []},
            max_question_chars=2000,
        )
    assert caught.value.code == "unsupported_field"
    assert caught.value.status_code == 400
    assert "single-turn" in caught.value.message


@pytest.mark.parametrize(
    "field",
    ["history", "messages", "session_id", "conversation_id", "context", "turns"],
)
def test_every_conversation_field_is_named_in_the_refusal(field):
    with pytest.raises(AgentRequestError) as caught:
        parse_agent_request(
            {"question": "why?", field: []}, max_question_chars=2000
        )
    assert caught.value.code == "unsupported_field"
    assert f"'{field}'" in caught.value.message


def test_an_unknown_field_is_refused_and_named():
    with pytest.raises(AgentRequestError) as caught:
        parse_agent_request(
            {"question": "why?", "temperature": 0.2}, max_question_chars=2000
        )
    assert caught.value.code == "unsupported_field"
    assert "'temperature'" in caught.value.message


def test_the_chip_name_travels_as_the_question_kind():
    """The progress stream labels its first stage from this (layer 3, streaming).

    The chip name is not the question's wording and not the patient's identity,
    so it can name a progress line without putting anything clinical on screen
    before the guardrails have run.
    """
    assert parse_agent_request(
        {"chip": "meds", "hadm_id": 90000009}, max_question_chars=2000
    )["kind"] == "meds"
    assert parse_agent_request(
        {"question": "why?"}, max_question_chars=2000
    )["kind"] is None


def test_the_closed_contract_still_accepts_the_three_shapes():
    """Closing the field set must not narrow what already worked."""
    assert parse_agent_request(
        {"question": "why?"}, max_question_chars=2000
    ) == {"question": "why?", "kind": None}
    assert parse_agent_request(
        {"chip": "risk", "hadm_id": 90000009}, max_question_chars=2000
    )
    assert parse_agent_request(
        {"hadm_id": 90000009}, max_question_chars=2000
    )


# The wording moved here from the website (layer 3, chain artifact), so these
# assert parity with the strings `demo/views.py::_question_for` used to build.
# If these literals need changing, the prompt changed — which moves answers, and
# is not something a refactor should do quietly.

def test_a_chip_and_admission_compose_what_the_website_used_to_build():
    assert parse_agent_request(
        {"chip": "risk", "hadm_id": 90000009}, max_question_chars=2000
    ) == {
        "question": (
            "Assess the 30-day readmission risk for this patient. "
            "For admission 90000009."
        ),
        "kind": "risk",
    }


def test_free_text_with_an_admission_gets_the_same_suffix_the_website_added():
    assert parse_agent_request(
        {"question": "Why was this patient flagged?", "hadm_id": 90000009},
        max_question_chars=2000,
    ) == {
        "question": "Why was this patient flagged? For admission 90000009.",
        "kind": None,
    }


def test_an_admission_alone_asks_the_default_question():
    """A patient selected with no chip and no text is still a question.

    Deliberately a different phrase from the risk chip plus an admission — the
    website had both, and both are preserved so moving the strings could not
    change what the model is asked.
    """
    assert parse_agent_request(
        {"hadm_id": 90000009}, max_question_chars=2000
    ) == {
        "question": "Assess the 30-day readmission risk for admission 90000009.",
        "kind": None,
    }


def test_the_admission_is_not_appended_twice():
    composed = parse_agent_request(
        {"chip": "meds", "hadm_id": 7}, max_question_chars=2000
    )["question"]
    assert composed.count("For admission 7.") == 1


def test_an_unknown_chip_is_refused_with_its_own_code():
    with pytest.raises(AgentRequestError) as error:
        parse_agent_request({"chip": "bogus", "hadm_id": 7}, max_question_chars=2000)

    assert error.value.code == "unknown_chip"
    assert error.value.status_code == 400


def test_the_compare_chip_is_gone():
    """It asked about a previous assessment that a single-turn product has no
    data for — a leftover from an earlier UX exercise, never offered in the
    console. Removing it from the table is not enough on its own: the name has
    to be refused, or a direct request still spends a credit asking the model
    something the system cannot answer (Gap 4).
    """
    with pytest.raises(AgentRequestError) as error:
        parse_agent_request(
            {"chip": "compare", "hadm_id": 7}, max_question_chars=2000
        )

    assert error.value.code == "unknown_chip"
    assert error.value.status_code == 400


@pytest.mark.parametrize("hadm_id", [0, -1, "9", 1.5, True])
def test_an_invalid_admission_is_refused(hadm_id):
    """True is in this list on purpose: bool is an int, so without an explicit
    check it would compose a question about admission 1."""
    with pytest.raises(AgentRequestError) as error:
        parse_agent_request(
            {"chip": "risk", "hadm_id": hadm_id}, max_question_chars=2000
        )

    assert error.value.code == "invalid_request"
    assert error.value.status_code == 400


def test_a_composed_question_over_the_limit_is_refused():
    """The limit applies to the question the model is asked, not to the parts.

    The suffix is appended before the check, so a free-text question that only
    fits without it is still refused — which is what the old shape did, since
    the caller's composed string was what arrived here.
    """
    with pytest.raises(AgentRequestError) as error:
        parse_agent_request(
            {"question": "x" * 10, "hadm_id": 90000009}, max_question_chars=20
        )

    assert error.value.code == "question_too_long"
    assert error.value.status_code == 413


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
        "sources": [{"cite": 1, "section": "discharge_medications",
                     "text": "warfarin 4 mg QD", "query": "medications"}],
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


def test_validate_agent_success_rejects_missing_sources():
    payload = _success_payload()
    del payload["sources"]

    with pytest.raises(AgentResponseError):
        validate_agent_success(payload)


def test_validate_agent_success_rejects_malformed_source_entry():
    """The browser indexes sources by `cite`; a non-int cite breaks the lookup."""
    payload = _success_payload()
    payload["sources"] = [{"cite": "1", "section": "s", "text": "t", "query": "q"}]

    with pytest.raises(AgentResponseError):
        validate_agent_success(payload)

    payload["sources"] = [{"cite": 1, "section": "s", "text": "t"}]

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