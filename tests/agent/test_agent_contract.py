import re
from pathlib import Path

import pytest

from services.agent.contracts import (
    RE_DERIVABLE_TOOLS,
    RESULT_KEPT_TOOLS,
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
    ) == {"question": "assess risk", "kind": None, "turns": []}


# The shape of a request is a decision, not an accident. These pin the refusal,
# because the failure they prevent is invisible: a dropped field produced a
# normal-looking answer, and the caller had no way to tell.
#
# `turns` is now the one accepted way to carry earlier turns. The names below are
# still refused, and each is refused by name, because a caller reaching for one
# of them believes the agent will remember a conversation it holds nothing about.

def test_a_conversation_field_is_refused_rather_than_ignored():
    with pytest.raises(AgentRequestError) as caught:
        parse_agent_request(
            {"question": "and his medications?", "history": []},
            max_question_chars=2000,
        )
    assert caught.value.code == "unsupported_field"
    assert caught.value.status_code == 400
    assert "layer" in caught.value.message or "turns" in caught.value.message


@pytest.mark.parametrize(
    "field",
    ["history", "messages", "session_id", "conversation_id", "context"],
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
    ) == {"question": "why?", "kind": None, "turns": []}
    assert parse_agent_request(
        {"chip": "risk", "hadm_id": 90000009}, max_question_chars=2000
    )
    assert parse_agent_request(
        {"hadm_id": 90000009}, max_question_chars=2000
    )


def test_earlier_turns_are_accepted_and_normalised():
    """The caller owns the conversation; this is how it hands one over."""
    parsed = parse_agent_request(
        {
            "question": "and the potassium?",
            "hadm_id": 90000009,
            "turns": [
                {"question": "  why flagged?  ", "answer": "  She was…  "},
                {"question": "by how much?", "answer": "0.31."},
            ],
        },
        max_question_chars=2000,
    )

    assert parsed["turns"] == [
        {"question": "why flagged?", "answer": "She was…", "tool_calls": []},
        {"question": "by how much?", "answer": "0.31.", "tool_calls": []},
    ]


def test_an_absent_turns_field_means_the_question_stands_alone():
    assert parse_agent_request(
        {"question": "why?"}, max_question_chars=2000
    )["turns"] == []


def test_a_turn_with_an_unknown_field_is_refused():
    """A carried field is one the model never sees while the caller believes
    it was sent — the failure the closed contract exists to prevent."""
    with pytest.raises(AgentRequestError) as caught:
        parse_agent_request(
            {"question": "why?",
             "turns": [{"question": "q", "answer": "a", "citations": []}]},
            max_question_chars=2000,
        )
    assert caught.value.code == "unsupported_field"
    assert "'citations'" in caught.value.message


def test_a_turn_needs_both_a_question_and_an_answer():
    for turn in ({"question": "q"}, {"answer": "a"}, {"question": "  ", "answer": "a"}):
        with pytest.raises(AgentRequestError) as caught:
            parse_agent_request(
                {"question": "why?", "turns": [turn]}, max_question_chars=2000
            )
        assert caught.value.code == "invalid_request"


def test_the_replayed_conversation_is_capped():
    turn = {"question": "q", "answer": "a"}
    with pytest.raises(AgentRequestError) as caught:
        parse_agent_request(
            {"question": "why?", "turns": [turn] * 7}, max_question_chars=2000
        )
    assert caught.value.code == "too_many_turns"
    assert caught.value.status_code == 413


def test_turns_must_be_a_list_of_objects():
    with pytest.raises(AgentRequestError):
        parse_agent_request(
            {"question": "why?", "turns": "history"}, max_question_chars=2000
        )
    with pytest.raises(AgentRequestError):
        parse_agent_request(
            {"question": "why?", "turns": ["q", "a"]}, max_question_chars=2000
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
        "turns": [],
    }


def test_free_text_with_an_admission_gets_the_same_suffix_the_website_added():
    assert parse_agent_request(
        {"question": "Why was this patient flagged?", "hadm_id": 90000009},
        max_question_chars=2000,
    ) == {
        "question": "Why was this patient flagged? For admission 90000009.",
        "kind": None,
        "turns": [],
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
        "turns": [],
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
        "tool_calls": [{
            "name": "predict_readmission",
            "args": {"hadm_id": 90000009},
            "response": {},
            "derivable": False,
        }],
        "a2ui": None,
        "sources": [{"cite": 1, "section": "discharge_medications",
                     "text": "warfarin 4 mg QD", "query": "medications"}],
        "model": "gemini-2.5-flash",
        "code_revision": "rev-1",
        "langfuse_trace_id": "a84f7e9903ab13d86b6ee075f2a91f60",
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


def test_validate_agent_success_rejects_malformed_tool_arguments():
    """A turn is stored to be replayed, so the arguments are part of the
    contract: a call whose arguments are not an object cannot be replayed."""
    payload = _success_payload()
    payload["tool_calls"][0]["args"] = "hadm_id=90000009"

    with pytest.raises(AgentResponseError):
        validate_agent_success(payload)


def test_validate_agent_success_requires_a_code_revision():
    """The revision is what makes a stored turn explainable after a deploy, so
    its absence is a contract failure rather than a cosmetic omission. An empty
    string is legitimate (a local run with no revision to report)."""
    payload = _success_payload()
    del payload["code_revision"]

    with pytest.raises(AgentResponseError):
        validate_agent_success(payload)

    payload = _success_payload()
    payload["code_revision"] = ""

    assert validate_agent_success(payload) is payload


def test_validate_agent_success_requires_a_trace_id_field():
    """The trace id is the pointer a stored turn uses to reach its run, so the
    field is part of the response contract. An empty string is legitimate and is
    the normal state when tracing is off; an absent field is not, because a
    caller stores whatever it is sent and would have to guess otherwise."""
    payload = _success_payload()
    del payload["langfuse_trace_id"]

    with pytest.raises(AgentResponseError):
        validate_agent_success(payload)

    payload = _success_payload()
    payload["langfuse_trace_id"] = None

    with pytest.raises(AgentResponseError):
        validate_agent_success(payload)

    # Tracing is a sink: off is a supported state, and it says so with "".
    payload = _success_payload()
    payload["langfuse_trace_id"] = ""

    assert validate_agent_success(payload) is payload


def test_validate_agent_success_requires_each_call_to_say_whether_it_is_derivable():
    """The flag decides what a storing caller keeps, and a caller that has to
    guess from the response's shape guesses wrong: `rag_search_sections` returns
    discharge-note text under section names rather than a 'passages' key."""
    payload = _success_payload()
    del payload["tool_calls"][0]["derivable"]

    with pytest.raises(AgentResponseError):
        validate_agent_success(payload)


def test_every_registered_tool_is_classified_as_derivable_or_kept():
    """Exhaustive by construction, because the default is a privacy decision.

    A new tool that is not classified here would be emitted with whatever the
    composition falls back to, and the failure that produces is invisible: note
    text accumulating in the web application's database. The tool names are read
    from the server that registers them rather than listed twice.
    """
    server = (Path(__file__).resolve().parents[2]
              / "services" / "mcp" / "server.py").read_text()
    registered = set(re.findall(r"server\.add_tool\((\w+)\)", server))

    assert registered, "no tools found — has registration moved?"
    assert registered == RE_DERIVABLE_TOOLS | RESULT_KEPT_TOOLS
    assert not RE_DERIVABLE_TOOLS & RESULT_KEPT_TOOLS


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