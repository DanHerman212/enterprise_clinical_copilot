"""Gap 1: a failure is reported for what it is.

The response says why a turn produced no text — the allowance ran out, or the
content filters made a decision, or the prompt never reached the model. The three
need different words, and a refusal must not be reported as something worth
retrying: at temperature 0 the same request reaches the same decision. These tests
are the acceptance for that, and for the finish reason reaching the record instead
of being inferred from an empty answer.
"""

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from services.agent import http as srv  # noqa: E402
from services.agent import model_turn  # noqa: E402


def _turn(text="", **metadata):
    return AIMessage(content=text, response_metadata=metadata)


def _state(message):
    return {
        "messages": [HumanMessage(content="what do the notes say?"), message],
        "tool_calls": [],
    }


def _client():
    return TestClient(srv.app, raise_server_exceptions=False)


@asynccontextmanager
async def _fake_toolbox():
    yield None


def _replying(message):
    async def fake_ask(box, question, on_event=None, question_kind=None, turns=None):
        return _state(message)
    return fake_ask


# --- what the response tells us ---------------------------------------------

def test_a_truncation_and_a_refusal_are_different_outcomes():
    assert model_turn.outcome(_turn(finish_reason="MAX_TOKENS")) == model_turn.TRUNCATED
    assert model_turn.outcome(_turn(finish_reason="SAFETY")) == model_turn.REFUSED
    assert model_turn.outcome(_turn(finish_reason="PROHIBITED_CONTENT")) == model_turn.REFUSED
    assert model_turn.outcome(_turn(finish_reason="STOP")) == model_turn.OK


def test_a_blocked_prompt_is_a_refusal_with_no_finish_reason_to_read():
    # No candidate was produced, so there is nothing to finish: the signal is on
    # the prompt feedback instead.
    blocked = _turn(prompt_feedback={"block_reason": "PROHIBITED_CONTENT"})
    assert model_turn.outcome(blocked) == model_turn.REFUSED
    assert model_turn.classify(blocked)[0] == "answer_refused"


def test_an_unblocked_prompt_is_not_mistaken_for_a_refusal():
    # BLOCK_REASON_UNSPECIFIED means "not blocked". Reading it as a refusal would
    # refuse every ordinary answer.
    unblocked = _turn(
        finish_reason="STOP",
        prompt_feedback={"block_reason": "BLOCK_REASON_UNSPECIFIED"},
    )
    assert model_turn.blocked_reason(unblocked) is None
    assert model_turn.outcome(unblocked) == model_turn.OK


def test_a_refusal_never_advises_retrying():
    code, sentence, reason = model_turn.classify(_turn(finish_reason="SAFETY"))
    assert code == "answer_refused"
    assert reason == "SAFETY"
    assert "retry" not in sentence.lower()


def test_a_truncation_does_advise_retrying():
    code, sentence, reason = model_turn.classify(_turn(finish_reason="MAX_TOKENS"))
    assert code == "answer_truncated"
    assert reason == "MAX_TOKENS"
    assert "retry" in sentence.lower()


def test_a_turn_with_no_signal_is_still_unavailable():
    code, sentence, reason = model_turn.classify(_turn())
    assert code == "answer_unavailable"
    assert sentence == model_turn.UNKNOWN_SENTENCE
    assert reason is None


# --- through the route, and into the record ---------------------------------

def test_a_refusal_reaches_the_caller_as_a_refusal():
    with patch.object(srv, "toolbox", _fake_toolbox), \
         patch.object(srv, "ask", _replying(_turn(finish_reason="SAFETY"))):
        response = _client().post("/ask", json={"question": "a blocked request"})

    assert response.status_code == 502
    body = response.json()
    assert body["error"] == "answer_refused"
    assert "retry" not in body["message"].lower()


def test_a_truncation_reaches_the_caller_as_a_truncation():
    with patch.object(srv, "toolbox", _fake_toolbox), \
         patch.object(srv, "ask", _replying(_turn(finish_reason="MAX_TOKENS"))):
        response = _client().post("/ask", json={"question": "a long question"})

    body = response.json()
    assert body["error"] == "answer_truncated"


def test_the_finish_reason_is_recorded_on_a_failure(caplog):
    with caplog.at_level("INFO", logger="services.agent.chain"):
        with patch.object(srv, "toolbox", _fake_toolbox), \
             patch.object(srv, "ask", _replying(_turn(finish_reason="SAFETY"))):
            _client().post("/ask", json={"question": "a blocked request"})

    record = json.loads(caplog.records[-1].message)
    assert record["finish_reason"] == "SAFETY"
    assert record["error"] == "answer_refused"
    assert record["outcome"] == "error"


def test_the_stream_tells_the_caller_the_same_thing():
    with patch.object(srv, "toolbox", _fake_toolbox), \
         patch.object(srv, "ask", _replying(_turn(finish_reason="SAFETY"))):
        response = _client().post("/ask/stream", json={"question": "a blocked request"})

    assert "answer_refused" in response.text
    assert "Please retry" not in response.text
