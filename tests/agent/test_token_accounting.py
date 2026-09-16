"""Gap 3: the numbers the response reports are recorded.

Every response reports what it cost — input, output, thinking and cached tokens —
and which model version served it. None of that reached the record, which left
layer 10 with nothing to build a token metric from and attributed an answer to the
model we asked for rather than the one that answered. These tests are the
acceptance for both.
"""

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from services.agent import chain  # noqa: E402
from services.agent import http as srv  # noqa: E402
from services.agent import model_turn  # noqa: E402
from services.mcp import config  # noqa: E402


def _usage(input_tokens=0, output_tokens=0, thinking=0, cached=0):
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "output_token_details": {"reasoning": thinking},
        "input_token_details": {"cache_read": cached},
    }


def _turn(text="The note describes pneumonia. ^[1]", usage=None,
          served="gemini-2.5-flash-002", finish="STOP"):
    return AIMessage(
        content=text,
        response_metadata={"finish_reason": finish, "model_name": served},
        usage_metadata=usage if usage is not None else _usage(),
    )


def _answering_state(usage=None, finish="STOP"):
    """A completed chain that survives the real guardrails and composition."""
    return {
        "messages": [
            HumanMessage(content="What do the notes say about the diagnosis?"),
            _turn(usage=usage, finish=finish),
        ],
        "tool_calls": [{
            "name": "rag_search",
            "args": {"hadm_id": 90000009, "query": "diagnosis"},
            "response": {"hadm_id": 90000009, "returned": 1, "passages": [
                {"id": "MT-1-DS_discharge_diagnosis_1",
                 "section": "discharge_diagnosis",
                 "text": "DISCHARGE DIAGNOSES: 1. Pneumonia."},
            ]},
        }],
    }


def _client():
    return TestClient(srv.app, raise_server_exceptions=False)


@asynccontextmanager
async def _fake_toolbox():
    yield None


def _replying(state):
    async def fake_ask(box, question, on_event=None, question_kind=None):
        return state
    return fake_ask


# --- the arithmetic ---------------------------------------------------------

def test_every_model_turn_is_counted_not_only_the_last():
    # Each turn resends the conversation and is billed for it, so the execution's
    # cost is the sum. Counting only the answering turn would understate a
    # tool-using question by most of its cost.
    state = {"messages": [
        HumanMessage(content="q"),
        _turn(text="", usage=_usage(input_tokens=800, output_tokens=20)),
        _turn(usage=_usage(input_tokens=900, output_tokens=40)),
    ]}
    tokens = model_turn.token_usage(state)
    assert tokens == {"input": 1700, "output": 60, "total": 1760,
                      "thinking": 0, "cached": 0}


def test_thinking_and_cached_tokens_are_reported_separately():
    # Thinking is billed as output, and cached input is billed at a discount:
    # both are worth seeing on their own rather than folded into a total.
    state = {"messages": [_turn(usage=_usage(
        input_tokens=3000, output_tokens=180, thinking=1024, cached=2048,
    ))]}
    tokens = model_turn.token_usage(state)
    assert tokens["thinking"] == 1024
    assert tokens["cached"] == 2048
    assert tokens["input"] == 3000


def test_an_unreported_usage_is_none_rather_than_zero():
    # A zero is a measurement somebody made. Absence is not, and a metric that
    # cannot tell them apart is worse than one with a gap in it.
    state = {"messages": [AIMessage(content="no usage reported")]}
    assert model_turn.token_usage(state) is None


def test_the_served_model_is_read_rather_than_assumed():
    assert model_turn.served_model(_turn()) == "gemini-2.5-flash-002"
    assert model_turn.served_model(_turn(served=None)) is None
    # The pin is what we asked for; these are not the same question, and only one
    # of them is evidence. Read from the config rather than restated here, because a
    # hard-coded pin is the thing this test exists to stop anyone doing.
    assert chain.MODEL_ID == config.GEMINI_MODEL


# --- into the record --------------------------------------------------------

def test_a_successful_execution_records_what_it_cost(caplog):
    state = _answering_state(usage=_usage(input_tokens=1200, output_tokens=90,
                                          thinking=300, cached=1024))
    with caplog.at_level("INFO", logger="services.agent.chain"):
        with patch.object(srv, "toolbox", _fake_toolbox), \
             patch.object(srv, "ask", _replying(state)):
            response = _client().post("/ask", json={"question": "notes?"})

    assert response.status_code == 200
    record = json.loads(caplog.records[-1].message)
    assert record["tokens"] == {"input": 1200, "output": 90, "total": 1290,
                                "thinking": 300, "cached": 1024}
    assert record["served_model"] == "gemini-2.5-flash-002"
    assert record["model"] == chain.MODEL_ID


def test_a_failed_execution_records_what_it_cost_too(caplog):
    # The failure is where a token metric is most interesting, so the fields
    # cannot be a success-only luxury.
    state = _answering_state(usage=_usage(input_tokens=700, output_tokens=0,
                                          thinking=700), finish="MAX_TOKENS")
    state["messages"][-1] = _turn(text="", usage=_usage(input_tokens=700,
                                                        thinking=700),
                                  finish="MAX_TOKENS")
    with caplog.at_level("INFO", logger="services.agent.chain"):
        with patch.object(srv, "toolbox", _fake_toolbox), \
             patch.object(srv, "ask", _replying(state)):
            response = _client().post("/ask", json={"question": "notes?"})

    assert response.status_code == 502
    record = json.loads(caplog.records[-1].message)
    assert record["error"] == "answer_truncated"
    assert record["tokens"]["input"] == 700
    assert record["tokens"]["thinking"] == 700


def test_an_unknown_usage_is_recorded_as_null(caplog):
    state = _answering_state()
    state["messages"][-1] = AIMessage(content="The note describes pneumonia. ^[1]")
    with caplog.at_level("INFO", logger="services.agent.chain"):
        with patch.object(srv, "toolbox", _fake_toolbox), \
             patch.object(srv, "ask", _replying(state)):
            _client().post("/ask", json={"question": "notes?"})

    record = json.loads(caplog.records[-1].message)
    assert record["tokens"] is None
    assert record["served_model"] is None
