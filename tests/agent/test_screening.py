"""Gap 5: content filtering is configured here, and a filtered response is recorded.

Two halves, and only one of them belongs to this layer. The thresholds are set per
category rather than inherited from the model, because the default differs between
models and is `OFF` on the ones this pin is due to move to — so without this, the
migration would remove filtering silently. What is deliberately *not* here is any
defence against injection or jailbreak: that is a policy question for layer 11.

The second half is that a refusal can now be counted rather than only experienced:
which categories were flagged is recorded, on the runs that were refused.
"""

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from services.agent import graph  # noqa: E402
from services.agent import http as srv  # noqa: E402
from services.agent import model_turn  # noqa: E402
from services.mcp.config import GEMINI_SAFETY_THRESHOLDS  # noqa: E402

# The four categories a request can configure. The CSAM and personal-data filters
# are not in this list because they are not configurable at all.
CONFIGURABLE = {
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
}


def _name(value):
    """An enum or a string, whichever the SDK handed back."""
    return str(getattr(value, "name", value))


def _turn(text="", **metadata):
    return AIMessage(content=text, response_metadata=metadata)


def _client():
    return TestClient(srv.app, raise_server_exceptions=False)


@asynccontextmanager
async def _fake_toolbox():
    yield None


# --- the thresholds are ours, not the model's --------------------------------

def test_every_configurable_category_is_decided_here():
    assert set(GEMINI_SAFETY_THRESHOLDS) == CONFIGURABLE


def test_the_client_carries_those_thresholds_per_category():
    llm = graph._build_llm(graph.MODEL_ID)
    pinned = {_name(k): _name(v) for k, v in llm.safety_settings.items()}
    assert pinned == GEMINI_SAFETY_THRESHOLDS


def test_no_category_is_left_to_the_model_default():
    # "Unspecified" and "OFF" both mean the model decides, which is the state this
    # gap was about — and the model this pin migrates to decides differently.
    assert "HARM_BLOCK_THRESHOLD_UNSPECIFIED" not in GEMINI_SAFETY_THRESHOLDS.values()
    assert "OFF" not in GEMINI_SAFETY_THRESHOLDS.values()


# --- a filtered response says so --------------------------------------------

def test_a_flagged_response_names_the_category_that_flagged_it():
    flagged = _turn(safety_ratings=[
        {"category": "HARM_CATEGORY_HATE_SPEECH", "blocked": False},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "blocked": True},
    ])
    assert model_turn.filtered_categories(flagged) == ["HARM_CATEGORY_DANGEROUS_CONTENT"]


def test_a_blocked_prompt_is_read_from_the_prompt_feedback():
    # Nothing reached the model, so there are no candidate ratings — the ones that
    # matter are attached to the prompt.
    blocked = _turn(prompt_feedback={
        "block_reason": "JAILBREAK",
        "safety_ratings": [
            {"category": "HARM_CATEGORY_JAILBREAK", "blocked": True},
        ],
    })
    assert model_turn.filtered_categories(blocked) == ["HARM_CATEGORY_JAILBREAK"]


def test_an_unflagged_response_is_none_rather_than_empty():
    # None means nothing was flagged; an empty list would be indistinguishable from
    # a response nobody scored.
    assert model_turn.filtered_categories(_turn()) is None
    assert model_turn.filtered_categories(_turn(safety_ratings=[])) is None


def test_a_refusal_records_which_category_refused_it(caplog):
    state = {
        "messages": [HumanMessage(content="a blocked request")],
        "tool_calls": [],
    }
    state["messages"].append(_turn(
        finish_reason="SAFETY",
        safety_ratings=[
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "blocked": True},
        ],
    ))

    async def fake_ask(box, question, on_event=None, question_kind=None):
        return state

    with caplog.at_level("INFO", logger="services.agent.chain"):
        with patch.object(srv, "toolbox", _fake_toolbox), \
             patch.object(srv, "ask", fake_ask):
            response = _client().post("/ask", json={"question": "a blocked request"})

    assert response.status_code == 502
    record = json.loads(caplog.records[-1].message)
    assert record["error"] == "answer_refused"
    assert record["finish_reason"] == "SAFETY"
    assert record["filtered"] == ["HARM_CATEGORY_DANGEROUS_CONTENT"]


def test_an_ordinary_answer_records_no_filtering(caplog):
    state = {
        "messages": [
            HumanMessage(content="What do the notes say about the diagnosis?"),
            _turn(text="The note describes pneumonia. ^[1]", finish_reason="STOP"),
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

    async def fake_ask(box, question, on_event=None, question_kind=None):
        return state

    with caplog.at_level("INFO", logger="services.agent.chain"):
        with patch.object(srv, "toolbox", _fake_toolbox), \
             patch.object(srv, "ask", fake_ask):
            response = _client().post("/ask", json={"question": "notes?"})

    assert response.status_code == 200
    assert json.loads(caplog.records[-1].message)["filtered"] is None
