"""The chain's identity, and the record of what each execution used (Gap 2).

Offline: the chain is stubbed, so these prove the *record* rather than the
answer. The two things worth guarding are that the model cannot be changed by
the environment, and that an execution which failed still leaves a record —
a log that only covers the answers that worked cannot explain the ones that
did not.
"""

import importlib
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from services.agent import chain, http as srv, stages  # noqa: E402


@asynccontextmanager
async def _fake_toolbox():
    yield None


def _client():
    return TestClient(srv.app, raise_server_exceptions=False)


def _state():
    """A completed chain that survives the real guardrails and composition."""
    return {
        "messages": [
            HumanMessage(content="What do the notes say about the diagnosis?"),
            AIMessage(content="The note describes pneumonia. ^[1]"),
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


def _records(caplog):
    """Every execution record in the captured log, parsed."""
    out = []
    for record in caplog.records:
        message = record.getMessage()
        if not message.startswith("{"):
            continue
        try:
            parsed = json.loads(message)
        except ValueError:
            continue
        if parsed.get("event") == "agent_execution":
            out.append(parsed)
    return out


# --- the identity -----------------------------------------------------------

def test_the_model_is_pinned_and_ignores_the_environment(monkeypatch):
    """The pin is the whole point: an environment default lets a deploy change
    the model with no commit anywhere, which is what makes an answer
    unreproducible. Reloaded with the variable set, the value must not move."""
    import services.mcp.config as config

    monkeypatch.setenv("GEMINI_MODEL", "gemini-imaginary-1")
    try:
        importlib.reload(config)
        assert config.GEMINI_MODEL != "gemini-imaginary-1"
        assert config.GEMINI_MODEL == "gemini-2.5-flash"
    finally:
        monkeypatch.delenv("GEMINI_MODEL", raising=False)
        importlib.reload(config)


def test_the_pinned_model_is_the_one_on_record():
    """The pin and its written reason move together.

    A test rather than a comment because a comment does not stop anyone: changing
    the model now means editing the record, and the record is where the reason and
    the evidence live.
    """
    import services.mcp.config as config

    assert config.MODEL_CHOICE["model"] == config.GEMINI_MODEL
    assert config.MODEL_CHOICE["decided"]
    assert config.MODEL_CHOICE["tier"]


def test_the_record_names_what_a_comparison_would_be_against():
    # The cheaper tier is named so the next person does not have to rediscover it,
    # and the evidence slot exists so that filling it in is a deliberate edit.
    import services.mcp.config as config

    assert config.MODEL_CHOICE["cheaper_alternative"] != config.GEMINI_MODEL
    assert "evidence" in config.MODEL_CHOICE


def test_the_chain_exposes_its_model_and_identity():
    assert chain.MODEL_ID == chain.MODEL_ID.strip()
    assert chain.CODE_REVISION.strip()
    # The record names the fields; the shape is asserted below against a real
    # record rather than against this tuple alone.
    assert "code_revision" in chain.RECORD_FIELDS
    assert "model" in chain.RECORD_FIELDS


def test_the_identity_comes_from_the_deployment():
    # The deploy's own value wins, and Cloud Run's revision name is the fallback.
    assert chain.resolve_code_revision({"CODE_REVISION": "abc1234"}) == "abc1234"
    assert chain.resolve_code_revision(
        {"K_REVISION": "agent-00032-7cv"}
    ) == "agent-00032-7cv"
    assert chain.resolve_code_revision(
        {"CODE_REVISION": "abc1234", "K_REVISION": "agent-00032-7cv"}
    ) == "abc1234"


def test_a_run_that_is_not_a_deployment_says_so():
    assert chain.resolve_code_revision({}) == "local"
    assert chain.resolve_code_revision(
        {"CODE_REVISION": "   ", "K_REVISION": ""}
    ) == "local"


def test_an_unexpanded_substitution_is_not_an_identity():
    # A $COMMIT_SHA that a build never substituted must not be reported as the
    # revision this answer came from — that is the failure this replaced.
    assert chain.resolve_code_revision({"CODE_REVISION": "$COMMIT_SHA"}) == "local"
    assert chain.resolve_code_revision(
        {"CODE_REVISION": "$COMMIT_SHA", "K_REVISION": "agent-00032-7cv"}
    ) == "agent-00032-7cv"


# --- the record -------------------------------------------------------------

def test_the_record_carries_the_identity_and_the_steps(caplog):
    with caplog.at_level("INFO", logger="services.agent.chain"):
        record = chain.record_execution(
            trace="abc123",
            question="What do the notes say?",
            stages=[
                {"stage": "planning", "tool": None, "ms": 12},
                {"stage": "tool", "tool": "rag_search", "ms": 480},
            ],
            duration_ms=910,
            outcome="ok",
            tool_calls=["rag_search"],
            guardrail_flags=2,
        )

    assert record["code_revision"] == chain.CODE_REVISION
    assert record["model"] == chain.MODEL_ID
    assert record["trace"] == "abc123"
    assert record["outcome"] == "ok"
    assert record["question_chars"] == len("What do the notes say?")
    assert record["duration_ms"] == 910
    assert [stage["stage"] for stage in record["stages"]] == ["planning", "tool"]
    assert record["tool_calls"] == ["rag_search"]
    assert record["guardrail_flags"] == 2
    assert "error" not in record
    # One line, and it is JSON: queryable by whatever storage layer 10 picks.
    assert len(caplog.records) == 1
    assert json.loads(caplog.records[0].getMessage()) == record


def test_a_failed_execution_still_records_what_it_had(caplog):
    """A record that only exists for answers that worked cannot explain the
    ones that did not."""
    with caplog.at_level("INFO", logger="services.agent.chain"):
        record = chain.record_execution(
            trace="-",
            question="Why?",
            stages=[{"stage": "planning", "tool": None, "ms": 5}],
            duration_ms=100,
            outcome="timeout",
            error="timeout",
        )

    assert record["outcome"] == "timeout"
    assert record["error"] == "timeout"
    assert record["tool_calls"] == []
    assert record["guardrail_flags"] == 0


def test_the_record_carries_no_question_or_answer_text(caplog):
    """The text is patient-derived and a log store is not the repository's
    privacy regime. Length and shape are recorded instead; if the text is ever
    needed for an incident review that is a deliberate decision with retention,
    not a side effect of adding a log line."""
    secret = "What medications was admission 90000009 discharged on?"

    with caplog.at_level("INFO", logger="services.agent.chain"):
        chain.record_execution(
            trace="-", question=secret, stages=[], duration_ms=1, outcome="ok",
        )

    line = caplog.records[0].getMessage()
    assert secret not in line
    assert "90000009" not in line
    assert json.loads(line)["question_chars"] == len(secret)


# --- one record per execution, both routes ----------------------------------

def test_a_blocking_execution_records_once(caplog):
    async def fake_ask(box, question, on_event=None, question_kind=None):
        on_event(stages.planning_event())
        on_event(stages.tool_event("rag_search"))
        return _state()

    with patch.object(srv, "toolbox", _fake_toolbox), \
         patch.object(srv, "ask", fake_ask), \
         caplog.at_level("INFO", logger="services.agent.chain"):
        resp = _client().post("/ask", json={"question": "diagnosis?"})

    assert resp.status_code == 200
    records = _records(caplog)
    assert len(records) == 1
    assert records[0]["outcome"] == "ok"
    # The stages the model ran, with timings, not just a count.
    assert [stage["stage"] for stage in records[0]["stages"]] == ["planning", "tool"]
    assert records[0]["stages"][1]["tool"] == "rag_search"
    assert records[0]["tool_calls"] == ["rag_search"]


def test_a_failed_blocking_execution_still_records_once(caplog):
    async def boom(box, question, on_event=None, question_kind=None):
        raise RuntimeError("upstream died")

    with patch.object(srv, "toolbox", _fake_toolbox), \
         patch.object(srv, "ask", boom), \
         caplog.at_level("INFO", logger="services.agent.chain"):
        resp = _client().post("/ask", json={"question": "diagnosis?"})

    assert resp.status_code == 502
    records = _records(caplog)
    assert len(records) == 1
    assert records[0]["outcome"] == "error"
    assert records[0]["error"] == "agent_failed"


def test_a_streamed_execution_records_once(caplog):
    async def fake_ask(box, question, on_event=None, question_kind=None):
        on_event(stages.planning_event())
        on_event(stages.tool_event("rag_search"))
        return _state()

    with patch.object(srv, "toolbox", _fake_toolbox), \
         patch.object(srv, "ask", fake_ask), \
         caplog.at_level("INFO", logger="services.agent.chain"):
        resp = _client().post("/ask/stream", json={"question": "diagnosis?"})

    assert resp.status_code == 200
    assert "event: answer" in resp.text
    records = _records(caplog)
    assert len(records) == 1
    assert records[0]["outcome"] == "ok"


def test_a_failed_streamed_execution_still_records_once(caplog):
    async def boom(box, question, on_event=None, question_kind=None):
        on_event(stages.tool_event("rag_search"))
        raise RuntimeError("upstream died")

    with patch.object(srv, "toolbox", _fake_toolbox), \
         patch.object(srv, "ask", boom), \
         caplog.at_level("INFO", logger="services.agent.chain"):
        resp = _client().post("/ask/stream", json={"question": "diagnosis?"})

    assert resp.status_code == 200
    records = _records(caplog)
    assert len(records) == 1
    assert records[0]["outcome"] == "error"


def test_a_rejected_request_is_not_an_execution(caplog):
    """Nothing ran, so there is nothing to record: the chain never started."""
    with caplog.at_level("INFO", logger="services.agent.chain"):
        resp = _client().post("/ask", json={"chip": "bogus", "hadm_id": 7})

    assert resp.status_code == 400
    assert _records(caplog) == []
