"""Per-path prose streaming: who may announce prose, and when.

UX Phase 2, part 1. The policy is not "stream the answer" — it is a decision per
path, and it is a safety decision before it is a latency one:

  - a command whose prose survives the guardrails by construction may be announced
    as it is written. `/risk`'s sentence is written from the projected result, and
    `/meds`'s narrative is generated in code from verified rows;
  - `/ehr` is model prose over retrieved sections — grounded, but unverified until
    the citation layer has resolved its markers — so it is buffered with free text;
  - and none of it happens unless the CALLER asked (`stream_prose`). The canvas
    that renders these frames today turns any event it does not recognise into a
    progress row, so streaming to a caller that did not ask would trade an answer
    for a progress log.

Offline: a stubbed toolbox, a stubbed model, and the real command table.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, AIMessageChunk  # noqa: E402

from services.agent import graph, http, routing, stages  # noqa: E402
from services.agent.contracts import AgentRequestError, parse_agent_request  # noqa: E402
from services.agent.deterministic import run_command  # noqa: E402

PREDICTION = {
    "hadm_id": 90000015,
    "probability": 0.0905,
    "threshold": 0.11,
    "decision": 0,
    "base_value": -1.767608,
    "top_factors": [
        {"feature": "prior_inpatient_days", "contribution": 0.3091,
         "direction": "increases"},
    ],
    "model_version": "readmission-final-20260917171317",
    "feature_source": "bigquery",
}

SECTIONS = {
    "hadm_id": 90000015,
    "returned": 1,
    "passages": [{"id": "n_bhc_1", "section": "brief_hospital_course",
                  "text": "The patient was admitted from the office."}],
}

RISK_PIECES = (
    "The readmission risk score is 0.33, ",
    "which is above the operating threshold of 0.11.",
)


class _RecordingToolbox:
    """Answers every call with one payload."""

    def __init__(self, payload):
        self.payload = payload
        self.calls: list[tuple[str, dict]] = []

    async def call(self, name, args):
        self.calls.append((name, dict(args)))
        return self.payload


class _RecordingModel:
    """Answers either way, and records which way it was asked.

    The recorded call is the assertion in two of the tests below: a command that
    must not stream has to reach the model through `ainvoke`, exactly as it did
    before streaming existed — not stream into frames that are then dropped.
    """

    def __init__(self, pieces) -> None:
        self.pieces = tuple(pieces)
        self.used: list[str] = []

    async def ainvoke(self, messages, config=None):
        self.used.append("ainvoke")
        return AIMessage(content="".join(self.pieces))

    async def astream(self, messages, config=None):
        self.used.append("astream")
        for piece in self.pieces:
            yield AIMessageChunk(content=piece)


def _run(wire, *, prose_stream, payload=PREDICTION, pieces=RISK_PIECES):
    """Run one command against stubs; return (state, model, delta events)."""
    events: list[dict] = []
    box = _RecordingToolbox(payload)
    model = _RecordingModel(pieces)
    command = routing.command_for(wire)
    topic = "a clinical topic" if command.topic_arg else None
    with patch.object(graph, "_build_llm", lambda name: model):
        state = asyncio.run(run_command(
            box, command, 90000064, topic=topic, question="a command",
            model="test-model", on_event=events.append, prose_stream=prose_stream,
        ))
    deltas = [e for e in events if e["stage"] == stages.STAGE_ANSWER_DELTA]
    return state, model, deltas


# --- the policy ---------------------------------------------------------------

def test_the_streaming_set_is_exactly_the_pre_validated_commands():
    """Written as a set rather than as flags read off each entry, so a command
    added later has to decide and this fails until it does."""
    # `/risk` used to be in here and is not any more: it has no prose to stream
    # (UX ruling 2026-10-13), so `/meds` is the whole set.
    assert {c.wire for c in routing.COMMANDS if c.streams_prose} == {"meds"}


def test_only_the_command_that_writes_its_own_prose_streams():
    """Streaming is granted per command, and only `/meds` has it.

    It composes its narrative in code from verified rows, so the text announced is the
    settled answer rather than a preview of something still being written. `/risk` used
    to be here and returns no prose at all now (UX ruling 2026-10-13).
    """
    assert routing.command_for("meds").streams_prose is True


def test_the_risk_command_has_no_prose_turn_at_all():
    """The widget is the whole answer, so no LLM is in this command's loop."""
    risk = routing.command_for("risk")
    assert risk.prose is False
    assert risk.formatting == ""


# --- what a caller that asked for it sees -------------------------------------

# Token-by-token announcement has no command left to announce for. `/risk` was the
# streaming case — its sentence was written from the projected result — and that
# sentence was deleted (UX ruling 2026-10-13), leaving `/meds`, which announces in
# one piece because its narrative is composed in code. The two tests that covered
# chunk-by-chunk streaming lived here and are gone with their subject. The effect is
# that `_formatting_turn`'s `announce` parameter is now unreachable through any
# command: when a command that genuinely streams is added, it comes back with a test,
# and this note is why there is not one in the meantime.


def test_the_medication_narrative_is_announced_not_the_extraction_json():
    """The model's reply for `/meds` is JSON. Announcing it would put
    `{"medications": …}` on a clinician's screen, so the prose that streams is the
    narrative written in code, and it is announced once, as the settled answer."""
    state, model, deltas = _run(
        "meds", prose_stream=True, payload={"returned": 0, "passages": []},
        pieces=('{"medications":', " []}"),
    )

    assert [d["text"] for d in deltas] == [state["deterministic_answer"]]
    assert "medications" not in deltas[0]["text"]
    # Nothing to stream from the model's turn, so it is not asked to stream.
    assert model.used == ["ainvoke"]


# --- what everyone else sees --------------------------------------------------

def test_nothing_is_announced_unless_the_caller_asks():
    """Default off — and off means the model is not asked to stream at all, so the
    call shape for a caller that did not opt in is the one it always had."""
    _, model, deltas = _run(
        "meds", prose_stream=False, payload={"returned": 0, "passages": []},
        pieces=('{"medications":', " []}"),
    )

    assert deltas == []
    assert model.used == ["ainvoke"]


def test_a_buffered_command_does_not_stream_however_hard_it_is_asked():
    """`/ehr` is model prose over retrieved sections: the summary is grounded, but
    it is not final while it is being written — its markers are resolved after the
    turn — so it waits with free text."""
    _, model, deltas = _run("summarize", prose_stream=True, payload=SECTIONS)

    assert deltas == []
    assert model.used == ["ainvoke"]


# --- the assumption, measured -------------------------------------------------

def test_a_changed_answer_is_logged_and_a_match_is_not(caplog):
    """The policy assumes a streaming command's prose survives the guardrails
    unchanged. This is where that assumption is checked against the finished
    answer, so a broken assumption is a log line rather than a clinician watching
    text move."""
    with caplog.at_level("WARNING", logger="services.agent.http"):
        http._check_no_mutation([], "served", "trace-quiet")
        http._check_no_mutation(["serv", "ed"], "served", "trace-quiet")

    assert caplog.text == ""

    with caplog.at_level("WARNING", logger="services.agent.http"):
        http._check_no_mutation(["The score is 0.14"], "The score is 0.33", "trace-1")

    assert "differs from the served answer" in caplog.text
    assert "trace-1" in caplog.text
    # Both texts, so the anomaly can be judged from the log alone.
    assert "0.14" in caplog.text
    assert "0.33" in caplog.text


# --- the opt-in at the edge ---------------------------------------------------

def test_the_opt_in_defaults_off_and_has_to_be_a_bool():
    """A `"false"` string is truthy, so it would stream while its own payload reads
    as off — the same reason `hadm_id` refuses `True`."""
    assert parse_agent_request(
        {"question": "why?"}, max_question_chars=2000
    )["stream_prose"] is False

    assert parse_agent_request(
        {"question": "why?", "stream_prose": True}, max_question_chars=2000
    )["stream_prose"] is True

    with pytest.raises(AgentRequestError) as raised:
        parse_agent_request(
            {"question": "why?", "stream_prose": "false"}, max_question_chars=2000
        )

    assert raised.value.code == "invalid_request"
