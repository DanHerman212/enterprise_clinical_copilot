"""Progress stages: what the chain announces, when, and what it never says.

Offline — a stub toolbox and a patched `ask`, so no Vertex credentials and no
MCP server are involved.

Three of these are guards against a specific dishonesty, not feature tests:

  - a refused tool call announces NOTHING, because announcing it would describe
    work that never started;
  - every tool the MCP server advertises has a label, so the wording cannot
    quietly drift behind the tool set;
  - a listener that raises cannot take the answer down with it.

Progressive *delivery* — frames arriving while the chain is still working — is
not testable here: TestClient buffers the whole response before returning it.
These tests assert the frames, their order, and the terminal frame. That the
frames actually arrive early is what the live test proves.
"""

import asyncio
import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from services.agent import http as srv  # noqa: E402
from services.agent import stages  # noqa: E402
from services.agent.graph import MAX_TOOL_CALLS_PER_TURN, _execute_tool_calls  # noqa: E402


class _StubToolbox:
    def __init__(self):
        self.calls = []

    async def call(self, name, args):
        self.calls.append((name, args))
        return {"ok": True}


@asynccontextmanager
async def _fake_toolbox():
    yield None


def _client():
    return TestClient(srv.app, raise_server_exceptions=False)


def _run(coro):
    return asyncio.run(coro)


def _frames(body):
    """Parse an SSE body into (event name, data) pairs, skipping keepalives."""
    frames = []
    for block in body.strip().split("\n\n"):
        lines = [l for l in block.split("\n") if l.strip() and not l.startswith(":")]
        if not lines:
            continue
        name = next((l[7:] for l in lines if l.startswith("event: ")), None)
        data = next((l[6:] for l in lines if l.startswith("data: ")), None)
        frames.append((name, json.loads(data)))
    return frames


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


# --- the vocabulary ---------------------------------------------------------

def test_the_stage_vocabulary_is_closed():
    """A client switches on these values, so the set is part of the contract."""
    assert set(stages.STAGES) == {
        "planning", "reviewing", "tool", "result", "verify", "answer", "error",
    }


def test_every_advertised_tool_has_a_progress_label():
    """The label table must keep up with the MCP server's tool list.

    A missing label does not break anything — the fallback is generic and still
    true — but the user gets a worse answer to "what is it doing?". This is the
    test that turns a silent omission into a failure.
    """
    from services.mcp.tools import (
        predict_readmission,
        rag_search,
        rag_search_sections,
    )

    advertised = {
        predict_readmission.__name__,
        rag_search.__name__,
        rag_search_sections.__name__,
    }
    assert advertised == set(stages.TOOL_LABELS)


def test_an_unknown_tool_falls_back_to_a_true_generic_label():
    assert stages.tool_label("brand_new_tool") == stages.LABEL_UNKNOWN_TOOL


def test_every_label_is_title_case():
    """House style, enforced rather than remembered.

    Title Case with the usual exception: an article or a short conjunction or
    preposition stays lowercase unless it opens the label. A set that mixes
    sentence case with Title Case reads as a mistake rather than as a style, and
    a label added without following the rule should fail here, not in review.

    Every offender is reported at once — a failure that names one word per run
    turns a five-second fix into five test runs.

    A numeral has no case, so a word beginning with a digit is accepted: the
    count labels are built at runtime, and "Found 3 Note Passages" is the rule
    applied to a number.
    """
    labels = sorted({
        stages.LABEL_PLANNING,
        stages.LABEL_REVIEWING,
        stages.LABEL_VERIFYING,
        stages.LABEL_UNKNOWN_TOOL,
        stages.LABEL_RESULT_FAILED,
        *stages.TOOL_LABELS.values(),
        *stages.QUESTION_KIND_LABELS.values(),
        *stages.RESULT_LABELS.values(),
        # Built at runtime, so a sample stands in for the form.
        stages.result_event("rag_search", {"returned": 3})["label"],
    })

    def breaks_the_rule(index, word):
        # A numeral has no case, so "Found 3 Note Passages" is the rule applied to
        # a number rather than an exception to it.
        if word[:1].isdigit():
            return False
        if index == 0:
            return not word[:1].isupper()
        if word.lower() in stages.SMALL_WORDS:
            return word != word.lower()
        return not word[:1].isupper()

    offenders = [
        f"{label!r} -> {word!r}"
        for label in labels
        for index, word in enumerate(label.split())
        if breaks_the_rule(index, word)
    ]

    assert not offenders, (
        "a stage label must be Title Case, with small words lowercase: "
        + ", ".join(offenders)
    )


# --- what the chain announces ----------------------------------------------

def test_a_stage_is_announced_for_each_executed_tool_call():
    box = _StubToolbox()
    events = []
    calls = [
        {"name": "predict_readmission", "args": {}, "id": "1"},
        {"name": "rag_search", "args": {}, "id": "2"},
    ]

    _run(_execute_tool_calls(box, calls, events.append))

    # A tool stage and then its result, per executed call.
    assert [e["stage"] for e in events] == [
        stages.STAGE_TOOL, stages.STAGE_RESULT,
        stages.STAGE_TOOL, stages.STAGE_RESULT,
    ]
    assert [e["label"] for e in events][0::2] == [
        stages.TOOL_LABELS["predict_readmission"],
        stages.TOOL_LABELS["rag_search"],
    ]
    assert [e["tool"] for e in events][0::2] == ["predict_readmission", "rag_search"]


def test_a_refused_call_announces_nothing():
    """Calls past the per-turn budget never reach the tool, so they must never
    be announced: a stage is a claim that work started."""
    box = _StubToolbox()
    events = []
    calls = [
        {"name": f"rag_search", "args": {}, "id": str(i)}
        for i in range(MAX_TOOL_CALLS_PER_TURN + 2)
    ]

    _run(_execute_tool_calls(box, calls, events.append))

    # Two events per executed call — the tool stage and its result — and the
    # refused call contributes none of either.
    assert len(events) == MAX_TOOL_CALLS_PER_TURN * 2
    assert len(box.calls) == MAX_TOOL_CALLS_PER_TURN


def test_no_listener_means_no_stage_work():
    """The single-response path passes no listener, and must be unaffected."""
    box = _StubToolbox()

    messages, recorded = _run(
        _execute_tool_calls(box, [{"name": "rag_search", "args": {}, "id": "1"}])
    )

    assert recorded[0]["name"] == "rag_search"
    assert len(messages) == 1


def test_a_failing_listener_does_not_take_the_answer_down():
    """Progress is a courtesy to the caller, never load-bearing: the worst case
    is a missing progress line, not a 502 caused by the reporting itself."""
    def broken_listener(event):
        raise RuntimeError("listener is broken")

    box = _StubToolbox()

    messages, recorded = _run(
        _execute_tool_calls(
            box, [{"name": "rag_search", "args": {}, "id": "1"}], broken_listener
        )
    )

    assert len(box.calls) == 1
    assert len(recorded) == 1
    assert len(messages) == 1


# --- the streaming route ----------------------------------------------------

def test_stream_sends_stages_then_one_answer_frame():
    async def fake_ask(box, question, on_event=None, question_kind=None, turns=None):
        on_event(stages.planning_event())
        on_event(stages.tool_event("rag_search"))
        return _state()

    with patch.object(srv, "toolbox", _fake_toolbox), \
         patch.object(srv, "ask", fake_ask):
        resp = _client().post("/ask/stream", json={"question": "diagnosis?"})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    frames = _frames(resp.text)
    assert [name for name, _ in frames] == ["planning", "tool", "verify", "answer"]
    assert frames[1][1]["label"] == stages.TOOL_LABELS["rag_search"]

    # Exactly one terminal frame, carrying the same object /ask returns.
    answer = frames[-1][1]
    assert set(answer) == {
        "question", "answer", "guardrail_flags", "tool_calls",
        "a2ui", "sources", "model", "code_revision", "mcp_transport",
        # The pointer to the run, streamed with the answer rather than only
        # recorded in the agent's log, so the site can store it beside the turn.
        "langfuse_trace_id",
    }
    assert answer["answer"].strip()

    # The streamed frame is what the site stores and later replays, so it has to
    # carry the same tool-call detail the single-response route carries.
    assert answer["tool_calls"][0]["args"] == {
        "hadm_id": 90000009, "query": "diagnosis",
    }
    assert answer["tool_calls"][0]["derivable"] is True


def test_the_answer_does_not_wait_for_a_keepalive_after_the_chain_finishes():
    """The relay must end when the chain ends, not when the next keepalive is
    due.

    The first version of this loop only noticed a finished chain between
    keepalives, so every answer waited out a full keepalive interval after the
    work was done — a progress stream that made the common case slower than not
    streaming at all. The timer here is the test: if the loop regresses, this
    takes a keepalive interval instead of milliseconds.
    """
    async def fake_ask(box, question, on_event=None, question_kind=None, turns=None):
        on_event(stages.planning_event())
        return _state()

    with patch.object(srv, "toolbox", _fake_toolbox), \
         patch.object(srv, "ask", fake_ask):
        started = time.monotonic()
        resp = _client().post("/ask/stream", json={"question": "diagnosis?"})
        elapsed = time.monotonic() - started

    assert resp.status_code == 200
    assert _frames(resp.text)[-1][0] == "answer"
    assert elapsed < srv.STREAM_KEEPALIVE_SECONDS


def test_stream_failures_before_it_opens_keep_their_status_codes():
    """Everything up to the first byte is still an ordinary request, so the
    caller gets a real status code rather than having to parse an event."""
    bad_json = _client().post(
        "/ask/stream", content=b"not json",
        headers={"Content-Type": "application/json"},
    )
    assert bad_json.status_code == 400
    assert bad_json.json()["error"] == "invalid_json"

    too_long = _client().post("/ask/stream", json={"question": "x" * 2001})
    assert too_long.status_code == 413
    assert too_long.json()["error"] == "question_too_long"


def test_stream_requires_an_identity_header_when_configured():
    with patch.object(srv, "REQUIRE_AUTH_HEADER", True):
        resp = _client().post("/ask/stream", json={"question": "risk?"})

    assert resp.status_code == 401
    assert resp.json()["error"] == "unauthenticated"


def test_stream_reports_a_mid_stream_failure_as_a_terminal_event():
    """Once the first byte is out the status code is spent, so a failure that
    happens after that arrives as an error frame carrying the same code and
    correlation id the single-response route would have put in its body."""
    async def boom(box, question, on_event=None, question_kind=None, turns=None):
        on_event(stages.tool_event("rag_search"))
        raise RuntimeError("https://secret-mcp-url/ask audience=projects/12345")

    with patch.object(srv, "toolbox", _fake_toolbox), \
         patch.object(srv, "ask", boom):
        resp = _client().post("/ask/stream", json={"question": "diagnosis?"})

    assert resp.status_code == 200
    frames = _frames(resp.text)
    assert [name for name, _ in frames] == ["tool", "error"]

    body = frames[-1][1]
    assert body["error"] == "agent_failed"
    assert len(body["correlation_id"]) == 12
    # Detail stays server-side (ECC-06) even on the streamed path.
    assert "secret-mcp-url" not in resp.text
    assert "RuntimeError" not in resp.text


def test_stream_refuses_to_ship_an_empty_answer():
    """The MAX_TOKENS failure raises nothing, so it is caught downstream: the
    final turn is empty and the caller gets answer_unavailable, not a blank
    frame that looks like a real answer."""
    async def empty_final_turn(box, question, on_event=None, question_kind=None, turns=None):
        state = _state()
        state["messages"] = [AIMessage(content="")]
        return state

    with patch.object(srv, "toolbox", _fake_toolbox), \
         patch.object(srv, "ask", empty_final_turn):
        resp = _client().post("/ask/stream", json={"question": "diagnosis?"})

    frames = _frames(resp.text)
    assert [name for name, _ in frames] == ["verify", "error"]
    assert frames[-1][1]["error"] == "answer_unavailable"
    assert not any(name == "answer" for name, _ in frames)


# --- result stages and request-derived planning labels -----------------------

class _PayloadToolbox:
    """Returns one canned payload, so a result label can be asserted."""

    def __init__(self, payload):
        self.payload = payload

    async def call(self, name, args):
        return self.payload


def _events_after_one_call(name, payload, on_event=None):
    events = []
    _run(_execute_tool_calls(
        _PayloadToolbox(payload), [{"name": name, "args": {}, "id": "1"}],
        events.append if on_event is None else on_event,
    ))
    return events


def test_a_result_stage_reports_the_count_the_call_returned():
    events = _events_after_one_call("rag_search", {"returned": 3, "passages": []})
    assert [e["stage"] for e in events] == [stages.STAGE_TOOL, stages.STAGE_RESULT]
    assert events[1]["label"] == "Found 3 Note Passages"


def test_a_result_stage_never_carries_a_clinical_value():
    """What 6.1 declined was showing a number before the guardrails had run.

    The tool's probability is the number the guarded answer will state, so
    putting it on the progress line ahead of the answer is that disclosure. The
    count of passages retrieved is not, and is the reason this event exists.
    """
    events = _events_after_one_call(
        "predict_readmission",
        {"probability": 0.085608, "threshold": 0.11, "top_factors": []},
    )
    result = events[1]
    assert result["label"] == stages.LABEL_RESULT_RISK
    encoded = json.dumps(result)
    assert "0.085608" not in encoded
    assert "threshold" not in encoded


def test_a_failed_call_says_so_rather_than_reporting_zero():
    """A tool that did not answer must not read as "found nothing"."""
    events = _events_after_one_call("rag_search", {"error": "search_failed"})
    assert events[1]["label"] == stages.LABEL_RESULT_FAILED
    assert events[1]["ok"] is False


def test_every_chip_has_a_planning_label():
    """The planning label cannot drift behind the chip set, the way the tool
    labels cannot drift behind the tool set."""
    from services.agent.questions import CHIP_QUESTIONS

    missing = [c for c in CHIP_QUESTIONS if c not in stages.QUESTION_KIND_LABELS]
    assert not missing, f"chips with no planning label: {missing}"


def test_free_text_gets_the_generic_planning_label():
    """What a free-text question is *about* is only knowable by the model, and
    asking the model is the narration this design does not take."""
    assert stages.planning_label(None) == stages.LABEL_PLANNING
    assert stages.planning_label("not_a_chip") == stages.LABEL_PLANNING
    assert stages.planning_event("meds")["label"] == "Reading the Medication Question"


def test_the_planning_label_follows_the_chip_in_the_request():
    """The chip survives from the request contract to the first progress frame."""
    seen = {}

    async def capture(box, question, on_event=None, question_kind=None, turns=None):
        seen["kind"] = question_kind
        on_event(stages.planning_event(question_kind))
        return _state()

    with patch.object(srv, "toolbox", _fake_toolbox), \
         patch.object(srv, "ask", capture):
        resp = _client().post(
            "/ask/stream", json={"chip": "meds", "hadm_id": 90000009}
        )

    assert seen["kind"] == "meds"
    assert _frames(resp.text)[0] == (
        "planning",
        {"stage": "planning", "label": "Reading the Medication Question"},
    )
