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

from services.agent.engine import graph
from services.agent import http
from services.agent.routing import routing
from services.agent.telemetry import stages  # noqa: E402
from services.agent.contracts import AgentRequestError, parse_agent_request  # noqa: E402
from services.agent.engine.model import text_of  # noqa: E402
from services.agent.engine.deterministic import run_command  # noqa: E402

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

def test_the_streaming_set_is_exactly_the_commands_that_asked_for_it():
    """Written as a set rather than as flags read off each entry, so a command
    added later has to decide and this fails until it does.

    Four now, and the two model-prose commands are the Phase 4 change: they were
    buffered while withholding the text was what carried the risk. `/risk` is still
    absent, having no prose at all (UX ruling 2026-10-13). `/labs` joined in Phase 4B
    and is model prose: the description of a course is written, and the card draws the
    values it describes.
    """
    assert {c.wire for c in routing.COMMANDS if c.streams_prose} == {
        "meds", "summarize", "literature", "labs",
    }


def test_the_two_prose_sources_are_declared_per_command():
    """`streams_prose` says there is prose to announce; `prose_is_model_output`
    says whose prose it is. `/meds` announces code output and `/ehr` announces the
    model's own reply, and the second flag is what keeps the extraction JSON off a
    clinician's screen."""
    meds = routing.command_for("meds")
    ehr = routing.command_for("summarize")

    assert meds.streams_prose is True
    assert meds.prose_is_model_output is False
    assert ehr.streams_prose is True
    assert ehr.prose_is_model_output is True


def test_no_command_is_a_widget_only_turn_any_more():
    """`/risk` was the widget-only command: its canvas was the whole answer, so it ran no
    formatting turn and built no model at all. It was retired on 2026-10-26 after the
    charter item that found it produced nothing from a blank composer, and every command
    left writes prose.

    Asserted rather than dropped because the streaming policy is stated in terms of it:
    a caller asks to be shown prose as it is written, and a command with no prose has
    nothing to show. The machinery for such a command (`prose=False`,
    `answer_in_surface`) stays in `deterministic` and `turn` for the day one comes back.
    """
    assert [command.wire for command in routing.COMMANDS if not command.prose] == []


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


def test_the_summary_streams_its_own_model_prose():
    """`/ehr`'s model reply IS the answer, so it is announced as it is written.

    This is the Phase 4 reversal. Under the Phase 2 policy this test asserted the
    opposite — that `/ehr` was buffered because its markers are resolved after the
    turn — and the risk it named is still real: what a caller watches arrive carries
    citations that are unresolved until the citation layer has run, and the settled
    answer may differ. What changed is where that is handled: not by withholding the
    text, but by the swap the terminal frame drives when the two disagree.
    """
    _, model, deltas = _run("summarize", prose_stream=True, payload=SECTIONS)

    assert "".join(d["text"] for d in deltas) == "".join(RISK_PIECES)
    assert model.used == ["astream"]


def test_the_literature_synthesis_streams_too():
    """`/pubmed` is the same case as `/ehr`: the synthesis is the answer, and the
    PMIDs it cites are resolved after the turn."""
    _, model, deltas = _run("literature", prose_stream=True, payload=SECTIONS)

    assert "".join(d["text"] for d in deltas) == "".join(RISK_PIECES)
    assert model.used == ["astream"]


# --- the swap, decided here ----------------------------------------------------

def test_a_settled_verdict_names_the_swap_the_client_owes():
    """The verdict is the server half of the Dual Swap Protocol: it says whether the
    caller is owed a notice, and it is computed here because only this process ran the
    guardrails. A client could see the text move; it could not say why."""
    quiet = {"answer": "served", "guardrail_flags": []}

    # Nothing announced, and announced-but-unchanged: nothing to reconcile.
    assert http._prose_settled([], quiet, "trace-none") is None
    assert http._prose_settled(["serv", "ed"], quiet, "trace-same") is None

    # Moved with nothing claiming responsibility — reformatting, not removal.
    assert http._prose_settled(
        ["served with  [1]"],
        {"answer": "served with [1]", "guardrail_flags": []},
        "trace-silent",
    ) == "silent"

    # Moved and a guardrail owns it: the caller must be told, not quietly corrected.
    assert http._prose_settled(
        ["The score is 0.14"],
        {"answer": "The score is 0.33", "guardrail_flags": ["risk_number"]},
        "trace-visible",
    ) == "visible"


def test_the_settled_verdict_is_logged_with_both_texts(caplog):
    """Both texts, so the swap can be judged from the log alone.

    At info rather than warning: movement is the expected outcome on a streaming path
    now, so a warning would fire on the normal case and stop meaning anything. What is
    worth watching is how often the visible one happens.
    """
    with caplog.at_level("INFO", logger="services.agent.http"):
        http._prose_settled(
            ["The score is 0.14"],
            {"answer": "The score is 0.33", "guardrail_flags": ["risk_number"]},
            "trace-1",
        )

    assert "visible swap" in caplog.text
    assert "trace-1" in caplog.text
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


# --- the free-text turn: what the hold is for ---------------------------------
#
# `agent_node` writes two different things through one call — the deliberation that
# picks a tool, and the answer the reader came for — so a chunk cannot be announced
# when it arrives: it might be either. `graph._streamed_turn` therefore holds the WHOLE
# turn and announces it only once the turn has settled as an answer. These tests are
# that rule stated as behaviour, and they run the helper directly rather than through
# the graph: the node's other work (the stage label, the state it returns) is covered
# where it already was.


def _chunk(text="", tool=None):
    if tool is None:
        return AIMessageChunk(content=text)
    return AIMessageChunk(
        content=text,
        tool_call_chunks=[{"name": tool, "args": "", "id": "call_1", "index": 0}],
    )


class _ChunkModel:
    """Streams a fixed list of chunks, and records that it was asked to stream."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.used: list[str] = []

    async def astream(self, messages, config=None):
        self.used.append("astream")
        for chunk in self.chunks:
            yield chunk


def _streamed(chunks):
    """Run the helper over these chunks; return (accumulated turn, announced text)."""
    events: list[dict] = []
    turn = asyncio.run(graph._streamed_turn(_ChunkModel(chunks), [], events.append))
    text = "".join(
        e["text"] for e in events if e["stage"] == stages.STAGE_ANSWER_DELTA
    )
    return turn, text


def test_an_answer_turn_is_announced_in_full_once_it_has_settled():
    turn, announced = _streamed([
        _chunk("The patient "), _chunk("was admitted "), _chunk("with sepsis."),
    ])

    assert announced == "The patient was admitted with sepsis."
    assert turn.content == "The patient was admitted with sepsis."


def test_a_routing_turn_is_never_announced():
    """A tool call in the opening chunks marks the turn as deliberation, so nothing
    of it reaches the caller and there is no text to withdraw afterwards."""
    turn, announced = _streamed([
        _chunk("I should check the notes first.", tool="search_notes"),
        _chunk(""),
    ])

    assert announced == ""
    assert graph._has_tool_call(turn)


def test_prose_written_BEFORE_a_tool_call_is_never_announced():
    """The reported defect: a paragraph on screen, then the run went back for evidence.

    A follow-up after `/ehr` opened with "I'll check the sections for an inpatient
    diagnosis", called a tool four times, and every paragraph it wrote on the way was
    drawn and then thrown away — the reader watched the answer appear and vanish four
    times in one turn. The old one-chunk lookahead could not cover it: the prose came
    first, so by the time the call arrived the text was already on screen.
    """
    turn, announced = _streamed([
        _chunk("I'll check the sections for "),
        _chunk("an inpatient diagnosis. "),
        _chunk("", tool="rag_search_sections"),
    ])

    assert announced == "", "deliberation reached the caller"
    assert graph._has_tool_call(turn)


def test_an_answer_of_any_length_is_announced():
    """The hold is a gate on ordering, not a reason to swallow a short answer."""
    _, announced = _streamed([_chunk("No evidence was found in the chart.")])

    assert announced == "No evidence was found in the chart."


def test_a_turn_of_empty_chunks_announces_nothing():
    """A turn that never says anything must not put an empty frame on the wire."""
    _, announced = _streamed([_chunk(""), _chunk(""), _chunk("")])

    assert announced == ""


# --- reading the text out of a chunk ------------------------------------------
#
# Gemini answers in content blocks, so `chunk.content` is usually a list. The streaming
# paths announced `str(chunk.content)` and put the envelope on the wire instead of the
# prose. Measured live against `/ehr` on 2026-10-14: 6,116 announced characters against
# a 1,021-character answer. `model_turn.text_of` is the one implementation of the
# unwrapping, and these are its cases.


def test_text_of_reads_a_plain_string():
    assert text_of("The patient was admitted.") == "The patient was admitted."


def test_text_of_reads_content_blocks():
    assert text_of([
        {"type": "text", "text": "Brief Hospital Course\n"},
        {"type": "text", "text": "The patient was admitted.", "index": 0},
    ]) == "Brief Hospital Course\nThe patient was admitted."


def test_text_of_ignores_blocks_that_carry_no_text():
    """A function call is not prose to announce, and neither is a thought signature."""
    assert text_of([
        {"type": "text", "text": "Checking the notes."},
        {"type": "function_call", "name": "rag_search_sections", "args": {}},
    ]) == "Checking the notes."


def test_text_of_stringifies_a_non_dict_entry():
    """The original unwrapping did this, and a plain list of strings depends on it."""
    assert text_of(["one", "two"]) == "onetwo"


def test_text_of_is_empty_for_shapes_it_does_not_know():
    assert text_of(None) == ""
    assert text_of(42) == ""


def test_a_block_shaped_chunk_is_announced_as_prose_not_as_its_repr():
    """The bug that module exists for: the delta is the sentence, not the block."""
    _, announced = _streamed([
        AIMessageChunk(content=[{"type": "text", "text": "The patient "}]),
        AIMessageChunk(content=[{"type": "text", "text": "was admitted."}]),
        AIMessageChunk(content=[{"type": "text", "text": " She recovered."}]),
    ])

    assert announced == "The patient was admitted. She recovered."
    assert "type" not in announced.replace("type-", "")
    assert "{" not in announced
