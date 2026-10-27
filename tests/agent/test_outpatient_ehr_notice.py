"""TF-6: `/ehr` on an outpatient record says so, and the turn's trace is not a dash.

Two defects met on one turn (90000051, `phase04-gate1-canary-20261023-153621`):

1. the formatting turn was handed a record with nothing in it and abstained —
   "I don't have an answer for that" — which named neither the encounter nor what
   follows from it; and
2. because that state carried no answer text, `graph.command_trace_output` returned
   an empty string, so the turn's trace row had no output while every other turn in
   the same run had one.

The fix is in the deterministic path, where the record-wide search already settles
that the absence is real. These tests pin both halves: the answer for an outpatient
record, and a non-empty answer on the state for every exit from a summarize turn.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from services.agent.engine import deterministic
from services.agent.engine import graph
from services.agent.engine import model as model_turn
from services.agent.routing import routing
from services.agent.presentation.a2ui import compose_presentation
from services.agent.engine.deterministic import run_command

SUMMARIZE = routing.command_for("summarize")
OUTPATIENT = sorted(routing.OUTPATIENT_HADM_IDS)


class _EmptyToolbox:
    """Every tool returns no passages — the record-wide fallback included."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def call(self, name, args):
        self.calls.append(name)
        return {"returned": 0, "passages": []}


class _RecordingModel:
    def __init__(self, reply: str = "a summary") -> None:
        self.reply = reply
        self.calls = 0

    def bind_tools(self, *args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("the formatting turn was given tools")

    async def ainvoke(self, messages, config=None):
        self.calls += 1
        from langchain_core.messages import AIMessage

        return AIMessage(content=self.reply)


def _run(hadm_id: int, *, reply: str = "a summary", model: object | None = None):
    box = _EmptyToolbox()
    used = model or _RecordingModel(reply)
    with patch.object(graph, "_build_llm", lambda name: used):
        state = asyncio.run(run_command(
            box, SUMMARIZE, hadm_id, question="/ehr", model="test-model",
        ))
    return state, box, used


@pytest.mark.parametrize("hadm_id", OUTPATIENT)
def test_an_outpatient_record_gets_the_encounter_notice(hadm_id):
    state, _, model = _run(hadm_id)

    assert state["deterministic_answer"] == routing.OUTPATIENT_EHR_NOTICE
    assert model.calls == 0, "an empty record is answered without a model turn"


@pytest.mark.parametrize("hadm_id", OUTPATIENT)
def test_the_notice_says_what_the_record_is_and_what_it_cannot_hold(hadm_id):
    state, _, _ = _run(hadm_id)
    notice = state["deterministic_answer"].lower()

    assert "outpatient encounter" in notice
    assert "no discharge summary" in notice
    # The absences are named, not implied by an abstention.
    assert "no hospital course" in notice
    assert "no discharge summary to review" in notice
    # And no stay is asserted as existing.
    assert "should be read as an inpatient stay" in notice


@pytest.mark.parametrize("hadm_id", OUTPATIENT)
def test_the_notice_is_not_verified_against_the_chart(hadm_id):
    state, _, _ = _run(hadm_id)
    out = compose_presentation("/ehr", state["deterministic_answer"],
                               state["tool_calls"], None, "summarize")

    assert out["verified"] is False
    assert out["scope"] is None


@pytest.mark.parametrize("hadm_id", OUTPATIENT)
def test_the_turns_trace_output_is_not_empty(hadm_id):
    """The trace row reads its output off the state; an empty answer leaves a dash."""
    state, _, _ = _run(hadm_id)

    assert graph.command_trace_output(state).strip() == routing.OUTPATIENT_EHR_NOTICE


def test_the_record_wide_search_still_runs_before_the_notice():
    """The notice is only correct once the record itself has been searched: an empty
    sectioned result alone cannot tell an outpatient note from an inpatient one whose
    sections were not found."""
    _, box, _ = _run(OUTPATIENT[0])

    assert "rag_search_sections" in box.calls
    assert "rag_search" in box.calls


def test_an_inpatient_empty_record_still_reaches_the_formatter():
    """The gate is scoped to outpatient records; an inpatient summary is the model's."""
    state, _, model = _run(90000015)

    assert model.calls == 1
    # No deterministic answer: the model's reply is the answer, and it is not empty.
    assert "deterministic_answer" not in state


def test_a_formatter_that_says_nothing_leaves_a_readable_answer():
    state, _, model = _run(90000015, reply="")

    assert model.calls == 1
    assert state["deterministic_answer"] == model_turn.SILENT_TURN_SENTENCE
    assert graph.command_trace_output(state).strip() == model_turn.SILENT_TURN_SENTENCE


def test_an_outpatient_formatter_that_says_nothing_uses_the_encounter_wording():
    """`/labs` and `/pubmed` still reach the model on an outpatient record; if it
    abstains, the sentence must describe an encounter, not an admission."""
    labs = routing.command_for("labs")
    box = _EmptyToolbox()
    used = _RecordingModel(reply="")
    with patch.object(graph, "_build_llm", lambda name: used):
        state = asyncio.run(run_command(
            box, labs, OUTPATIENT[0], question="/labs", model="test-model",
        ))

    assert state["deterministic_answer"] == model_turn.OUTPATIENT_SILENT_TURN_SENTENCE
    assert graph.command_trace_output(state).strip()


class _PassageToolbox:
    """The sectioned search finds something, so there is a summary to write."""

    async def call(self, name, args):
        return {"returned": 1, "passages": [{"text": "Discharged home in stable "
                                                    "condition.", "id": "p1"}]}


def test_an_outpatient_record_with_a_note_is_still_summarised():
    """The gate is empty-only: a record that has text keeps the formatting turn."""
    used = _RecordingModel(reply="**Hospital course.** A short visit.")
    with patch.object(graph, "_build_llm", lambda name: used):
        state = asyncio.run(run_command(
            _PassageToolbox(), SUMMARIZE, OUTPATIENT[0], question="/ehr",
            model="test-model",
        ))

    assert used.calls == 1
    assert "deterministic_answer" not in state
