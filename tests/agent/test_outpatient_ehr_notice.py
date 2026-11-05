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
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from services.agent.engine import deterministic
from services.agent.engine import model as model_gateway
from services.agent.engine import graph
from services.agent.engine import model as model_turn
from services.agent.routing import encounter, routing
from services.agent.postprocess import answer
from services.agent.presentation.a2ui import compose_presentation
from services.agent.engine.deterministic import run_command

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluation.outpatient_oracle import OUTPATIENT_HADM_IDS  # noqa: E402

SUMMARIZE = routing.command_for("summarize")
OUTPATIENT = sorted(OUTPATIENT_HADM_IDS)


def _classify(hadm_id: int) -> None:
    """What the pre-flight records before the command runs.

    These tests enter `run_command` directly, skipping `admission_context.hydrate`, so
    the encounter type has to be set here — which is the point of the ContextVar being
    request-scoped: the value the command gate reads is the one the pre-flight wrote.
    """
    encounter.set_current(
        encounter.OUTPATIENT if hadm_id in OUTPATIENT_HADM_IDS else encounter.INPATIENT
    )


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
    _classify(hadm_id)
    with patch.object(model_gateway, "build_llm", lambda name: used):
        state = asyncio.run(run_command(
            box, SUMMARIZE, hadm_id, question="/ehr", model="test-model",
        ))
    return state, box, used


@pytest.mark.parametrize("hadm_id", OUTPATIENT)
def test_an_outpatient_record_gets_the_encounter_notice(hadm_id):
    state, _, model = _run(hadm_id)

    assert state["deterministic_answer"] == answer.OUTPATIENT_EHR_NOTICE
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
def test_the_notice_is_verified_against_the_encounter(hadm_id):
    """The refusal rests on the record, so it reports as checked against the record.

    Until 2026-11-05 this test asserted the opposite (`verified is False`, `scope is
    None`) and it described the code correctly: the notice cites no passage, so the
    composer read it as synthesis nobody could vouch for and withdrew the badge. That is
    the wrong ruling for this answer. `encounter: outpatient` came from the pre-flight
    manifest — the row the readmission model was scored on — and the sectioned search
    that established the record has nothing to summarise is on the turn. A refusal to
    describe a stay is the correct answer to `/ehr` on a visit, and it is grounded.

    The dataset says the same thing in `dev_t7_01`'s behaviour note: *"It reports
    `verified: true`. That flag says the answer was checked against this patient's
    record; it does not say an inpatient risk was verified, and none was."*
    """
    state, _, _ = _run(hadm_id)

    assert state["manifest_answered"] is True
    out = compose_presentation(
        "/ehr", state["deterministic_answer"], state["tool_calls"], None, "summarize",
        context_grounded=bool(state.get("manifest_answered")),
    )

    assert out["verified"] is True
    assert out["scope"] == "patient_chart"
    assert out["scope_suppressed"] is False


@pytest.mark.parametrize("hadm_id", OUTPATIENT)
def test_the_turns_trace_output_is_not_empty(hadm_id):
    """The trace row reads its output off the state; an empty answer leaves a dash."""
    state, _, _ = _run(hadm_id)

    assert graph.command_trace_output(state).strip() == answer.OUTPATIENT_EHR_NOTICE


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
    assert state["deterministic_answer"] == answer.SILENT_TURN_SENTENCE
    assert graph.command_trace_output(state).strip() == answer.SILENT_TURN_SENTENCE


def test_an_outpatient_formatter_that_says_nothing_uses_the_encounter_wording():
    """`/labs` and `/pubmed` still reach the model on an outpatient record; if it
    abstains, the sentence must describe an encounter, not an admission."""
    labs = routing.command_for("labs")
    box = _EmptyToolbox()
    used = _RecordingModel(reply="")
    _classify(OUTPATIENT[0])
    with patch.object(model_gateway, "build_llm", lambda name: used):
        state = asyncio.run(run_command(
            box, labs, OUTPATIENT[0], question="/labs", model="test-model",
        ))

    assert state["deterministic_answer"] == answer.OUTPATIENT_SILENT_TURN_SENTENCE
    assert graph.command_trace_output(state).strip()


class _PassageToolbox:
    """The sectioned search finds something, so there is a summary to write."""

    async def call(self, name, args):
        return {"returned": 1, "passages": [{"text": "Discharged home in stable "
                                                    "condition.", "id": "p1"}]}


def test_an_outpatient_record_with_a_note_gets_the_notice_not_a_summary():
    """The gate is the RECORD, not the emptiness of the retrieval.

    Until 2026-11-05 this test asserted the opposite, and it was right about the code and
    wrong about the contract: `/ehr` on an outpatient record with one readable passage
    reached the formatter, which `EHR_FORMATTING` had been told to answer under a renamed
    heading, and the rubric calls a renamed heading a responsiveness defect. A command's
    answer shape is not the model's to renegotiate, so the branch is decided on the
    encounter and the notice is served whatever the retrieval returned (`dev_t7_01`).

    The retrieval still happened and its call is still on the state: the turn retrieved
    something, and a trace that could not show what would be a turn an audit cannot follow.
    """
    used = _RecordingModel(reply="**Clinical course.** A short visit.")
    _classify(OUTPATIENT[0])
    with patch.object(model_gateway, "build_llm", lambda name: used):
        state = asyncio.run(run_command(
            _PassageToolbox(), SUMMARIZE, OUTPATIENT[0], question="/ehr",
            model="test-model",
        ))

    assert used.calls == 0, "an outpatient /ehr answer is composed, not written"
    assert state["deterministic_answer"] == answer.OUTPATIENT_EHR_NOTICE
    assert state["command"] == "summarize"
    assert state["tool_calls"], "the retrieval stays on the state"


def test_an_inpatient_record_with_a_note_still_reaches_the_formatter():
    """The counterpart, so the gate above is a rule about encounters rather than about
    records that happen to have one passage."""
    used = _RecordingModel(reply="**Presentation & diagnosis.** A short stay.")
    _classify(90000015)
    with patch.object(model_gateway, "build_llm", lambda name: used):
        state = asyncio.run(run_command(
            _PassageToolbox(), SUMMARIZE, 90000015, question="/ehr",
            model="test-model",
        ))

    assert used.calls == 1
    assert "deterministic_answer" not in state
