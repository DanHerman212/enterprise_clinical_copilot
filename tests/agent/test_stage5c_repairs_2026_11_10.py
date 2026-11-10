"""The three repairs Gate 2 of the 2026-11-10 run asked for.

Each of these is a live failure that a unit test can hold still:

  * a driver answer said "the model does not provide a specific numeric contribution"
    over a payload carrying one, because the projection withheld it (`mcp_client`);
  * a turn that had retrieved the discharge list, or retrieved passages, then declined,
    and the reader was told the chart did not cover the question (`graph`);
  * a refusal about another admission was served with an empty trace id, so the harness
    counted the turn as unattributed (`graph`).

The fourth failure of that run — the prompt forbidding the very absence the record
established — has no unit here beyond the contract tests in `test_prompt_contract.py`;
it is a wording defect whose proof is the re-run.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from services.agent.engine import graph
from services.agent.postprocess import answer as answer_text
from services.agent.postprocess.evidence_state import SessionEvidenceState

ADMISSION = 90000067


# --- one retry when a decline contradicts the record the session holds ---------------

class _StubModel:
    """A model that answers with whatever it was told to, counting its calls."""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[list] = []

    async def ainvoke(self, messages, *_args, **_kwargs):
        self.calls.append(list(messages))
        return AIMessage(content=self.reply)


def _declined_state(*, record: SessionEvidenceState | None, text: str = '{"evidence_found": false}'):
    return {
        "messages": [HumanMessage(content="What is she going home on?"),
                     AIMessage(content=text)],
        "tool_calls": [{"name": "discharge_medications", "args": {"hadm_id": ADMISSION},
                        "response": {"hadm_id": ADMISSION, "passages": [],
                                     "dedicated_section_present": True}}],
        "replayed_tool_calls": [],
        "question": "What is she going home on?",
        "allowed_tools": frozenset({"discharge_medications"}),
        "retrieval_demanded": False,
        "tool_attempts": 1,
        "evidence_state": record,
    }


def _record_with_an_estimate() -> SessionEvidenceState:
    return SessionEvidenceState(
        hadm_id=ADMISSION,
        prediction={"hadm_id": ADMISSION, "probability": 0.112419,
                    "threshold": 0.11, "top_factors": []},
    )


def test_a_decline_over_a_record_holding_evidence_is_asked_once_more(monkeypatch):
    """The record IS the answer's evidence, so the decline is checkable and wrong."""
    stub = _StubModel("She is going home on metoprolol and furosemide.")
    monkeypatch.setattr(graph.model_gateway, "build_llm", lambda *_a, **_k: stub)
    state = _declined_state(record=_record_with_an_estimate())

    corrected = asyncio.run(graph._correct_invalid_decline(
        state, model="stub", allowed=frozenset({"discharge_medications"})))

    assert corrected == "She is going home on metoprolol and furosemide."
    assert len(stub.calls) == 1, "one retry, not a loop"
    assert graph.final_text(state) == corrected, (
        "the corrected turn has to be the message the settlement and the trace read")


def test_the_retry_is_told_which_object_is_unavailable_and_why(monkeypatch):
    stub = _StubModel("She is going home on metoprolol.")
    monkeypatch.setattr(graph.model_gateway, "build_llm", lambda *_a, **_k: stub)
    state = _declined_state(record=_record_with_an_estimate())

    asyncio.run(graph._correct_invalid_decline(
        state, model="stub", allowed=frozenset({"discharge_medications"})))

    instruction = stub.calls[0][-1]
    assert isinstance(instruction, HumanMessage)
    flattened = " ".join(instruction.content.split())
    assert '{"evidence_found": false}' in flattened
    assert "already run" in flattened, (
        "the retry must say the retrieval has happened, or the model searches again")
    assert "Do not return the object again" in flattened


def test_the_retry_is_unbound(monkeypatch):
    """What is missing is not a retrieval: the results are already in the conversation.

    `_StubModel` has no `bind_tools`, so a bound retry would raise rather than answer.
    """
    stub = _StubModel("A prose answer.")
    built: list = []

    def _build(*args, **kwargs):
        built.append(args)
        return stub

    monkeypatch.setattr(graph.model_gateway, "build_llm", _build)
    state = _declined_state(record=_record_with_an_estimate())

    corrected = asyncio.run(graph._correct_invalid_decline(
        state, model="stub", allowed=frozenset({"discharge_medications"})))

    assert corrected == "A prose answer."
    assert not hasattr(_StubModel, "bind_tools")
    assert len(built) == 1


def test_a_second_decline_is_honoured(monkeypatch):
    """A record can be present and still not answer the question."""
    stub = _StubModel('{"evidence_found": false}')
    monkeypatch.setattr(graph.model_gateway, "build_llm", lambda *_a, **_k: stub)
    state = _declined_state(record=_record_with_an_estimate())
    before = graph.final_text(state)

    corrected = asyncio.run(graph._correct_invalid_decline(
        state, model="stub", allowed=frozenset({"discharge_medications"})))

    assert corrected is None
    assert graph.final_text(state) == before, "a failed retry leaves the turn as it was"


def test_the_retry_waits_for_the_record_to_have_something_in_it():
    """A cold turn declines against nothing, which is not a contradiction."""
    assert graph._record_has_content(SessionEvidenceState(hadm_id=ADMISSION)) is False
    assert graph._record_has_content(_record_with_an_estimate()) is True


def test_the_decline_object_is_read_as_a_decline_and_prose_is_not():
    """The trigger is the object, not any short reply: a withheld turn has no text."""
    assert graph.read_evidence_reply('{"evidence_found": false}').declined is True
    assert graph.read_evidence_reply("").declined is True
    assert graph.read_evidence_reply("Metoprolol.").declined is False


# --- a refusal is a turn, and a turn has a trace -----------------------------------

class _Span:
    def __init__(self):
        self.tags: list[str] = []
        self.updates: list[dict] = []

    def update(self, metadata=None, **_kwargs):
        self.updates.append(dict(metadata or {}))


def test_a_foreign_admission_refusal_carries_its_trace_id_and_tags(monkeypatch):
    """`dev_t7_03` turn 2 was served with `""` and counted as unattributed."""
    span = _Span()
    tags: list[tuple] = []

    @contextlib.contextmanager
    def _observation(_name, **_kwargs):
        yield span

    monkeypatch.setattr(graph.observability, "observation", _observation)
    monkeypatch.setattr(graph.observability, "span_trace_id",
                        lambda target: "trace-refusal" if target is not None else None)
    monkeypatch.setattr(graph.observability, "current_trace_id", lambda: "")
    monkeypatch.setattr(graph.observability, "current_turn_span", lambda: None)
    monkeypatch.setattr(graph.observability, "stamp_trace_tags",
                        lambda target, values: tags.append((target, list(values))))

    out = asyncio.run(graph.ask(
        object(), "What did admission 90000099 document?", hadm_id=ADMISSION))

    assert out["deterministic_answer"] == answer_text.foreign_admission_refusal()
    assert out["langfuse_trace_id"] == "trace-refusal"
    stamped = tags[0][1]
    assert "tools:0" in stamped
    assert "foreign_admission_refused" in stamped


def test_the_refusal_is_still_not_answered_by_the_model(monkeypatch):
    """No model call, no tool call: the refusal is decided before either."""
    def _explode(*_args, **_kwargs):
        raise AssertionError("a foreign-admission refusal must not reach the model")

    monkeypatch.setattr(graph, "build_graph", _explode)

    out = asyncio.run(graph.ask(
        object(), "What did admission 90000099 document?", hadm_id=ADMISSION))

    assert out["tool_calls"] == []
    assert out["deterministic_answer"] == answer_text.foreign_admission_refusal()
