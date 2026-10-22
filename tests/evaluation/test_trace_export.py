"""A run's traces, read back turn by turn with their spans.

The record a reviewer needs is the SPAN ROWS: the root's input and output, then one
observation per tool call with what it was given and what it returned. The trace row
alone says only that a turn happened. Section 4 of the phase specification is explicit
that the export labels each record with its turn index, because the two halves of a
thread are shaped identically and which one produced a defect is the first question
asked of them.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from evaluation.platform.trace_export import export_item, export_turn, trace_ids_for_turns

TURN_ONE = "a" * 32
TURN_TWO = "b" * 32


def _trace(name: str, *, spans: int = 1, answer: str = "ok"):
    observations = [
        SimpleNamespace(id=f"obs-{index}", name=f"step-{index}", type="SPAN",
                        input={"step": index}, output={"result": index},
                        level=None, status_message=None, start_time="t0",
                        end_time="t1")
        for index in range(spans)
    ]
    return SimpleNamespace(name=name, input="the question", output=answer,
                           latency=1.5, observations=observations,
                           scores=[SimpleNamespace(name="verdict", value=1.0,
                                                   comment="pass")])


class _Client:
    def __init__(self, traces: dict[str, object], *, fail: set[str] | None = None):
        self.api = self
        self.trace = self
        self._traces = traces
        self._fail = fail or set()

    def get(self, trace_id, request_options=None):
        if trace_id in self._fail:
            raise RuntimeError("gateway timeout")
        return self._traces[trace_id]


def test_a_turn_record_carries_its_spans_and_its_scores():
    client = _Client({TURN_ONE: _trace("free_text.turn1", spans=2)})

    record = export_turn(client, TURN_ONE, turn=1)

    assert record["turn"] == 1
    assert record["input"] == "the question"
    assert record["output"] == "ok"
    assert [span["name"] for span in record["spans"]] == ["step-0", "step-1"]
    assert record["spans"][1]["input"] == {"step": 1}
    assert record["spans"][1]["output"] == {"result": 1}
    assert record["scores"] == [{"name": "verdict", "value": 1.0, "comment": "pass"}]


def test_every_turn_of_an_item_is_exported_and_labelled():
    client = _Client({TURN_ONE: _trace("turn1"), TURN_TWO: _trace("turn2")})

    records = export_item(client, {"turn_trace_ids": [TURN_ONE, TURN_TWO]})

    assert [record["turn"] for record in records] == [1, 2]
    assert [record["trace_id"] for record in records] == [TURN_ONE, TURN_TWO]


def test_a_turn_with_no_trace_id_is_reported_rather_than_dropped():
    """Half a thread is a finding. Omitting it from the export hides the finding."""
    client = _Client({TURN_ONE: _trace("turn1")})

    records = export_item(client, {"turn_trace_ids": [TURN_ONE, ""]})

    assert records[1] == {"turn": 2, "trace_id": "",
                          "error": "no trace id for this turn"}


def test_an_unreadable_trace_does_not_lose_the_export():
    """A timeout is not evidence about the trace, so the run keeps going."""
    client = _Client({TURN_ONE: _trace("turn1")}, fail={TURN_TWO})

    records = export_item(client, {"turn_trace_ids": [TURN_ONE, TURN_TWO]})

    assert records[0]["name"] == "turn1"
    assert records[1]["error"] == "unreadable: RuntimeError"


def test_an_artifact_recorded_before_the_list_existed_still_exports():
    assert trace_ids_for_turns({"langfuse_trace_id": TURN_ONE}) == [TURN_ONE]


@pytest.mark.parametrize("output", [{}, {"turn_trace_ids": []}])
def test_no_ids_at_all_yields_one_empty_turn(output):
    assert trace_ids_for_turns(output) == [""]
