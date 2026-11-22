"""A corrected answer must say so, on the trace and in the run's own records.

`guardrail_flags` has ridden on the response payload and on the execution log line since
the guardrails existed, and the trace never carried it: on 2026-11-22 a baseline of 32
turns recorded zero flags, and the only way to know whether any answer had been rewritten
before the reader saw it was to join the log to the trace by hand.

Two facts had nowhere to go at all. The comparison pass's own outcome — `verified`,
`verified_empty`, `extraction_degraded` — was translated into a sentence and then
discarded, so a trace held the table and not the reason it looked as it did. The count of
medication rows the extraction withheld travelled no further than the grid's own
`dropped_count`.

These tests pin the three writes, the one ordering constraint that makes them possible
(they are computed before the span closes), and the rule that a clean turn stamps nothing.
"""

from __future__ import annotations

from services.agent import turn
from services.agent.telemetry import observability


class _Span:
    """A span that keeps what it was told, in the order it was told."""

    def __init__(self) -> None:
        self.metadata: list[dict] = []

    def update(self, **kwargs):
        self.metadata.append(kwargs)


def _state(**overrides) -> dict:
    state = {
        "guardrail_flags": [],
        "comparator_outcome": None,
        "command_payload": None,
    }
    state.update(overrides)
    return state


# --- the three facts, computed once ------------------------------------------------


def test_the_guardrails_verdict_and_the_settlement_verdict_are_one_list():
    """Both passes judge the same answer, and the payload names each fact once.

    `sanitize_answer_markers` settles the text and reports what it deleted; the guardrails
    then read that settled text and report their own findings. A turn can produce the same
    flag from both — a deleted marker that left the sentence ungrounded is reported by
    each — and listing it twice reads as two faults.
    """
    signals = turn._turn_signals(
        _state(guardrail_flags=["ungrounded_after_settlement"]),
        {"flags": ["ungrounded_after_settlement", "dose_rewritten"]},
        {},
    )

    assert signals["flags"] == ["ungrounded_after_settlement", "dose_rewritten"]


def test_a_chart_whose_scope_was_withheld_carries_the_flag_only_this_layer_sees():
    signals = turn._turn_signals(
        _state(), {"flags": []}, {"scope_suppressed": True})

    assert signals["flags"] == ["chart_synthesis_without_citations"]


def test_the_withheld_rows_are_counted_from_the_command_payload():
    """The rows the extraction extracted and verification refused.

    They are never drawn — that is the point of refusing them — so the count is the only
    trace that they existed.
    """
    signals = turn._turn_signals(
        _state(command_payload={"medications": [{"medication": "metformin"}],
                                "dropped": [{"medication": "glipizide"},
                                            {"medication": "insulin"}]}),
        {"flags": []}, {},
    )

    assert signals["meds_dropped_count"] == 2


def test_a_turn_with_no_command_payload_withholds_nothing():
    """Every turn that is not a `/meds` turn, including the refusals."""
    assert turn._turn_signals(_state(), {"flags": []}, {})["meds_dropped_count"] == 0


# --- the tags a non-clean turn carries ----------------------------------------------


def test_a_clean_turn_carries_no_outcome_markers():
    tags = turn._signal_tags(
        {"flags": [], "comparator_outcome": "verified", "meds_dropped_count": 0})

    assert tags == []


def test_each_signal_has_its_own_marker():
    tags = turn._signal_tags({
        "flags": ["dose_rewritten"],
        "comparator_outcome": "extraction_degraded",
        "meds_dropped_count": 1,
    })

    assert tags == ["guardrail:rescued", "comparator:degraded", "meds:dropped"]


def test_a_comparison_that_verified_nothing_is_not_a_degraded_one():
    """"Read the abstracts and found no rate" is a finding, not a failure."""
    tags = turn._signal_tags(
        {"flags": [], "comparator_outcome": "verified_empty", "meds_dropped_count": 0})

    assert tags == []


# --- what reaches the span ----------------------------------------------------------


def test_the_span_records_the_three_facts():
    span = _Span()

    observability.record_turn_signals(
        guardrail_flags=["dose_rewritten"], comparator_outcome="extraction_degraded",
        meds_dropped_count=2, span=span,
    )

    (update,) = span.metadata
    assert update["metadata"] == {
        "guardrail_flags": ["dose_rewritten"],
        "comparator_outcome": "extraction_degraded",
        "meds_dropped_count": 2,
    }


def test_the_tools_count_survives_and_the_markers_ride_beside_it():
    """`tools:` is the count the trace drawer has always read; the rest are outcomes.

    The trace-level tag list cannot be appended to after the trace exists, so both land in
    metadata: one under the key it has always used, and the outcome markers under their
    own so a run can count rescued turns without opening every trace.
    """
    span = _Span()

    observability.stamp_trace_tags(
        span, ["tools:3", "guardrail:rescued", "meds:dropped"])

    (update,) = span.metadata
    assert update["metadata"] == {
        "tools": "3",
        "tags": ["guardrail:rescued", "meds:dropped"],
    }


def test_a_turn_with_no_span_records_nothing_and_fails_nothing(monkeypatch):
    """Rule 1: tracing is a sink, never a dependency."""
    monkeypatch.setattr(observability, "current_turn_span", lambda: None)

    observability.record_turn_signals(guardrail_flags=["dose_rewritten"])
    observability.stamp_trace_tags(None, ["guardrail:rescued"])


# --- what reaches the log line ------------------------------------------------------


def test_the_execution_record_carries_the_two_new_fields():
    """The record and its declared field list cannot drift apart.

    The Cloud Logging line is the only per-turn record that survives the process, and a
    field written but not declared is a field no reader knows to look for.
    """
    assert "comparator_outcome" in observability.RECORD_FIELDS
    assert "meds_dropped_count" in observability.RECORD_FIELDS


def test_the_response_fields_report_the_outcome_and_the_withheld_rows():
    state = _state(command_payload={"dropped": [{"medication": "glipizide"}]},
                   comparator_outcome="extraction_degraded")

    fields = turn._response_fields(state)

    assert fields["comparator_outcome"] == "extraction_degraded"
    assert fields["meds_dropped_count"] == 1
