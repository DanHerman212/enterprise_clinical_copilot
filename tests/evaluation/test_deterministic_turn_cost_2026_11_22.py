"""Gate C2, fix 4: a turn that made no model call is priced at nothing, not at "unknown".

A trace with no GENERATION on it used to be one thing only — an unpriceable turn — and the
float billed for it is reported as `unattributed` rather than zero. The outpatient `/ehr`
encounter notice and the foreign-admission refusal are the other thing: they answer with no
model call at all, so their price IS zero and the missing generation is the evidence for it
rather than a gap in the tracing.

The distinction is load-bearing in both directions. Calling a free turn unpriceable throws
away a fact the service recorded, and costs four retries to do it; calling an unpriceable
turn free moves real spend out of the column the arms are compared on. So the flag is
required, and its absence is what keeps the old behaviour.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from evaluation.platform import trace_export


class _TraceApi:
    def __init__(self, trace) -> None:
        self._trace = trace
        self.calls = 0

    def get(self, _trace_id, **_kwargs):
        self.calls += 1
        return self._trace


class _Client:
    def __init__(self, trace) -> None:
        self.api = SimpleNamespace(trace=_TraceApi(trace))


def _trace(*, metadata: dict | None = None, latency: float | None = 2.5):
    observations = []
    if metadata is not None:
        observations.append(SimpleNamespace(metadata=metadata))
    return SimpleNamespace(observations=observations, latency=latency)


@pytest.fixture
def no_sleep(monkeypatch):
    """Record the sleeps rather than serving them, so a retry is visible and instant."""
    slept: list[float] = []
    monkeypatch.setattr(trace_export.time, "sleep", lambda delay: slept.append(delay))
    return slept


def test_a_deterministic_turn_is_priced_at_zero_and_not_retried(no_sleep):
    client = _Client(_trace(metadata={"deterministic_turn": True}))

    usage = trace_export.turn_usage(client, "trace-1", turn=1, attempts=4, delay=1.5)

    assert usage["cost_usd"] == 0.0
    assert usage["unattributed"] is False
    assert usage["reason"] is None
    assert usage["error"] is None
    assert usage["generations"] == 0
    assert usage["models"] == [] and usage["unpriced_models"] == []
    # One read, no sleeping: the absence of a generation is the answer, not a race.
    assert client.api.trace.calls == 1
    assert no_sleep == []


def test_the_zeroed_usage_states_every_count_rather_than_leaving_them_unset():
    """`scenario_usage` sums these straight into an item total, so None would poison it."""
    client = _Client(_trace(metadata={"deterministic_turn": True}))

    usage = trace_export.turn_usage(client, "trace-1", turn=1, attempts=1)

    assert {key: usage[key] for key in ("input", "cached", "output", "cache_write",
                                        "thinking", "total", "billable_input",
                                        "billable_output")} == {
        "input": 0, "cached": 0, "output": 0, "cache_write": 0,
        "thinking": 0, "total": 0, "billable_input": 0, "billable_output": 0,
    }
    assert usage["latency_ms"] == 2500.0
    assert usage["signals"] == {"deterministic_turn": True}


def test_a_zero_generation_trace_without_the_flag_stays_unattributed(no_sleep):
    """An older revision, or a generation that never ingested, is the same shape on the
    wire. Report it unpriceable, and keep retrying in case it lands."""
    client = _Client(_trace(metadata={"guardrail_flags": []}))

    usage = trace_export.turn_usage(client, "trace-1", turn=1, attempts=3, delay=1.5)

    assert usage["cost_usd"] is None
    assert usage["unattributed"] is True
    assert usage["reason"] == "no GENERATION observation in this trace"
    assert client.api.trace.calls == 3
    assert no_sleep == [1.5, 1.5]


def test_a_trace_with_no_metadata_at_all_stays_unattributed(no_sleep):
    client = _Client(_trace(metadata=None))

    usage = trace_export.turn_usage(client, "trace-1", turn=1, attempts=1)

    assert usage["unattributed"] is True
    assert usage["cost_usd"] is None


def test_the_flag_is_false_on_a_turn_that_called_the_model(no_sleep):
    """`deterministic_turn: False` is a claim too — this turn was not free.

    The generation is what prices it, and the false flag must not short-circuit the read.
    """
    generation = SimpleNamespace(
        type="GENERATION", model="gemini-3.8-flash",
        usage_details={"input": 100, "output": 20, "total": 120},
    )
    trace = SimpleNamespace(
        observations=[SimpleNamespace(metadata={"deterministic_turn": False}), generation],
        latency=3.0,
    )
    client = _Client(trace)

    usage = trace_export.turn_usage(client, "trace-1", turn=1, attempts=1)

    assert usage["generations"] == 1
    assert usage["input"] == 100
    assert usage["cost_usd"] is not None
    assert usage["unattributed"] is False
