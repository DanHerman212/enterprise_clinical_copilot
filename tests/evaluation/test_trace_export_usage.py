"""Cost is read from the traces, and an unread trace is not a free one.

The agent's wire contract carries no token counts and Sprint 03 is not allowed to add
them, so every price in a run comes from the GENERATION observations the tracer recorded
for that turn. Three things therefore have to be true, and each one is a way the report
could lie:

* every generation in a turn counts, because a turn is several model calls — a command's
  extraction, each reasoning round, the formatting turn — and summing one of them would
  understate the bill by a factor that varies per model;
* every turn in a scenario counts, because a two-turn item made two requests;
* a trace that cannot be read, or holds nothing to price, reports `None` and is counted
  unattributed. A zero would be indistinguishable from a free turn in the column the
  arms are compared on.
"""

from __future__ import annotations

import pytest

from evaluation.platform import pricing, trace_export


class _Observation:
    def __init__(self, kind: str, model: str, usage: dict | None) -> None:
        self.type = kind
        self.model = model
        self.usage_details = usage


class _Trace:
    def __init__(self, observations: list[_Observation]) -> None:
        self.observations = observations


class _TraceApi:
    def __init__(self, traces: dict[str, _Trace | Exception]) -> None:
        self._traces = traces
        self.reads: list[str] = []

    def get(self, trace_id: str, **_kwargs):
        self.reads.append(trace_id)
        found = self._traces.get(trace_id)
        if isinstance(found, Exception):
            raise found
        if found is None:
            raise KeyError(trace_id)
        return found


class _Api:
    def __init__(self, traces: dict[str, _Trace | Exception]) -> None:
        self.trace = _TraceApi(traces)


class _Client:
    def __init__(self, traces: dict[str, _Trace | Exception]) -> None:
        self.api = _Api(traces)


def _generation(model: str, *, tokens_in: int, tokens_out: int,
                cached: int | None = None, thinking: int | None = None) -> _Observation:
    usage = {"input": tokens_in, "output": tokens_out}
    if cached is not None:
        usage["cached"] = cached
    if thinking is not None:
        usage["thinking"] = thinking
    return _Observation("GENERATION", model, usage)


FLASH = "gemini-3.1-flash-lite"
RATE = pricing.rate_for(FLASH)


def test_every_generation_of_every_turn_is_summed(monkeypatch):
    """A 2-turn scenario: 3 ReAct generations in turn 1, 2 in turn 2.

    This is the shape the harness actually produces: `/ehr` runs a command and a loop,
    and the second turn adds its own calls on top. Summing the root observation, or the
    first generation, would report less than half the bill.
    """
    monkeypatch.setattr(trace_export.time, "sleep", lambda _s: None)
    client = _Client({
        "trace-1": _Trace([
            _generation(FLASH, tokens_in=1000, tokens_out=100),
            _generation(FLASH, tokens_in=2000, tokens_out=200, thinking=150),
            _generation(FLASH, tokens_in=3000, tokens_out=300),
        ]),
        "trace-2": _Trace([
            _generation(FLASH, tokens_in=400, tokens_out=40),
            _generation(FLASH, tokens_in=500, tokens_out=50),
        ]),
    })

    usage = trace_export.scenario_usage(
        client, {"turn_trace_ids": ["trace-1", "trace-2"]})

    assert usage["turn_count"] == 2
    assert usage["generations"] == 5
    assert usage["input"] == 6900
    assert usage["output"] == 690
    assert usage["thinking"] == 150
    expected = (6900 * RATE.input + 690 * RATE.output) / 1e6
    assert usage["cost_usd"] == pytest.approx(expected)
    assert usage["unattributed"] is False
    assert [turn["turn"] for turn in usage["turns"]] == [1, 2]


def test_the_thinking_sub_count_does_not_change_the_price(monkeypatch):
    """Same totals, once with reasoning reported and once without."""
    monkeypatch.setattr(trace_export.time, "sleep", lambda _s: None)
    with_thinking = _Client({"t": _Trace([
        _generation(FLASH, tokens_in=100, tokens_out=200, thinking=180)])})
    without = _Client({"t": _Trace([_generation(FLASH, tokens_in=100, tokens_out=200)])})

    a = trace_export.scenario_usage(with_thinking, {"turn_trace_ids": ["t"]})
    b = trace_export.scenario_usage(without, {"turn_trace_ids": ["t"]})

    assert a["cost_usd"] == pytest.approx(b["cost_usd"])
    assert a["thinking"] == 180
    assert b["thinking"] is None


def test_a_cached_read_is_priced_at_the_cached_rate(monkeypatch):
    monkeypatch.setattr(trace_export.time, "sleep", lambda _s: None)
    client = _Client({"t": _Trace([
        _generation(FLASH, tokens_in=10_000, tokens_out=0, cached=8_000)])})

    usage = trace_export.scenario_usage(client, {"turn_trace_ids": ["t"]})

    assert usage["cached"] == 8_000
    assert usage["cost_usd"] == pytest.approx(
        (2_000 * RATE.input + 8_000 * RATE.cached_read) / 1e6)


def test_span_observations_are_not_priced(monkeypatch):
    """A SPAN has no usage; only a GENERATION does."""
    monkeypatch.setattr(trace_export.time, "sleep", lambda _s: None)
    client = _Client({"t": _Trace([
        _Observation("SPAN", FLASH, {"input": 5000, "output": 5000}),
        _Observation("EVENT", FLASH, None),
        _generation(FLASH, tokens_in=10, tokens_out=10)])})

    usage = trace_export.scenario_usage(client, {"turn_trace_ids": ["t"]})

    assert usage["generations"] == 1
    assert usage["input"] == 10


def test_a_missing_trace_id_is_unattributed_not_zero(monkeypatch):
    monkeypatch.setattr(trace_export.time, "sleep", lambda _s: None)
    usage = trace_export.scenario_usage(_Client({}), {"turn_trace_ids": [""]})

    assert usage["cost_usd"] is None
    assert usage["unattributed"] is True
    assert "no trace id" in (usage["reason"] or "")
    assert usage["turns"][0]["error"] == "no trace id for this turn"


def test_an_unreadable_trace_is_unattributed_not_zero(monkeypatch):
    monkeypatch.setattr(trace_export.time, "sleep", lambda _s: None)
    client = _Client({"t": RuntimeError("store timed out")})

    usage = trace_export.scenario_usage(client, {"turn_trace_ids": ["t"]},
                                        attempts=2)

    assert usage["cost_usd"] is None
    assert usage["unattributed"] is True
    assert "unreadable" in (usage["reason"] or "")


def test_a_trace_with_no_generations_is_unattributed_not_zero(monkeypatch):
    """Read successfully, held nothing to price — a fact about tracing, not a price."""
    monkeypatch.setattr(trace_export.time, "sleep", lambda _s: None)
    client = _Client({"t": _Trace([_Observation("SPAN", "", None)])})

    usage = trace_export.scenario_usage(client, {"turn_trace_ids": ["t"]},
                                        attempts=2)

    assert usage["cost_usd"] is None
    assert "no GENERATION observation" in (usage["reason"] or "")


def test_ingestion_lag_is_polled_before_giving_up(monkeypatch):
    """A read straight after the answer can precede the generation observations."""

    class _Late(_Trace):
        def __init__(self) -> None:
            super().__init__([])
            self.calls = 0

    late = _Late()
    sleeps: list[float] = []
    monkeypatch.setattr(trace_export.time, "sleep", sleeps.append)

    class _LateApi(_TraceApi):
        def get(self, trace_id, **_kwargs):
            late.calls += 1
            if late.calls < 3:
                return _Trace([])
            late.observations = [_generation(FLASH, tokens_in=10, tokens_out=10)]
            return late

    client = _Client({})
    client.api.trace = _LateApi({})

    usage = trace_export.scenario_usage(client, {"turn_trace_ids": ["t"]},
                                        attempts=4, delay=1.5)

    assert late.calls == 3
    assert sleeps == [1.5, 1.5]
    assert usage["cost_usd"] is not None


def test_one_unpriced_model_makes_the_scenario_unattributed(monkeypatch):
    """Half a price is worse than no price: it reads as a cheap item."""
    monkeypatch.setattr(trace_export.time, "sleep", lambda _s: None)
    client = _Client({"t": _Trace([
        _generation(FLASH, tokens_in=100, tokens_out=100),
        _generation("gemini-9-imaginary", tokens_in=100, tokens_out=100)])})

    usage = trace_export.scenario_usage(client, {"turn_trace_ids": ["t"]})

    assert usage["cost_usd"] is None
    assert usage["unattributed"] is True
    assert usage["turns"][0]["unpriced_models"] == ["gemini-9-imaginary"]


def test_a_generation_with_no_model_id_is_unattributed(monkeypatch):
    monkeypatch.setattr(trace_export.time, "sleep", lambda _s: None)
    client = _Client({"t": _Trace([_generation("", tokens_in=100, tokens_out=100)])})

    usage = trace_export.scenario_usage(client, {"turn_trace_ids": ["t"]})

    assert usage["cost_usd"] is None
    assert usage["turns"][0]["unpriced_models"] == ["(unreported)"]


def test_two_models_in_one_scenario_are_each_priced_at_their_own_rate(monkeypatch):
    monkeypatch.setattr(trace_export.time, "sleep", lambda _s: None)
    client = _Client({"t": _Trace([
        _generation(FLASH, tokens_in=1000, tokens_out=100),
        _generation("xai/grok-4.7", tokens_in=1000, tokens_out=100)])})

    usage = trace_export.scenario_usage(client, {"turn_trace_ids": ["t"]})

    assert usage["models"] == ["gemini-3.1-flash-lite", "xai/grok-4.7"]
    assert usage["cost_usd"] == pytest.approx(
        (1000 * RATE.input + 100 * RATE.output) / 1e6
        + (1000 * 2.00 + 100 * 6.00) / 1e6)
