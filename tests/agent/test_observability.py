"""The tracing sink: off by default, and unable to fail a run.

The failure these tests exist for is ECC-17. With tracing enabled, a run read an
attribute the callback handler did not expose and raised *after* the model and
tool spend had completed, so the only configuration that could fail was the one
dev never exercised — a 502 that cost a full answer's spend. The rule that came
out of it is that tracing degrades to "no trace", never to a failed request, and
these tests hold each entry point to that.

The other property worth pinning is the shape of the off state: no keys means no
handler and no id, and it must be silent. An unconfigured environment is normal
(dev runs, the site's fixture mode, CI) and must not spam warnings.
"""

from __future__ import annotations

import logging

import pytest

from services.agent import observability

KEYS = {
    "LANGFUSE_HOST": "https://observability.example.invalid",
    "LANGFUSE_PUBLIC_KEY": "pk-test",
    "LANGFUSE_SECRET_KEY": "sk-test",
}


@pytest.fixture
def no_keys(monkeypatch):
    for name in KEYS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def with_keys(monkeypatch):
    for name, value in KEYS.items():
        monkeypatch.setenv(name, value)


def test_unconfigured_is_off_without_a_handler(no_keys):
    """No keys: no handler, and nothing to explain.

    `no_keys` is explicit rather than implied by the ambient environment. It was
    implicit until 2026-10-17, when a shell that had exported the real credentials
    for a live run made this test fail — the test was asserting a property of the
    process it happened to be started in, not of the code.
    """
    assert observability.handler() is None


def test_a_partial_configuration_is_not_configured(no_keys, monkeypatch):
    """Two of three variables is a misconfiguration, not a preference.

    Half-configured tracing is worse than none: the SDK initialises and ships
    nowhere, so traces vanish while every signal says tracing is on.
    """
    monkeypatch.setenv("LANGFUSE_HOST", KEYS["LANGFUSE_HOST"])
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", KEYS["LANGFUSE_PUBLIC_KEY"])

    assert observability.configured() is False
    assert observability.handler() is None


def test_blank_values_count_as_absent(no_keys, monkeypatch):
    monkeypatch.setenv("LANGFUSE_HOST", "  ")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", KEYS["LANGFUSE_PUBLIC_KEY"])
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", KEYS["LANGFUSE_SECRET_KEY"])

    assert observability.configured() is False


def test_the_trace_id_is_empty_when_tracing_is_off(no_keys):
    assert observability.trace_id() == ""
    assert observability.trace_id(object()) == ""


def test_a_handler_without_the_attribute_yields_no_id(with_keys, monkeypatch):
    """The ECC-17 case, reduced: the attribute the SDK used to expose is gone."""

    class HandlerWithoutTraceId:
        pass

    class FakeClient:
        def get_current_trace_id(self):
            raise RuntimeError("no current trace outside a run")

    monkeypatch.setattr("langfuse.get_client", lambda: FakeClient())

    assert observability.trace_id(HandlerWithoutTraceId()) == ""


def test_the_client_fallback_is_used_when_the_handler_has_no_id(with_keys, monkeypatch):
    class BareHandler:
        pass

    class FakeClient:
        def get_current_trace_id(self):
            return "trace-from-client"

    monkeypatch.setattr("langfuse.get_client", lambda: FakeClient())

    assert observability.trace_id(BareHandler()) == "trace-from-client"


def test_the_handlers_own_id_wins(with_keys):
    class Handler:
        last_trace_id = "trace-from-handler"

    assert observability.trace_id(Handler()) == "trace-from-handler"


def test_a_raising_handler_does_not_propagate(with_keys):
    class Exploding:
        @property
        def last_trace_id(self):
            raise RuntimeError("attribute exploded")

    # The fallback still runs, so this returns "" rather than raising.
    assert observability.trace_id(Exploding()) in ("", "trace")


def test_building_a_handler_cannot_raise(with_keys, monkeypatch):
    """If the SDK is broken or absent, the run continues without tracing."""

    def boom(*args, **kwargs):
        raise ImportError("langfuse.langchain is not installed")

    monkeypatch.setattr("langfuse.langchain.CallbackHandler", boom, raising=False)

    assert observability.handler() is None


def test_flush_is_a_no_op_without_configuration(no_keys):
    observability.flush()  # must not raise


def test_flush_failure_does_not_propagate(with_keys, monkeypatch):
    class BadClient:
        def flush(self):
            raise RuntimeError("network down")

    monkeypatch.setattr("langfuse.get_client", lambda: BadClient())

    observability.flush()  # logged, not raised


def test_the_off_state_is_silent(no_keys, caplog):
    """Dev runs, fixture mode and CI all run unconfigured. That is normal."""
    with caplog.at_level(logging.WARNING, logger="services.agent.observability"):
        observability.handler()
        observability.trace_id()
        observability.flush()

    assert caplog.records == []


# --- span I/O: a span that names a step and records nothing about it ----------


class _RecordingSpan:
    """A span that keeps what was written to it."""

    def __init__(self):
        self.updates: list[dict] = []

    def update(self, **fields):
        self.updates.append(fields)


class _RecordingClient:
    """The client, recording both ends: the trace row and the span it hands back."""

    def __init__(self, span=None):
        self.io: list[dict] = []
        self.opened: list[dict] = []
        self._span = span or _RecordingSpan()

    def set_current_trace_io(self, **fields):
        self.io.append(fields)

    def start_as_current_observation(self, **fields):
        self.opened.append(fields)
        span = self._span

        class _Manager:
            def __enter__(self_inner):
                return span

            def __exit__(self_inner, *exc):
                return False

        return _Manager()


def test_trace_io_reaches_the_trace_row_and_the_root_span(with_keys, monkeypatch):
    """The defect this closes: the table row had both ends and the root span had none.

    `set_current_trace_io` fills the columns in the trace list. The root observation is
    a different record and was left with `input: null, output: null`, so an exported
    span tree — the artefact root-cause analysis actually reads — began with nothing.
    """
    client = _RecordingClient()
    monkeypatch.setattr("langfuse.get_client", lambda: client)
    span = _RecordingSpan()

    observability.set_trace_io("what is her risk?", "16.7%", span=span)

    assert client.io == [{"input": "what is her risk?", "output": "16.7%"}]
    assert span.updates == [{"input": "what is her risk?", "output": "16.7%"}]


def test_a_trace_level_failure_does_not_cost_the_span_its_io(with_keys, monkeypatch):
    """Two records, two failures. One must not take the other down (rule 1)."""

    class HalfBroken(_RecordingClient):
        def set_current_trace_io(self, **fields):
            raise RuntimeError("no active trace")

    client = HalfBroken()
    monkeypatch.setattr("langfuse.get_client", lambda: client)
    span = _RecordingSpan()

    observability.set_trace_io("q", "a", span=span)

    assert span.updates == [{"input": "q", "output": "a"}]


def test_a_span_opens_with_its_input(with_keys, monkeypatch):
    """A tool span records the arguments it was given, at the moment it opens."""
    client = _RecordingClient()
    monkeypatch.setattr("langfuse.get_client", lambda: client)

    with observability.observation("tool.rag_search",
                                   input={"hadm_id": 90000015}) as span:
        assert span is not None

    assert client.opened[0]["name"] == "tool.rag_search"
    assert client.opened[0]["input"] == {"hadm_id": 90000015}


def test_a_span_records_its_output(with_keys, monkeypatch):
    client = _RecordingClient()
    monkeypatch.setattr("langfuse.get_client", lambda: client)
    span = _RecordingSpan()

    observability.record_span_output(span, {"passages": []})

    assert span.updates == [{"output": {"passages": []}}]


def test_a_tool_error_is_levelled_error(with_keys, monkeypatch):
    """`MCPToolbox.call` returns errors as values, so nothing else marks the failure.

    A tool that answered `{"error": ...}` reaches Langfuse as a normal payload; without
    the level, a span tree read by statuses alone reports a clean turn.
    """
    span = _RecordingSpan()

    observability.record_span_output(span, {"error": "timeout"}, is_error=True)

    assert span.updates == [{"output": {"error": "timeout"}, "level": "ERROR"}]


def test_recording_output_cannot_raise(with_keys):
    class Exploding:
        def update(self, **fields):
            raise RuntimeError("span detached")

    observability.record_span_output(Exploding(), {"ok": True})  # logged, not raised


def test_a_traced_tool_call_records_both_ends(with_keys, monkeypatch):
    """The command path's tool span: arguments in, payload out.

    `_execute_tool_calls(traced=True)` is the deterministic `/risk`, `/labs`, `/meds`
    path, whose tools are the only ones that run under a span this module opened. It
    passed neither the arguments nor the result, so a census dump showed that
    `rag_search` ran and not what was searched for or what came back.
    """
    from services.agent import graph

    client = _RecordingClient()
    monkeypatch.setattr("langfuse.get_client", lambda: client)

    class _Box:
        async def call(self, name, arguments):
            return {"hadm_id": arguments["hadm_id"], "passages": []}

    import asyncio

    asyncio.run(graph._execute_tool_calls(
        _Box(),
        [{"name": "rag_search", "args": {"hadm_id": 90000015, "query": "lactate"},
          "id": "call-1"}],
        traced=True,
    ))

    assert client.opened[0]["name"] == "tool.rag_search"
    assert client.opened[0]["input"] == {"hadm_id": 90000015, "query": "lactate"}
    assert client._span.updates == [
        {"output": {"hadm_id": 90000015, "passages": []}},
    ]
