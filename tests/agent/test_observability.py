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


def test_unconfigured_is_off_without_a_handler():
    """No keys: no handler, and nothing to explain."""
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
