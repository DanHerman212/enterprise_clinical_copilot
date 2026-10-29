"""The root trace span must still be open when the served answer is written.

Gate 2, 2026-10-29. Every trace in the run carried `input: null, output: null`, and the
root observation did too: `graph.ask` opened the span, and `observability.set_trace_io`
was called in `turn.compose_success` — after `ask` had returned, so after the `with`
block had closed. The SDK skips a write to a closed observation (verified against
langfuse 4.14.4: `span.update()` on an ended span is ignored) and `set_current_trace_io`
has no trace to address, so the write landed nowhere. Child spans were unaffected:
LangChain's handler reports its own runs, which is why every generation and tool span in
those traces had both fields and only the root was empty.

The fix moves the span up to `turn.turn_trace`, which the routes hold open across the
chain AND the composition. These tests pin the property the move exists for: at the
moment the answer is recorded, a span is current — and the text recorded is the guarded
answer, not the model's draft.
"""

from __future__ import annotations

import contextlib
import sys
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from services.agent import turn  # noqa: E402
from services.agent.telemetry import observability  # noqa: E402


class _Span:
    """A Langfuse observation, reduced to what the trace writer touches."""

    def __init__(self, name: str, input=None) -> None:  # noqa: A002
        self.name = name
        self.input = input
        self.output = None
        self.metadata = None
        self.closed = False

    def update(self, **fields) -> None:
        # The real object accepts the call and drops it once the span has ended. That
        # is the behaviour this file exists to stay on the right side of, so the fake
        # refuses to record anything after close rather than accepting it.
        if self.closed:
            return
        for key, value in fields.items():
            setattr(self, key, value)


class _Client:
    def __init__(self) -> None:
        self.spans: list[_Span] = []
        self.trace_io: list[dict] = []
        self.open_now: list[_Span] = []

    def start_as_current_observation(self, *, as_type, name, input=None):  # noqa: A002
        span = _Span(name, input)
        self.spans.append(span)
        client = self

        @contextlib.contextmanager
        def _open():
            client.open_now.append(span)
            try:
                yield span
            finally:
                client.open_now.remove(span)
                span.closed = True

        return _open()

    def set_current_trace_io(self, *, input=None, output=None) -> None:  # noqa: A002
        self.trace_io.append({
            "input": input,
            "output": output,
            "inside_a_span": bool(self.open_now),
        })

    def get_current_trace_id(self) -> str:
        return "trace-from-fake"


def _install(monkeypatch) -> _Client:
    client = _Client()
    module = types.SimpleNamespace(
        get_client=lambda: client,
        propagate_attributes=lambda **kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setitem(sys.modules, "langfuse", module)
    monkeypatch.setattr(observability, "configured", lambda: True)
    return client


def _parsed(**overrides) -> dict:
    parsed = {"kind": None, "hadm_id": None, "turns": [], "run_tag": "trace-test",
              "intent": None}
    parsed.update(overrides)
    return parsed


def test_a_turn_publishes_its_root_span_while_it_runs(monkeypatch):
    _install(monkeypatch)

    assert observability.current_turn_span() is None
    with turn.turn_trace(_parsed()) as span:
        assert observability.current_turn_span() is span
    assert observability.current_turn_span() is None


def test_the_answer_is_written_while_the_root_span_is_still_open(monkeypatch):
    """The property the whole fix is for: no write after the span has ended."""
    client = _install(monkeypatch)

    with turn.turn_trace(_parsed()):
        observability.set_trace_io("What is the readmission risk?", "0.21.")

    assert client.trace_io == [{
        "input": "What is the readmission risk?",
        "output": "0.21.",
        "inside_a_span": True,
    }]
    root = client.spans[0]
    assert root.name == "free_text.turn1"
    assert root.input == "What is the readmission risk?"
    assert root.output == "0.21."


def test_the_recorded_output_is_the_guarded_answer_not_the_models_draft(monkeypatch):
    """A trace is the record of what was served, so the guardrails must come first.

    The draft states a readmission risk of 0.14 with no prediction behind it, which
    `guard_answer` strips. Before the span moved, the only text available inside the
    span was this draft — which is why the write moved to `compose_success` in the first
    place, and why moving it without moving the span left the root empty.
    """
    client = _install(monkeypatch)
    draft = "The readmission risk is 0.14."

    from langchain_core.messages import AIMessage, HumanMessage

    state = {
        "messages": [HumanMessage(content="What is the readmission risk?"),
                     AIMessage(content=draft)],
        "tool_calls": [],
    }
    with turn.turn_trace(_parsed()) as span:
        payload = _run_composition(state)

    assert "0.14" in draft
    assert "0.14" not in payload["answer"]
    assert span.output == payload["answer"], "the trace must carry the served prose"
    assert span.input == "What is the readmission risk?"
    assert client.trace_io and client.trace_io[-1]["inside_a_span"] is True
    assert client.trace_io[-1]["output"] == payload["answer"]


def test_nothing_is_recorded_on_the_trace_when_tracing_is_off(monkeypatch):
    """Rule 1: an unconfigured process records nothing and fails nothing."""
    for name in ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)

    with turn.turn_trace(_parsed()) as span:
        assert span is None
        observability.set_trace_io("q", "a")


def _run_composition(state: dict) -> dict:
    """`compose_success` for a free-text turn, awaited without a model call."""
    import asyncio

    return asyncio.run(turn.compose_success(
        "What is the readmission risk?", state, "trace-1",
    ))
