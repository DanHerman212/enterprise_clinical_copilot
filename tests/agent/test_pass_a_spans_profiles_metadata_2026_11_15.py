"""Pass A, 2026-11-15: one span per turn, budgeted `/ehr` calls, and a trace that says
what a turn could call.

Three changes, and each closes a gap that made a live run harder to read than it should
have been:

* the inner `step.command.*` / `step.free_text.*` span opened with no input, closed with
  no output, and held no generation of its own — the tool calls and the model turns were
  already under the turn span — so it was a payload-free row above every turn. Skipped
  now, not renamed;
* `/ehr` ran its query generator, its abstract selector and its four-section synthesis on
  the served thinking level and the 16,384-token ceiling. The two helper steps now run on
  `fast_helper` and the synthesis on `ehr_synthesis`;
* a trace recorded the tools a turn RAN and never the tools it COULD run, so a mis-bound
  turn and a correctly bound turn that chose nothing looked identical.
"""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

from services.agent.engine import deterministic
from services.agent.engine import gateway
from services.agent.engine import graph
from services.agent.engine import model as model_gateway
from services.agent.routing import routing
from services.mcp import config

PIN = config.AGENT_MODEL


# --------------------------------------------------------------------------------------
# Helpers: a span spy, and the two `ask` routes with their model work replaced.
# --------------------------------------------------------------------------------------


class _Span:
    """A span that keeps what was written to it."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.updates: list[dict] = []

    def update(self, **fields) -> None:
        self.updates.append(fields)


class _SpanSpy:
    """Records every span opened through `observability.observation`."""

    def __init__(self) -> None:
        self.opened: list[str] = []

    def observation(self, name: str, *, as_type: str = "span",
                    trace_metadata: dict | None = None, input=None):
        self.opened.append(name)
        return contextlib.nullcontext(_Span(name))


def _patch_observability(monkeypatch, turn_root):
    """Route every trace write to the spy, and answer `current_turn_span` fixed."""
    spy = _SpanSpy()
    stamped: list[tuple[object, list[str]]] = []
    monkeypatch.setattr(graph.observability, "observation", spy.observation)
    monkeypatch.setattr(graph.observability, "current_turn_span", lambda: turn_root)
    monkeypatch.setattr(
        graph.observability, "stamp_trace_tags",
        lambda span, tags: stamped.append((span, tags)))
    monkeypatch.setattr(graph.observability, "span_trace_id", lambda span: "")
    monkeypatch.setattr(graph.observability, "current_trace_id", lambda: "")
    monkeypatch.setattr(graph.observability, "trace_id", lambda handler=None: "")
    return spy, stamped


async def _empty_replay(toolbox, turns, *, hadm_id=None, record_sink=None):
    return [], [], []


def _ask_command(monkeypatch, *, turn_root):
    """One `/ehr` turn through `ask`, with the command and the replay stubbed."""
    spy, stamped = _patch_observability(monkeypatch, turn_root)

    async def _run(toolbox, command, hadm_id, **kwargs):
        return {"messages": [], "tool_calls": [], "command": command.wire}

    monkeypatch.setattr(graph, "replay_messages", _empty_replay)
    monkeypatch.setattr(graph, "run_command", _run)
    state = asyncio.run(graph.ask(
        object(), "summarise the record", question_kind="summarize", hadm_id=90000001,
    ))
    return spy, stamped, state


class _SettledGraph:
    """A compiled graph that answers in prose without calling anything."""

    async def ainvoke(self, state, config=None):
        return {**state, "messages": [AIMessage(content="The record says so.")]}


def _ask_free_text(monkeypatch, *, turn_root):
    spy, stamped = _patch_observability(monkeypatch, turn_root)
    monkeypatch.setattr(graph, "replay_messages", _empty_replay)
    monkeypatch.setattr(graph, "build_graph", lambda *a, **k: _SettledGraph())
    state = asyncio.run(graph.ask(object(), "what changed overnight?"))
    return spy, stamped, state


# --------------------------------------------------------------------------------------
# 1. The inner span is skipped when the route owns the turn span.
# --------------------------------------------------------------------------------------


def test_the_command_path_opens_no_second_span_when_the_turn_is_open(monkeypatch):
    """The row carried no input, no output and no generation — so it is not opened.

    Measured on `067c06f743450d62006ff3a79928da79`: `free_text.turn3` (the route's span)
    held the question and the answer, an inner `free_text.turn3` beside it held neither,
    and the `LangGraph` chain beneath the inner one held the messages. The inner row was
    pure clutter, so the branch runs on the turn span itself.
    """
    turn_root = _Span("command.summarize")
    spy, stamped, state = _ask_command(monkeypatch, turn_root=turn_root)

    assert spy.opened == []
    assert state["command"] == "summarize"


def test_the_command_path_still_opens_its_own_span_with_no_turn(monkeypatch):
    """Off the routes — a direct caller — the span is the only one the turn will have."""
    spy, _, _ = _ask_command(monkeypatch, turn_root=None)

    assert spy.opened == ["command.summarize"]


def test_the_turn_span_still_receives_the_tool_tag(monkeypatch):
    """Skipping the span must not cost the tag: it is addressed to `root_span`.

    `tools:0` is what the triage view filters on, and it was stamped on the inner span
    before this change. With the inner span gone the write has to reach the turn.
    """
    turn_root = _Span("command.summarize")
    _, stamped, _ = _ask_command(monkeypatch, turn_root=turn_root)

    assert len(stamped) == 1
    span, tags = stamped[0]
    assert span is turn_root
    assert "tools:0" in tags


def test_the_free_text_path_opens_no_second_span_when_the_turn_is_open(monkeypatch):
    turn_root = _Span("free_text.turn1")
    spy, _, _ = _ask_free_text(monkeypatch, turn_root=turn_root)

    assert spy.opened == []


def test_the_free_text_path_still_opens_its_own_span_with_no_turn(monkeypatch):
    spy, _, _ = _ask_free_text(monkeypatch, turn_root=None)

    assert spy.opened == ["free_text.turn1"]


# --------------------------------------------------------------------------------------
# 2. Call profiles: stated once, wired to `/ehr` and nowhere else.
# --------------------------------------------------------------------------------------


def test_the_three_profiles_are_the_three_budgets_the_directive_names():
    helper = gateway.call_profile(gateway.CALL_PROFILE_FAST_HELPER)
    synthesis = gateway.call_profile(gateway.CALL_PROFILE_EHR_SYNTHESIS)
    default = gateway.call_profile(gateway.CALL_PROFILE_DEFAULT)

    # Thinking budgets only. A capped reply is not a shorter reply: it is one that stops
    # mid-sentence, which is a worse answer rather than a faster one.
    assert (helper.thinking_budget, helper.max_output_tokens) == (0, None)
    assert (synthesis.thinking_budget, synthesis.max_output_tokens) == (512, None)
    # The default sets NOTHING, which is what keeps every other route on the baseline.
    assert (default.thinking_budget, default.max_output_tokens) == (None, None)


def test_an_unknown_profile_name_is_the_default_rather_than_an_error():
    """A performance control must not be able to stop an answer."""
    assert gateway.call_profile("nonsense").name == gateway.CALL_PROFILE_DEFAULT
    assert gateway.call_profile(None).name == gateway.CALL_PROFILE_DEFAULT


def test_the_profiles_are_spelled_for_the_family_that_serves_them():
    profile = gateway.profile(PIN)

    assert gateway.call_profile_parameters(
        profile, gateway.call_profile(gateway.CALL_PROFILE_FAST_HELPER),
    ) == {"thinking_budget": 0}
    assert gateway.call_profile_parameters(
        profile, gateway.call_profile(gateway.CALL_PROFILE_EHR_SYNTHESIS),
    ) == {"thinking_budget": 512}
    # The default adds no parameter at all, so it cannot move a baseline.
    assert gateway.call_profile_parameters(
        profile, gateway.call_profile(gateway.CALL_PROFILE_DEFAULT),
    ) == {}


def test_a_refused_budget_falls_back_to_the_level_the_profile_accepts():
    """The second attempt drops the budget and uses the family's lowest level."""
    profile = gateway.profile(PIN)

    for name in (gateway.CALL_PROFILE_FAST_HELPER, gateway.CALL_PROFILE_EHR_SYNTHESIS):
        parameters = gateway.call_profile_parameters(
            profile, gateway.call_profile(name), fallback=True)
        assert "thinking_budget" not in parameters
        assert parameters["reasoning_effort"] == "low"


def test_the_default_client_carries_no_budget_and_the_served_ceiling():
    llm = model_gateway.build_llm(PIN)

    assert type(llm).__name__ == "ChatGoogleGenerativeAI"
    assert llm.thinking_budget is None
    assert llm.max_output_tokens == config.GEMINI_MAX_OUTPUT_TOKENS


def test_the_ehr_synthesis_client_keeps_its_thinking_budget_and_no_ceiling():
    llm = model_gateway.build_llm(
        PIN, call_profile=model_gateway.CALL_PROFILE_EHR_SYNTHESIS)

    assert llm.thinking_budget == 512
    # No profile sells a reply short: the deployment's own ceiling still applies.
    assert llm.max_output_tokens == config.GEMINI_MAX_OUTPUT_TOKENS


def test_the_ehr_helper_client_thinks_not_at_all():
    llm = model_gateway.build_extraction_llm(
        PIN, call_profile=model_gateway.CALL_PROFILE_FAST_HELPER)

    assert llm.thinking_budget == 0
    assert llm.max_output_tokens == config.GEMINI_MAX_OUTPUT_TOKENS
    # The content filter is still the extraction one: this is the same constructor.
    assert llm.reasoning_effort == gateway.EXTRACTION_REASONING_EFFORT


def test_the_route_decides_the_profile_and_the_trace_records_the_same_one():
    """`/ehr` is budgeted for its synthesis, `/labs` for its answer, `/meds` not at all.

    Both readers of `model.FORMATTING_PROFILES` are asserted here against the route alone,
    which is how the trace and the client construction are kept from disagreeing.
    """
    assert graph._call_profile_for("summarize") == gateway.CALL_PROFILE_EHR_SYNTHESIS
    assert graph._call_profile_for("labs") == gateway.CALL_PROFILE_FAST_HELPER
    for kind in ("meds", "literature", None):
        assert graph._call_profile_for(kind) == gateway.CALL_PROFILE_DEFAULT


def test_the_ehr_helper_steps_are_built_on_the_fast_helper_profile(monkeypatch):
    """Both `/ehr` helper calls ask for the profile at the call site.

    The suite's offline seam is put back to the module default for this test, so the
    production branch is the one under assertion, and the constructor it reaches is patched
    to record the keyword it was given.
    """
    seen: dict = {}

    def _build(model, **kwargs):
        seen["model"] = model
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(deterministic, "extraction_llm_factory",
                        deterministic._DEFAULT_EXTRACTION_FACTORY)
    monkeypatch.setattr(model_gateway, "build_extraction_llm", _build)
    deterministic._ehr_helper_llm("test-model")

    assert seen == {
        "model": "test-model",
        "call_profile": gateway.CALL_PROFILE_FAST_HELPER,
    }


def test_a_replaced_extraction_seam_still_decides_the_helper_client(monkeypatch):
    """The seam wins over the profile.

    `conftest` replaces the factory with `lambda model: None` so no `/ehr` test reaches a
    model. If the profile were applied unconditionally, that replacement would be bypassed
    and every one of those tests would build a real client.
    """
    monkeypatch.setattr(deterministic, "extraction_llm_factory", lambda model: None)

    assert deterministic._ehr_helper_llm("test-model") is None


@pytest.mark.parametrize(
    "wire, expected",
    [("summarize", gateway.CALL_PROFILE_EHR_SYNTHESIS),
     ("labs", gateway.CALL_PROFILE_FAST_HELPER),
     ("meds", None),
     ("literature", None)],
)
def test_the_formatting_turn_names_the_profile_of_its_own_command(
        wire, expected, monkeypatch):
    """Every command's formatting turn, at the profile its job wants.

    `/ehr` writes the briefing and keeps a thinking budget; `/labs` describes a panel that
    is drawn beside it. `/meds` is absent from the table on purpose: its formatting turn is
    the medication extraction, and de-thinking it would change what the grid contains rather
    than how fast it arrives.

    `None` is the assertion that matters as much as the profiles: a command with no entry
    calls the factory with no keyword at all, so nothing added to this table can move
    `/meds` or `/pubmed`.
    """
    seen: dict = {}

    def _build(model, **kwargs):
        seen.update(kwargs)
        return _FormattingModel()

    monkeypatch.setattr(model_gateway, "build_llm", _build)
    monkeypatch.setattr(deterministic, "extraction_llm_factory", lambda model: None)
    asyncio.run(deterministic.run_command(
        _Toolbox(), routing.command_for(wire), 90000023,
        topic="heart failure readmission" if wire == "literature" else None,
        question="a command", model="test-model",
    ))

    assert seen.get("call_profile") == expected


def test_the_medication_and_laboratory_turns_keep_their_own_answers(monkeypatch):
    """`/meds` is unchanged, `/labs` keeps its own reply, and neither loses its payload."""
    monkeypatch.setattr(model_gateway, "build_llm",
                        lambda model, **_kw: _FormattingModel())

    meds = asyncio.run(deterministic.run_command(
        _Toolbox(), routing.command_for("meds"), 90000023, question="/meds",
        model="test-model"))
    labs = asyncio.run(deterministic.run_command(
        _Toolbox(), routing.command_for("labs"), 90000023, question="/labs",
        model="test-model"))

    assert meds["command_payload"] == {
        "medications": [], "dropped": [], "returned_passages": 0, "sections": [],
    }
    assert labs["command"] == "labs"
    # Both still ran a formatting turn and still wrote its reply into the answer.
    assert meds["messages"][-1].content
    assert labs["messages"][-1].content


def test_a_free_text_turn_runs_no_profile_at_all():
    """Free text and `/pubmed` reason over prose, so they keep the served allowance.

    Nothing in the free-text path names a profile, which is why this reads the client the
    engine would build rather than a call record: the absence is the assertion.
    """
    llm = graph._build_llm(PIN)

    assert gateway.profile_for(llm).name == PIN
    assert llm.thinking_budget is None
    assert llm.max_output_tokens == config.GEMINI_MAX_OUTPUT_TOKENS


class _Toolbox:
    """A toolbox that answers every call with an empty result."""

    async def call(self, name, arguments):
        return {"returned": 0, "passages": []}


class _FormattingModel:
    """A model that answers the formatting turn in prose and binds nothing."""

    def bind_tools(self, *args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("the formatting turn must not be given tools")

    async def ainvoke(self, messages, config=None):
        return AIMessage(content="A summary.")

    async def astream(self, messages, config=None):
        yield AIMessage(content="A summary.")


# --------------------------------------------------------------------------------------
# 3. The refused-budget retry: one attempt, only for a budget refusal.
# --------------------------------------------------------------------------------------


class _Refusing:
    """A client whose first call fails the way a refused budget fails."""

    def __init__(self, message: str) -> None:
        self.message = message
        self.calls = 0

    async def ainvoke(self, messages, config=None, **kwargs):
        self.calls += 1
        raise RuntimeError(self.message)


class _Answering:
    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, messages, config=None, **kwargs):
        self.calls += 1
        return AIMessage(content="ok")


def _wrapper(primary, fallback):
    return gateway._BudgetFallback(
        primary, lambda: fallback, gateway.profile(PIN))


def test_a_refused_thinking_budget_is_retried_once_without_it():
    primary = _Refusing("400 INVALID_ARGUMENT: thinking_budget is not supported")
    fallback = _Answering()

    answer = asyncio.run(_wrapper(primary, fallback).ainvoke([]))

    assert answer.content == "ok"
    assert (primary.calls, fallback.calls) == (1, 1)


def test_a_safety_refusal_is_not_retried():
    """Retrying a refusal the provider has already answered would send it twice."""
    primary = _Refusing("safety: the response was blocked")
    fallback = _Answering()

    with pytest.raises(RuntimeError):
        asyncio.run(_wrapper(primary, fallback).ainvoke([]))

    assert fallback.calls == 0


def test_the_fallback_client_is_built_at_most_once():
    builds: list[int] = []

    def _rebuild():
        builds.append(1)
        return _Answering()

    primary = _Refusing("400 INVALID_ARGUMENT: thinking budget unsupported")
    client = gateway._BudgetFallback(primary, _rebuild, gateway.profile(PIN))

    asyncio.run(client.ainvoke([]))
    asyncio.run(client.ainvoke([]))

    assert len(builds) == 1


def test_a_stream_that_has_already_announced_is_not_retried():
    """Two attempts after a frame has been shown would print the prose twice."""

    class _HalfStream:
        async def astream(self, messages, config=None, **kwargs):
            yield AIMessage(content="the first sentence")
            raise RuntimeError("400 INVALID_ARGUMENT: thinking_budget refused")

    fallback = _Answering()
    client = _wrapper(_HalfStream(), fallback)

    async def _drain():
        seen = []
        with pytest.raises(RuntimeError):
            async for chunk in client.astream([]):
                seen.append(chunk)
        return seen

    assert len(asyncio.run(_drain())) == 1
    assert fallback.calls == 0


# --------------------------------------------------------------------------------------
# 4. Metadata: the profile, the budget, the bound set and the routing answers.
# --------------------------------------------------------------------------------------


def test_the_metadata_carries_the_profile_the_budget_and_the_bound_set():
    meta = graph._trace_metadata(
        90000001, None, "run-x", None, None,
        allowed_tools=frozenset({"rag_search", "get_lab_trends"}),
        predicates={
            "risk_driver": False,
            "wants_literature": True,
            "lab_driver": True,
            "follow_up_declared": False,
            "risk_score": True,
        },
        call_profile=gateway.CALL_PROFILE_DEFAULT,
    )

    assert meta["llm_profile"] == "default"
    assert meta["thinking_budget"] is None
    # Sorted, so two turns of one route compare equal in an export.
    assert meta["allowed_tools"] == ["get_lab_trends", "rag_search"]
    assert meta["risk_driver"] is False
    assert meta["wants_literature"] is True
    assert meta["lab_driver"] is True
    assert meta["follow_up_declared"] is False
    assert meta["risk_score"] is True


def test_the_route_level_metadata_names_the_profile_from_the_route_alone():
    """`turn.py` calls `turn_scope` before any question shape is known."""
    _, command_meta = graph.turn_scope("summarize", 90000001)
    assert command_meta["llm_profile"] == "ehr_synthesis"
    assert command_meta["thinking_budget"] == 512
    assert command_meta["allowed_tools"] == []
    # `/pubmed` declares its subject, so the one predicate that is a fact about the
    # route is answered here rather than left absent.
    _, literature_meta = graph.turn_scope("literature", 90000001)
    assert literature_meta["llm_profile"] == "default"
    assert literature_meta["wants_literature"] is True


class _GraphConfig:
    """The run config `ainvoke` was handed, so the metadata can be read back."""

    last: dict = {}


class _ConfigRecordingGraph(_SettledGraph):
    async def ainvoke(self, state, config=None):
        _GraphConfig.last = dict((config or {}).get("metadata") or {})
        return await super().ainvoke(state, config)


def test_the_free_text_run_config_carries_the_metadata(monkeypatch):
    """The five answers and the bound set reach the run, so a trace can be read back.

    Read off the config `ainvoke` was handed rather than off the span: the metadata is
    what Langfuse stores, and the config is the only place it is visible before the SDK
    writes it.
    """
    _GraphConfig.last = {}
    _patch_observability(monkeypatch, None)
    monkeypatch.setattr(graph, "replay_messages", _empty_replay)
    monkeypatch.setattr(graph, "build_graph", lambda *a, **k: _ConfigRecordingGraph())

    asyncio.run(graph.ask(object(), "what changed overnight?"))

    assert _GraphConfig.last["llm_profile"] == "default"
    assert _GraphConfig.last["thinking_budget"] is None
    assert _GraphConfig.last["allowed_tools"] == sorted(routing.FREETEXT_TOOLS)
    assert _GraphConfig.last["risk_driver"] is False
    assert _GraphConfig.last["wants_literature"] is False
    assert _GraphConfig.last["lab_driver"] is False
    assert _GraphConfig.last["follow_up_declared"] is False
    assert _GraphConfig.last["risk_score"] is False
