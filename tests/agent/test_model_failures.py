"""Gap 8: a failing model, rather than a failing description of one.

Section 3.4 of the layer 4 document describes what happens when a model call goes wrong,
and until this file nothing constructed one. The tests that did exist patch `ask` — the
route's seam — which is the right level for testing the route and the wrong level for
testing the chain: with `ask` replaced, the tool loop, `final_message`, `final_text` and
the classification never run, so the graph's own handling of a bad turn was the part
nothing exercised.

So these inject at the model. `_build_llm` is replaced with a model that fails in one
named way and the request arrives through `/ask` as a caller would send it, which means
everything between the route and the client runs for real.

Two things are deliberately not tested here. `max_retries` belongs to the SDK: the fake
stands in for the client *after* the attempts have been spent, so what is asserted is the
route's handling of a failure that got out, not the retry policy — that stays the open
half of gap 2. And the injected failures are the ones that fail *silently* or *late*: an
exception is easy, a model that returns HTTP 200 with nothing in it is not.
"""

import asyncio
import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage  # noqa: E402
from langchain_google_genai.chat_models import GoogleRateLimitError  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from services.agent import graph  # noqa: E402
from services.agent import http as srv  # noqa: E402

RECORD_LOGGER = "services.agent.chain"


class _EmptyToolbox:
    """No tools, because the failure under test is the model's.

    `_tools` reads this mapping, so an empty one satisfies the real contract rather than
    skipping it — and that matters: a toolbox that raised would produce the same 502 from
    outside, and the tests below would pass while testing something else entirely.
    """

    _tools: dict = {}


@asynccontextmanager
async def _fake_toolbox():
    yield _EmptyToolbox()


class _FailingModel:
    """A model that fails in one chosen way.

    Duck-typed on purpose: `build_graph` calls `bind_tools` once and the agent node calls
    `ainvoke` once per turn, and those two are the whole contract needed to put a bad turn
    in front of the chain. If the graph grows a third call this raises `AttributeError`,
    which is the failure a test should have.
    """

    def __init__(self, mode: str, delay: float = 0.0):
        self.mode = mode
        self.delay = delay
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages, **kwargs):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.mode == "stalls":
            # A slow model, not a broken one: it answers, eventually. The point of the
            # test is that the deadline fires first, and that is only a meaningful claim
            # if this turn would have been served had it been given the time.
            return AIMessage(content="An answer that arrived too late to be useful.")
        if self.mode == "raises":
            # The 429 gap 8 named, with the kind of detail the SDK puts in it: a quota
            # line naming a project and a region.
            raise GoogleRateLimitError(
                "Error calling model (RESOURCE_EXHAUSTED): 429 quota exceeded for "
                "project trim-icon-498815-a0 in us"
            )
        if self.mode == "empty":
            # What MAX_TOKENS looks like from here: HTTP 200, no text, no exception.
            return AIMessage(
                content="", response_metadata={"finish_reason": "MAX_TOKENS"}
            )
        if self.mode == "safety":
            return AIMessage(
                content="",
                response_metadata={
                    "finish_reason": "SAFETY",
                    "safety_ratings": [
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "blocked": True},
                    ],
                },
            )
        raise AssertionError(f"unknown failure mode {self.mode!r}")


def _client():
    return TestClient(srv.app, raise_server_exceptions=False)


def _with(model, **patches):
    """Patch the model and the toolbox, and whatever else a test needs."""
    stack = [patch.object(graph, "_build_llm", lambda name: model),
             patch.object(srv, "toolbox", _fake_toolbox)]
    stack += [patch.object(srv, name, value) for name, value in patches.items()]
    return stack


def _post(model, question="Was this patient readmitted?", **patches):
    from contextlib import ExitStack

    with ExitStack() as stack:
        for patcher in _with(model, **patches):
            stack.enter_context(patcher)
        return _client().post("/ask", json={"question": question})


def _record(caplog):
    return json.loads(caplog.records[-1].message)


# --- the model raises --------------------------------------------------------

def test_a_rate_limited_call_is_not_served_as_an_answer(caplog):
    """A 429 that survived the SDK's retries must fail, not produce something plausible.

    The detail in the exception names a project, a region and a quota — the caller gets a
    stable code and a correlation id that pairs with the log line, and nothing else
    (ECC-06).
    """
    model = _FailingModel("raises")
    with caplog.at_level("INFO", logger=RECORD_LOGGER):
        response = _post(model)

    assert model.calls == 1, "the model was never reached, so nothing here was tested"
    assert response.status_code == 502
    body = response.json()
    assert body["error"] == "agent_failed"
    assert body["correlation_id"]
    assert "RESOURCE_EXHAUSTED" not in response.text
    assert "trim-icon-498815-a0" not in response.text
    record = _record(caplog)
    assert record["outcome"] == "error"
    assert record["error"] == "agent_failed"


# --- the model stalls -------------------------------------------------------

def test_a_stalled_call_is_cut_off_by_the_deadline(caplog):
    """The 504, which nothing tested before this file.

    The deadline is a spend control: a call that never returns must not keep billing until
    the caller gives up, and the caller must be told which limit was hit rather than
    getting a connection that simply stops.
    """
    model = _FailingModel("stalls", delay=10)
    started = time.monotonic()
    with caplog.at_level("INFO", logger=RECORD_LOGGER):
        response = _post(model, ASK_TIMEOUT_SECONDS=1.0)
    elapsed = time.monotonic() - started

    assert response.status_code == 504
    body = response.json()
    assert body["error"] == "timeout"
    assert "1s" in body["message"], "the caller is not told which limit it hit"
    assert elapsed < 5, f"the deadline did not fire; the request took {elapsed:.1f}s"
    record = _record(caplog)
    assert record["outcome"] == "timeout"
    assert record["error"] == "timeout"


# --- the model returns nothing ----------------------------------------------

def test_an_empty_turn_is_reported_as_truncated_rather_than_served(caplog):
    """MAX_TOKENS is a 200 with no text and no exception, so nothing raises on its own.

    This is the chain's half of the failure: `final_text` must not fall back to an earlier
    message, and the response's own reason has to reach both the caller and the record.
    """
    model = _FailingModel("empty")
    with caplog.at_level("INFO", logger=RECORD_LOGGER):
        response = _post(model, question="A long question")

    assert model.calls == 1, "the model was never reached, so nothing here was tested"
    assert response.status_code == 502
    assert response.json()["error"] == "answer_truncated"
    record = _record(caplog)
    assert record["finish_reason"] == "MAX_TOKENS"
    assert record["outcome"] == "error"


def test_a_safety_finish_is_reported_as_a_refusal_with_its_category(caplog):
    """A refusal is not a transient failure, and the words have to agree with that.

    At temperature 0 the same question reaches the same decision, so a caller told to
    retry would be told to do something that cannot work. Which category refused it is
    recorded, which is what makes refusals countable rather than only experienced.
    """
    model = _FailingModel("safety")
    with caplog.at_level("INFO", logger=RECORD_LOGGER):
        response = _post(model)

    assert model.calls == 1, "the model was never reached, so nothing here was tested"
    assert response.status_code == 502
    body = response.json()
    assert body["error"] == "answer_refused"
    assert "retry" not in body["message"].lower()
    record = _record(caplog)
    assert record["finish_reason"] == "SAFETY"
    assert record["filtered"] == ["HARM_CATEGORY_DANGEROUS_CONTENT"]
