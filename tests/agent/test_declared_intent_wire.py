"""The declared intent, on the wire and on the artifacts.

The reported defect (2026-10-26): a chip under a `/labs` answer produced two loop
responses. The surviving trace showed the replay was intact and the first pass had
already answered, so the only branch that could still have sent the run back for
evidence is `route`'s `follow_up_declared` — which is false when the request never
carried `intent`.

`test_followup_routing.py` pins `route` given a declaration, and the website's own
suites pin the console and the BFF. What none of them covered is the seam between
them: `/ask` and `/ask/stream` parse the body, then hand the run to `turn.run_chain`,
which is where a keyword can go missing without any of the three sides noticing. The
streamed route is the one the console uses and the one with the second copy of the
request body, so it is the one where a field is easiest to lose.

Offline: the model is never built, `ask` and `toolbox` are patched at the route, and
what is asserted is what the graph would have been told.
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from services.agent.ingress import http as srv  # noqa: E402

#: The chip the production defect came from, verbatim.
THE_CHIP = ("How does the abnormal discharge Red blood cells (3.47 x10^6/uL) compare to "
            "its reference range and trend?")


@asynccontextmanager
async def _fake_toolbox():
    yield None


def _completed_state():
    """A finished run that survives the real guardrails and composition."""
    return {
        "messages": [
            HumanMessage(content=THE_CHIP),
            AIMessage(content="The red cell count was 3.47 x10^6/uL, below its range. ^[1]"),
        ],
        "tool_calls": [{
            "name": "rag_search",
            "args": {"hadm_id": 90000015, "query": "red blood cells"},
            "response": {"hadm_id": 90000015, "returned": 1, "passages": [
                {"id": "MT-1-LB_labs_1", "section": "labs",
                 "text": "Discharge labs: RBC 3.47 x10^6/uL."},
            ]},
        }],
    }


def _posted(path, body):
    """Post one turn and return what the graph was handed."""
    seen = {}

    async def fake_ask(box, question, **kwargs):
        seen.update(kwargs)
        seen["question"] = question
        return _completed_state()

    with patch.object(srv, "toolbox", _fake_toolbox), \
         patch.object(srv, "ask", fake_ask):
        response = TestClient(srv.app, raise_server_exceptions=False).post(path, json=body)
    return response, seen


def test_the_plain_route_hands_the_declaration_to_the_graph():
    response, seen = _posted("/ask", {"hadm_id": 90000015, "question": THE_CHIP,
                                      "intent": "follow_up"})

    assert response.status_code == 200
    assert seen["intent"] == "follow_up", (
        "the route accepted the declaration and then did not pass it on, so the graph "
        "cannot tell a chip from a typed question")


def test_the_streamed_route_hands_it_over_too():
    """The console's route: `stream_prose` and `follow_ups` are added to the same body
    a few lines later, and a field carried on one branch and not the other is how a fix
    ships, passes the suite, and changes nothing in production."""
    response, seen = _posted("/ask/stream", {"hadm_id": 90000015, "question": THE_CHIP,
                                             "intent": "follow_up",
                                             "stream_prose": True})

    assert response.status_code == 200
    assert seen["intent"] == "follow_up"
    assert seen["prose_stream"] is True


def test_an_ordinary_question_declares_nothing():
    """Absent means absent. A turn no control spoke for must not arrive at the graph
    with a value that would exempt it from the retrieval it may need."""
    response, seen = _posted("/ask/stream", {"hadm_id": 90000015, "question": THE_CHIP})

    assert response.status_code == 200
    assert "intent" not in seen, (
        "a question with no control behind it reached the graph with a declaration")


def test_the_declaration_is_on_the_trace_and_on_the_execution_record():
    """The artifact is the point: the trace of the failing turn could not say whether
    the declaration had arrived, so the one fact the routing turned on was unanswerable
    after the fact."""
    from services.agent.engine import graph
    from services.agent import turn

    meta = graph._trace_metadata(90000015, None, None, None, "follow_up")
    assert meta["intent"] == "follow_up"
    assert "intent:follow_up" in meta["langfuse_tags"]

    # ...and the conversation record, which is where the failures are read from: a
    # declaration carried only on the success line is missing from every run that went
    # wrong.
    staging = turn.StageLog(intent="follow_up")
    with patch.object(turn.chain, "record_execution",
                      wraps=turn.chain.record_execution) as spy:
        staging.record("trace-1", THE_CHIP, "ok")
    assert spy.call_args.kwargs["intent"] == "follow_up"
