"""Replayed tool results carry the same projection a live result does.

`tool_node` renders `_presented_for_model(payload)`; the replay path rendered the raw
payload, so a follow-up turn saw a *different* result from the turn that fetched it. Two
things the prompt tells the model to rely on were missing from every replayed turn:

  - the ordinal `_abnormality_rank` on a laboratory panel — the only ranking the prompt
    permits, and the field `dev_t3_03` needs on Turn 2;
  - `probability_display` / `threshold_display` in place of the estimate's floats.

Offline: a toolbox whose `call` returns a stored payload, and the real replay path.
"""

import asyncio
import json

from langchain_core.messages import ToolMessage

from services.agent.engine import graph


class _Box:
    """A toolbox whose `call` returns the payload the replayed turn stored."""

    def __init__(self, payloads):
        self._payloads = payloads

    async def call(self, name, arguments):  # noqa: ANN001 - mirrors MCPToolbox.call
        return self._payloads[name]


def _turn(tool, payload, *, question=None, **args):
    return [{
        "question": question or f"/{tool}",
        "answer": "…",
        "tool_calls": [{"name": tool, "args": args, "payload": payload}],
    }]


def _body(messages) -> str:
    """The rendered `<tool_result>` body from a replayed transcript."""
    return next(
        message.content for message in messages if isinstance(message, ToolMessage)
    )


#: A panel in the shape `get_lab_trends` returns. Red cells are the furthest outside their
#: interval (1.26 reference-range widths), then RDW (1.00), then hemoglobin (0.57) — so a
#: model reading the rank names red cells first, and a model reading the digits is free to
#: get it wrong.
RONCO_PANEL = {
    "hadm_id": 90000108,
    "trends": [
        {"label": "Hemoglobin", "reference_low": 12.0, "reference_high": 15.5,
         "decimals": 1, "series": [{"day": 9.1, "value": 10.0, "flag": "low"}]},
        {"label": "Red blood cells", "reference_low": 4.2, "reference_high": 5.4,
         "decimals": 2, "series": [{"day": 9.1, "value": 2.69, "flag": "low"}]},
        {"label": "RDW", "reference_low": 11.5, "reference_high": 14.5,
         "decimals": 1, "series": [{"day": 9.1, "value": 17.5, "flag": "high"}]},
    ],
}


def test_a_replayed_lab_panel_carries_the_series_and_its_citation_numbers():
    """A replayed panel is the tool's own payload plus the numbering, and nothing else.

    The replay path used to have to reproduce the projection's `_abnormality_rank`, which
    is gone: the panel now reaches the model exactly as the tool returned it, with the
    citation fields the renderer adds for every passage-shaped unit.
    """
    messages, _ = asyncio.run(graph.replay_messages(
        _Box({"get_lab_trends": RONCO_PANEL}),
        _turn("get_lab_trends", RONCO_PANEL, question="/labs", hadm_id=90000108),
    ))

    body = _body(messages)
    assert "_abnormality_rank" not in body, body
    assert "excursion" not in body, body

    panel = json.loads(body.split("\n", 1)[1].rsplit("\n", 1)[0])
    assert [trend["cite"] for trend in panel["trends"]] == [1, 2, 3]
    assert [trend["label"] for trend in panel["trends"]] == [
        "Hemoglobin", "Red blood cells", "RDW"]


def test_a_replayed_estimate_shows_display_strings_and_no_float():
    """The other half of the same coupling: the estimate is projected on replay too."""
    payload = {
        "probability": 0.316788,
        "threshold": 0.25,
        "probability_display": "31.7%",
        "threshold_display": "25.0%",
        "base_value": 0.1,
    }

    messages, _ = asyncio.run(graph.replay_messages(
        _Box({"predict_readmission": payload}),
        _turn("predict_readmission", payload, question="/ehr", hadm_id=90000023),
    ))

    body = _body(messages)
    assert "31.7%" in body, body
    assert "0.316788" not in body, body
    assert "base_value" not in body, body


def test_the_recorded_replay_still_holds_what_the_tool_returned():
    """Projection is the model's copy, never the record.

    The evidence a later turn is judged against, and the guardrail that checks an answer
    against it, read the recorded call. If projection leaked into it, the `excursion` this
    test's own panel is ranked by would become part of "what the tool said".
    """
    _, calls = asyncio.run(graph.replay_messages(
        _Box({"get_lab_trends": RONCO_PANEL}),
        _turn("get_lab_trends", RONCO_PANEL, question="/labs", hadm_id=90000108),
    ))

    assert calls[0]["response"] == RONCO_PANEL
    assert "_abnormality_rank" not in json.dumps(calls[0]["response"])
