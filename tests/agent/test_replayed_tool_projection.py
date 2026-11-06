"""What a replayed tool result carries into the next turn's window.

A replayed turn used to hand its evidence forward as a `ToolMessage` body, and the
projection that withholds the estimate's floats had to be reapplied there or a follow-up
saw a different result from the turn that fetched it (2026-11-04).

The tool pair is retired (Phase 1 Sprint 1.4): a replayed turn now contributes its
question, its answer, and its calls, and the evidence reaches the model through the
session record — `render_for_prompt()`, injected per round by `agent_node`. So these
tests ask their question of the record: what the model reads is
`hydrate_from_calls(hadm_id, projected calls).render_for_prompt()`, and the projection has
to have been applied BEFORE ingestion, or the record hands the floats straight back.

Offline: a toolbox whose `call` returns a stored payload, and the real replay path.
"""

import asyncio
import json

from services.agent.engine import graph
from services.agent.engine.mcp_client import _presented_call
from services.agent.postprocess.evidence_state import SessionEvidenceState


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


def _replay(tool, payload, *, hadm_id, question=None):
    """The replayed calls, and the record block the next turn's model would read."""
    _, calls, _articles = asyncio.run(graph.replay_messages(
        _Box({tool: payload}), _turn(tool, payload, question=question, hadm_id=hadm_id),
        hadm_id=hadm_id,
    ))
    record = SessionEvidenceState.hydrate_from_calls(
        hadm_id, [_presented_call(call) for call in calls])
    return calls, record, record.render_for_prompt()


#: A panel in the shape `get_lab_trends` returns. Red cells are the furthest outside their
#: interval (1.26 reference-range widths), then RDW (1.00), then hemoglobin (0.57) — the
#: order the prompt asks a ranking answer to derive from the flags and the bounds.
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
    """A replayed panel reaches the next turn as numbered record items.

    The replay path used to have to reproduce the projection's `_abnormality_rank`, which
    is gone: the panel travels as the tool returned it, and the positions are the record's.
    """
    _calls, _record, block = _replay(
        "get_lab_trends", RONCO_PANEL, question="/labs", hadm_id=90000108)

    assert "_abnormality_rank" not in block, block
    assert "excursion" not in block, block
    # One header per analyte, numbered 1..3 in the order the panel gave them.
    headers = [line for line in block.splitlines() if line.startswith("[^[")]
    assert len(headers) == 3, block
    assert headers[0].startswith("[^[1] | tool=get_lab_trends")
    # And the trajectories themselves, in the same order.
    assert block.index("Hemoglobin") < block.index("Red blood cells") < block.index("RDW")


def test_a_replayed_estimate_shows_display_strings_and_no_float():
    """The other half of the same coupling: the estimate is projected before ingestion."""
    payload = {
        "probability": 0.316788,
        "threshold": 0.25,
        "probability_display": "31.7%",
        "threshold_display": "25.0%",
        "base_value": 0.1,
    }

    _calls, record, block = _replay(
        "predict_readmission", payload, question="/ehr", hadm_id=90000023)

    assert "31.7%" in block, block
    assert "0.316788" not in block, block
    assert "base_value" not in block, block
    assert "probability" not in record.prediction, (
        "the record must hold what the projection allowed, not the tool's floats")


def test_the_recorded_replay_still_holds_what_the_tool_returned():
    """Projection is the model's copy, never the record the caller stores.

    The evidence a later turn is judged against, and the guardrail that checks an answer
    against it, read the recorded call. If projection leaked into it, the `excursion` this
    test's own panel is ranked by would become part of "what the tool said".
    """
    calls, _record, _block = _replay(
        "get_lab_trends", RONCO_PANEL, question="/labs", hadm_id=90000108)

    assert calls[0]["response"] == RONCO_PANEL
    assert "_abnormality_rank" not in json.dumps(calls[0]["response"])
