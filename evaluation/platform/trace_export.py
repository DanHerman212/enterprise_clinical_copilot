"""Read a finished run's traces back, with their spans, for root-cause analysis.

The harness records trace ids; it does not record what is inside them. That gap is
what made the 2026-10-21 census dump unreadable: it carried turn 1's id and an
eight-character prefix for turn 2, so a reviewer could open half a thread and no
script could open the other half at all.

What a reviewer needs from a turn is its SPANS — the root span's input and output,
then one observation per tool call with the arguments it was given and the payload it
returned. Those are the rows that say why an answer was wrong; the trace row alone
says only that something ran.

Two shapes matter here and both are handled:

* An item with no trace id for a turn. The record is returned with `error` set rather
  than dropped, because a missing turn is the finding, not an absence of one.
* A read that fails. The store answers a single-trace read slowly under load (measured
  at fourteen seconds) and a timeout is not evidence about the trace, so the record
  says `unreadable` and the run continues.

Nothing here raises for the same reason the agent's tracing never raises: an export is
a convenience for a reviewer, and losing one turn's spans must not lose the run's.
"""

from __future__ import annotations

from typing import Any

#: Fields copied off an observation. Chosen because they are what a failure is read
#: from: which step it was, what went in, what came out, and whether it was levelled.
_SPAN_FIELDS = ("id", "name", "type", "input", "output", "level",
                "status_message", "start_time", "end_time")


def trace_ids_for_turns(output: dict[str, Any]) -> list[str]:
    """Every turn's trace id for one item, in turn order.

    Falls back to the two ids earlier runs recorded, so an artifact written before the
    list existed still yields both turns — and one whose second id was never captured
    yields the first alone. That shape cannot say whether a second turn ran without a
    trace or never ran, so it is reported as a single turn rather than as an empty one:
    inventing a gap is as wrong as hiding it.
    """
    ids = [str(value or "") for value in (output.get("turn_trace_ids") or [])]
    if ids:
        return ids
    legacy = [str(output.get("langfuse_trace_id") or "")]
    second = str(output.get("turn_2_trace_id") or "")
    if second:
        legacy.append(second)
    return legacy


def export_turn(client: Any, trace_id: str, *, turn: int,
                timeout: int = 90) -> dict[str, Any]:
    """One turn's trace, its spans and its scores, labelled with the turn index.

    `turn` is carried on the record rather than inferred from position by the caller:
    the spans of a two-turn thread are the same shape, and which turn produced them is
    the first thing a reviewer asks of them.
    """
    if not trace_id:
        return {"turn": turn, "trace_id": "", "error": "no trace id for this turn"}
    try:
        trace = client.api.trace.get(
            trace_id, request_options={"timeout_in_seconds": timeout})
    except Exception as exc:  # noqa: BLE001 — an export may not lose the run
        return {"turn": turn, "trace_id": trace_id,
                "error": f"unreadable: {type(exc).__name__}"}

    spans = []
    for observation in getattr(trace, "observations", None) or []:
        spans.append({name: getattr(observation, name, None) for name in _SPAN_FIELDS})

    return {
        "turn": turn,
        "trace_id": trace_id,
        "name": getattr(trace, "name", None),
        "input": getattr(trace, "input", None),
        "output": getattr(trace, "output", None),
        "latency": getattr(trace, "latency", None),
        "spans": spans,
        "scores": [
            {"name": getattr(score, "name", None),
             "value": getattr(score, "value", None),
             "comment": getattr(score, "comment", None)}
            for score in (getattr(trace, "scores", None) or [])
        ],
    }


def export_item(client: Any, output: dict[str, Any], **kwargs: Any) -> list[dict[str, Any]]:
    """Every turn of one item, in order, each with its spans attached."""
    return [
        export_turn(client, trace_id, turn=index + 1, **kwargs)
        for index, trace_id in enumerate(trace_ids_for_turns(output))
    ]
