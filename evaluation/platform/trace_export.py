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

import time
from typing import Any

from evaluation.platform import pricing

#: Fields copied off an observation. Chosen because they are what a failure is read
#: from: which step it was, what went in, what came out, and whether it was levelled.
_SPAN_FIELDS = ("id", "name", "type", "input", "output", "level",
                "status_message", "start_time", "end_time")

#: The observation type that carries token usage. A `SPAN` and an `EVENT` carry none, and
#: a `GENERATION` that a provider answered without reporting usage carries none either —
#: which is a different thing from zero, and the difference is what `unattributed` says.
GENERATION_TYPE = "GENERATION"


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


# ===========================================================================
# Token usage, read back from the generations a turn actually made.
#
# The agent's wire contract (`AgentSuccess`) carries no token counts, and Sprint 03 is
# explicitly forbidden from adding them: the numbers a run is costed from are the ones
# the provider reported to the tracer, so they are read from the trace rather than
# accepted from the service being measured. That also makes the cost independent of the
# candidate: a model that reports its usage sloppily is costed from what the tracer saw.
#
# A turn is not one model call. `/ehr` runs a command that may extract JSON, then a
# reasoning loop that calls tools round by round, then a formatting turn; a `/meds` turn
# adds an extraction call of its own. Summing only the first generation, or only the
# root, understates the bill by a factor that varies per model — which would make the
# cheaper arm look like the cheaper arm by accident of call shape. So every GENERATION
# observation inside every turn's trace is summed, and the count is reported beside the
# total so a reader can see how many calls produced it.
# ===========================================================================


#: The names Langfuse's LangChain handler gives the counts it splits out of a provider's
#: usage block, and the term this harness calls each one.
#:
#: The handler subtracts a sub-category from its parent before storing it — `output`
#: becomes the tokens the reader saw and `output_reasoning` the ones the model spent
#: thinking — which is the opposite convention to the raw provider payload, where reasoning
#: is inside `output`. Read without translating, a turn whose model thought for 3,929
#: tokens and answered in 163 was costed at 163 tokens of output: `trace_export` reported
#: `agent_cost_usd 0.001632` on 2026-11-13 where the trace's own `calculated_total_cost`
#: was `0.01636575`, a factor of ten, in the column the arms are compared on.
#:
#: The mapping is by name rather than by arithmetic, so nothing is inferred: the two counts
#: are reported, and `pricing.billable_output` decides from `total` whether the reasoning
#: has already been billed once.
_LANGFUSE_SUBCOUNTS: tuple[tuple[str, str], ...] = (
    ("output_reasoning", "thinking"),
    ("input_cache_read", "cached"),
    ("input_cache_write", "cache_write"),
)


def _langfuse_terms(block: dict[str, Any]) -> dict[str, Any]:
    """One stored usage block, with Langfuse's sub-category names read as ours.

    A name that is already present is left alone, because a block that reports both
    spellings is reporting the same count twice and the first one wins.
    """
    out = dict(block)
    for source, target in _LANGFUSE_SUBCOUNTS:
        if source in out and target not in out:
            out[target] = out.pop(source)
    return out


def _observation_usage(observation: Any) -> dict[str, int | None]:
    """One generation's usage, normalised. Reads `usage_details` then `usage`."""
    block: Any = getattr(observation, "usage_details", None)
    if block is None:
        block = getattr(observation, "usage", None)
    if block is not None and hasattr(block, "model_dump"):
        try:
            block = block.model_dump()
        except Exception:  # noqa: BLE001 — a usage block may not fail an export
            block = None
    if not isinstance(block, dict):
        return pricing.normalise_usage(None)
    return pricing.normalise_usage(_langfuse_terms(block))


def _is_generation(observation: Any) -> bool:
    value = getattr(observation, "type", None)
    kind = str(getattr(value, "value", value) or "").upper()
    return kind == GENERATION_TYPE


def generation_usages(trace: Any) -> list[dict[str, Any]]:
    """Every generation in one trace, as `{model, usage}` pairs, in observation order."""
    found: list[dict[str, Any]] = []
    for observation in getattr(trace, "observations", None) or []:
        if not _is_generation(observation):
            continue
        found.append({
            "model": str(getattr(observation, "model", "") or "").strip(),
            "usage": _observation_usage(observation),
        })
    return found


def _totals(generations: list[dict[str, Any]]) -> dict[str, Any]:
    """Token totals and cost for a set of generations.

    `cost_usd` is `None` — not zero — as soon as one generation cannot be priced, and the
    reason is reported: an unpriced model would otherwise be costed at nothing and make
    the run look cheap in exactly the column it is being compared on.
    """
    merged = pricing.merge_usage(generation["usage"] for generation in generations)
    total = 0.0
    priced = True
    unpriced: list[str] = []
    for generation in generations:
        model = generation["model"]
        if not model:
            unpriced.append("(unreported)")
            priced = False
            continue
        try:
            total += pricing.cost_usd(model, generation["usage"])
        except pricing.UnknownModelError:
            unpriced.append(model)
            priced = False
    return {
        **merged,
        # What the cost was computed from, stated beside the counts it was computed from:
        # `output` is what the reader saw and `billable_output` is what the provider
        # charges for, and on a model that thinks they are different numbers.
        "billable_input": pricing.billable_input(merged),
        "billable_output": pricing.billable_output(merged),
        "cost_usd": total if priced else None,
        "models": sorted({generation["model"] for generation in generations
                          if generation["model"]}),
        "unpriced_models": sorted(set(unpriced)),
        "generations": len(generations),
    }


def turn_usage(client: Any, trace_id: str, *, turn: int,
               attempts: int = 4, delay: float = 1.5,
               timeout: int = 90) -> dict[str, Any]:
    """One turn's token usage and cost, from the generations inside its trace.

    The poll is short and bounded on purpose. Langfuse ingests asynchronously, so a read
    immediately after the answer has landed can return a trace whose GENERATION
    observations have not arrived yet — and reporting that as "no usage" would cost an
    item at zero for a reason that fixes itself in a second. Four attempts with a 1.5s
    beat covers ingestion lag without holding the run's summary open indefinitely.

    "No generations found" stays an unattributed result rather than a zero: it means the
    trace was read and held nothing to price, which is a fact about the tracing, not a
    price of zero.
    """
    if not trace_id:
        return {"turn": turn, "trace_id": "", "error": "no trace id for this turn",
                **_totals([]), "cost_usd": None, "unattributed": True,
                "reason": "no trace id for this turn"}

    last_error = ""
    for attempt in range(max(1, attempts)):
        try:
            trace = client.api.trace.get(
                trace_id, request_options={"timeout_in_seconds": timeout})
        except Exception as exc:  # noqa: BLE001 — an export may not lose the run
            last_error = f"unreadable: {type(exc).__name__}"
        else:
            generations = generation_usages(trace)
            if generations:
                summary = _totals(generations)
                return {
                    "turn": turn,
                    "trace_id": trace_id,
                    **summary,
                    "unattributed": summary["cost_usd"] is None,
                    "reason": (f"unpriced model(s): {summary['unpriced_models']}"
                               if summary["cost_usd"] is None else None),
                    "error": None,
                }
            last_error = "no GENERATION observation in this trace"
        if attempt < attempts - 1:
            time.sleep(delay)

    return {"turn": turn, "trace_id": trace_id, "error": last_error,
            **_totals([]), "cost_usd": None, "unattributed": True,
            "reason": last_error}


def scenario_usage(client: Any, output: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Every turn of one scenario, summed, with the per-turn detail kept.

    The sum is across turns: a two-turn item's cost is what both requests cost, and a
    per-turn mean is only meaningful when the numerator is honest about how many turns
    each item actually ran.
    """
    ids = trace_ids_for_turns(output)
    turns = [turn_usage(client, trace_id, turn=index + 1, **kwargs)
             for index, trace_id in enumerate(ids)]
    merged = pricing.merge_usage(
        {key: turn.get(key) for key in ("input", "cached", "output", "cache_write",
                                        "thinking", "total")}
        for turn in turns)
    costs = [turn.get("cost_usd") for turn in turns]
    priced_values = [float(cost) for cost in costs if isinstance(cost, (int, float))]
    priced = bool(turns) and len(priced_values) == len(costs)
    reasons = [str(turn["reason"]) for turn in turns if turn.get("reason")]
    return {
        "turns": turns,
        "turn_count": len(turns),
        **merged,
        "billable_input": pricing.billable_input(merged),
        "billable_output": pricing.billable_output(merged),
        "cost_usd": (sum(priced_values) if priced else None),
        "generations": sum(int(turn.get("generations") or 0) for turn in turns),
        "models": sorted({model for turn in turns for model in turn.get("models") or []}),
        "unattributed": not priced,
        "reason": ("; ".join(dict.fromkeys(reasons)) if reasons else None),
    }

