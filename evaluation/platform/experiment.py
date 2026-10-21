"""The experiment loop, with every score written onto the agent's own trace.

Langfuse's `run_experiment` opens a trace of its own for each dataset item
(`experiment-item-run`) and writes the item's scores there. Our task talks to a
deployed service that has already opened a trace and tagged it `run:<run name>`. Two
traces per item, and the scores land on the untagged one — which is why the Scores
view's trace-tag filter matched nothing on 2026-10-21 while 814 scores existed.

This module runs the same experiment without that wrapper. The item and its scores
are created against the `langfuse_trace_id` the service returns, so:

* the Scores view filters by run tag, because score and tag are on one trace; and
* opening a score shows the prompts, tool calls and model output that produced it,
  rather than an empty container beside a link.

The service's id is the only acceptable link. If a response carries none, the item keeps
its verdict but is counted `unattributed` and its scores are not written: the number was
measured, and the record cannot say against what. A run with any unattributed item
refuses to pass, because it is a run whose evidence cannot be opened.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from langfuse import Evaluation

from evaluation.platform import evaluators as ev

#: The field the service returns for the trace it opened while answering.
TRACE_FIELD = "langfuse_trace_id"


@dataclass
class ItemOutcome:
    """One item's result, in the shape `summarise_item_results` reads."""

    item_id: str
    trace_id: str | None
    evaluations: list[Evaluation] = field(default_factory=list)
    dataset_run_id: str | None = None
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class DatasetRun:
    """What `run_linked` hands back: the counts, and the per-item record."""

    stats: dict[str, Any]
    item_results: list[ItemOutcome]
    dataset_run_id: str | None


def _field(item: Any, name: str) -> Any:
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)


def _unscored(reason: str) -> Evaluation:
    """The same 'no verdict to trust' record the judge path uses.

    A distinct name from `judge_error` would need a second branch in the summariser;
    the thing being said is identical — this item was not measured — so it is said in
    the same words and counted the same way.
    """
    return Evaluation(name=ev.JUDGE_ERROR, value=1.0,
                      comment=f"unscored — {reason}", data_type="NUMERIC")


def _evaluate(evaluators: list[Callable[..., list[Evaluation]]],
              composite: Callable[..., Evaluation] | None,
              *, item_input: Any, output: dict[str, Any],
              expected_output: Any, item_metadata: Any) -> list[Evaluation]:
    """Assertions first, then the composite — the order the rubric requires."""
    kwargs = dict(input=item_input or {}, output=output or {},
                  expected_output=expected_output, metadata=item_metadata)
    evaluations: list[Evaluation] = []
    for evaluator in evaluators:
        evaluations.extend(evaluator(**kwargs))

    if composite is not None:
        evaluations.append(composite(evaluations=evaluations, **kwargs))
    return evaluations


def run_linked(
    client: Any,
    *,
    dataset_name: str,
    run_name: str,
    items: list[Any],
    task: Callable[..., dict[str, Any]],
    evaluators: list[Callable[..., list[Evaluation]]],
    composite: Callable[..., Evaluation] | None = None,
    metadata: dict[str, Any] | None = None,
    description: str | None = None,
    max_concurrency: int = 2,
) -> DatasetRun:
    """Run one dataset, writing each item's scores onto the trace that answered it."""
    total = len(items)
    done = 0

    def one(item: Any) -> ItemOutcome:
        nonlocal done
        item_id = str(_field(item, "id") or "")
        item_input = _field(item, "input")
        item_metadata = _field(item, "metadata")
        expected_output = _field(item, "expected_output")

        try:
            output = task(item=item) or {}
        except Exception as exc:  # noqa: BLE001 — one bad item must not end the run
            outcome = ItemOutcome(item_id=item_id, trace_id=None,
                                  evaluations=[_unscored(f"task failed: {exc}")],
                                  error=str(exc))
            done += 1
            print(f"  [{done}/{total}] {item_id[:8]} ERROR {exc}", flush=True)
            return outcome

        trace_id = str(output.get(TRACE_FIELD) or "") or None
        evaluations = _evaluate(evaluators, composite, item_input=item_input,
                                output=output, expected_output=expected_output,
                                item_metadata=item_metadata)

        if trace_id is None:
            # No id, no attribution. The verdict stands — the answer was measured — but it
            # is not written anywhere a reader could open it, and the run says so.
            print(f"  {item_id[:8]} no {TRACE_FIELD} in the response; "
                  "scores not written", flush=True)

        run_id: str | None = None
        try:
            row = client.api.dataset_run_items.create(
                run_name=run_name,
                dataset_item_id=item_id,
                run_description=description,
                metadata=metadata,
                trace_id=trace_id,
            )
            run_id = getattr(row, "dataset_run_id", None)
        except Exception as exc:  # noqa: BLE001
            print(f"  {item_id[:8]} run item failed: {exc}", flush=True)

        if trace_id is not None:
            for evaluation in evaluations:
                client.create_score(
                    trace_id=trace_id,
                    name=evaluation.name,
                    value=evaluation.value,
                    comment=evaluation.comment,
                    data_type=evaluation.data_type,
                )

        done += 1
        verdict = next((e.value for e in evaluations if e.name == ev.VERDICT), None)
        state = "unscored" if verdict is None else f"verdict={verdict:g}"
        print(f"  [{done}/{total}] {item_id[:8]} {state}", flush=True)
        return ItemOutcome(item_id=item_id, trace_id=trace_id, evaluations=evaluations,
                           dataset_run_id=run_id, output=output)

    with ThreadPoolExecutor(max_workers=max(1, max_concurrency)) as pool:
        outcomes = list(pool.map(one, items))

    stats = ev.summarise_item_results(outcomes)
    # An answer we obtained but cannot point at. A task that failed is already counted
    # `unscored`; counting it here as well would report one gap twice.
    stats["unattributed"] = sum(
        1 for outcome in outcomes if outcome.trace_id is None and outcome.error is None)
    dataset_run_id = next((o.dataset_run_id for o in outcomes if o.dataset_run_id), None)

    return DatasetRun(stats=stats, item_results=outcomes, dataset_run_id=dataset_run_id)


def retry_unscored(
    client: Any,
    *,
    items: list[Any],
    outcomes: list[ItemOutcome],
    evaluators: list[Callable[..., list[Evaluation]]],
    composite: Callable[..., Evaluation] | None = None,
) -> dict[str, Any]:
    """Re-judge items that settled unscored, once, at concurrency 1.

    Only the evaluators run again. The task is not re-sent: the answer exists, and
    re-running it would spend a second call, a second trace and a second attribution for
    a defect in the JUDGE. That is also why this is safe to do at the end of a run — a
    2.9-hour census that loses one item to a slow judge must not be thrown away, and one
    quiet retry is the cheapest way to keep it.

    Only scores the first pass did not write are written now, so the trace keeps the
    `judge_error` as the record of the attempt that failed and gains the verdict the
    second attempt earned. Returns the updated stats.
    """
    by_id = {str(_field(item, "id") or ""): item for item in items}
    recovered = 0
    for outcome in outcomes:
        names = {evaluation.name for evaluation in outcome.evaluations}
        if ev.VERDICT in names or ev.JUDGE_ERROR not in names:
            continue
        item = by_id.get(outcome.item_id)
        if item is None or outcome.trace_id is None:
            continue
        fresh = _evaluate(evaluators, composite,
                          item_input=_field(item, "input"), output=outcome.output,
                          expected_output=_field(item, "expected_output"),
                          item_metadata=_field(item, "metadata"))
        verdict = next((e for e in fresh if e.name == ev.VERDICT), None)
        if verdict is None:
            print(f"  retry {outcome.item_id[:8]}: still unscored", flush=True)
            continue
        for evaluation in fresh:
            if evaluation.name in names:
                continue
            client.create_score(trace_id=outcome.trace_id, name=evaluation.name,
                                value=evaluation.value, comment=evaluation.comment,
                                data_type=evaluation.data_type)
        outcome.evaluations = fresh
        recovered += 1
        print(f"  retry {outcome.item_id[:8]}: verdict={verdict.value:g}", flush=True)

    print(f"  retry pass recovered {recovered} item(s)", flush=True)
    stats = ev.summarise_item_results(outcomes)
    stats["unattributed"] = sum(
        1 for outcome in outcomes if outcome.trace_id is None and outcome.error is None)
    return stats


def record_run_scores(
    client: Any,
    *,
    dataset_run_id: str | None,
    outcomes: list[ItemOutcome],
    run_evaluators: list[Callable[..., list[Evaluation]]],
) -> None:
    """The run-level gate, written against the run once every item is final.

    Separate from `run_linked` because a retry pass can change an item after the loop
    ends, and a gate score written before that would say a run failed when it did not —
    the Datasets view reads this number, not the console.
    """
    if dataset_run_id is None:
        return
    for evaluator in run_evaluators:
        for evaluation in evaluator(item_results=outcomes):
            client.create_score(dataset_run_id=dataset_run_id, name=evaluation.name,
                                value=evaluation.value, comment=evaluation.comment,
                                data_type=evaluation.data_type)
