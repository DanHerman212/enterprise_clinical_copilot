"""Delete one experiment run and every trace it produced.

Used to remove `pilot-verify-20261020-175015`, whose runs were produced by a harness
that was wrong (10 of 30 suite-2 items failed on assertion defects, and the judge's
reasoning was not recorded), so the data cannot be explained and must not be left in
the audit trail.

Two sources, because one experiment run creates traces in two places:
  * `experiment-item-run` traces, reachable through the dataset run's items
  * the agent's own traces, tagged `run:<run-name>`

    PROJECT_ID=... LANGFUSE_HOST=... LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=... \
    ./.venv/bin/python evaluation/platform/purge_run.py --run-name pilot-verify-...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

def _datasets() -> tuple[str, ...]:
    """Every dataset a run can be on.

    Imported rather than restated: the gated suites were added to `run_evals` and not
    here, so a purge of a dev-suite run deleted the traces and left the dataset run
    behind — half a deletion, reported as a success (2026-10-29).
    """
    from evaluation.platform.run_evals import (
        DATASETS as DEFAULTS,
        DEV_SUITE_DATASET,
        HOLDOUT_SUITE_DATASET,
    )

    return (*DEFAULTS, DEV_SUITE_DATASET, HOLDOUT_SUITE_DATASET)


def _run_item_trace_ids(run) -> list[str]:
    ids: list[str] = []
    for item in (getattr(run, "dataset_run_items", None) or []):
        trace_id = getattr(item, "trace_id", None)
        if trace_id:
            ids.append(str(trace_id))
    return ids


def purge(run_name: str, datasets: tuple[str, ...]) -> dict[str, int]:
    from langfuse import Langfuse

    client = Langfuse()
    counts = {"dataset_runs": 0, "tagged_traces": 0, "deleted_traces": 0}

    # 1. Dataset runs, and the experiment traces they point at.
    trace_ids: set[str] = set()
    for dataset in datasets:
        try:
            run = client.get_dataset_run(dataset_name=dataset, run_name=run_name)
        except Exception:  # noqa: BLE001 — absent run is not an error here
            print(f"  no run {run_name!r} on {dataset}")
            continue
        ids = _run_item_trace_ids(run)
        trace_ids.update(ids)
        client.delete_dataset_run(dataset_name=dataset, run_name=run_name)
        counts["dataset_runs"] += 1
        print(f"  deleted dataset run {dataset}/{run_name} ({len(ids)} item traces)")

    # 2. Agent-side traces for the same run, which the dataset run does not own.
    page = 1
    while True:
        response = client.api.trace.list(tags=[f"run:{run_name}"], limit=100, page=page)
        rows = getattr(response, "data", None) or []
        if not rows:
            break
        for row in rows:
            trace_ids.add(str(row.id))
            counts["tagged_traces"] += 1
        page += 1

    if trace_ids:
        ordered = sorted(trace_ids)
        for start in range(0, len(ordered), 50):
            batch = ordered[start:start + 50]
            client.api.trace.delete_multiple(trace_ids=batch)
            counts["deleted_traces"] += len(batch)
    client.flush()
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", required=True)
    args = ap.parse_args(argv)

    print(f"purging run {args.run_name}")
    counts = purge(args.run_name, _datasets())
    print(f"  {counts}")

    from langfuse import Langfuse

    client = Langfuse()
    remaining = client.api.trace.list(tags=[f"run:{args.run_name}"], limit=1)
    print(f"  traces still tagged run:{args.run_name}: "
          f"{getattr(remaining.meta, 'total_items', 'unknown')}")
    client.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
