"""Upload the multi-model selection slice to Langfuse as one dataset, idempotently.

The slice is authored in `evaluation/datasets/selection_slice.jsonl`; this script is the
only thing that writes it to Langfuse. Zero LLM calls: it reads the file, resolves each
record's pre-flight manifest from BigQuery through the same decoder the service uses, and
writes items.

Default is a dry validation (what would be written, and whether the file holds up);
`--commit` performs the writes and then reads the dataset back to count what landed.

    PROJECT_ID=... LANGFUSE_HOST=... LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=... \\
    ./.venv/bin/python evaluation/platform/sync_selection_slice.py --commit

Item identity is the scenario id (`sel_t1_01` … `sel_t7_02`). Langfuse items are immutable
in place and this version cannot delete a dataset, so a re-run deletes the items it
recognises and recreates them — re-running the sync updates the slice instead of doubling
it.

The dataset name is version-locked: `ecc-model-selection-slice-v1`. A later slice gets a new
name rather than being rewritten in place, because a run's scores are only comparable with
the items that were there when it ran.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evaluation.platform.selection_slice import (  # noqa: E402
    EXPECTED_ITEMS,
    PER_TRAJECTORY,
    SELECTION_SLICE_PATH,
    SELECTED_HADM_IDS,
    TRAJECTORIES,
    item_payload,
    read_rows,
    validate_rows,
)

DATASET = "ecc-model-selection-slice-v1"
DESCRIPTION = (
    "Multi-model selection slice: 14 scenarios, 5 previously unused records, "
    "2 per trajectory across the 7 interaction and adversarial trajectories"
)
RUBRIC_VERSION = "v8"


def _manifest_provider():
    from evaluation.platform.fixtures import manifest_from_bigquery  # noqa: PLC0415

    return manifest_from_bigquery


def build_items(*, manifests: dict[int, dict]) -> list[dict]:
    """Every row as a Langfuse item payload: input, expected_output, metadata.

    The manifest travels in `input` because it is pre-flight context the service gives the
    agent before any tool runs — if it is not on the item, the judge never sees it.
    """
    rows = read_rows()
    validate_rows(rows)
    payloads: list[dict] = []
    for row in rows:
        manifest = manifests.get(int(row["hadm_id"])) or {}
        expectations = item_payload(row, manifest)["expectations"]
        metadata = dict(row.get("metadata") or {})
        metadata.update({
            "scenario_id": row["id"],
            "rubric_version": RUBRIC_VERSION,
            "trajectory_index": TRAJECTORIES[row["trajectory"]],
            "hadm_id": str(row["hadm_id"]),
            "archetype": row["archetype"],
            "purpose": "multi-model-selection-slice",
        })
        payloads.append({
            "id": row["id"],
            "input": {
                "scenario_id": row["id"],
                # `patient_id` is the field the task reads the record from, and the
                # manifest is keyed by it. `hadm_id` travels beside it because the
                # directive names it, and `task.DatasetItemInput` reads `patient_id`.
                "patient_id": str(row["hadm_id"]),
                "hadm_id": str(row["hadm_id"]),
                "trajectory": row["trajectory"],
                "trajectory_index": TRAJECTORIES[row["trajectory"]],
                "workflow_type": row["workflow_type"],
                "command": row.get("command"),
                "turns": list(row["turns"]),
                "patient_manifest": manifest,
                # The assertions evaluator reads these from the item input; without them
                # the deterministic half of the score is a no-op that reports ok.
                "expectations": expectations,
            },
            "expected_output": row["expected_output"],
            "metadata": metadata,
        })
    return payloads


def existing_item_ids(client, dataset_name: str) -> list[str]:
    """Every item id in the dataset, paged. Empty when the dataset does not exist."""
    ids: list[str] = []
    page = 1
    while True:
        response = client.api.dataset_items.list(
            dataset_name=dataset_name, limit=100, page=page)
        rows = getattr(response, "data", None) or []
        if not rows:
            return ids
        ids.extend(str(row.id) for row in rows)
        page += 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commit", action="store_true",
                    help="write the dataset to Langfuse (default: validate only)")
    ap.add_argument("--verify", action="store_true",
                    help="only read the dataset back and report its item count")
    ap.add_argument("--dataset", default=DATASET)
    args = ap.parse_args(argv)

    if args.verify:
        from langfuse import Langfuse  # noqa: PLC0415

        ids = existing_item_ids(Langfuse(), args.dataset)
        print(f"{args.dataset}: {len(ids)} item(s)")
        print("  " + " ".join(sorted(ids)) if ids else "  (empty)")
        return 0 if len(ids) == EXPECTED_ITEMS else 1

    rows = read_rows()
    validate_rows(rows)
    print(f"{SELECTION_SLICE_PATH.relative_to(REPO)}: {len(rows)} scenarios")
    for name, index in TRAJECTORIES.items():
        count = sum(1 for row in rows if row["trajectory"] == name)
        print(f"  trajectory {index}: {name} — {count} scenario(s) "
              f"(expected {PER_TRAJECTORY})")
    print("  records: " + ", ".join(str(hadm) for hadm in SELECTED_HADM_IDS))

    provider = _manifest_provider()
    hadm_ids = sorted({int(row["hadm_id"]) for row in rows})
    manifests = provider(hadm_ids)
    missing = [hadm for hadm in hadm_ids if not manifests.get(hadm)]
    if missing:
        raise SystemExit(f"no manifest for {missing}; refusing to write items that "
                         "would reach the judge without pre-flight context")

    payloads = build_items(manifests=manifests)
    if len(payloads) != EXPECTED_ITEMS:
        raise SystemExit(f"{len(payloads)} items built, expected {EXPECTED_ITEMS}")

    if not args.commit:
        print(f"\nwould write {len(payloads)} items to dataset {args.dataset!r} "
              f"(rubric {RUBRIC_VERSION}) — pass --commit to write")
        return 0

    from langfuse import Langfuse  # noqa: PLC0415

    client = Langfuse()
    client.create_dataset(
        name=args.dataset,
        description=DESCRIPTION,
        metadata={
            "rubric_version": RUBRIC_VERSION,
            "purpose": "multi-model-selection-slice",
            "scenario_count": EXPECTED_ITEMS,
            "patient_count": len(SELECTED_HADM_IDS),
            "source": "evaluation/datasets/selection_slice.jsonl",
        },
        # No input/expected_output schema: this dataset's expected output carries the
        # ground-truth facts and the behavioural expectation, which no envelope model
        # describes. Declaring one silently drops the other.
    )

    wanted = {payload["id"] for payload in payloads}
    stale = [item_id for item_id in existing_item_ids(client, args.dataset)
             if item_id in wanted]
    for item_id in stale:
        client.api.dataset_items.delete(item_id)
    if stale:
        print(f"replaced {len(stale)} existing item(s)")

    for payload in payloads:
        client.create_dataset_item(
            dataset_name=args.dataset,
            id=payload["id"],
            input=payload["input"],
            expected_output=payload["expected_output"],
            metadata=payload["metadata"],
        )
    client.flush()

    ids = existing_item_ids(client, args.dataset)
    print(f"{args.dataset}: {len(ids)} item(s) after upload "
          f"(expected {EXPECTED_ITEMS})")
    if len(ids) != EXPECTED_ITEMS:
        print("  count mismatch: " + " ".join(sorted(ids)))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
