"""Upload the Tier 2 holdout suite to Langfuse as one dataset, idempotently.

The twin of `sync_dev_suite.py`, for the 15 rows in
`evaluation/datasets/holdout_suite.jsonl`. Separate rather than generalised because the
two suites are validated against OPPOSITE record sets: `dev_suite.validate_rows` refuses
any row whose record is a holdout, and this module refuses any row whose record is not
one. One shared validator would have to be told which side it was checking, and the
whole point of the split is that neither file can drift into the other's cohort by
accident.

Zero LLM calls: it reads the file, resolves each record's pre-flight manifest from
BigQuery through the same decoder the service uses, and writes items.

Default is a dry validation; `--commit` performs the writes and then reads the dataset
back to count what landed.

    PROJECT_ID=... LANGFUSE_HOST=... LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=... \\
    ./.venv/bin/python evaluation/platform/sync_holdout_suite.py --commit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evaluation.platform.dev_suite import (  # noqa: E402
    DEV_HADM_IDS,
    HOLDOUT_HADM_IDS,
    TRAJECTORIES,
    item_payload,
)

DATASET = "ecc-tier2-holdout-suite"
HOLDOUT_SUITE_PATH = REPO / "evaluation" / "datasets" / "holdout_suite.jsonl"
DESCRIPTION = ("Tier 2 blind holdout suite: 15 scenarios over 4 unseen records, "
               "7 interaction and adversarial trajectories")
RUBRIC_VERSION = "v6"
EXPECTED_ITEMS = 15


class HoldoutSuiteError(ValueError):
    """A row the holdout suite cannot run. Raised at load, not at scoring time."""


def read_rows(path: Path | None = None) -> list[dict[str, Any]]:
    """Every row, parsed. No model, no network."""
    source = path or HOLDOUT_SUITE_PATH
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(source.read_text().splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            rows.append(json.loads(text))
        except json.JSONDecodeError as exc:
            raise HoldoutSuiteError(f"{source.name} line {number}: {exc}") from exc
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> None:
    """The invariants that make the suite a HOLDOUT, checked before it is uploaded.

    The two cohort rules are the point of the file, so they are enforced here rather
    than left to review:

    * every record is one of the reserved holdout ids;
    * no record is one of the dev ids — the suites must not share a patient, or the
      comparison between them measures memorisation instead of generalisation.

    The rest mirror `dev_suite.validate_rows`: unique ids, a known trajectory, turns
    that fit the workflow type, and an `expected_output` that is never absent.
    """
    if len(rows) != EXPECTED_ITEMS:
        raise HoldoutSuiteError(
            f"{len(rows)} row(s), expected {EXPECTED_ITEMS}")

    seen: set[str] = set()
    for row in rows:
        scenario = str(row.get("id") or "").strip()
        if not scenario:
            raise HoldoutSuiteError("every scenario needs an id")
        if scenario in seen:
            raise HoldoutSuiteError(f"duplicate scenario id {scenario}")
        seen.add(scenario)

        hadm = str(row.get("hadm_id") or "")
        if hadm in DEV_HADM_IDS:
            raise HoldoutSuiteError(f"{scenario}: {hadm} is a dev record")
        if hadm not in HOLDOUT_HADM_IDS:
            raise HoldoutSuiteError(f"{scenario}: {hadm} is not a holdout record")

        trajectory = str(row.get("trajectory") or "")
        if trajectory not in TRAJECTORIES:
            raise HoldoutSuiteError(f"{scenario}: unknown trajectory {trajectory!r}")

        turns = row.get("turns")
        if not isinstance(turns, list) or not turns:
            raise HoldoutSuiteError(f"{scenario}: turns must be a non-empty list")
        if len(turns) > 3:
            raise HoldoutSuiteError(f"{scenario}: {len(turns)} turns exceeds 3")

        expected = row.get("expected_output")
        if not isinstance(expected, dict) or not expected:
            raise HoldoutSuiteError(f"{scenario}: expected_output is mandatory")

        workflow = row.get("workflow_type")
        wanted = {1: {"single_probe"}, 2: {"command_plus_followup"},
                  3: {"multi_turn"}}[len(turns)]
        if workflow not in wanted:
            raise HoldoutSuiteError(
                f"{scenario}: {len(turns)} turn(s) need workflow_type in {wanted}, "
                f"got {workflow!r}")


def build_items(*, manifests: dict[int, dict]) -> list[dict[str, Any]]:
    """Every row as a Langfuse item payload. Same shape as the dev suite's."""
    rows = read_rows()
    validate_rows(rows)
    payloads: list[dict[str, Any]] = []
    for row in rows:
        manifest = manifests.get(int(row["hadm_id"])) or {}
        expectations = item_payload(row, manifest)["expectations"]
        metadata = dict(row.get("metadata") or {})
        metadata.update({
            "scenario_id": row["id"],
            "rubric_version": RUBRIC_VERSION,
            "trajectory_index": TRAJECTORIES[row["trajectory"]],
        })
        payloads.append({
            "id": row["id"],
            "input": {
                "scenario_id": row["id"],
                "patient_id": str(row["hadm_id"]),
                "hadm_id": str(row["hadm_id"]),
                "trajectory": row["trajectory"],
                "trajectory_index": TRAJECTORIES[row["trajectory"]],
                "workflow_type": row["workflow_type"],
                "command": row.get("command"),
                "turns": list(row["turns"]),
                "patient_manifest": manifest,
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
    records = sorted({str(row["hadm_id"]) for row in rows})
    print(f"{HOLDOUT_SUITE_PATH.relative_to(REPO)}: {len(rows)} scenarios")
    for name, index in TRAJECTORIES.items():
        count = sum(1 for row in rows if row["trajectory"] == name)
        print(f"  trajectory {index}: {name} — {count} scenario(s)")
    print(f"  records: {', '.join(records)}")
    overlap = sorted(set(records) & set(DEV_HADM_IDS))
    if overlap:
        raise SystemExit(f"dev records in the holdout suite: {overlap}")

    from evaluation.platform.fixtures import manifest_from_bigquery  # noqa: PLC0415

    manifests = manifest_from_bigquery([int(hadm) for hadm in records])
    missing = [hadm for hadm in records if not manifests.get(int(hadm))]
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
        metadata={"rubric_version": RUBRIC_VERSION, "workflow": "tier2-holdout-suite",
                  "source": "evaluation/datasets/holdout_suite.jsonl"},
        # No input/expected_output schema, for the same reason the dev sync omits one:
        # this dataset's expected output carries ground-truth facts and a behavioural
        # expectation that no envelope model describes.
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
