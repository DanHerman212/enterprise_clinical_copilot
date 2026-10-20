"""Seed the three datasets into Langfuse.

Default is a dry validation: build every item, check counts and manifests, print what
*would* be created. `--commit` performs the writes.

Why the default is dry: this Langfuse version has no dataset deletion (verified
2026-10-20 — `DELETE /api/public/datasets/{name}` returns 405), so a dataset created
with the wrong schema is permanent. Validate first, write second.

`--replace` exists because items are immutable in place: the dataset cannot be deleted,
so the only way to change what an item carries is to delete the items and recreate them.
Re-seeding without it appends a second copy of every item and the pass rate is then
computed over a doubled set.

    PROJECT_ID=... LANGFUSE_HOST=... LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=... \
    ./.venv/bin/python evaluation/platform/seed.py \
        --datasets suite2-adversarial-v1 --replace --commit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evaluation.platform.fixtures import (  # noqa: E402
    EXPECTED_COUNTS,
    all_datasets,
)
from evaluation.platform.schemas import (  # noqa: E402
    DatasetItemInput,
    DeterministicExpectations,
)

DESCRIPTIONS = {
    "clinical-census-pilot-v2": "1 patient x 4 workflows, two turns each (smoke)",
    "clinical-census-full-v1": "82 patients x 4 workflows, two turns each (census)",
    "suite2-adversarial-v1": "30 single-probe modality and boundary cases",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commit", action="store_true",
                    help="write the datasets to Langfuse (default: validate only)")
    ap.add_argument("--datasets", default="",
                    help="comma-separated subset to seed (default: all three)")
    ap.add_argument("--replace", action="store_true",
                    help="delete existing items before creating (items are immutable)")
    args = ap.parse_args(argv)

    datasets = all_datasets()
    if args.datasets.strip():
        wanted = [n.strip() for n in args.datasets.split(",") if n.strip()]
        unknown = [n for n in wanted if n not in datasets]
        if unknown:
            raise SystemExit(f"unknown dataset(s): {unknown}; have {sorted(datasets)}")
        datasets = {n: datasets[n] for n in wanted}

    for name, items in datasets.items():
        print(f"{name}: {len(items)} items (expected {EXPECTED_COUNTS[name]})")
        sample = items[0]
        print(f"  sample patient={sample.patient_id} turns={len(sample.turns)} "
              f"manifest_keys={len(sample.patient_manifest)}")

    if not args.commit:
        print("\nvalidation only — pass --commit to write to Langfuse")
        return 0

    from langfuse import Langfuse  # noqa: PLC0415

    client = Langfuse()
    for name, items in datasets.items():
        if args.replace:
            removed = 0
            page = 1
            while True:
                response = client.api.dataset_items.list(
                    dataset_name=name, limit=100, page=page)
                rows = getattr(response, "data", None) or []
                if not rows:
                    break
                for row in rows:
                    client.api.dataset_items.delete(str(row.id))
                    removed += 1
                page += 1
            print(f"removed {removed} existing items from {name}")

        client.create_dataset(
            name=name,
            description=DESCRIPTIONS.get(name, ""),
            metadata={"rubric_version": "v5", "workflow": "platform-owned"},
            input_schema=DatasetItemInput.model_json_schema(),
            # The dataset asserts ENVELOPE expectations, not a finished answer. Using
            # TaskOutput here made the server reject every census item, because an
            # item with no expected answer cannot satisfy a required-answer shape.
            expected_output_schema=DeterministicExpectations.model_json_schema(),
        )
        for item in items:
            client.create_dataset_item(
                dataset_name=name,
                input=item.model_dump(),
                expected_output=DeterministicExpectations(
                    **item.expectations).model_dump(),
                metadata={"command": item.command,
                          "workflow_type": item.workflow_type},
            )
        client.flush()
        print(f"seeded {name}: {len(items)} items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
