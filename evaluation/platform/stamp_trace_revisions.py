"""Backward fill: stamp the serving revision onto existing Langfuse traces.

Reads evaluation/results/trace_revisions_271026.json, and for each attributed
trace appends a `rev:<revision>` tag and `code_revision` metadata via the
Langfuse ingestion endpoint (the only write path for an existing trace id).

Additive: existing tags and metadata are read first and re-sent unchanged.
Idempotent: skips traces that already carry the tag.

    .venv/bin/python evaluation/platform/stamp_trace_revisions.py [--dry-run] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from uuid import uuid4

from langfuse import Langfuse
from langfuse.api.ingestion.types import CreateEventBody, CreateEventEvent

MAP_PATH = pathlib.Path("evaluation/results/trace_revisions_271026.json")


def load_map() -> dict[str, str]:
    payload = json.loads(MAP_PATH.read_text())
    out: dict[str, str] = {}
    for run in payload["runs"].values():
        for trace_id, record in run["attributed"].items():
            out[trace_id] = record["code_revision"]
    return out


def stamp(client: Langfuse, trace_id: str, revision: str, *, dry_run: bool) -> str:
    trace = client.api.trace.get(trace_id)
    tags = list(trace.tags or [])
    tag = f"rev:{revision}"
    metadata = dict(trace.metadata or {})
    already = tag in tags and metadata.get("code_revision") == revision
    if already:
        return "skip"
    if dry_run:
        return "would-stamp"
    if tag not in tags:
        tags.append(tag)
    metadata["code_revision"] = revision
    client.api.ingestion.batch(
        batch=[
            CreateEventEvent(
                id=str(uuid4()),
                type="trace-create",
                timestamp=trace.timestamp.isoformat(),
                body=CreateEventBody(
                    id=trace_id,
                    name=trace.name,
                    timestamp=trace.timestamp,
                    metadata=metadata,
                    tags=tags,
                    environment=trace.environment,
                ),
            )
        ]
    )
    return "stamped"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    wanted = load_map()
    if args.limit:
        wanted = dict(list(wanted.items())[: args.limit])
    print(f"traces to stamp: {len(wanted)}")

    client = Langfuse()
    counts = {"stamped": 0, "skip": 0, "would-stamp": 0, "error": 0}
    errors: list[tuple[str, str]] = []

    for i, (trace_id, revision) in enumerate(sorted(wanted.items()), 1):
        try:
            outcome = stamp(client, trace_id, revision, dry_run=args.dry_run)
        except Exception as exc:  # noqa: BLE001 - report and continue
            outcome = "error"
            errors.append((trace_id, f"{type(exc).__name__}: {exc}"))
        counts[outcome] += 1
        if i % 25 == 0 or i == len(wanted):
            print(f"  {i}/{len(wanted)}  {counts}", flush=True)

    client.flush()
    print("RESULT:", counts)
    for trace_id, err in errors[:10]:
        print("  ERR", trace_id, err)
    return 0 if counts["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
