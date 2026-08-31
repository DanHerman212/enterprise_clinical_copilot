"""deploy_synthetic_rag — deploy the newest synthetic tree-ah index to the
demo RAG endpoint on a small (cheap) machine.

Task 4c: after the synthetic rag-ingest pipeline builds the index, point the
live demo endpoint at it. Nothing is deployed today, so there is no undeploy
step — we just create/find `readmission-rag-index` and deploy.

Because the synthetic index is a SMALL shard (~190 vectors), it does NOT need
the e2-standard-16 that the 555k-vector medium shard required — a small
machine is valid and ~$0.09/hr vs ~$0.38/hr.

Usage:
    .venv/bin/python scripts/deploy_synthetic_rag.py [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import aiplatform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp_server.config import LOCATION, PROJECT  # noqa: E402

# Small-shard machine; the medium-shard e2-standard-16 minimum does not apply.
MACHINE = os.environ.get("INDEX_MACHINE_TYPE", "e2-standard-2")


def newest_synthetic_tree_index() -> tuple[str, str]:
    """Return (resource_name, display_name) of the newest rag-tree-ah-* index."""
    aiplatform.init(project=PROJECT, location=LOCATION)
    candidates = []
    for idx in aiplatform.MatchingEngineIndex.list():
        if idx.display_name.startswith("rag-tree-ah-"):
            created = idx.gca_resource.create_time
            vectors = idx.gca_resource.index_stats.vectors_count if (
                idx.gca_resource.index_stats) else 0
            candidates.append((created, idx.resource_name, idx.display_name, vectors))
    if not candidates:
        raise SystemExit("no rag-tree-ah-* index found — run the synthetic ingest first")
    # Newest first, print all for transparency, pick the latest.
    candidates.sort(reverse=True)
    for created, name, display, vectors in candidates:
        print(f"  {display:32s} vectors={vectors:<8} {name.split('/')[-1]}")
    _, name, display, vectors = candidates[0]
    if vectors > 100_000:
        raise SystemExit(
            f"refusing to deploy {display} ({vectors} vectors): "
            "looks like the real corpus, not synthetic"
        )
    print(f"→ deploying newest synthetic tree-ah: {display} ({vectors} vectors)")
    return name, display


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    index_name, display = newest_synthetic_tree_index()
    if args.dry_run:
        print(f"dry-run: would deploy {display} to endpoint "
              f"'readmission-rag-index' machine={MACHINE}")
        return 0

    # Reuse the exact deploy path from deploy_rag_endpoint.py.
    os.environ["INDEX_ID"] = index_name.split("/")[-1]
    os.environ["ENDPOINT_NAME"] = "readmission-rag-index"
    os.environ["DEPLOYED_INDEX_ID"] = "rag_tree_ah"
    os.environ["INDEX_MACHINE_TYPE"] = MACHINE
    sys.path.insert(0, os.path.dirname(__file__))
    import deploy_rag_endpoint  # noqa: E402

    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] "
          f"deploying {display} …")
    return deploy_rag_endpoint.main()


if __name__ == "__main__":
    raise SystemExit(main())
