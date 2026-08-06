"""§6: Build Vertex AI Vector Search indexes from the ingest JSONL.

Two-index approach (guide §6, D3): a small BRUTE_FORCE index gives exact
neighbors (ground truth) at pennies; the full TREE_AH index is the approximate
index that serves the demo. Compare the two at §10/§12 to separate "retrieval
is broken" from "approximation is imperfect".

Brute-force is built over a 2,000-record sample for now; the guide intends it
over the 20-patient cohort, which is built later at §10. The sample stands in
until then.

    python scripts/deploy_index.py --mode brute-force [--sample 2000]
    python scripts/deploy_index.py --mode tree-ah

Both build to a CREATED index; the script fails loudly if the final datapoint
count does not match expectations (silent shortfall = dropped chunks).

Env:  INGEST_URI   override the source JSONL directory (default GCS ingest/)
"""

from __future__ import annotations

import argparse
import gzip
import sys
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import aiplatform, storage
from google.cloud.aiplatform.matching_engine.matching_engine_index_config import (
    DistanceMeasureType,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.notes import CACHE_DIR  # noqa: E402

PROJECT = "trim-icon-498815-a0"
LOCATION = "us-east1"
BUCKET = "trim-icon-498815-a0-mlops"
DIMENSIONS = 768
DISTANCE = DistanceMeasureType.DOT_PRODUCT_DISTANCE
INGEST_URI = f"gs://{BUCKET}/rag/embeddings/ingest/"
VALIDATION_DIR = f"gs://{BUCKET}/rag/embeddings/validation/"
DEFAULT_SAMPLE = 2000
APPROXIMATE_NEIGHBORS = 40  # tune against recall at §12


def write_and_upload_sample(sample: int) -> str:
    """Take the first `sample` records from the local ingest and upload them.

    Vector Search only accepts .json/.csv/.avro extensions. The object is
    served gzip-encoded so the file stays small without a re-upload.
    """
    src = CACHE_DIR / "embed_ingest.jsonl.gz"
    if not src.exists():
        raise SystemExit("Local ingest cache missing; re-run scripts/embed_chunks.py first.")
    tmp = Path("/tmp") / f"sample_{sample}.json"
    with gzip.open(src, "rt", encoding="utf-8") as fin, \
            open(tmp, "w", encoding="utf-8") as fout:
        for index, line in enumerate(fin):
            if index >= sample:
                break
            fout.write(line)
    dest = f"rag/embeddings/validation/sample_{sample}.json"
    blob = storage.Client(project=PROJECT).bucket(BUCKET).blob(dest)
    blob.upload_from_filename(str(tmp))
    print(f"Uploaded sample → gs://{BUCKET}/{dest}")
    return VALIDATION_DIR


def verify(index: aiplatform.MatchingEngineIndex, expected: int, label: str) -> None:
    """Block until CREATED and assert the datapoint count."""
    print(f"{label}: waiting for build…")
    index.wait()
    index.refresh()
    state = index.gca_resource.state.name
    vectors = index.gca_resource.index_stats.vectors_count
    print(f"{label}: state={state}, vectors={vectors}")
    if state != "READY":
        raise SystemExit(f"{label} did not reach READY (state={state})")
    if vectors != expected:
        raise SystemExit(f"{label}: expected {expected} vectors, got {vectors} "
                         "- chunks were dropped")
    print(f"{label} verified: {vectors} vectors at {DIMENSIONS} dims")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["brute-force", "tree-ah"], required=True)
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    args = parser.parse_args()

    aiplatform.init(project=PROJECT, location=LOCATION)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    if args.mode == "brute-force":
        uri = write_and_upload_sample(args.sample)
        display = f"rag-brute-force-{ts}"
        index = aiplatform.MatchingEngineIndex.create_brute_force_index(
            display_name=display,
            contents_delta_uri=uri,
            dimensions=DIMENSIONS,
            distance_measure_type=DISTANCE,
        )
        expected = args.sample
    else:
        display = f"rag-tree-ah-{ts}"
        index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
            display_name=display,
            contents_delta_uri=INGEST_URI,
            dimensions=DIMENSIONS,
            approximate_neighbors_count=APPROXIMATE_NEIGHBORS,
            distance_measure_type=DISTANCE,
        )
        expected = 555770

    print(f"Created {display}: {index.resource_name}")
    verify(index, expected, display)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
