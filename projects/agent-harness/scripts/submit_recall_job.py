"""Submit the recall@k job as a Vertex AI custom job.

The job runs fully in the cloud (reads the ingest + chunks from GCS, embeds
queries, computes exact neighbors, queries the live tree-AH endpoint, writes
the report back to GCS). Nothing touches a developer machine.

Usage:
    .venv/bin/python projects/agent-harness/scripts/submit_recall_job.py
"""

import os
import sys
from datetime import datetime, timezone

from google.cloud import aiplatform

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PROJECT = os.environ.get("PROJECT_ID", "trim-icon-498815-a0")
LOCATION = "us-east1"
IMAGE = os.environ.get(
    "RAG_IMAGE_URI",
    f"{LOCATION}-docker.pkg.dev/{PROJECT}/readmission/rag-ingest:latest",
)
SERVICE_ACCOUNT = os.environ.get(
    "PIPELINE_SA", f"mlops-pipeline@{PROJECT}.iam.gserviceaccount.com")
STAGING = f"gs://{PROJECT}-mlops/pipeline-root"

CHUNKS = ("gs://trim-icon-498815-a0-mlops/pipeline-root/778397675435/"
          "rag-ingest-20260806173635/chunk-notes_-7096656101919162368/chunks")
INGEST = ("gs://trim-icon-498815-a0-mlops/pipeline-root/778397675435/"
          "rag-ingest-20260806173635/embed-chunks_6738401953363001344/ingest")
ENDPOINT = ("projects/778397675435/locations/us-east1/"
            "indexEndpoints/4397109727197134848")
OUT_DIR = f"gs://{PROJECT}-mlops/rag/recall"


def main() -> int:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    aiplatform.init(project=PROJECT, location=LOCATION, staging_bucket=STAGING)
    job = aiplatform.CustomJob(
        display_name=f"rag-recall-k-{ts}",
        worker_pool_specs=[{
            "machine_spec": {"machine_type": "e2-standard-8"},
            "replica_count": 1,
            "container_spec": {
                "image_uri": IMAGE,
                "command": ["python", "/app/pipelines/recall_k.py"],
                "args": [
                    "--chunks", CHUNKS,
                    "--ingest", INGEST,
                    "--endpoint", ENDPOINT,
                    "--deployed-id", "rag_tree_ah",
                    "--out-dir", OUT_DIR,
                    "--num-queries", "100",
                    "--top-k", "10",
                ],
            },
        }],
    )
    job.run(service_account=SERVICE_ACCOUNT)
    print(f"recall@k job: {job.resource_name}")
    print(f"report → {OUT_DIR}/recall_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
