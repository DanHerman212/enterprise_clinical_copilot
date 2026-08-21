"""chain_rag_deploy.py — wait for the hybrid rag-ingest pipeline, then deploy.

Runs while Dan is away: polls the given PipelineJob to a terminal state and,
ONLY on SUCCEEDED, launches the RAG endpoint deployment (deploy_synthetic_rag),
which auto-picks the newest rag-tree-ah-* index (the hybrid one just built).

On FAILED/CANCELLED it exits non-zero with the failing task's name so the next
session has a clear starting point. Idempotent: deploy_rag_endpoint reports and
exits 0 if the index is already deployed.

Usage (from projects/agent-harness):
  ../../.venv/bin/python scripts/chain_rag_deploy.py \
      projects/778397675435/locations/us-east1/pipelineJobs/<run>
"""

import subprocess
import sys
import time

from google.api_core.exceptions import ServiceUnavailable
from google.cloud.aiplatform_v1 import PipelineServiceClient

LOCATION = "us-east1"
TERMINAL = {
    "PIPELINE_STATE_SUCCEEDED",
    "PIPELINE_STATE_FAILED",
    "PIPELINE_STATE_CANCELLED",
}


def _get_job(client: PipelineServiceClient, name: str):
    for attempt in range(10):
        try:
            return client.get_pipeline_job(name=name)
        except ServiceUnavailable:
            time.sleep(10 * (attempt + 1))
    raise SystemExit("get_pipeline_job kept failing with 503 after retries")


def _deploy() -> int:
    print("\n=== pipeline SUCCEEDED → deploying RAG index endpoint ===", flush=True)
    return subprocess.call(
        [sys.executable, "scripts/deploy_synthetic_rag.py"], cwd=".")


def main() -> int:
    name = sys.argv[1]
    client = PipelineServiceClient(
        client_options={"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"})

    prev: dict[str, str] = {}
    while True:
        job = _get_job(client, name)
        jstate = job.state.name
        tasks = {t.task_name: t.state.name for t in job.job_detail.task_details}
        if not prev or any(prev.get(k) != v for k, v in tasks.items()):
            print(f"[{time.strftime('%H:%M:%S')}] job={jstate} "
                  f"tasks={ {k: v.split('_')[-1] for k, v in tasks.items()} }",
                  flush=True)
        prev = tasks

        if jstate == "PIPELINE_STATE_SUCCEEDED":
            return _deploy()
        if jstate in ("PIPELINE_STATE_FAILED", "PIPELINE_STATE_CANCELLED"):
            failed = [k for k, v in tasks.items() if v == "TASK_STATE_FAILED"]
            print(f"\n=== pipeline {jstate.split('_')[-1]} ===", flush=True)
            if failed:
                print("failed task(s):", failed, flush=True)
            return 1
        time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
