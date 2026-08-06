"""Watch a RAG ingest pipeline run to completion and report the outcome.

Polls the Vertex PipelineJob until a terminal state, prints task-state
transitions, dumps any failed task's error, and on success lists the created
Vector Search indexes (the verification the build_index component enforced).

Usage: .venv/bin/python scripts/watch_pipeline.py <pipeline_job_full_name>
"""

import sys
import time

from google.api_core.exceptions import ServiceUnavailable
from google.cloud import aiplatform
from google.cloud.aiplatform_v1 import PipelineServiceClient

PROJECT = "trim-icon-498815-a0"
LOCATION = "us-east1"

TERMINAL = {
    "PIPELINE_STATE_SUCCEEDED",
    "PIPELINE_STATE_FAILED",
    "PIPELINE_STATE_CANCELLED",
}


def get_job(client: PipelineServiceClient, name: str):
    """Poll with retry: gRPC 503s are transient and must not kill the watcher."""
    for attempt in range(10):
        try:
            return client.get_pipeline_job(name=name)
        except ServiceUnavailable:
            time.sleep(10 * (attempt + 1))
    raise SystemExit("get_pipeline_job kept failing with 503 after retries")


def main() -> int:
    name = sys.argv[1]
    client = PipelineServiceClient(
        client_options={"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"}
    )
    prev: dict[str, str] = {}
    iteration = 0
    while True:
        iteration += 1
        job = get_job(client, name)
        jstate = job.state.name
        tasks = {t.task_name: t.state.name for t in job.job_detail.task_details}
        if iteration == 1 or any(prev.get(k) != v for k, v in tasks.items()):
            print(f"[{time.strftime('%H:%M:%S')}] job={jstate}")
            for key, value in sorted(tasks.items()):
                print(f"    {key:32s} {value}")
            print(flush=True)
        prev = tasks

        failed = [t for t in job.job_detail.task_details
                  if t.state.name == "PIPELINE_TASK_STATE_FAILED"]
        if failed:
            for task in failed:
                print(f"=== FAILED TASK: {task.task_name} ===", flush=True)
                print(f"    error: {task.error}", flush=True)
            return 1

        if jstate in TERMINAL:
            print(f"=== PIPELINE TERMINAL: {jstate} ===", flush=True)
            if jstate == "PIPELINE_STATE_SUCCEEDED":
                aiplatform.init(project=PROJECT, location=LOCATION)
                for index in aiplatform.MatchingEngineIndex.list():
                    stats = index.gca_resource.index_stats
                    print(f"  index: {index.display_name}  vectors={stats.vectors_count}")
            return 0
        time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
