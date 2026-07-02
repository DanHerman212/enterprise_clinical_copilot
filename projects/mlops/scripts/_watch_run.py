"""One-off pipeline run monitor. Polls task states and stops on the first
task failure (dumping its error) or when the pipeline reaches a terminal state.
Usage: .venv/bin/python projects/mlops/scripts/_watch_run.py <pipeline_job_name>
"""
import sys
import time

from google.cloud.aiplatform_v1 import PipelineServiceClient

NAME = sys.argv[1]
client = PipelineServiceClient(
    client_options={"api_endpoint": "us-east1-aiplatform.googleapis.com"}
)
TERMINAL = {
    "PIPELINE_STATE_SUCCEEDED",
    "PIPELINE_STATE_FAILED",
    "PIPELINE_STATE_CANCELLED",
}

prev = {}
for i in range(160):
    job = client.get_pipeline_job(name=NAME)
    jstate = job.state.name
    tasks = {t.task_name: t.state.name for t in job.job_detail.task_details}
    if i == 0 or any(prev.get(k) != v for k, v in tasks.items()):
        print(f"[{time.strftime('%H:%M:%S')}] job={jstate}")
        for k, v in sorted(tasks.items()):
            print(f"    {k:26s} {v}")
        print(flush=True)
    prev = tasks
    failed = [t for t in job.job_detail.task_details if t.state.name == "FAILED"]
    if failed:
        for t in failed:
            print("=== FAILED TASK:", t.task_name, "===")
            print("  error:", t.error, flush=True)
        break
    if jstate in TERMINAL:
        print("=== PIPELINE TERMINAL:", jstate, "===", flush=True)
        break
    time.sleep(30)
