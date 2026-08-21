"""inspect_failed_pipeline.py — print the failed task's error from a KFP run."""

import sys

from google.cloud.aiplatform_v1 import PipelineServiceClient

NAME = sys.argv[1] if len(sys.argv) > 1 else (
    "projects/778397675435/locations/us-east1/pipelineJobs/rag-ingest-20260821115227")


def main() -> None:
    client = PipelineServiceClient(
        client_options={"api_endpoint": "us-east1-aiplatform.googleapis.com"})
    job = client.get_pipeline_job(name=NAME)
    for t in job.job_detail.task_details:
        state = t.state.name
        print(f"\n== {t.task_name}: {state}")
        if state != "TASK_STATE_FAILED":
            continue
        print("error:", t.error.message[:2000] if t.error.message else "(no error message)")
        # executor detail may carry the container output
        for k, v in t.outputs.items():
            print("output:", k, "=", v)


if __name__ == "__main__":
    main()
