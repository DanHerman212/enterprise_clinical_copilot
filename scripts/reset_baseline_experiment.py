"""One-shot cleanup: delete the `readmission-30d` Vertex AI experiment
(all runs, including `baseline-v1`) and remove every GCS object under
`gs://<bucket>/artifacts/baseline-v1/`.

Idempotent — safe to re-run after a partial delete.
Run from project root:

    python scripts/reset_baseline_experiment.py
"""
from __future__ import annotations

import sys

from google.cloud import aiplatform, storage
from google.api_core import exceptions as gapi_exc

# Make `src` importable when invoked as a script.
sys.path.insert(0, ".")
from src import config  # noqa: E402


def _delete_experiment() -> None:
    aiplatform.init(project=config.PROJECT_ID, location=config.VERTEX_REGION)
    try:
        exp = aiplatform.Experiment(config.EXPERIMENT_NAME)
    except gapi_exc.NotFound:
        print(f"[experiment] `{config.EXPERIMENT_NAME}` not found — nothing to delete.")
        return

    # Delete every run under the experiment first.
    runs = aiplatform.ExperimentRun.list(experiment=config.EXPERIMENT_NAME)
    for run in runs:
        print(f"[experiment] deleting run `{run.name}`")
        try:
            run.delete(delete_backing_tensorboard_run=True)
        except TypeError:
            # Older SDKs use a different kwarg name; fall back gracefully.
            run.delete()

    # Then the experiment itself.
    print(f"[experiment] deleting experiment `{config.EXPERIMENT_NAME}`")
    exp.delete()


def _delete_baseline_artifacts() -> None:
    client = storage.Client(project=config.PROJECT_ID)
    bucket = client.bucket(config.GCS_BUCKET)
    prefix = "artifacts/baseline-v1/"
    blobs = list(client.list_blobs(bucket, prefix=prefix))
    if not blobs:
        print(f"[gcs] no objects under gs://{config.GCS_BUCKET}/{prefix}")
        return
    for blob in blobs:
        print(f"[gcs] deleting gs://{config.GCS_BUCKET}/{blob.name}")
        blob.delete()


if __name__ == "__main__":
    _delete_experiment()
    _delete_baseline_artifacts()
    print("Done.")
