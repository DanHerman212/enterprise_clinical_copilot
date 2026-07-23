"""
register_serving_model.py — register an existing GCS serving bundle in the
Vertex AI Model Registry behind the pre-built XGBoost container.

Use this to (re)register a model from a completed pipeline's serving bundle
(model.bst + manifest.json) without re-running the whole training pipeline —
e.g. after changing the serving pattern. No managed explanation spec is
attached; feature attributions are computed client-side with native TreeSHAP
(see pipelines.serving.ReadmissionExplainer).

Usage (from repo root):
    .venv/bin/python projects/mlops/scripts/register_serving_model.py [BUNDLE_URI]

Environment:
    BUNDLE_URI     — GCS dir containing model.bst + manifest.json
    SERVING_IMAGE  — override the pre-built container image
"""

import os
import sys
from datetime import datetime, timezone

from google.cloud import aiplatform

PROJECT = "trim-icon-498815-a0"
LOCATION = "us-east1"
DEFAULT_IMAGE = "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.2-1:latest"

DEFAULT_BUNDLE = (
    "gs://trim-icon-498815-a0-mlops/pipeline-root/778397675435/"
    "readmission-training-20260720164335/"
    "register-model_3063486647661232128/serving_model"
)


def main() -> None:
    bundle_uri = (
        sys.argv[1] if len(sys.argv) > 1 else os.environ.get("BUNDLE_URI", DEFAULT_BUNDLE)
    ).rstrip("/")
    image = os.environ.get("SERVING_IMAGE", DEFAULT_IMAGE)

    aiplatform.init(project=PROJECT, location=LOCATION)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    display_name = f"readmission-final-{ts}"

    print(f"Registering {display_name}")
    print(f"  Bundle:  {bundle_uri}")
    print(f"  Serving: {image}  (pre-built XGBoost; native TreeSHAP client-side)")
    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=bundle_uri,
        serving_container_image_uri=image,
        labels={"pipeline": "readmission-training", "stage": "final"},
    )
    print(f"Registered: {model.resource_name}")


if __name__ == "__main__":
    main()
