"""
deploy_cpr.py — Build, push, register, and deploy the readmission Custom
Prediction Routine (CPR) to a Vertex AI endpoint.

The endpoint returns probability + threshold decision + native-TreeSHAP
attributions in a single response, while keeping Vertex's traffic control and
model monitoring.

Usage (from repo root):
    .venv/bin/python projects/mlops/scripts/deploy_cpr.py [--build-only]

Env:
    BUNDLE_URI     — GCS dir with model.bst + manifest.json [+ threshold.json]
    IMAGE_URI      — override the output image URI
    ENDPOINT_NAME  — Vertex endpoint display name (default: readmission-endpoint)
    MACHINE_TYPE   — default: n1-standard-2
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import aiplatform
from google.cloud.aiplatform.prediction import LocalModel

# Import the predictor class from the CPR source dir.
CPR_SRC = str(Path(__file__).resolve().parents[1] / "pipelines" / "serving" / "cpr")
sys.path.insert(0, CPR_SRC)
from predictor import ReadmissionPredictor  # noqa: E402

PROJECT = "trim-icon-498815-a0"
LOCATION = "us-east1"
REPO = "readmission"
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "readmission-endpoint")
MACHINE_TYPE = os.environ.get("MACHINE_TYPE", "n1-standard-2")

IMAGE_URI = os.environ.get(
    "IMAGE_URI",
    f"{LOCATION}-docker.pkg.dev/{PROJECT}/{REPO}/readmission-cpr:latest",
)

# The serving bundle is discovered from the latest pipeline provenance record
# (readmission-final-*). Set BUNDLE_URI to override (e.g. to pin a specific run).
BUNDLE_URI_OVERRIDE = os.environ.get("BUNDLE_URI")
FINAL_MODEL_PREFIX = "readmission-final-"


def discover_bundle_uri() -> str:
    """Return the artifact_uri of the newest readmission-final-* provenance record."""
    models = [
        m for m in aiplatform.Model.list(order_by="create_time desc")
        if m.display_name.startswith(FINAL_MODEL_PREFIX)
    ]
    if not models:
        raise SystemExit(
            f"No '{FINAL_MODEL_PREFIX}*' model found; run the training pipeline first "
            "or set BUNDLE_URI explicitly."
        )
    latest = models[0]
    uri = latest.gca_resource.artifact_uri.rstrip("/")
    print(f"Discovered bundle from {latest.display_name}: {uri}")
    return uri


def build() -> LocalModel:
    print(f"Building CPR image: {IMAGE_URI}")
    local_model = LocalModel.build_cpr_model(
        CPR_SRC,
        IMAGE_URI,
        predictor=ReadmissionPredictor,
        requirements_path=os.path.join(CPR_SRC, "requirements.txt"),
    )
    print("Pushing image …")
    local_model.push_image()
    return local_model


def main() -> None:
    aiplatform.init(project=PROJECT, location=LOCATION)
    local_model = build()
    if "--build-only" in sys.argv:
        print("Build-only: done.")
        return

    bundle_uri = BUNDLE_URI_OVERRIDE.rstrip("/") if BUNDLE_URI_OVERRIDE else discover_bundle_uri()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    display_name = f"readmission-cpr-{ts}"
    print(f"Registering model: {display_name}")
    model = aiplatform.Model.upload(
        local_model=local_model,
        display_name=display_name,
        artifact_uri=bundle_uri,
        labels={"pipeline": "readmission-training", "stage": "cpr"},
    )
    print(f"Registered: {model.resource_name}")

    # Reuse or create the endpoint; clear any existing deployments.
    endpoints = [
        ep for ep in aiplatform.Endpoint.list(order_by="create_time desc")
        if ep.display_name == ENDPOINT_NAME
    ]
    ep = endpoints[0] if endpoints else aiplatform.Endpoint.create(display_name=ENDPOINT_NAME)
    print(f"Endpoint: {ep.resource_name}")
    for dm in ep.list_models():
        print(f"  Undeploying stale model {dm.id} …")
        ep.undeploy(deployed_model_id=dm.id)

    print("Deploying CPR model (5–10 min) …")
    model.deploy(
        endpoint=ep,
        deployed_model_display_name=display_name,
        machine_type=MACHINE_TYPE,
        min_replica_count=1,
        max_replica_count=1,
        traffic_percentage=100,
    )
    print(f"Deployed. Endpoint: {ep.resource_name}")


if __name__ == "__main__":
    main()
