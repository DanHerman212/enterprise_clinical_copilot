"""
deploy_cpr.py — Register and deploy the readmission Custom Prediction Routine
(CPR) to a Vertex AI endpoint.

The CPR serving image is built on Cloud Build (native linux/amd64) and is
content-addressed: it is rebuilt only when the CPR source (Dockerfile,
predictor.py, requirements.txt) changes. A newly trained model reuses the same
image — only the serving bundle (artifact_uri) changes.

The endpoint returns probability + threshold decision + native-TreeSHAP
attributions in a single response, while keeping Vertex's traffic control and
model monitoring.

Usage (from repo root):
    .venv/bin/python projects/mlops/scripts/deploy_cpr.py [--build-only] [--force-build]

Env:
    BUNDLE_URI     — GCS dir with model.bst + manifest.json [+ threshold.json]
    IMAGE_URI      — override the output image repo (tag is ignored/recomputed)
    ENDPOINT_NAME  — Vertex endpoint display name (default: readmission-endpoint)
    MACHINE_TYPE   — default: n1-standard-2
"""

import hashlib
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import aiplatform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_project_id  # noqa: E402

PROJECT = get_project_id()
LOCATION = "us-east1"
REPO = "readmission"
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "readmission-endpoint")
MACHINE_TYPE = os.environ.get("MACHINE_TYPE", "n1-standard-2")

CPR_SRC = Path(__file__).resolve().parents[1] / "pipelines" / "serving" / "cpr"
CLOUDBUILD = CPR_SRC / "cloudbuild.yaml"

# Image repo (no tag). The concrete tag is a content hash of the CPR source, so
# the image is reused across model versions and only rebuilt when it changes.
IMAGE_REPO = os.environ.get(
    "IMAGE_URI",
    f"{LOCATION}-docker.pkg.dev/{PROJECT}/{REPO}/readmission-cpr",
).split(":")[0]

# CPR serving-container contract (matches LocalModel.build_cpr_model output).
PREDICT_ROUTE = "/predict"
HEALTH_ROUTE = "/health"
CONTAINER_PORT = 8080

# Files whose contents determine the image tag (rebuild only when they change).
HASH_INPUTS = ["Dockerfile", "predictor.py", "requirements.txt"]

# The serving bundle is discovered from the latest pipeline provenance record
# (readmission-final-*). Set BUNDLE_URI to override (e.g. to pin a specific run).
BUNDLE_URI_OVERRIDE = os.environ.get("BUNDLE_URI")
FINAL_MODEL_PREFIX = "readmission-final-"


def image_tag() -> str:
    """Return a stable content hash over the CPR source that defines the image."""
    h = hashlib.sha256()
    for name in HASH_INPUTS:
        h.update(name.encode())
        h.update((CPR_SRC / name).read_bytes())
    return h.hexdigest()[:12]


def image_exists(image: str) -> bool:
    """True if the tagged image already exists in Artifact Registry."""
    r = subprocess.run(
        ["gcloud", "artifacts", "docker", "images", "describe", image,
         "--format=value(image_summary.digest)"],
        capture_output=True, text=True,
    )
    return r.returncode == 0 and bool(r.stdout.strip())


def cloud_build(tag: str) -> None:
    """Build & push the CPR image on Cloud Build (linux/amd64), tagged + latest."""
    print(f"Cloud Build: {IMAGE_REPO}:{tag}")
    subprocess.run(
        ["gcloud", "builds", "submit", str(CPR_SRC),
         "--project", PROJECT,
         "--config", str(CLOUDBUILD),
         "--substitutions", f"_IMAGE={IMAGE_REPO},_TAG={tag}"],
        check=True,
    )


def ensure_image(force: bool = False) -> str:
    """Return the CPR image URI, building on Cloud Build only if needed."""
    tag = image_tag()
    image = f"{IMAGE_REPO}:{tag}"
    if not force and image_exists(image):
        print(f"CPR image up-to-date, reusing: {image}")
        return image
    cloud_build(tag)
    return image


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


def main() -> None:
    aiplatform.init(project=PROJECT, location=LOCATION)
    image = ensure_image(force="--force-build" in sys.argv)
    if "--build-only" in sys.argv:
        print(f"Build-only: {image}")
        return

    bundle_uri = BUNDLE_URI_OVERRIDE.rstrip("/") if BUNDLE_URI_OVERRIDE else discover_bundle_uri()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    display_name = f"readmission-cpr-{ts}"
    print(f"Registering model: {display_name}")
    model = aiplatform.Model.upload(
        display_name=display_name,
        serving_container_image_uri=image,
        serving_container_predict_route=PREDICT_ROUTE,
        serving_container_health_route=HEALTH_ROUTE,
        serving_container_ports=[CONTAINER_PORT],
        # The CPR predictor downloads the bundle via storage.Client(), which
        # needs a project. Pin it explicitly so worker boot never depends on
        # ambient metadata-server project resolution (the "Model server never
        # became ready" failure was storage.Client() unable to resolve one).
        serving_container_environment_variables={"GOOGLE_CLOUD_PROJECT": PROJECT},
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
