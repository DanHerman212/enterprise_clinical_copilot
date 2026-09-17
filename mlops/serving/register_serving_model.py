"""
register_serving_model.py — register an existing GCS serving bundle in the
Vertex AI Model Registry behind the pre-built XGBoost container.

Use this to (re)register a model from a completed pipeline's serving bundle
(model.bst + manifest.json) without re-running the whole training pipeline —
e.g. after changing the serving pattern. No managed explanation spec is
attached; feature attributions are computed client-side with native TreeSHAP
(see mlops.serving.readmission_explainer).

This is the *manual* path, so it must not be able to take the newest position
that everything downstream resolves. The model that serves is the bundle from
the most recent successful pipeline run; a record written by hand has no run
behind it, no gate metrics and no digests of its own. It therefore publishes
under its own name and its own stage label (readmission-manual-*, stage=manual)
and no resolver selects it — the two paths cannot be confused for each other.

Usage (from repo root):
    .venv/bin/python mlops/serving/register_serving_model.py [BUNDLE_URI]

Environment:
    BUNDLE_URI     — GCS dir containing model.bst + manifest.json (required
                     unless passed as the positional argument)
    SERVING_IMAGE  — the pre-built container image (required: no mutable
                     `:latest` default — ECC-59)
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import aiplatform

# Repo root, so `import mlops...` resolves: the parent of the package goes on
# the path, not the package directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mlops.data.config import get_project_id  # noqa: E402

PROJECT = get_project_id()
LOCATION = "us-east1"


def main() -> None:
    bundle_uri = (
        sys.argv[1] if len(sys.argv) > 1 else os.environ.get("BUNDLE_URI", "")
    ).rstrip("/")
    if not bundle_uri:
        sys.exit(
            "BUNDLE_URI is required (positional arg or env): the gs:// dir of a "
            "completed pipeline's serving bundle (model.bst + manifest.json)."
        )
    image = os.environ.get("SERVING_IMAGE")
    if not image:
        sys.exit(
            "SERVING_IMAGE is required (ECC-59): the pre-built xgboost-cpu.2-1:latest "
            "container is mutable. Pass a versioned/digest-pinned serving image."
        )

    aiplatform.init(project=PROJECT, location=LOCATION)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    # Own namespace on purpose: nothing resolves readmission-manual-*.
    display_name = f"readmission-manual-{ts}"

    print(f"Registering {display_name}")
    print(f"  Bundle:  {bundle_uri}")
    print(f"  Serving: {image}  (pre-built XGBoost; native TreeSHAP client-side)")
    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=bundle_uri,
        serving_container_image_uri=image,
        labels={"pipeline": "readmission-training", "stage": "manual"},
    )
    print(f"Registered: {model.resource_name}")
    print("  Not selected by the deploy script or the application: both resolve")
    print("  pipeline registrations (readmission-final-*, stage=final) only.")


if __name__ == "__main__":
    main()
