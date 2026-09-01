"""
deploy_endpoint.py — Deploy or teardown the readmission model endpoint.

The FeatureOnlineStore/FeatureView path was removed (2026-08-31, ECC-54): the
Vertex Feature Store was a dev experiment and is not used in production —
serving reads features from BigQuery (FEATURE_SOURCE=bigquery).

Usage (from repo root):
    # Deploy
    .venv/bin/python projects/mlops/scripts/deploy_endpoint.py

    # Teardown everything (endpoint, keeps models)
    .venv/bin/python projects/mlops/scripts/deploy_endpoint.py --teardown

    # Teardown including registered models
    .venv/bin/python projects/mlops/scripts/deploy_endpoint.py --teardown --include-models

Environment:
    ENDPOINT_NAME   — Vertex endpoint display name (default: readmission-endpoint)
    MODEL_ID        — Override model resource name (default: latest readmission-final-*)
    MACHINE_TYPE    — default: n1-standard-2
    MIN_REPLICAS    — default: 1
    MAX_REPLICAS    — default: 1
    TRAFFIC_SPLIT   — default: 100
"""

import json
import os
import sys
import time
from pathlib import Path

from google.cloud import aiplatform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_project_id  # noqa: E402

PROJECT = get_project_id()
LOCATION = "us-east1"

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "readmission-endpoint")
MACHINE_TYPE = os.environ.get("MACHINE_TYPE", "n1-standard-2")
MIN_REPLICAS = int(os.environ.get("MIN_REPLICAS", "1"))
MAX_REPLICAS = int(os.environ.get("MAX_REPLICAS", "1"))
TRAFFIC_SPLIT = int(os.environ.get("TRAFFIC_SPLIT", "100"))

# ---------------------------------------------------------------------------
# Model / Endpoint helpers
# ---------------------------------------------------------------------------
def _latest_model() -> aiplatform.Model:
    models = aiplatform.Model.list(order_by="create_time desc")
    for m in models:
        if m.display_name.startswith("readmission-final-"):
            print(f"Model: {m.display_name}  ({m.resource_name})")
            return m
    sys.exit("No readmission-final-* model found in registry.")


def _endpoint() -> aiplatform.Endpoint:
    endpoints = aiplatform.Endpoint.list(order_by="create_time desc")
    for ep in endpoints:
        if ep.display_name == ENDPOINT_NAME:
            print(f"Reusing endpoint: {ep.resource_name}")
            return ep
    ep = aiplatform.Endpoint.create(display_name=ENDPOINT_NAME)
    print(f"Created endpoint: {ep.resource_name}")
    return ep


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Deploy the latest readmission-final model to the endpoint.

    The model is registered behind the pre-built XGBoost container with no
    managed explanation spec (attributions are computed client-side with
    native TreeSHAP), so it deploys cleanly via the standard model.deploy().
    """
    aiplatform.init(project=PROJECT, location=LOCATION)

    model = aiplatform.Model(os.environ["MODEL_ID"]) if os.environ.get("MODEL_ID") else _latest_model()
    ep = _endpoint()

    deployed = {dm.model: dm.id for dm in ep.list_models()}
    if model.resource_name in deployed:
        print(f"Already deployed (deployedModelId={deployed[model.resource_name]}).")
        print(f"Endpoint: {ep.resource_name}")
    else:
        print("Deploying model to endpoint (this takes 5–10 min) …")
        model.deploy(
            endpoint=ep,
            deployed_model_display_name=f"readmission-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}",
            machine_type=MACHINE_TYPE,
            min_replica_count=MIN_REPLICAS,
            max_replica_count=MAX_REPLICAS,
            traffic_percentage=TRAFFIC_SPLIT,
        )
        print(f"Deployed. Endpoint: {ep.resource_name}")


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------
def teardown(*, include_models: bool = False) -> None:
    """Undeploy all models from all matching endpoints and delete them.

    Set include_models=True to also delete registered models from the registry.
    """
    aiplatform.init(project=PROJECT, location=LOCATION)

    # --- Endpoints ---
    endpoints = aiplatform.Endpoint.list(order_by="create_time desc")
    matched = [ep for ep in endpoints if ep.display_name == ENDPOINT_NAME]
    if not matched:
        print("No matching endpoints found.")
    for ep in matched:
        models = list(ep.list_models())
        for dm in models:
            print(f"Undeploying model {dm.id} from {ep.display_name} …")
            ep.undeploy(deployed_model_id=dm.id)
        print(f"Deleting endpoint: {ep.display_name} ({ep.resource_name}) …")
        ep.delete()
        print(f"  Deleted.")

    # --- Models (optional — keep for reference by default) ---
    if include_models:
        models = aiplatform.Model.list(order_by="create_time desc")
        for m in models:
            if m.display_name.startswith("readmission-final-"):
                print(f"Deleting model: {m.display_name} …")
                m.delete()
                print(f"  Deleted.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if "--teardown" in sys.argv:
        teardown(include_models="--include-models" in sys.argv)
    else:
        main()

