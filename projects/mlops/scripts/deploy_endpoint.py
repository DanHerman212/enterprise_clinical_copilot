"""
deploy_endpoint.py — Deploy or teardown the readmission model endpoint + FeatureView.

Usage (from repo root):
    # Deploy
    .venv/bin/python projects/mlops/scripts/deploy_endpoint.py [--skip-feature-view]

    # Teardown everything (endpoints + FeatureStore, keeps models)
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

from google.cloud import aiplatform

PROJECT = "trim-icon-498815-a0"
LOCATION = "us-east1"
PARENT = f"projects/{PROJECT}/locations/{LOCATION}"

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "readmission-endpoint")
MACHINE_TYPE = os.environ.get("MACHINE_TYPE", "n1-standard-2")
MIN_REPLICAS = int(os.environ.get("MIN_REPLICAS", "1"))
MAX_REPLICAS = int(os.environ.get("MAX_REPLICAS", "1"))
TRAFFIC_SPLIT = int(os.environ.get("TRAFFIC_SPLIT", "100"))

FEATURE_VIEW_SRC = os.environ.get(
    "FEATURE_VIEW_SRC",
    f"bq://{PROJECT}.readmission.analytics_dataset_encoded",
)

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
# FeatureView (current-gen, BigQuery-backed, via v1beta1 low-level API)
# ---------------------------------------------------------------------------
def _setup_feature_view() -> str | None:
    """Idempotent: create or reuse FeatureOnlineStore + FeatureView.

    Returns the FeatureView resource name or None on failure.
    """
    try:
        from google.cloud.aiplatform_v1beta1 import (
            FeatureOnlineStoreAdminServiceClient,
            FeatureOnlineStoreServiceClient,
            types as fv_types,
        )
    except ImportError:
        print("FeatureView: v1beta1 SDK not available; skipping (non-fatal).")
        return None

    admin = FeatureOnlineStoreAdminServiceClient(
        client_options={"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"}
    )

    # --- FeatureOnlineStore ---
    fos_id = "readmission_online_store"
    fos_name = f"{PARENT}/featureOnlineStores/{fos_id}"

    try:
        fos = admin.get_feature_online_store(name=fos_name)
        print(f"FeatureOnlineStore exists: {fos_name}")
    except Exception:
        print(f"Creating FeatureOnlineStore: {fos_name} …")
        op = admin.create_feature_online_store(
            parent=PARENT,
            feature_online_store_id=fos_id,
            feature_online_store=fv_types.FeatureOnlineStore(
                bigtable=fv_types.FeatureOnlineStoreBigtable(
                    auto_scaling=fv_types.FeatureOnlineStoreBigtableAutoScaling(
                        min_node_count=1,
                        max_node_count=3,
                        cpu_utilization_target=50,
                    )
                )
            ),
        )
        fos = op.result()
        print(f"FeatureOnlineStore ready: {fos.name}")

    # --- FeatureView ---
    fv_id = "readmission_features"
    fv_name = f"{fos_name}/featureViews/{fv_id}"

    try:
        fv = admin.get_feature_view(name=fv_name)
        print(f"FeatureView exists: {fv_name}")
    except Exception:
        print(f"Creating FeatureView: {fv_name} → {FEATURE_VIEW_SRC} …")
        op = admin.create_feature_view(
            parent=fos_name,
            feature_view_id=fv_id,
            feature_view=fv_types.FeatureView(
                big_query_source=fv_types.FeatureViewBigQuerySource(
                    uri=FEATURE_VIEW_SRC,
                    entity_id_columns=["hadm_id"],
                )
            ),
        )
        fv = op.result()
        print(f"FeatureView ready: {fv.name}")

    print(f"FeatureView: {fv.name}")
    print(f"  Entity key: hadm_id")
    print(f"  Source:     {FEATURE_VIEW_SRC}")
    return fv.name


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

    if "--skip-feature-view" not in sys.argv:
        _setup_feature_view()


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------
def teardown(*, include_models: bool = False) -> None:
    """Undeploy all models from all matching endpoints and delete them.

    Also tears down FeatureOnlineStore + FeatureView if they exist.
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

    # --- FeatureOnlineStore + FeatureView ---
    try:
        from google.cloud.aiplatform_v1beta1 import (
            FeatureOnlineStoreAdminServiceClient,
        )
        admin = FeatureOnlineStoreAdminServiceClient(
            client_options={"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"}
        )
        fos_name = f"{PARENT}/featureOnlineStores/readmission_online_store"
        try:
            admin.delete_feature_online_store(name=fos_name, force=True)
            print(f"Deleting FeatureOnlineStore: {fos_name} …")
        except Exception:
            print(f"FeatureOnlineStore not found (already deleted): {fos_name}")
    except ImportError:
        print("FeatureView teardown skipped (v1beta1 SDK not available).")

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

