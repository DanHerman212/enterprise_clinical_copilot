"""
smoke_test.py — Test the deployed readmission endpoint with a real patient.

Usage (from repo root):
    .venv/bin/python projects/mlops/scripts/smoke_test.py [hadm_id]

If no hadm_id is given, picks a random one from the encoded dataset.

The endpoint (pre-built XGBoost container) returns the calibrated probability;
feature attributions are computed locally with native TreeSHAP via the shared
serving glue (pipelines.serving.ReadmissionExplainer). The endpoint probability
is cross-checked against the local booster for parity.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# Make the serving glue importable (projects/mlops on the path).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google.cloud import aiplatform, bigquery
from pipelines.serving import ReadmissionExplainer

PROJECT = "trim-icon-498815-a0"
LOCATION = "us-east1"
ENDPOINT_NAME = "readmission-endpoint"
TABLE = "readmission.analytics_dataset_encoded"

# GCS serving bundle (model.bst + manifest.json) from the latest pipeline run.
BUNDLE_URI = os.environ.get(
    "BUNDLE_URI",
    "gs://trim-icon-498815-a0-mlops/pipeline-root/778397675435/"
    "readmission-training-20260720164335/"
    "register-model_3063486647661232128/serving_model",
)
MANIFEST_URI = f"{BUNDLE_URI.rstrip('/')}/manifest.json"


def _load_manifest() -> dict:
    raw = subprocess.check_output(["gsutil", "cat", MANIFEST_URI], text=True)
    return json.loads(raw)


def _fetch_patient(hadm_id: int) -> dict:
    """Fetch one row from BigQuery, return as a dict of column→value."""
    client = bigquery.Client(project=PROJECT)
    query = f"SELECT * FROM {TABLE} WHERE hadm_id = @hid LIMIT 1"
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("hid", "INT64", hadm_id)]
    )
    rows = list(client.query(query, job_config=job_config).result())
    if not rows:
        sys.exit(f"No row found for hadm_id={hadm_id}")
    return dict(rows[0].items())


def _assemble_features(row: dict, feature_order: list[str]) -> list[float]:
    """Build the feature array in the exact order expected by the model."""
    features = []
    for col in feature_order:
        val = row.get(col)
        if val is None:
            features.append(float("nan"))
        else:
            features.append(float(val))
    return features


def _get_endpoint() -> aiplatform.Endpoint:
    aiplatform.init(project=PROJECT, location=LOCATION)
    endpoints = aiplatform.Endpoint.list(order_by="create_time desc")
    for ep in endpoints:
        if ep.display_name == ENDPOINT_NAME:
            return ep
    sys.exit(f"No endpoint found with display_name='{ENDPOINT_NAME}'")


def _predict(endpoint: aiplatform.Endpoint, features: list[float]) -> dict:
    """Call the endpoint, return raw response."""
    return endpoint.predict(instances=[features])


def main() -> None:
    manifest = _load_manifest()
    feature_order: list[str] = manifest["feature_order"]
    groups: dict[str, list[str]] = manifest.get("groups", {})
    print(f"Manifest: {len(feature_order)} features, {len(groups)} parent groups")

    # Pick a patient
    if len(sys.argv) > 1:
        hadm_id = int(sys.argv[1])
    else:
        client = bigquery.Client(project=PROJECT)
        row = next(client.query(f"SELECT hadm_id FROM {TABLE} LIMIT 1").result())
        hadm_id = row.hadm_id
    print(f"Patient: hadm_id={hadm_id}")

    # Fetch row and assemble features
    row = _fetch_patient(hadm_id)
    features = _assemble_features(row, feature_order)
    print(f"Features: {len(features)} values, sample: {features[:5]}...")

    # --- Prediction from the deployed endpoint (enterprise serving spine) ---
    endpoint = _get_endpoint()
    print(f"Endpoint: {endpoint.resource_name}")
    response = _predict(endpoint, features)
    predictions = response.predictions
    probability = float(predictions[0]) if isinstance(predictions, list) else float(predictions)
    print(f"\n  Risk (endpoint probability): {probability:.4f}  ({probability*100:.1f}%)")

    # --- Explanation via native TreeSHAP (local serving glue) ---
    explainer = ReadmissionExplainer.from_gcs(BUNDLE_URI)
    explanation = explainer.explain(features)

    # Parity check: local booster vs deployed endpoint.
    delta = abs(explanation.probability - probability)
    parity = "OK" if delta < 1e-3 else f"WARN (Δ={delta:.4g})"
    print(f"  Local booster probability:   {explanation.probability:.4f}  [parity {parity}]")

    print(f"\n  Top TreeSHAP factors (aggregated from one-hot groups, logit space):")
    for name, score in explanation.top(10):
        direction = "↑ risk" if score > 0 else "↓ risk"
        print(f"    {name:30s} {score:+.4f}  {direction}")
    print(f"\n  Base value (logit): {explanation.base_value:+.4f}")
    print(f"  Model: {response.deployed_model_id}")


if __name__ == "__main__":
    main()
