"""
smoke_test.py — Test the deployed readmission CPR endpoint with a real patient.

Usage (from repo root):
    .venv/bin/python projects/mlops/scripts/smoke_test.py [hadm_id]

If no hadm_id is given, picks a random one from the encoded dataset.

The Custom Prediction Routine (CPR) endpoint returns, in a single response, the
calibrated probability, the threshold decision, and native-TreeSHAP feature
attributions aggregated to parent features. Missing values are sent as JSON null.
"""

import json
import os
import subprocess
import sys

from google.cloud import aiplatform, bigquery

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


def _assemble_features(row: dict, feature_order: list[str]) -> list:
    """Build the feature array in model order; missing -> None (JSON null -> NaN)."""
    features = []
    for col in feature_order:
        val = row.get(col)
        features.append(None if val is None else float(val))
    return features


def _get_endpoint() -> aiplatform.Endpoint:
    aiplatform.init(project=PROJECT, location=LOCATION)
    for ep in aiplatform.Endpoint.list(order_by="create_time desc"):
        if ep.display_name == ENDPOINT_NAME:
            return ep
    sys.exit(f"No endpoint found with display_name='{ENDPOINT_NAME}'")


def main() -> None:
    manifest = _load_manifest()
    feature_order: list[str] = manifest["feature_order"]
    groups: dict[str, list[str]] = manifest.get("groups", {})
    print(f"Manifest: {len(feature_order)} features, {len(groups)} parent groups")

    if len(sys.argv) > 1:
        hadm_id = int(sys.argv[1])
    else:
        client = bigquery.Client(project=PROJECT)
        row = next(client.query(f"SELECT hadm_id FROM {TABLE} LIMIT 1").result())
        hadm_id = row.hadm_id
    print(f"Patient: hadm_id={hadm_id}")

    row = _fetch_patient(hadm_id)
    features = _assemble_features(row, feature_order)
    n_missing = sum(1 for v in features if v is None)
    print(f"Features: {len(features)} values, {n_missing} missing (null)")

    endpoint = _get_endpoint()
    print(f"Endpoint: {endpoint.resource_name}")

    # Single call returns probability + threshold decision + attributions.
    response = endpoint.predict(instances=[features])
    pred = response.predictions[0]
    probability = float(pred["probability"])
    decision = int(pred["prediction"])
    threshold = float(pred["threshold"])

    print(f"\n  Risk (probability):  {probability:.4f}  ({probability*100:.1f}%)")
    print(f"  Decision:            {decision}  (threshold {threshold:.2f})")
    print(f"  Base value (logit):  {float(pred['base_value']):+.4f}")

    print(f"\n  Top TreeSHAP factors (parent-aggregated, logit space):")
    for factor in pred["top_factors"][:10]:
        score = float(factor["attribution"])
        direction = "↑ risk" if score > 0 else "↓ risk"
        print(f"    {factor['feature']:30s} {score:+.4f}  {direction}")
    print(f"\n  Model: {response.deployed_model_id}")


if __name__ == "__main__":
    main()
