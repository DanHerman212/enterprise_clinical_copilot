"""
smoke_test.py — Test the deployed readmission CPR endpoint with a real patient.

Usage (from repo root):
    .venv/bin/python projects/mlops/scripts/smoke_test.py [hadm_id]
    .venv/bin/python projects/mlops/scripts/smoke_test.py 20924467 --write-fixture

If no hadm_id is given, picks a random one from the encoded dataset.

The Custom Prediction Routine (CPR) endpoint returns, in a single response, the
calibrated probability, the threshold decision, and native-TreeSHAP feature
attributions aggregated to parent features. Missing values are sent as JSON null.

--write-fixture records the observed values into the agent-harness Tier 1 fixture
(tests/fixtures/expected.json) so those tests never hardcode a probability. Re-run
it after any retraining pass; existing patients in the fixture are preserved.
"""

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

from google.cloud import aiplatform, bigquery, storage

PROJECT = "trim-icon-498815-a0"
LOCATION = "us-east1"
ENDPOINT_NAME = "readmission-endpoint"
TABLE = "readmission.analytics_dataset_encoded"

# Serving bundle is discovered from the newest readmission-final-* provenance
# record; set BUNDLE_URI to override (e.g. to pin a specific run).
BUNDLE_URI_OVERRIDE = os.environ.get("BUNDLE_URI")

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = REPO_ROOT / "projects" / "agent-harness" / "tests" / "fixtures" / "expected.json"

# Never assert float equality against a served model — serving-side numeric drift
# is real, and an exact-match failure reads like an integration bug.
FIXTURE_TOLERANCE = 1e-4


def _discover_bundle() -> tuple[str, str]:
    """(artifact_uri, display_name) of the newest readmission-final-* record."""
    models = [
        m for m in aiplatform.Model.list(order_by="create_time desc")
        if m.display_name.startswith("readmission-final-")
    ]
    if not models:
        sys.exit("No 'readmission-final-*' model found; run the pipeline or set BUNDLE_URI.")
    return models[0].gca_resource.artifact_uri.rstrip("/"), models[0].display_name


def _load_manifest(bundle_uri: str) -> dict:
    """Read manifest.json from GCS.

    Uses the storage client rather than shelling out to `gsutil`. Once
    aiplatform has opened its gRPC channels, any fork() — which is what
    subprocess does — can deadlock in gRPC's pthread_atfork handler: the child
    never spawns and the parent sleeps forever. It is racy, so it looks like an
    intermittent hang rather than a crash.
    """
    bucket_name, _, prefix = bundle_uri[len("gs://"):].partition("/")
    blob = storage.Client(project=PROJECT).bucket(bucket_name).blob(f"{prefix}/manifest.json")
    return json.loads(blob.download_as_text())


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


def _write_fixture(
    hadm_id: int,
    probability: float,
    decision: int,
    threshold: float,
    base_value: float,
    model_version: str,
) -> None:
    """Upsert one patient into the Tier 1 expected-values fixture."""
    if FIXTURE_PATH.exists():
        fixture = json.loads(FIXTURE_PATH.read_text())
        previous = fixture.get("patients", {}).get(str(hadm_id))
    else:
        fixture = {}
        previous = None

    fixture["model_version"] = model_version
    fixture["threshold"] = threshold
    fixture["generated_at"] = date.today().isoformat()
    fixture["generated_by"] = "projects/mlops/scripts/smoke_test.py --write-fixture"
    fixture.setdefault("patients", {})[str(hadm_id)] = {
        "probability": round(probability, 6),
        "decision": decision,
        "base_value": round(base_value, 6),
        "tolerance": FIXTURE_TOLERANCE,
    }

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")

    rel = FIXTURE_PATH.relative_to(REPO_ROOT)
    if previous and abs(previous.get("probability", 0.0) - probability) > FIXTURE_TOLERANCE:
        print(f"\n  Fixture: {rel}")
        print(f"    hadm_id={hadm_id} probability {previous['probability']:.4f} -> "
              f"{probability:.4f}  (model changed — expected after retraining)")
    else:
        print(f"\n  Fixture: {rel}  (hadm_id={hadm_id} recorded)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("hadm_id", nargs="?", type=int,
                        help="admission id; defaults to an arbitrary row")
    parser.add_argument("--write-fixture", action="store_true",
                        help="record observed values into the Tier 1 expected.json fixture")
    args = parser.parse_args()

    aiplatform.init(project=PROJECT, location=LOCATION)
    if BUNDLE_URI_OVERRIDE:
        bundle_uri = BUNDLE_URI_OVERRIDE.rstrip("/")
        model_version = os.path.basename(bundle_uri) or bundle_uri
    else:
        bundle_uri, model_version = _discover_bundle()
    print(f"Bundle: {bundle_uri}")
    manifest = _load_manifest(bundle_uri)
    feature_order: list[str] = manifest["feature_order"]
    groups: dict[str, list[str]] = manifest.get("groups", {})
    print(f"Manifest: {len(feature_order)} features, {len(groups)} parent groups")

    if args.hadm_id is not None:
        hadm_id = args.hadm_id
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
    base_value = float(pred["base_value"])

    print(f"\n  Risk (probability):  {probability:.4f}  ({probability*100:.1f}%)")
    print(f"  Decision:            {decision}  (threshold {threshold:.2f})")
    print(f"  Base value (logit):  {base_value:+.4f}")

    print(f"\n  Top TreeSHAP factors (parent-aggregated, logit space):")
    for factor in pred["top_factors"][:10]:
        score = float(factor["attribution"])
        direction = "↑ risk" if score > 0 else "↓ risk"
        print(f"    {factor['feature']:30s} {score:+.4f}  {direction}")
    print(f"\n  Model: {response.deployed_model_id}")

    if args.write_fixture:
        _write_fixture(hadm_id, probability, decision, threshold, base_value, model_version)


if __name__ == "__main__":
    main()
