"""
test_cpr_local.py — Run the built CPR image on a local endpoint and verify it
returns probability + attributions for a real patient, before deploying to Vertex.

Usage (from repo root):
    .venv/bin/python projects/mlops/scripts/test_cpr_local.py [hadm_id]

Sends the patient two ways to prove the container's JSON handling:
  1. positional list with NaN->null for genuinely-missing values
  2. named dict with null for missing values
Expected probability for hadm_id=20924467 is ~0.1314.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from google.cloud import aiplatform, bigquery
from google.cloud.aiplatform.prediction import LocalModel

CPR_SRC = str(Path(__file__).resolve().parents[1] / "pipelines" / "serving" / "cpr")
sys.path.insert(0, CPR_SRC)
from predictor import ReadmissionPredictor  # noqa: E402

PROJECT = "trim-icon-498815-a0"
LOCATION = "us-east1"
TABLE = "readmission.analytics_dataset_encoded"
IMAGE_URI = f"{LOCATION}-docker.pkg.dev/{PROJECT}/readmission/readmission-cpr:latest"
BUNDLE_URI_OVERRIDE = os.environ.get("BUNDLE_URI")
CREDS = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")


def _discover_bundle_uri() -> str:
    """artifact_uri of the newest readmission-final-* provenance record."""
    models = [
        m for m in aiplatform.Model.list(order_by="create_time desc")
        if m.display_name.startswith("readmission-final-")
    ]
    if not models:
        sys.exit("No 'readmission-final-*' model found; run the pipeline or set BUNDLE_URI.")
    return models[0].gca_resource.artifact_uri.rstrip("/")


def _manifest(bundle_uri: str) -> dict:
    raw = subprocess.check_output(
        ["gsutil", "cat", f"{bundle_uri}/manifest.json"], text=True
    )
    return json.loads(raw)


def _patient(hadm_id: int, feature_order: list[str]):
    client = bigquery.Client(project=PROJECT)
    cfg = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("hid", "INT64", hadm_id)]
    )
    rows = list(
        client.query(
            f"SELECT * FROM {TABLE} WHERE hadm_id=@hid LIMIT 1", cfg
        ).result()
    )
    if not rows:
        sys.exit(f"No row for hadm_id={hadm_id}")
    row = dict(rows[0].items())
    positional = [row.get(c) if row.get(c) is not None else None for c in feature_order]
    positional = [None if v is None else float(v) for v in positional]
    named = {c: (None if row.get(c) is None else float(row.get(c))) for c in feature_order}
    return positional, named


def main() -> None:
    hadm_id = int(sys.argv[1]) if len(sys.argv) > 1 else 20924467
    aiplatform.init(project=PROJECT, location=LOCATION)
    bundle_uri = BUNDLE_URI_OVERRIDE.rstrip("/") if BUNDLE_URI_OVERRIDE else _discover_bundle_uri()
    print(f"Bundle: {bundle_uri}")
    feature_order = _manifest(bundle_uri)["feature_order"]
    positional, named = _patient(hadm_id, feature_order)
    n_missing = sum(1 for v in positional if v is None)
    print(f"Patient {hadm_id}: {len(positional)} features, {n_missing} missing (null)")

    local_model = LocalModel.build_cpr_model(
        CPR_SRC,
        IMAGE_URI,
        predictor=ReadmissionPredictor,
        requirements_path=os.path.join(CPR_SRC, "requirements.txt"),
    )

    with local_model.deploy_to_local_endpoint(
        artifact_uri=BUNDLE_URI, credential_path=CREDS
    ) as endpoint:
        for label, instance in (("positional-list", positional), ("named-dict", named)):
            resp = endpoint.predict(
                request=json.dumps({"instances": [instance]}),
                headers={"Content-Type": "application/json"},
            )
            body = json.loads(resp.content)
            pred = body["predictions"][0]
            print(f"\n[{label}] probability={pred['probability']:.4f} "
                  f"prediction={pred['prediction']} threshold={pred['threshold']}")
            print("  top factors:")
            for f in pred["top_factors"][:5]:
                print(f"    {f['feature']:28s} {f['attribution']:+.4f}")


if __name__ == "__main__":
    main()
