"""build_synthetic_fixtures — capture SYNTHETIC demo payloads for offline UI dev.

Synthetic twin of build_fixtures.py. The demo is real-system-on-synthetic-
data, so the fixtures the UI renders while endpoints are down must also be
synthetic. This runs the SAME serving predictor (ReadmissionPredictor, the
bundle on GCS) on the synthetic cohort's feature rows, so probability,
threshold, base_value and the native-TreeSHAP top_factors are exactly what the
deployed endpoint will return for that synthetic patient.

Provenance is recorded per fixture so the UI is never built against data we
cannot defend (here: fully synthetic, marked as such).

RAG fixtures: the synthetic index is built by the rag-ingest pipeline; real
passage scores for the synthetic primary patient are captured after deploy.
Until then, rag_*.json is left to the honest-empty path in fixtures.py.

Usage (from projects/agent-harness):
  ../../.venv/bin/python scripts/build_synthetic_fixtures.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parents[1]
COHORT_SOURCE = HARNESS_ROOT / "eval" / "results" / "synthetic_cohort.json"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "demo_fixtures"
SITE_FIXTURES = Path(os.environ.get(
    "SITE_FIXTURES",
    HARNESS_ROOT.parents[1].parent / "danielmherman" / "demo" / "data" / "demo_fixtures",
))
PROJECT = "trim-icon-498815-a0"
LOCATION = "us-east1"

# Synthetic patients to emit per-patient predict fixtures for (the demo
# primary + a borderline + a high, so each band's card is buildable).
PREDICT_PATIENTS = [90000009, 90000017, 90000001]


def _bundle() -> tuple[str, str]:
    """(bundle_uri, model_version) of the newest readmission-final-* model."""
    from google.cloud import aiplatform
    aiplatform.init(project=PROJECT, location=LOCATION)
    models = [
        m for m in aiplatform.Model.list(order_by="create_time desc")
        if m.display_name.startswith("readmission-final-")
    ]
    if not models:
        raise SystemExit("No 'readmission-final-*' model found.")
    return models[0].gca_resource.artifact_uri.rstrip("/"), models[0].display_name


def _run_predictions(feature_rows: dict[int, dict], version: str) -> dict[int, dict]:
    """Run the REAL serving predictor on synthetic feature dicts.

    The predictor's preprocess accepts named dicts in manifest feature order
    (JSON null -> NaN), so we pass the synthetic feature dict straight in and
    get the same numbers the deployed endpoint will return.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                           / "mlops" / "pipelines" / "serving" / "cpr"))
    from predictor import ReadmissionPredictor  # noqa: E402

    uri, _ = _bundle()
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            p = ReadmissionPredictor()
            p.load(uri)
            for hadm_id, features in feature_rows.items():
                matrix = p.preprocess({"instances": [features]})
                probs, contribs = p.predict(matrix)
                out = p.postprocess((probs, contribs))["predictions"][0]
                results[hadm_id] = _shape(out, hadm_id, version)
        finally:
            os.chdir(cwd)
    return results


def _shape(out: dict, hadm_id: int, version: str) -> dict:
    """Shape one prediction like the predict_readmission tool response."""
    top_factors = [
        {
            "feature": f["feature"],
            "contribution": round(float(f["attribution"]), 4),
            "direction": "increases" if float(f["attribution"]) > 0 else "decreases",
        }
        for f in out["top_factors"][:5]
    ]
    return {
        "hadm_id": hadm_id,
        "probability": round(float(out["probability"]), 6),
        "threshold": float(out["threshold"]),
        "decision": int(out["prediction"]),
        "base_value": round(float(out["base_value"]), 6),
        "top_factors": top_factors,
        "model_version": version,
        "feature_source": "synthetic",
        "provenance": "SYNTHETIC — local run of the serving predictor on "
                      "synthetic cohort features (2026-08-20)",
    }


def main() -> int:
    if not COHORT_SOURCE.exists():
        raise SystemExit(f"synthetic cohort not found: {COHORT_SOURCE}")

    data = json.loads(COHORT_SOURCE.read_text())
    patients_in = data["patients"]
    feature_rows = {int(p["hadm_id"]): p["features"] for p in patients_in}
    print(f"source: {COHORT_SOURCE} ({len(patients_in)} patients)")

    version = _bundle()[1]
    results = _run_predictions(feature_rows, version)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []

    for hadm_id in PREDICT_PATIENTS:
        payload = results[hadm_id]
        path = OUT_DIR / f"predict_{hadm_id}.json"
        path.write_text(json.dumps(payload, indent=2))
        written.append(path.name)
        print(f"predict {hadm_id}: probability={payload['probability']} "
              f"decision={payload['decision']} -> {path.name}")

    cohort_risk = {str(h): results[h] for h in sorted(feature_rows)}
    path = OUT_DIR / "cohort_risk.json"
    path.write_text(json.dumps(cohort_risk, indent=2))
    written.append(path.name)
    print(f"cohort_risk: {len(cohort_risk)} synthetic patients -> {path.name}")

    # Also ship to the site's fixture dir (sibling workspace).
    if SITE_FIXTURES:
        SITE_FIXTURES.mkdir(parents=True, exist_ok=True)
        for name in written:
            (SITE_FIXTURES / name).write_text((OUT_DIR / name).read_text())
        print(f"copied {len(written)} fixtures -> {SITE_FIXTURES}")

    print(f"\nwrote {len(written)} files to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
