"""A2: score the full test split locally and write readmission.test_predictions.

The A4 note-signal probe needs the model's probability for every test admission,
to isolate the population the model called low-risk. No prediction table exists,
and scoring 49k rows through a live endpoint would be slow and cost money for no
benefit — the model is a file. So this script scores locally.

Local scoring is only trustworthy if it provably matches what the endpoint
would return. Two measures ensure that:

  * It does not reimplement anything. It imports ReadmissionPredictor — the
    exact class the CPR container runs — and calls its load/preprocess/predict.
  * Before scoring anything it scores the anchor admission (hadm_id 20924467)
    and requires the probability verified against the live endpoint, 0.131398,
    to six decimals. Wrong bundle, wrong feature order, or different
    missing-value handling all produce plausible-looking probabilities and no
    error; the anchor turns that silent failure into a hard stop.

Note: the current bundle comes from the dry run, not a fully tuned model
(AUCPR ~0.41 vs 0.33 baseline). The output table records model_version, and the
gate's conclusions are conditional on that model; re-run this script when a
fully trained model lands.

Writes:  readmission.test_predictions (WRITE_TRUNCATE)
         docs/probes/test_predictions_summary.json

Usage:
    python scripts/score_test_split.py
Env:
    BUNDLE_URI  — pin a specific serving bundle (default: discover from the
                  newest readmission-final-* model registry record)
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from google.cloud import aiplatform, bigquery

HARNESS = Path(__file__).resolve().parents[1]
CPR_SRC = HARNESS.parent / "mlops" / "pipelines" / "serving" / "cpr"
sys.path.insert(0, str(CPR_SRC))

from predictor import ReadmissionPredictor  # noqa: E402

PROJECT = "trim-icon-498815-a0"
LOCATION = "us-east1"
FINAL_MODEL_PREFIX = "readmission-final-"
FEATURE_TABLE = f"{PROJECT}.readmission.analytics_dataset_encoded"
OUTPUT_TABLE = f"{PROJECT}.readmission.test_predictions"
ARTIFACT_PATH = HARNESS / "docs" / "probes" / "test_predictions_summary.json"

# Verified against the live endpoint on 2026-08-03, before it was torn down.
ANCHOR_HADM_ID = 20924467
ANCHOR_PROBABILITY = 0.131398
ANCHOR_TOLERANCE = 5e-7  # match to six decimal places


def resolve_bundle() -> tuple[str, str]:
    """Return (bundle_uri, model_version), honouring the BUNDLE_URI override."""
    aiplatform.init(project=PROJECT, location=LOCATION)
    override = os.environ.get("BUNDLE_URI")
    if override:
        return override.rstrip("/"), "(BUNDLE_URI override)"
    models = [
        m for m in aiplatform.Model.list(order_by="create_time desc")
        if m.display_name.startswith(FINAL_MODEL_PREFIX)
    ]
    if not models:
        raise SystemExit(f"No '{FINAL_MODEL_PREFIX}*' model in the registry "
                         "and no BUNDLE_URI set.")
    latest = models[0]
    return latest.gca_resource.artifact_uri.rstrip("/"), latest.display_name


def fetch_rows(client: bigquery.Client, where: str) -> list[dict]:
    query = f"SELECT * FROM `{FEATURE_TABLE}` WHERE {where}"
    return [dict(row) for row in client.query(query).result()]


def score(predictor: ReadmissionPredictor, rows: list[dict]) -> np.ndarray:
    """Probabilities for feature rows, via the predictor's own code path."""
    # Dict instances: preprocess() picks columns by manifest feature order and
    # maps SQL NULL (None) to NaN, exactly as the serving container does.
    matrix = predictor.preprocess({"instances": rows})
    probs, _ = predictor.predict(matrix)
    return probs


def main() -> int:
    bundle_uri, model_version = resolve_bundle()
    print(f"Bundle:  {bundle_uri}")
    print(f"Model:   {model_version}")

    predictor = ReadmissionPredictor()
    with tempfile.TemporaryDirectory() as tmp:
        with contextlib.chdir(tmp):  # load() downloads bundle files into cwd
            predictor.load(bundle_uri)

        threshold = predictor._threshold
        print(f"Threshold from bundle: {threshold}")

        client = bigquery.Client(project=PROJECT)

        # -- anchor: prove the local path matches the verified endpoint output
        anchor_rows = fetch_rows(client, f"hadm_id = {ANCHOR_HADM_ID}")
        if len(anchor_rows) != 1:
            print(f"FAILED: expected 1 row for anchor hadm_id {ANCHOR_HADM_ID}, "
                  f"got {len(anchor_rows)}.")
            return 1
        anchor_prob = float(score(predictor, anchor_rows)[0])
        drift = abs(anchor_prob - ANCHOR_PROBABILITY)
        if drift > ANCHOR_TOLERANCE:
            print(f"FAILED anchor check: local={anchor_prob:.6f}, "
                  f"endpoint-verified={ANCHOR_PROBABILITY}, drift={drift:.2e}.\n"
                  "Local scoring does NOT match production; nothing written.")
            return 1
        print(f"Anchor check passed: {anchor_prob:.6f} == {ANCHOR_PROBABILITY} "
              f"(drift {drift:.2e})")

        # -- score the full test split
        print("Fetching test split…")
        rows = fetch_rows(client, "split_name = 'test'")
        print(f"  {len(rows)} rows")
        probs = score(predictor, rows)

    scored_at = datetime.now(timezone.utc).isoformat()
    records = [
        {
            "hadm_id": row["hadm_id"],
            "probability": round(float(prob), 6),
            "decision": int(float(prob) >= threshold),
            "readmission_30d": int(row["readmission_30d"]),
            "threshold": threshold,
            "model_version": model_version,
            "scored_at": scored_at,
        }
        for row, prob in zip(rows, probs)
    ]

    schema = [
        bigquery.SchemaField("hadm_id", "INT64"),
        bigquery.SchemaField("probability", "FLOAT64"),
        bigquery.SchemaField("decision", "INT64"),
        bigquery.SchemaField("readmission_30d", "INT64"),
        bigquery.SchemaField("threshold", "FLOAT64"),
        bigquery.SchemaField("model_version", "STRING"),
        bigquery.SchemaField("scored_at", "TIMESTAMP"),
    ]
    job = client.load_table_from_json(
        records,
        OUTPUT_TABLE,
        job_config=bigquery.LoadJobConfig(
            schema=schema, write_disposition="WRITE_TRUNCATE"
        ),
    )
    job.result()

    # -- verify the write independently, then summarise
    written = next(iter(client.query(
        f"SELECT COUNT(*) AS n FROM `{OUTPUT_TABLE}`").result()))["n"]
    if written != len(records):
        print(f"FAILED: wrote {len(records)} records but table holds {written}.")
        return 1

    y = np.array([r["readmission_30d"] for r in records])
    d = np.array([r["decision"] for r in records])
    quadrants = {
        "true_positive": int(((d == 1) & (y == 1)).sum()),
        "false_positive": int(((d == 1) & (y == 0)).sum()),
        "false_negative": int(((d == 0) & (y == 1)).sum()),
        "true_negative": int(((d == 0) & (y == 0)).sum()),
    }
    summary = {
        "generated_utc": scored_at,
        "model_version": model_version,
        "model_note": "dry-run model, not fully tuned; re-score after HPO",
        "bundle_uri": bundle_uri,
        "threshold": threshold,
        "anchor": {
            "hadm_id": ANCHOR_HADM_ID,
            "expected": ANCHOR_PROBABILITY,
            "actual": round(anchor_prob, 6),
        },
        "rows_scored": len(records),
        "readmission_rate": round(float(y.mean()), 4),
        "quadrants": quadrants,
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nWrote {OUTPUT_TABLE} ({written} rows) and {ARTIFACT_PATH}")
    print(f"  readmission rate: {summary['readmission_rate']}")
    print(f"  quadrants at threshold {threshold}:")
    for name, count in quadrants.items():
        print(f"    {name:<16}{count:>8}")
    print("  gate population (A4) = false_negative + true_negative "
          f"= {quadrants['false_negative'] + quadrants['true_negative']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
