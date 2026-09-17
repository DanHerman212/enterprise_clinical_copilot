"""load_synthetic_features — load the synthetic cohort feature rows into BigQuery.

Task 5e: the predict path reads features from its configured table, which
defaults to readmission.hybrid_features (the hybrid demo cohort). The 24
synthetic patients (90000001+) do not exist in the real-MIMIC table, so they
cannot be scored from it. This writes their 49 feature rows to a
clearly-synthetic table
(readmission.synthetic_features) with the SAME schema, so the predict path can
be pointed at it (or the rows merged) without touching the real dataset.

Source: evaluation/agent/results/synthetic_cohort.json — each patient has a `features`
dict keyed by the manifest feature names (the same names as the encoded table
columns), plus hadm_id / band / probability / threshold.

Idempotent: WRITE_TRUNCATE.

Usage (from services):
  ../../.venv/bin/python scripts/load_synthetic_features.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from google.cloud import bigquery

PROJECT = "trim-icon-498815-a0"
DATASET = "readmission"
OUTPUT_TABLE = f"{PROJECT}.{DATASET}.synthetic_features"
RESULTS = Path(__file__).resolve().parents[2] / "evaluation" / "agent" / "results" / "synthetic_cohort.json"

# Bookkeeping columns the real encoded table carries (subject_id, split_name,
# readmission_30d) so the synthetic table is shape-compatible if ever joined.
#
# The 49 feature names are the model's contract, imported rather than retyped.
# This table exists to be scored by the trained booster, and that only works while
# it presents exactly the vocabulary the booster was trained on — so the list is
# derived from the contract, not copied from it.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlops.data.encoding import feature_order  # noqa: E402

_FEATURE_NAMES = feature_order()

_SCHEMA = [bigquery.SchemaField("hadm_id", "INT64")]
_SCHEMA += [bigquery.SchemaField(name, "FLOAT64") for name in _FEATURE_NAMES]
_SCHEMA += [
    bigquery.SchemaField("subject_id", "INT64"),
    bigquery.SchemaField("split_name", "STRING"),
    bigquery.SchemaField("readmission_30d", "INT64"),
]


def main() -> int:
    if not RESULTS.exists():
        raise SystemExit(f"synthetic cohort not found: {RESULTS}")

    data = json.loads(RESULTS.read_text())
    patients = data["patients"]
    print(f"source: {RESULTS} ({len(patients)} patients)")

    rows = []
    for p in patients:
        f = p["features"]
        row = {"hadm_id": int(p["hadm_id"]), "split_name": "test"}
        for name in _FEATURE_NAMES:
            # Missing values are legitimate (model reads null as NaN); absent
            # keys become NULL rather than 0.
            row[name] = f.get(name)
        # Synthetic bookkeeping: no real subject linkage; label is synthetic too.
        row["subject_id"] = None
        row["readmission_30d"] = 0
        rows.append(row)

    client = bigquery.Client(project=PROJECT)
    job = client.load_table_from_json(
        rows,
        OUTPUT_TABLE,
        job_config=bigquery.LoadJobConfig(
            schema=_SCHEMA, write_disposition="WRITE_TRUNCATE"
        ),
    )
    job.result()
    n = next(iter(client.query(
        f"SELECT COUNT(*) AS n FROM `{OUTPUT_TABLE}`").result()))["n"]
    if n != len(rows):
        print(f"FAILED: wrote {len(rows)} rows but table holds {n}.")
        return 1

    # Sanity: a random sample must have exactly the 49 model features non-null.
    got = next(iter(client.query(
        f"SELECT COUNT(*) AS n FROM `{OUTPUT_TABLE}` "
        f"WHERE age IS NULL OR gender IS NULL OR insurance_unknown IS NULL"
    ).result()))["n"]
    print(f"rows with missing key features: {got} (expect 0)")

    print(f"OK: {OUTPUT_TABLE} ({n} synthetic feature rows)")
    print("\nNext: point the predict path at this table — set FEATURE_TABLE="
          f"{PROJECT}.{DATASET}.synthetic_features on the deployed agent, "
          "or merge rows into analytics_dataset_encoded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
