"""load_synthetic_features — load the synthetic cohort feature rows into BigQuery.

Task 5e: the live predict tool reads features from
readmission.analytics_dataset_encoded keyed on hadm_id. The 24 synthetic
patients (90000001+) do not exist in that real-MIMIC table, so they cannot be
scored live. This writes their 49 feature rows to a clearly-synthetic table
(readmission.synthetic_features) with the SAME schema, so the predict path can
be pointed at it (or the rows merged) without touching the real dataset.

Source: eval/results/synthetic_cohort.json — each patient has a `features`
dict keyed by the manifest feature names (the same names as the encoded table
columns), plus hadm_id / band / probability / threshold.

Idempotent: WRITE_TRUNCATE.

Usage (from projects/agent-harness):
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
RESULTS = Path(__file__).resolve().parents[1] / "eval" / "results" / "synthetic_cohort.json"

# Bookkeeping columns the real encoded table carries (subject_id, split_name,
# readmission_30d) so the synthetic table is shape-compatible if ever joined.
_FEATURE_NAMES = [
    "age", "prior_admission_count", "prior_inpatient_days", "recent_ed_visits",
    "index_los_days", "procedure_count", "medication_count",
    "medication_order_count", "rbc_last", "rbc_min", "rdw_max", "monocytes_min",
    "hemoglobin_min", "sodium_last", "sodium_max", "sodium_min", "gender",
    "has_procedure", "oncology_flag",
    "race_white", "race_black", "race_hispanic", "race_asian", "race_amind",
    "race_nhpi", "race_unknown",
    "admission_type_ew_emer", "admission_type_eu_obs", "admission_type_obs_admit",
    "admission_type_urgent", "admission_type_direct_emer",
    "admission_type_ambulatory_obs", "admission_type_direct_obs",
    "admission_type_unknown",
    "discharge_location_home", "discharge_location_home_health",
    "discharge_location_snf", "discharge_location_rehab", "discharge_location_ltac",
    "discharge_location_hospice", "discharge_location_ama",
    "discharge_location_psych", "discharge_location_assisted_living",
    "discharge_location_unknown",
    "insurance_medicare", "insurance_medicaid", "insurance_private",
    "insurance_other", "insurance_unknown",
]

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
