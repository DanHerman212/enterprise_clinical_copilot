"""load_hybrid_notes — push the hybrid (real MTSamples) notes + features into BigQuery.

Hybrid twin of load_synthetic_notes.py + load_synthetic_features.py. Writes to
DISTINCT tables (readmission.hybrid_*) so the synthetic tables stay intact as a
fallback until the swap is validated end-to-end.

Writes (WRITE_TRUNCATE, idempotent):
  readmission.hybrid_notes     (hadm_id INT64, note_id STRING, text STRING)
  readmission.hybrid_split     (hadm_id INT64, split_name STRING)   # all 'test'
  readmission.hybrid_features  (hadm_id + 49 model features + bookkeeping cols)

Source:
  eval/results/hybrid_notes.json    {n, patients:[{hadm_id, archetype, band,
                                                   variant, note}]}
  eval/results/hybrid_cohort.json   {seed, n, patients:[{hadm_id, band,
                                                   probability, threshold, features}]}

Usage (from projects/agent-harness):
  ../../.venv/bin/python scripts/load_hybrid_notes.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from google.cloud import bigquery

PROJECT = "trim-icon-498815-a0"
DATASET = "readmission"
NOTES_TABLE = f"{PROJECT}.{DATASET}.hybrid_notes"
SPLIT_TABLE = f"{PROJECT}.{DATASET}.hybrid_split"
FEATURES_TABLE = f"{PROJECT}.{DATASET}.hybrid_features"
NOTES_SRC = Path(__file__).resolve().parents[1] / "eval" / "results" / "hybrid_notes.json"
COHORT_SRC = Path(__file__).resolve().parents[1] / "eval" / "results" / "hybrid_cohort.json"

# Same feature-name list as the synthetic loader (49 model features).
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


def main() -> int:
    if not NOTES_SRC.exists() or not COHORT_SRC.exists():
        raise SystemExit(
            "hybrid sources not found — run scripts/build_hybrid_cohort.py first")

    notes_doc = json.loads(NOTES_SRC.read_text())
    cohort = json.loads(COHORT_SRC.read_text())
    notes_patients = notes_doc["patients"]
    cohort_patients = cohort["patients"]
    print(f"notes: {len(notes_patients)}  features: {len(cohort_patients)}")

    # Build notes + split rows from hybrid_notes.json.
    notes = [
        {"hadm_id": int(p["hadm_id"]), "note_id": f"MT-{p['variant']}-DS",
         "text": p["note"]}
        for p in notes_patients
    ]
    splits = [{"hadm_id": int(p["hadm_id"]), "split_name": "test"}
              for p in notes_patients]

    # Build feature rows from hybrid_cohort.json (49 features + bookkeeping).
    feature_rows = []
    for p in cohort_patients:
        f = p["features"]
        row = {"hadm_id": int(p["hadm_id"]), "split_name": "test"}
        for name in _FEATURE_NAMES:
            row[name] = f.get(name)
        row["subject_id"] = None
        row["readmission_30d"] = 0
        feature_rows.append(row)

    client = bigquery.Client(project=PROJECT)

    notes_schema = [
        bigquery.SchemaField("hadm_id", "INT64"),
        bigquery.SchemaField("note_id", "STRING"),
        bigquery.SchemaField("text", "STRING"),
    ]
    split_schema = [
        bigquery.SchemaField("hadm_id", "INT64"),
        bigquery.SchemaField("split_name", "STRING"),
    ]
    features_schema = [bigquery.SchemaField("hadm_id", "INT64")]
    features_schema += [bigquery.SchemaField(name, "FLOAT64") for name in _FEATURE_NAMES]
    features_schema += [
        bigquery.SchemaField("subject_id", "INT64"),
        bigquery.SchemaField("split_name", "STRING"),
        bigquery.SchemaField("readmission_30d", "INT64"),
    ]

    for rows, table, schema, label in [
        (notes, NOTES_TABLE, notes_schema, "notes"),
        (splits, SPLIT_TABLE, split_schema, "split"),
        (feature_rows, FEATURES_TABLE, features_schema, "features"),
    ]:
        job = client.load_table_from_json(
            rows, table,
            job_config=bigquery.LoadJobConfig(
                schema=schema, write_disposition="WRITE_TRUNCATE"))
        job.result()
        n = next(iter(client.query(
            f"SELECT COUNT(*) AS n FROM `{table}`").result()))["n"]
        ok = n == len(rows)
        print(f"{'OK' if ok else 'FAILED'}: {table} ({n} rows)")
        if not ok:
            return 1

    # Sanity: the exact JOIN the rag-ingest pipeline runs must return all 24.
    joined = next(iter(client.query(
        f"""
        SELECT COUNT(*) AS n
        FROM `{NOTES_TABLE}` AS d
        JOIN `{SPLIT_TABLE}` AS a ON d.hadm_id = a.hadm_id
        WHERE a.split_name = 'test'
        """).result()))["n"]
    print(f"joined notes for split 'test': {joined} (expect 24)")
    if joined != len(notes_patients):
        return 1

    print("\nNext: re-run the rag-ingest pipeline pointed at the hybrid tables —")
    print("  NOTES_TABLE_REF=readmission.hybrid_notes "
          "SPLIT_TABLE_REF=readmission.hybrid_split")
    print("  and point the predict path at readmission.hybrid_features "
          "(FEATURE_TABLE).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
