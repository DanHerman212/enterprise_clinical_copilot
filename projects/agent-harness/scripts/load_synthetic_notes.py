"""load_synthetic_notes — push the synthetic discharge notes into BigQuery.

Task 4 of the synthetic cohort: the KFP rag-ingest pipeline reads notes from
BigQuery (JOIN on hadm_id, WHERE split_name='test'), so the synthetic corpus
must be loaded there with the same shape the pipeline queries.

Writes two tables (WRITE_TRUNCATE, so re-runs are idempotent):
  readmission.synthetic_notes   (hadm_id INT64, note_id STRING, text STRING)
  readmission.synthetic_split   (hadm_id INT64, split_name STRING)  # all 'test'

Source: eval/results/synthetic_notes.json  {n, patients:[{hadm_id, archetype,
band, variant, note}]}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from google.cloud import bigquery

PROJECT = "trim-icon-498815-a0"
NOTES_TABLE = f"{PROJECT}.readmission.synthetic_notes"
SPLIT_TABLE = f"{PROJECT}.readmission.synthetic_split"
RESULTS = Path(__file__).resolve().parents[1] / "eval" / "results" / "synthetic_notes.json"


def main() -> int:
    data = json.loads(RESULTS.read_text())
    patients = data["patients"]
    print(f"source: {RESULTS}")
    print(f"patients: {len(patients)}")

    notes = [
        {
            "hadm_id": int(p["hadm_id"]),
            "note_id": f"SYN-{p['hadm_id']}-DS",
            "text": p["note"],
        }
        for p in patients
    ]
    splits = [
        {"hadm_id": int(p["hadm_id"]), "split_name": "test"}
        for p in patients
    ]

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

    job = client.load_table_from_json(
        notes,
        NOTES_TABLE,
        job_config=bigquery.LoadJobConfig(
            schema=notes_schema, write_disposition="WRITE_TRUNCATE"
        ),
    )
    job.result()
    n_notes = next(iter(client.query(
        f"SELECT COUNT(*) AS n FROM `{NOTES_TABLE}`").result()))["n"]
    if n_notes != len(notes):
        print(f"FAILED: wrote {len(notes)} records but table holds {n_notes}.")
        return 1

    job = client.load_table_from_json(
        splits,
        SPLIT_TABLE,
        job_config=bigquery.LoadJobConfig(
            schema=split_schema, write_disposition="WRITE_TRUNCATE"
        ),
    )
    job.result()
    n_split = next(iter(client.query(
        f"SELECT COUNT(*) AS n FROM `{SPLIT_TABLE}`").result()))["n"]
    if n_split != len(splits):
        print(f"FAILED: wrote {len(splits)} split rows but table holds {n_split}.")
        return 1

    # sanity: the exact JOIN the pipeline runs should return all 24
    joined = next(iter(client.query(
        f"""
        SELECT COUNT(*) AS n
        FROM `{NOTES_TABLE}` AS d
        JOIN `{SPLIT_TABLE}` AS a ON d.hadm_id = a.hadm_id
        WHERE a.split_name = 'test'
        """
    ).result()))["n"]
    print(f"joined notes for split 'test': {joined}")
    if joined != len(patients):
        print("FAILED: join does not recover all synthetic notes.")
        return 1

    print(f"OK: {NOTES_TABLE} ({n_notes} rows) + {SPLIT_TABLE} ({n_split} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
