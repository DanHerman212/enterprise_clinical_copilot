"""prune_rag_datapoints.py — cheap incremental removal of pruned demo patients
from the kept Vector Search index and the BigQuery chunk/feature store.

The common-sense alternative to re-running the rag-ingest pipeline for a small
data change: after prune_inclusion_violations.py drops patients from the local
cohort artifacts, this removes the SAME patients from the served-system side so
nothing excluded remains:

  1. compute the datapoint ids (chunk ids) for the removed patients' notes,
     from BigQuery hybrid_notes (the exact source the index was built from),
  2. index.remove_datapoints() on the kept index resource (no endpoint needed),
  3. DELETE the patients from BigQuery hybrid_notes / hybrid_split /
     hybrid_features.

Run from projects/agent-harness.
"""
import sys
from pathlib import Path

from google.cloud import aiplatform, bigquery

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.chunking import DEFAULT_MAX_CHARS, chunk_note  # noqa: E402
from rag.embed import datapoint_id  # noqa: E402
from pipelines.components.chunk_notes import DEFAULT_SECTIONS as WHITELIST  # noqa: E402
from scripts.prune_inclusion_violations import REMOVE  # noqa: E402

PROJECT = "trim-icon-498815-a0"
LOCATION = "us-east1"
INDEX_RESOURCE = ("projects/trim-icon-498815-a0/locations/us-east1/"
                  "indexes/2805052074549575680")
DATASET = "readmission"
NOTES = f"{PROJECT}.{DATASET}.hybrid_notes"
SPLIT = f"{PROJECT}.{DATASET}.hybrid_split"
FEATURES = f"{PROJECT}.{DATASET}.hybrid_features"
PACK_TO = 700  # must match pipelines/components/chunk_notes.py default


def main() -> int:
    hadm_ids = sorted(REMOVE)
    in_clause = ", ".join(str(h) for h in hadm_ids)

    bq = bigquery.Client(project=PROJECT)

    # 1. Chunk the removed patients' notes exactly as the ingest pipeline does.
    sql = (f"SELECT hadm_id, note_id, text FROM `{NOTES}` "
           f"WHERE hadm_id IN ({in_clause})")
    rows = list(bq.query(sql).result())
    print(f"notes in BigQuery for {len(hadm_ids)} removed patients: {len(rows)}")
    dp_ids = set()
    for r in rows:
        note = {"hadm_id": r["hadm_id"], "note_id": r["note_id"],
                "text": r["text"]}
        for c in chunk_note(note, max_chars=DEFAULT_MAX_CHARS, pack_to=PACK_TO):
            if c.section in WHITELIST:
                dp_ids.add(datapoint_id(c.chunk_id))
    print(f"datapoint ids to remove from index: {len(dp_ids)}")

    # 2. Remove datapoints from the kept index resource (endpoint not needed).
    #    Only possible if the index was created with StreamUpdate enabled; a
    #    batch-mode index rejects it (400 "StreamUpdate is not enabled").
    aiplatform.init(project=PROJECT, location=LOCATION)
    index = aiplatform.MatchingEngineIndex(INDEX_RESOURCE)
    try:
        print("calling index.remove_datapoints…")
        index.remove_datapoints(datapoint_ids=sorted(dp_ids))
        print("remove_datapoints submitted (async update on the index resource).")
    except Exception as exc:  # noqa: BLE001 — report and continue
        print(f"index.remove_datapoints FAILED: {type(exc).__name__}: {exc}")
        print("The kept index was not built with StreamUpdate; it cannot be "
              "incrementally updated. Rebuild it (ideally with "
              "IndexUpdateMode.STREAM_UPDATE) to enable cheap future prunes.")

    # 3. Delete the patients from the BigQuery source-of-truth tables.
    for table in (NOTES, SPLIT, FEATURES):
        d = bq.query(f"DELETE FROM `{table}` WHERE hadm_id IN ({in_clause})")
        d.result()
        print(f"deleted {d.num_dml_affected_rows} rows from "
              f"{table.split('.')[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
