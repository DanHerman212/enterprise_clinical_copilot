"""count_hybrid_chunks.py — replicate chunk_notes locally to get EXPECTED_VECTORS.

Reads the hybrid notes from BigQuery (same SQL + whitelist + chunker the
rag-ingest chunk_notes component runs), and prints the exact number of chunks
the tree-ah index must hold. Feed that number to EXPECTED_VECTORS on the
pipeline submit.
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google.cloud import bigquery  # noqa: E402
from rag.chunking import DEFAULT_MAX_CHARS, chunk_note  # noqa: E402

PROJECT = "trim-icon-498815-a0"
NOTES = f"{PROJECT}.readmission.hybrid_notes"
SPLIT = f"{PROJECT}.readmission.hybrid_split"
WHITELIST = (
    "history_of_present_illness", "past_medical_history", "family_history",
    "social_history", "physical_exam", "brief_hospital_course",
    "discharge_condition", "discharge_diagnosis", "discharge_medications",
    "medications_on_admission", "discharge_disposition", "discharge_instructions",
)


def main() -> int:
    client = bigquery.Client(project=PROJECT)
    sql = (
        f"SELECT d.hadm_id, d.note_id, d.text FROM `{NOTES}` AS d "
        f"JOIN `{SPLIT}` AS a ON d.hadm_id = a.hadm_id "
        f"WHERE a.split_name = 'test'"
    )
    section_counts: Counter[str] = Counter()
    seen: set[str] = set()
    total = 0
    for row in client.query(sql).result():
        note = {"hadm_id": row["hadm_id"], "note_id": row["note_id"],
                "text": row["text"]}
        for chunk in chunk_note(note, max_chars=DEFAULT_MAX_CHARS, pack_to=1):
            if chunk.section not in WHITELIST:
                continue
            if chunk.chunk_id in seen:
                print(f"DUP: {chunk.chunk_id}")
            seen.add(chunk.chunk_id)
            section_counts[chunk.section] += 1
            total += 1
    print(f"TOTAL chunks: {total}")
    print(f"unique ids: {len(seen)}")
    print(f"by section: {dict(section_counts)}")
    print(f"\nEXPECTED_VECTORS={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
