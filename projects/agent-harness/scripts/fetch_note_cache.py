"""Fetch test-split discharge notes from BigQuery into the local cache.

One full scan of the discharge table (~3.5 GB, ~$0.02 on-demand), then the
probes iterate the cache for free. See rag/notes.py for why the cache lives
outside the repo.

The fetch refuses to write a cache that fails its own sanity checks:
  * every hadm_id must appear exactly once (verified fact of this corpus;
    a join gone wrong shows up here as duplicates)
  * no empty note text

Usage:
    python scripts/fetch_note_cache.py
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import bigquery

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.notes import CACHE_DIR, MANIFEST_PATH, NOTES_PATH  # noqa: E402

PROJECT = "trim-icon-498815-a0"

QUERY = """
SELECT
    d.hadm_id,
    d.note_id,
    d.text
FROM `trim-icon-498815-a0.mimiciv_note.discharge` AS d
JOIN `trim-icon-498815-a0.readmission.analytics_dataset_encoded` AS a
    ON d.hadm_id = a.hadm_id
WHERE a.split_name = 'test'
"""


def main() -> int:
    client = bigquery.Client(project=PROJECT)
    print("Querying test-split discharge notes…")
    rows = client.query(QUERY).result()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = NOTES_PATH.with_suffix(".tmp")

    hadm_counts: Counter[int] = Counter()
    note_count = 0
    total_chars = 0
    empty_notes = 0

    with gzip.open(tmp_path, "wt", encoding="utf-8") as out:
        for row in rows:
            record = {
                "hadm_id": row["hadm_id"],
                "note_id": row["note_id"],
                "text": row["text"],
            }
            out.write(json.dumps(record) + "\n")
            hadm_counts[record["hadm_id"]] += 1
            note_count += 1
            total_chars += len(record["text"] or "")
            if not (record["text"] or "").strip():
                empty_notes += 1
            if note_count % 5000 == 0:
                print(f"  …{note_count} notes")

    duplicates = {h: c for h, c in hadm_counts.items() if c > 1}
    if duplicates:
        tmp_path.unlink()
        print(f"FAILED: {len(duplicates)} hadm_ids have multiple notes; "
              "expected exactly one each. Cache not written.")
        return 1
    if empty_notes:
        tmp_path.unlink()
        print(f"FAILED: {empty_notes} notes have empty text. Cache not written.")
        return 1

    # Write data before manifest: a crash between the two leaves a cache that
    # iter_notes() rejects (no manifest), never one it silently accepts.
    tmp_path.rename(NOTES_PATH)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "query": QUERY.strip(),
        "note_count": note_count,
        "distinct_hadm_ids": len(hadm_counts),
        "total_chars": total_chars,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")

    size_mb = NOTES_PATH.stat().st_size / 1e6
    print(f"\nWrote {NOTES_PATH}  ({size_mb:.0f} MB compressed)")
    print(f"  notes:        {note_count}")
    print(f"  hadm_ids:     {len(hadm_counts)} (all unique)")
    print(f"  total chars:  {total_chars:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
