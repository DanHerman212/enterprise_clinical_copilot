"""Build chunks over the full test-split note cache (guide §4).

Local-first: reads the note cache and writes chunks to a local file, so we can
measure the real chunk count (the §5 cost input) without any GCP write. The
BigQuery write of readmission.note_chunks happens at deployment time.

Idempotent: chunk_ids are deterministic, so re-running produces identical rows
and never orphans an index.

    .venv/bin/python scripts/build_chunks.py [--max-chars N] [--limit N]

Writes:  ~/.cache/.../chunks.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.chunking import DEFAULT_MAX_CHARS, chunk_note  # noqa: E402
from rag.notes import CACHE_DIR, iter_notes, read_manifest  # noqa: E402

CHUNKS_PATH = CACHE_DIR / "chunks.jsonl.gz"

# Narrative/assessment sections worth indexing. Metadata (name, dates, sex) and
# lab-line noise (pertinent_results) add cost without retrieval value; the demo
# story lives in the sections a clinician writes. Override with --sections.
DEFAULT_SECTIONS = (
    "history_of_present_illness",
    "past_medical_history",
    "family_history",
    "social_history",
    "physical_exam",
    "brief_hospital_course",
    "discharge_condition",
    "discharge_diagnosis",
    "discharge_medications",
    "medications_on_admission",
    "discharge_disposition",
    "discharge_instructions",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--pack-to", type=int, default=None,
                        help="greedily merge pieces into chunks up to this size")
    parser.add_argument("--sections", default=",".join(DEFAULT_SECTIONS),
                        help="comma-separated section whitelist")
    parser.add_argument("--limit", type=int, default=None,
                        help="chunk only the first N notes (sizing quick pass)")
    args = parser.parse_args()

    whitelist = {s.strip() for s in args.sections.split(",") if s.strip()}
    manifest = read_manifest()
    print(f"Cache: {manifest['note_count']} notes  "
          f"(pack_to={args.pack_to}, {len(whitelist)} sections)")

    section_counts: Counter[str] = Counter()
    lengths: list[int] = []
    notes_with_zero_chunks = 0
    note_count = 0

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CHUNKS_PATH.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as out:
        for note in iter_notes():
            if args.limit is not None and note_count >= args.limit:
                break
            note_count += 1
            chunks = [
                chunk for chunk in chunk_note(note, max_chars=args.max_chars,
                                             pack_to=args.pack_to)
                if chunk.section in whitelist
            ]
            if not chunks:
                notes_with_zero_chunks += 1
            for chunk in chunks:
                section_counts[chunk.section] += 1
                lengths.append(len(chunk.text))
                out.write(json.dumps(chunk.__dict__) + "\n")
            if note_count % 5000 == 0:
                print(f"  …{note_count} notes")
    tmp.rename(CHUNKS_PATH)

    total = sum(section_counts.values())
    print(f"\nWrote {CHUNKS_PATH}")
    print(f"  notes chunked:    {note_count}")
    print(f"  notes w/ 0 chunks:{notes_with_zero_chunks} "
          f"({100*notes_with_zero_chunks/note_count:.1f}%)")
    print(f"  total chunks:     {total:,}")
    print(f"  chunks/note:      {total/note_count:.1f}")
    print(f"  chunk chars:      median {statistics.median(lengths):.0f}  "
          f"mean {statistics.mean(lengths):.0f}  max {max(lengths)}")
    print(f"\n  {'section':<28}{'chunks':>9}{'share':>8}")
    for section, count in section_counts.most_common():
        print(f"    {section:<26}{count:>9}{100*count/total:>7.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
