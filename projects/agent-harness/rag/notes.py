"""Local cache of test-split discharge notes.

The evidence-gate probes (A1, A4) read the same 378 MB of note text over and
over while rules are tuned. One BigQuery scan fills this cache; everything after
that reads from disk.

The cache lives under ~/.cache, NOT in the repo, because the repo sits on the
Desktop and the Desktop is iCloud-synced. MIMIC note text must not end up in
cloud sync. (The repo's data/ directory is gitignored, but gitignore does not
stop iCloud.)

Format: gzipped JSONL, one object per note: {"hadm_id", "note_id", "text"}.
A manifest JSON sits beside it recording counts, so a probe can fail loudly on a
stale or partial cache instead of quietly analysing half a corpus.
"""

from __future__ import annotations

import gzip
import json
import os
from collections.abc import Iterator
from pathlib import Path

CACHE_DIR = Path(
    os.environ.get(
        "NOTE_CACHE_DIR",
        Path.home() / ".cache" / "enterprise_clinical_copilot",
    )
)
NOTES_PATH = CACHE_DIR / "discharge_test_split.jsonl.gz"
MANIFEST_PATH = CACHE_DIR / "discharge_test_split.manifest.json"


def read_manifest() -> dict:
    """Load the cache manifest, or raise with instructions if there is none."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"No note cache manifest at {MANIFEST_PATH}. "
            "Run scripts/fetch_note_cache.py first."
        )
    return json.loads(MANIFEST_PATH.read_text())


def iter_notes() -> Iterator[dict]:
    """Yield cached notes, verifying the count against the manifest.

    A truncated gzip file otherwise reads as a short-but-valid corpus, and every
    statistic computed from it would be plausible and wrong.
    """
    expected = read_manifest()["note_count"]
    seen = 0
    with gzip.open(NOTES_PATH, "rt", encoding="utf-8") as handle:
        for line in handle:
            seen += 1
            yield json.loads(line)
    if seen != expected:
        raise RuntimeError(
            f"Note cache is corrupt: manifest says {expected} notes, "
            f"file contains {seen}. Re-run scripts/fetch_note_cache.py."
        )
