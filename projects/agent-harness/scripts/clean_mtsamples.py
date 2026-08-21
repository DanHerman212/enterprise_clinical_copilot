"""clean_mtsamples.py — strip MTSamples page chrome from the crawled notes.

The crawler's body-extraction anchor failed on ~70 of 108 notes, leaving the
site nav header (Home / Sitemap / specialty dropdown / "Educational Disclaimer")
prepended to the clinical body. This pass re-cuts each note at the FIRST line
that begins a clinical section, discarding everything before it.

Anchor strategy (most robust first):
1. `Intended for:` — the per-note metadata line that directly precedes the
   clinical body. Covers notes whose body opens with a heading that is either
   non-colon ("ADMISSION DIAGNOSES\n1. ...") or not in the MIMIC list
   ("HISTORY OF ILLNESS:", "LONG-TERM GOALS:", "COMPLICATIONS:").
2. `_BODY_START` — MIMIC-style clinical headings, only applied when nav chrome
   is still present (never truncates an already-clean note at a mid-body
   heading like "DIAGNOSIS:" that happens to appear later).

Target files in place (data/mtsamples/*.txt → *.txt rewritten), so downstream
steps see clean notes. Raw-text/no-git posture unchanged.

Usage (from projects/agent-harness):
  ../../.venv/bin/python scripts/clean_mtsamples.py
"""

import re
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = HARNESS_ROOT / "data" / "mtsamples"

# Any of these at line start marks the clinical body. Order matters: longer
# phrases first so they win over shorter substrings. (HISTORY OF PRESENT
# ILLNESS before PRESENT ILLNESS, DISCHARGE DIAGNOSIS before DIAGNOSIS.)
_BODY_START = re.compile(
    r"^(?:"
    r"ADMISSION DIAGNOSIS|ADMITTING DIAGNOSIS|ADMISSION DIAGNOSES|"
    r"DISCHARGE DIAGNOSIS|DISCHARGE DIAGNOSES|FINAL DIAGNOSIS|"
    r"HISTORY OF PRESENT ILLNESS|PRESENT ILLNESS|CHIEF COMPLAINT|"
    r"REASON FOR ADMISSION|SUMMARY|SUMMARY OF ADMISSION|DIAGNOSIS"
    r")\s*:",
    re.IGNORECASE | re.MULTILINE,
)

# Metadata / site-chrome strings that indicate the clinical body hasn't started
# yet. Used to decide whether a fallback cut is safe.
_NAV_MARKERS = (
    "Sample Name", "Medical Specialty", "Educational Disclaimer",
    "Transcribed Medical Transcription", "Intended for",
)

_ANCHOR = "Intended for:"


def clean(text: str) -> str:
    # Anchor 1: the per-note metadata line that always precedes the body.
    i = text.find(_ANCHOR)
    if i != -1:
        after = text[i + len(_ANCHOR):]
        nl = after.find("\n")
        body = after[nl + 1:] if nl != -1 else after
        return body.strip()

    # Anchor 2: MIMIC-style clinical heading — only if nav chrome is present,
    # so we never cut a clean note at a mid-body heading.
    if any(mk in text for mk in _NAV_MARKERS):
        m = _BODY_START.search(text)
        if m:
            return text[m.start():].strip()

    return text.strip()


def main() -> int:
    changed = leaked = 0
    nav_markers = _NAV_MARKERS
    for path in sorted(DATA_DIR.glob("*.txt")):
        if path.name == "labels.json":
            continue
        before = path.read_text(encoding="utf-8")
        after = clean(before)
        if len(after) != len(before):
            changed += 1
        path.write_text(after, encoding="utf-8")
        if any(mk in after for mk in nav_markers):
            leaked += 1
            print(f"  still-leaking: {path.stem} ({len(after)} chars)")
    print(f"cleaned {changed}/108 notes; {leaked} still contain nav chrome")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
