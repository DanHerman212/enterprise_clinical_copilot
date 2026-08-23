"""build_hybrid_cohort_108.py — extend the hybrid cohort from 24 to 108 notes.

Keeps the existing 24-patient cohort EXACTLY as-is (same hadm_ids 90000001-
90000024, same features/probabilities/provenance — the live demo depends on
those and they stay stable). Scores the remaining 84 MTSamples notes from
data/mtsamples/manifest.json with the same story-anchored fill + served model
(fill_features._provisional_row / _score / band_of), assigns hadm_ids
90000025-90000108 (band, then probability asc — same scheme as the 24), and
merges everything back into the artifacts the rest of the pipeline consumes:

  eval/results/hybrid_cohort.json   (108 patients — features contract)
  eval/results/hybrid_notes.json    (108 patients — notes contract)
  data/hybrid/provenance.json       (108 entries — per-feature provenance)

This is the eval corpus for the 324-question revalidation (see
docs/eval_sample_size.md): every curated note becomes a scoreable patient.

Usage (from projects/agent-harness):
  ../../.venv/bin/python scripts/build_hybrid_cohort_108.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.fill_features import (  # noqa: E402
    FEATURES,
    THRESHOLD,
    _provisional_row,
    _score,
    band_of,
)

HARNESS = Path(__file__).resolve().parents[1]
DATA_DIR = HARNESS / "data" / "mtsamples"
HYBRID_DIR = HARNESS / "data" / "hybrid"
MANIFEST = DATA_DIR / "manifest.json"
COHORT_OUT = HARNESS / "eval" / "results" / "hybrid_cohort.json"
NOTES_OUT = HARNESS / "eval" / "results" / "hybrid_notes.json"
PROVENANCE_OUT = HYBRID_DIR / "provenance.json"

_BAND_RANK = {"low": 0, "borderline": 1, "high": 2}
N_EXPECTED_EXISTING = 24
N_EXPECTED_TOTAL = 108


def main() -> int:
    if not (COHORT_OUT.exists() and NOTES_OUT.exists() and PROVENANCE_OUT.exists()):
        raise SystemExit(
            "run scripts/build_hybrid_cohort.py first (24-patient baseline)")

    cohort = json.loads(COHORT_OUT.read_text())
    notes_doc = json.loads(NOTES_OUT.read_text())
    provenance = json.loads(PROVENANCE_OUT.read_text())

    existing = cohort["patients"]
    existing_notes = notes_doc["patients"]
    if len(existing) != N_EXPECTED_EXISTING:
        raise SystemExit(
            f"expected {N_EXPECTED_EXISTING} baseline patients, got {len(existing)}")
    if len(existing_notes) != N_EXPECTED_EXISTING:
        raise SystemExit(f"baseline notes mismatch: {len(existing_notes)}")

    existing_sids = {p["archetype"].split(":")[1] for p in existing}
    manifest = json.loads(MANIFEST.read_text())
    all_sids = [str(f["id"]) for f in manifest["files"]]
    if len(all_sids) != N_EXPECTED_TOTAL:
        raise SystemExit(f"expected {N_EXPECTED_TOTAL} manifest notes, got {len(all_sids)}")

    missing = existing_sids - set(all_sids)
    if missing:
        raise SystemExit(f"baseline sids missing from manifest: {sorted(missing)}")

    new_sids = [s for s in all_sids if s not in existing_sids]
    if len(new_sids) != N_EXPECTED_TOTAL - N_EXPECTED_EXISTING:
        raise SystemExit(
            f"expected {N_EXPECTED_TOTAL - N_EXPECTED_EXISTING} new notes, "
            f"got {len(new_sids)}")

    # Score each new note with the SAME fill + served model as the baseline.
    scored: list[dict] = []
    for sid in new_sids:
        text = (DATA_DIR / f"{sid}.txt").read_text(encoding="utf-8")
        row, prov = _provisional_row(text)
        score = _score(row)
        prob = score["probability"]
        scored.append({
            "sid": sid, "text": text, "row": row, "prov": prov,
            "prob": prob, "band": band_of(prob),
            "top": score["top_factors"][:5],
        })

    # Deterministic order for the new ids: band, then probability asc.
    ordered = sorted(scored, key=lambda r: (_BAND_RANK[r["band"]], r["prob"]))
    next_hadm = max(p["hadm_id"] for p in existing) + 1

    patients = list(existing)        # first 24 unchanged (demo-stable)
    notes = list(existing_notes)
    for i, r in enumerate(ordered):
        hadm_id = next_hadm + i
        patients.append({
            "hadm_id": hadm_id,
            "archetype": f"mtsamples:{r['sid']}",
            "band": r["band"],
            "probability": round(r["prob"], 6),
            "threshold": THRESHOLD,
            "features": {k: r["row"][k] for k in FEATURES},
        })
        notes.append({
            "hadm_id": hadm_id,
            "archetype": f"mtsamples:{r['sid']}",
            "band": r["band"],
            "variant": r["sid"],
            "note": r["text"],
        })
        provenance[str(hadm_id)] = {
            "sid": r["sid"],
            "band": r["band"],
            "probability": round(r["prob"], 4),
            "top_factors": [
                {"feature": t["feature"], "attribution": round(t["attribution"], 4)}
                for t in r["top"]
            ],
            "provenance": r["prov"],
        }

    # Write merged artifacts (n is updated on the container docs).
    cohort["n"] = len(patients)
    cohort["patients"] = patients
    notes_doc["n"] = len(notes)
    notes_doc["patients"] = notes
    COHORT_OUT.write_text(json.dumps(cohort, indent=1) + "\n")
    NOTES_OUT.write_text(json.dumps(notes_doc, indent=1) + "\n")
    PROVENANCE_OUT.write_text(json.dumps(provenance, indent=1) + "\n")

    bands = Counter(p["band"] for p in patients)
    new_bands = Counter(r["band"] for r in ordered)
    print(f"wrote {COHORT_OUT.relative_to(HARNESS.parents[1])} "
          f"({len(patients)} patients)")
    print(f"wrote {NOTES_OUT.relative_to(HARNESS.parents[1])}")
    print(f"wrote {PROVENANCE_OUT.relative_to(HARNESS.parents[1])}")
    print(f"bands (all {len(patients)}): {dict(bands)}")
    print(f"bands (new {len(ordered)}):   {dict(new_bands)}")
    print("\nnew patients:")
    for p in patients[N_EXPECTED_EXISTING:]:
        print(f"  {p['hadm_id']} {p['band']:<10} {p['probability']:.4f}  "
              f"{p['archetype']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
