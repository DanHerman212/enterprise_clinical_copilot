"""build_hybrid_cohort.py — build the hybrid demo cohort from 24 real MTSamples notes.

Ties the Step-2 analysis into the artifact the rest of the demo consumes:

  * 24 notes from data/mtsamples/selection_24.json (8 low / 8 borderline / 8 high)
  * story-anchored 49-feature rows via fill_features._provisional_row (deterministic)
  * risk = the real served model via fill_features._score (identical numbers the
    deployed endpoint returns — same bundle, threshold, group aggregation)
  * hadm_ids 90000001-90000024 (same scheme as the synthetic cohort, so the site
    seed / fixtures / name-assignment machinery is unchanged)

Writes (all gitignored data, plus committed eval artifacts):
  eval/results/hybrid_cohort.json  (features contract, mirrors synthetic_cohort.json)
  eval/results/hybrid_notes.json   (notes contract, mirrors synthetic_notes.json)
  data/hybrid/provenance.json      (per-feature provenance + top_factors, transparency)

Usage (from projects/agent-harness):
  ../../.venv/bin/python scripts/build_hybrid_cohort.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.fill_features import (  # noqa: E402
    THRESHOLD,
    _provisional_row,
    _score,
    band_of,
)

HARNESS = Path(__file__).resolve().parents[1]
DATA_DIR = HARNESS / "data" / "mtsamples"
HYBRID_DIR = HARNESS / "data" / "hybrid"
SELECTION = DATA_DIR / "selection_24.json"
COHORT_OUT = HARNESS / "eval" / "results" / "hybrid_cohort.json"
NOTES_OUT = HARNESS / "eval" / "results" / "hybrid_notes.json"

# Band order for hadm_id assignment (low 90000001-90000008, etc.), mirroring the
# synthetic cohort's 8/8/8 layout. Within a band, ascending probability.
_BAND_RANK = {"low": 0, "borderline": 1, "high": 2}
_FIRST_ID = 90_000_001


def main() -> int:
    if not SELECTION.exists():
        raise SystemExit(f"selection not found: {SELECTION}")

    selection = json.loads(SELECTION.read_text())
    if len(selection) != 24:
        raise SystemExit(f"expected 24 selected notes, got {len(selection)}")

    # deterministic order: band (low<borderline<high), then probability asc.
    ordered = sorted(
        selection, key=lambda s: (_BAND_RANK[s["band"]], s["prob"]))

    patients: list[dict] = []
    notes: list[dict] = []
    provenance: dict[str, dict] = {}

    for i, sel in enumerate(ordered):
        sid = sel["sid"]
        hadm_id = _FIRST_ID + i
        text = (DATA_DIR / f"{sid}.txt").read_text(encoding="utf-8")

        row, prov = _provisional_row(text)
        score = _score(row)
        prob = score["probability"]
        band = band_of(prob)

        if band != sel["band"]:
            print(f"WARN {sid}: selection band {sel['band']} but fill scores "
                  f"{band} (prob {prob:.4f}) — using scored band")

        patients.append({
            "hadm_id": hadm_id,
            "archetype": f"mtsamples:{sid}",
            "band": band,
            "probability": round(prob, 6),
            "threshold": THRESHOLD,
            "features": {k: row[k] for k in sel_order()},
        })
        notes.append({
            "hadm_id": hadm_id,
            "archetype": f"mtsamples:{sid}",
            "band": band,
            "variant": sid,
            "note": text,
        })
        provenance[str(hadm_id)] = {
            "sid": sid,
            "band": band,
            "probability": round(prob, 4),
            "top_factors": [
                {"feature": t["feature"], "attribution": round(t["attribution"], 4)}
                for t in score["top_factors"][:5]
            ],
            "provenance": prov,
        }

    HYBRID_DIR.mkdir(parents=True, exist_ok=True)
    COHORT_OUT.parent.mkdir(parents=True, exist_ok=True)

    cohort = {"seed": 42, "n": len(patients), "patients": patients}
    notes_doc = {"n": len(notes), "patients": notes}
    COHORT_OUT.write_text(json.dumps(cohort, indent=1) + "\n")
    NOTES_OUT.write_text(json.dumps(notes_doc, indent=1) + "\n")
    (HYBRID_DIR / "provenance.json").write_text(
        json.dumps(provenance, indent=1) + "\n")

    from collections import Counter
    bands = Counter(p["band"] for p in patients)
    print(f"wrote {COHORT_OUT.relative_to(HARNESS.parents[1])} ({len(patients)} patients)")
    print(f"wrote {NOTES_OUT.relative_to(HARNESS.parents[1])}")
    print(f"bands: {dict(bands)}")
    print("\npatients:")
    for p in patients:
        print(f"  {p['hadm_id']} {p['band']:<10} {p['probability']:.4f}  "
              f"{p['archetype']}")
    return 0


def sel_order() -> list[str]:
    from scripts.fill_features import FEATURES  # noqa: PLC0415
    return FEATURES


if __name__ == "__main__":
    raise SystemExit(main())
