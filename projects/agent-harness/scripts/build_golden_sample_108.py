"""build_golden_sample_108.py — 324-question golden sample over the hybrid 108.

Reads eval/results/hybrid_cohort.json (108 scored patients) and writes the
golden sample in the shape collect.py consumes:

  eval/results/golden_sample_hybrid_108.json
    {"seed": 42, "n": 108, "patients": [{hadm_id, probability, band, archetype}]}

collect.py then runs 3 prompts (risk / meds / summarize) per patient =
108 x 3 = 324 agent runs. See docs/eval_sample_size.md for why 324.

Usage (from projects/agent-harness):
  ../../.venv/bin/python scripts/build_golden_sample_108.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
COHORT_SRC = HARNESS / "eval" / "results" / "hybrid_cohort.json"
OUT = HARNESS / "eval" / "results" / "golden_sample_hybrid_108.json"


def main() -> int:
    if not COHORT_SRC.exists():
        raise SystemExit(f"cohort not found: {COHORT_SRC} — run "
                         f"build_hybrid_cohort_108.py first")
    cohort = json.loads(COHORT_SRC.read_text())
    patients = [
        {
            "hadm_id": p["hadm_id"],
            "probability": p["probability"],
            "band": p["band"],
            "archetype": p["archetype"],
        }
        for p in cohort["patients"]
    ]
    doc = {"seed": 42, "n": len(patients), "patients": patients}
    OUT.write_text(json.dumps(doc, indent=1) + "\n")

    from collections import Counter
    bands = Counter(p["band"] for p in patients)
    print(f"wrote {OUT.relative_to(HARNESS.parents[1])} ({len(patients)} patients)")
    print(f"eval questions: {len(patients) * 3} (3 prompts each)")
    print(f"bands: {dict(bands)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
