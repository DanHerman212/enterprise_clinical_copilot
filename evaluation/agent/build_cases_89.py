"""Build the evaluation case set from the cohort the deployment actually serves.

Which cohort artifact is authoritative is not obvious from the filenames, and
getting it wrong is the kind of mistake that produces a whole run of
`unknown_patient` traces before anyone reads a filename closely:

  * `hybrid_cohort_v2.json` — 89 patients, with `probability` and the
    `readmission_30d` label. This is what `load_hybrid_notes.py` loads when the
    notes are written to BigQuery, so it is the cohort the served data is built
    from.
  * `hybrid_cohort.pre-inclusion-prune.json` — 108 patients, the state before
    `prune_inclusion_violations.py` removed the patients whose notes violated the
    training cohort's inclusion criteria (the risk model was trained on adults,
    so a note with neonatal content is not a patient this system should serve).
  * `golden_sample_hybrid_108.json` — 108 patients, and the file the old runner
    defaults to. It is a pre-prune sample and is now wrong: it contains patients
    the deployment no longer serves.

The set is a **census, not a sample**. `sample.py` drew a risk-weighted subset
of 100 from 3,402 scored admissions because the agent eval needed a bounded set;
here the cohort is the served population and it is small, so every patient is
evaluated. Nothing is sampled out, which also means the pass rate is a statement
about the whole served cohort rather than about a drawn subset of it.

Bands are derived rather than read: the v2 artifact carries no `band` field, so
the cut points are applied here at the same values `sample.py` uses, which keeps
the stratification language the same across both case sets.

Usage (harness root):
    .venv/bin/python evaluation/agent/build_cases_89.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
COHORT = RESULTS / "hybrid_cohort_v2.json"
OUT = RESULTS / "cases_89.json"

# The same cut points sample.py samples against, so "low / borderline / high"
# means the same thing in both artifacts.
BANDS = (("low", 0.0, 0.10), ("borderline", 0.10, 0.20), ("high", 0.20, 1.01))

# The three clinical prompts the harness has always asked. Kept verbatim and in
# this order because the wording is part of the case set's identity: change it
# and the pass rate stops being comparable with any earlier run.
PROMPTS = (
    "risk", "meds", "summarize",
)


def band_for(probability: float) -> str:
    for name, lo, hi in BANDS:
        if lo <= probability < hi:
            return name
    return "high"


def main() -> int:
    cohort = json.loads(COHORT.read_text())
    patients = cohort["patients"]

    cases = []
    for p in patients:
        probability = p["probability"]
        cases.append({
            "hadm_id": p["hadm_id"],
            "probability": probability,
            "band": band_for(probability),
            # Kept because it is what makes an error rate interpretable: 13 of
            # 89 positives means a "high risk" miss and a "low risk" miss are
            # not the same finding, and the label is the only way to tell which
            # one a failure was.
            "readmission_30d": p.get("readmission_30d"),
        })

    counts: dict[str, int] = {}
    for c in cases:
        counts[c["band"]] = counts.get(c["band"], 0) + 1

    payload = {
        "source": COHORT.name,
        "built": datetime.now(timezone.utc).isoformat(),
        "n": len(cases),
        "threshold": cohort.get("threshold"),
        "seed": cohort.get("seed"),
        "prompts": list(PROMPTS),
        "cases_expected": len(cases) * len(PROMPTS),
        "labels_positive": sum(1 for c in cases if c.get("readmission_30d") == 1),
        "band_counts": counts,
        "patients": cases,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"wrote {OUT}")
    print(f"  patients {payload['n']} (from {payload['source']})")
    print(f"  bands {counts}")
    print(f"  positives {payload['labels_positive']}/{payload['n']}  "
          f"threshold {payload['threshold']}")
    print(f"  cases to run {payload['cases_expected']} "
          f"({payload['n']} x {len(PROMPTS)} prompts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
