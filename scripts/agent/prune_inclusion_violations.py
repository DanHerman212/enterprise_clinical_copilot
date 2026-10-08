"""prune_inclusion_violations.py — remove cohort-inclusion violations from the
hybrid-108 demo corpus.

The risk model was trained on an ADULT-ONLY cohort (all patients >= 18). The
2026-08-25 coherence scan (scripts/coherence_scan.py) flagged notes that are
neonates/infants/toddlers or that filled an age < 18 — these are out of
distribution and must not appear in the demo. Each id below was confirmed by
manual review of the note text (see session doc 2026-08-25).

Backs up the three artifacts to *.pre-inclusion-prune.json, drops the flagged
patients (other entries kept byte-identical), and updates the n counts.

Usage (from services):
  ../../.venv/bin/python scripts/prune_inclusion_violations.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[2]

# hadm_ids removed 2026-08-25 (coherence_scan + manual note review):
#   neonate/infant notes: 90000006 (14mo), 90000007 (10mo), 90000021 (1mo),
#     90000038 (ex-34wk preemie), 90000062 (newborn, Kowalski), 90000083
#     (10wk infant), 90000085 (ex-32wk preemie)
#   age < 18:              90000042 (11), 90000066 (12), 90000098 (10)
#   AMA discharge:         90000072, 90000089, 90000096
#   elective + planned
#     hospital return:     90000093 (scheduled follow-up -> re-admit for
#                          debridement/revision)
#   hospice discharge:     90000008, 90000075
#   pediatric slip-through: 90000032 (3yo boy), 90000080 (3.5yo boy) — missed
#                          by the first pass; caught by the year-old check
#   sex/pronoun mismatch:   90000061 (male CVA note labeled F)
#   expired in hospital:    90000039 (expired 12/24/08), 90000107 (pronounced
#                          expired 08/21/08) — found 2026-10-08 by the
#                          coherence_scan expired check added that day. Both
#                          slipped through because the feature row cannot carry
#                          the signal: the generator omits discharge_location
#                          on death, so no discharge_location_* flag is set.
REMOVE = {
    90000006, 90000007, 90000021, 90000038, 90000042,
    90000062, 90000066, 90000083, 90000085, 90000098,
    90000072, 90000089, 90000096, 90000093,
    90000008, 90000075,
    90000032, 90000080, 90000061,
    90000039, 90000107,
}

FILES = [
    HARNESS / "evaluation" / "agent" / "results" / "hybrid_cohort.json",
    HARNESS / "evaluation" / "agent" / "results" / "hybrid_notes.json",
    # Moved from data/hybrid/ to data/agent/hybrid/. The old path was still
    # listed here, so on 2026-10-08 this script pruned two artifacts and then
    # died on the third — a partial prune, which is the one outcome it must
    # never produce. See the guard in main().
    HARNESS / "data" / "agent" / "hybrid" / "provenance.json",
]


def main() -> int:
    # All or nothing. An exclusion criterion has to land in every store or in
    # none of them: pruning the cohort list while leaving the note corpus (or
    # the provenance record) intact leaves the sources disagreeing about who is
    # in the demo, and nothing downstream can tell which of them is right.
    missing = [p for p in FILES if not p.exists()]
    if missing:
        for path in missing:
            print(f"MISSING: {path}")
        print(
            "refusing to prune a partial set — the artifacts would disagree "
            "with each other. Restore the missing artifact or correct FILES."
        )
        return 1

    for path in FILES:
        backup = path.with_suffix(".pre-inclusion-prune.json")
        if not backup.exists():
            shutil.copy2(path, backup)
        data = json.loads(path.read_text())

        if path.name == "provenance.json":
            dropped = {k for k in data if int(k) in REMOVE}
            for k in dropped:
                del data[k]
            print(f"{path.name}: removed {len(dropped)} entries "
                  f"({sorted(int(k) for k in dropped)})")
        else:
            before = len(data["patients"])
            data["patients"] = [
                p for p in data["patients"] if p["hadm_id"] not in REMOVE
            ]
            data["n"] = len(data["patients"])
            removed = before - len(data["patients"])
            print(f"{path.name}: {before} -> {len(data['patients'])} "
                  f"(removed {removed})")

        path.write_text(json.dumps(data, indent=2) + "\n")
    print("done. originals backed up as *.pre-inclusion-prune.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
