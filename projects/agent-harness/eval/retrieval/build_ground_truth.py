"""build_ground_truth — note-structure census for the retrieval eval.

Ground truth for section recall: for every demo-cohort patient, parse the
discharge note and record which summary sections actually exist. A meds
question on a patient whose note has NO discharge_medications / instructions
section must not count as a recall miss — the honest answer is "not
available".

Reads BigQuery only (no index endpoint), so it can run while the serving
endpoints are still deploying.

Usage (from projects/agent-harness):
    ../../.venv/bin/python eval/retrieval/build_ground_truth.py \
        --cohort /Users/danherman/Desktop/danielmherman/demo/data/demo_cohort.json \
        --out /tmp/ground_truth.json
"""

import argparse
import json
import os
import sys

# Harness root (parent of eval/retrieval/) so mcp_server/rag are importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from mcp_server.tools.rag_search import _fetch_note
from rag.sections import parse_note

SUMMARY_SECTIONS = (
    "brief_hospital_course",
    "discharge_diagnosis",
    "discharge_medications",
    "discharge_instructions",
    "discharge_summary",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True,
                        help="path to demo_cohort.json (patients[].hadm_id)")
    parser.add_argument("--out", default="/tmp/ground_truth.json")
    args = parser.parse_args()

    patients = json.load(open(args.cohort))["patients"]
    rows: list[dict] = []
    counts = {s: {"present": 0, "absent": 0} for s in SUMMARY_SECTIONS}
    missing_notes = 0

    for p in patients:
        hadm_id = int(p["hadm_id"])
        text = _fetch_note(hadm_id)
        sections: set[str] = set()
        if text:
            sections = {s.name for s in parse_note(text).sections}
        else:
            missing_notes += 1
        rows.append({
            "hadm_id": hadm_id,
            "display_name": p.get("display_name", ""),
            "note_len": len(text) if text else 0,
            "sections": sorted(sections),
        })
        for s in SUMMARY_SECTIONS:
            counts[s]["present" if s in sections else "absent"] += 1

    with open(args.out, "w") as handle:
        json.dump(rows, handle, indent=2)

    n = len(rows)
    print(f"wrote {n} rows -> {args.out} (missing_notes={missing_notes})")
    print("section presence (ground truth):")
    for s, c in counts.items():
        print(f"  {s:26s} present={c['present']:3d} absent={c['absent']:3d} "
              f"present_rate={c['present'] / n:.0%}")
    meds_ok = sum(
        1 for r in rows
        if {"discharge_medications", "discharge_instructions"} & set(r["sections"])
    )
    print(f"meds-bearing notes (meds OR instructions present): {meds_ok}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
