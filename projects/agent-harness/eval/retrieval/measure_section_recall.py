"""measure_section_recall — retrieval-level recall over the demo cohort.

For each patient, run the live `rag_search_sections` path (the deterministic
summary retrieval) and compare the sections it returns against the parsed
ground truth. Reports recall per section:

  recall(section) = |patients whose note HAS section AND retriever returned it|
                    -----------------------------------------------------------
                    |patients whose note HAS section|

Patients whose note lacks a section are excluded from that section's recall
(honest "not available" is not a miss). Also reports spurious returns (a
section returned for a patient whose note lacks it) and per-patient section
drops, so a single patient can't hide.

Requires the RAG index endpoint to be deployed (see scripts/launch_endpoints.sh).

Usage (from projects/agent-harness):
    ../../.venv/bin/python eval/retrieval/measure_section_recall.py \
        --ground-truth /tmp/ground_truth.json [--runs 3]
"""

import argparse
import importlib
import json
import os
import sys
from collections import Counter

# Harness root (parent of eval/retrieval/) so mcp_server/rag are importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

# `mcp_server.tools.rag_search` is shadowed by the re-exported function of the
# same name (tools/__init__), so import the module explicitly.
rs = importlib.import_module("mcp_server.tools.rag_search")
_search_sections = rs._search_sections

SUMMARY_SECTIONS = (
    "brief_hospital_course",
    "discharge_diagnosis",
    "discharge_medications",
    "discharge_instructions",
    "discharge_summary",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--runs", type=int, default=1,
                        help="repeat per patient (retrieval is stochastic); "
                             "report best/typical")
    args = parser.parse_args()

    rows = json.load(open(args.ground_truth))
    present = Counter()       # patients whose note has the section
    hit = Counter()           # patients where the retriever returned it
    spurious = Counter()      # returned for a patient whose note lacks it
    per_patient: list[dict] = []

    for row in rows:
        hadm_id = row["hadm_id"]
        truth = set(row["sections"])
        got: Counter[str] = Counter()
        for _ in range(args.runs):
            res = _search_sections(hadm_id)
            for p in res.get("passages", []):
                got[p["section"]] += 1
        # A section counts as retrieved if any run returned it.
        got_set = {s for s, c in got.items() if c > 0}
        dropped = sorted(truth & set(SUMMARY_SECTIONS) - got_set)
        per_patient.append({
            "hadm_id": hadm_id,
            "name": row.get("display_name", ""),
            "truth": sorted(truth & set(SUMMARY_SECTIONS)),
            "retrieved": sorted(got_set),
            "dropped": dropped,
        })
        for s in SUMMARY_SECTIONS:
            if s in truth:
                present[s] += 1
                if s in got_set:
                    hit[s] += 1
            elif s in got_set:
                spurious[s] += 1

    n = len(rows)
    print(f"patients={n} runs={args.runs}")
    print(f"{'section':26s} {'has_section':>11s} {'recalled':>8s} {'recall':>7s}  spurious")
    for s in SUMMARY_SECTIONS:
        tot = present[s]
        rec = hit[s]
        recall = rec / tot if tot else float("nan")
        print(f"{s:26s} {tot:11d} {rec:8d} {recall:7.1%}  {spurious[s]:d}")

    # Per-patient drops that matter for the demo (meds/instructions).
    meds_notes = [r for r in per_patient
                  if {"discharge_medications", "discharge_instructions"}
                  & set(r["truth"])]
    dropped_meds = [r for r in meds_notes
                    if not ({"discharge_medications", "discharge_instructions"}
                            & set(r["retrieved"]))]
    print(f"\nmeds-bearing patients: {len(meds_notes)}; "
          f"retriever dropped BOTH meds+instructions for: {len(dropped_meds)}")
    for r in dropped_meds[:10]:
        print(f"  {r['hadm_id']} {r['name']}: truth={r['truth']} got={r['retrieved']}")

    any_dropped = [r for r in per_patient if r["dropped"]]
    print(f"\npatients with >=1 dropped summary section: {len(any_dropped)}/{n}")
    for r in any_dropped[:15]:
        print(f"  {r['hadm_id']} {r['name']}: dropped={r['dropped']}")

    with open("/tmp/section_recall_per_patient.json", "w") as handle:
        json.dump(per_patient, handle, indent=2)
    print("\nper-patient detail -> /tmp/section_recall_per_patient.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
