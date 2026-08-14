"""Validate a judge version against the ACTUAL human-labeled pilot cases.

The pilot (pilot.md) selected 12 cases from the v1 judge pool and we hand-
labeled all 12 PASS. This validates any judge file against those SAME 12 cases
(not a re-selected sample). Usage: validate the v2 judge against the frozen
human labels.

Usage (harness root): .venv/bin/python eval/validate_judge.py
"""

import json
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
JUDGED = HARNESS / "eval" / "results" / "judged.jsonl"
HUMAN_OUT = HARNESS / "eval" / "results" / "human_labels" / "human_labels.jsonl"

# The exact 12 pilot cases from pilot.md, as human-labeled (all PASS).
PILOT = [
    (27016685, "risk"), (29914278, "summarize"), (23576068, "risk"),
    (27645629, "meds"), (29847993, "meds"), (20132486, "risk"),
    (29916192, "risk"), (25242454, "meds"), (27645629, "summarize"),
    (23082454, "meds"), (23571330, "meds"), (21545230, "risk"),
]
HUMAN_VERDICT = "PASS"


def _kappa(tp, tn, fp, fn):
    n = tp + tn + fp + fn
    if n == 0:
        return 1.0
    po = (tp + tn) / n
    row_pass, row_fail = tp + fn, fp + tn
    col_pass, col_fail = tp + fp, fn + tn
    pe = (row_pass / n) * (col_pass / n) + (row_fail / n) * (col_fail / n)
    return (po - pe) / (1 - pe) if pe != 1 else 0.0


def main() -> int:
    judged = {}
    for l in JUDGED.read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            judged[(r.get("hadm_id"), r.get("prompt"))] = r.get("judge", {})

    rows = []
    tp = fp = tn = fn = 0
    missing = []
    print(f"{'case':<5}{'hadm/prompt':<22}{'human':<7}{'judge':<7}dims")
    for i, (hadm, prompt) in enumerate(PILOT, 1):
        j = judged.get((hadm, prompt), {})
        judge_v = j.get("verdict")
        if judge_v is None:
            missing.append((hadm, prompt))
            print(f"{i:<5}{hadm}/{prompt:<14}{HUMAN_VERDICT:<7}{'MISSING':<7}{j}")
            continue
        if judge_v == "PASS":
            tp += 1
        else:
            fn += 1
        print(f"{i:<5}{hadm}/{prompt:<14}{HUMAN_VERDICT:<7}{judge_v:<7}{j.get('dimensions')}")
        rows.append({"hadm_id": hadm, "prompt": prompt,
                     "human_verdict": HUMAN_VERDICT, "judge_verdict": judge_v,
                     "judge_dims": j.get("dimensions")})

    print("\n=== JUDGE vs HUMAN on the FROZEN 12 pilot cases ===")
    print(f"human PASS / judge PASS = {tp}   human PASS / judge FAIL = {fn}")
    print(f"observed agreement: {tp / (tp + fn):.1%}")
    print(f"Cohen's kappa: {_kappa(tp, tn, fp, fn):.2f}")
    if missing:
        print("MISSING judge rows for:", missing)

    HUMAN_OUT.parent.mkdir(parents=True, exist_ok=True)
    HUMAN_OUT.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote frozen human labels -> {HUMAN_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
