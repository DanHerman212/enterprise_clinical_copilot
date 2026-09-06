"""P1b/P2 — save the agreed human labels and compute judge agreement.

Reconstructs the pilot case selection (same seed/order as pilot.md), writes
human_labels.jsonl (human verdict per case), and reports judge-vs-human
agreement: Cohen's kappa, confusion matrix, judge precision/recall, and the
safety-critical false-pass / false-fail rates.

Human verdicts are the ones agreed in the labeling session (12/12 PASS).

Usage (harness root): .venv/bin/python eval/save_human_labels.py
"""

import json
import random
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
TRACES = HARNESS / "eval" / "results" / "traces.jsonl"
JUDGED = HARNESS / "eval" / "results" / "judged.jsonl"
OUT = HARNESS / "eval" / "results" / "human_labels" / "human_labels.jsonl"
SEED = 20260814
N_PER_CLASS = 6

HUMAN_VERDICT = "PASS"  # agreed for all 12 pilot cases


def _pick_cases() -> list[tuple]:
    judged = [json.loads(l) for l in JUDGED.read_text().splitlines() if l.strip()]
    traces = [json.loads(l) for l in TRACES.read_text().splitlines() if l.strip()]
    tmap = {(t["hadm_id"], t["prompt"]): t for t in traces if "error" not in t}
    by_verdict = {"PASS": [], "FAIL": []}
    for rec in judged:
        j = rec.get("judge", {})
        if "error" in rec or j.get("error"):
            continue
        t = tmap.get((rec["hadm_id"], rec["prompt"]))
        if t:
            by_verdict.setdefault(j.get("verdict"), []).append((rec, j, t))
    rng = random.Random(SEED)
    picked = []
    for verdict in ("PASS", "FAIL"):
        pool = by_verdict.get(verdict, [])
        rng.shuffle(pool)
        by_prompt = {}
        for x in pool:
            by_prompt.setdefault(x[2]["prompt"], []).append(x)
        order = ["risk", "meds", "summarize"]
        chosen = []
        while len(chosen) < N_PER_CLASS:
            progressed = False
            for p in order:
                if by_prompt.get(p) and len(chosen) < N_PER_CLASS:
                    chosen.append(by_prompt[p].pop(0))
                    progressed = True
            if not progressed:
                break
        picked.extend(chosen)
    rng.shuffle(picked)
    return picked


def _kappa(tp: int, tn: int, fp: int, fn: int) -> float:
    n = tp + tn + fp + fn
    if n == 0:
        return 1.0
    po = (tp + tn) / n
    row_pass = tp + fn
    row_fail = fp + tn
    col_pass = tp + fp
    col_fail = fn + tn
    pe = (row_pass / n) * (col_pass / n) + (row_fail / n) * (col_fail / n)
    return (po - pe) / (1 - pe) if pe != 1 else 0.0


def main() -> int:
    picked = _pick_cases()
    rows = []
    tp = fp = tn = fn = 0
    for rec, j, t in picked:
        judge_v = j.get("verdict")
        human_v = HUMAN_VERDICT
        # human is gold (rows), judge is prediction (cols)
        if human_v == "PASS":
            if judge_v == "PASS":
                tp += 1
            else:
                fn += 1
        else:
            if judge_v == "PASS":
                fp += 1
            else:
                tn += 1
        rows.append({
            "hadm_id": rec["hadm_id"],
            "prompt": rec["prompt"],
            "human_verdict": human_v,
            "judge_verdict": judge_v,
            "judge_dims": j.get("dimensions"),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    print(f"Wrote {len(rows)} human labels -> {OUT}\n")
    print("=== P2 — JUDGE vs HUMAN AGREEMENT (human = gold) ===")
    print(f"confusion: human PASS/judge PASS={tp}  human PASS/judge FAIL={fn}  "
          f"human FAIL/judge PASS={fp}  human FAIL/judge FAIL={tn}")
    print(f"observed agreement: {(tp + tn) / (tp + tn + fp + fn):.2%}")
    print(f"Cohen's kappa: {_kappa(tp, tn, fp, fn):.2f}")
    print(f"judge precision (of its PASSes, correct): {tp / (tp + fp) if tp + fp else 'n/a'}")
    print(f"judge recall of true PASSes: {tp / (tp + fn) if tp + fn else 'n/a'}")
    print(f"judge FALSE-PASS rate: {fp / (fp + tn) if fp + tn else 0:.0%}  (dangerous direction)")
    print(f"judge FALSE-FAIL rate: {fn / (tp + fn) if tp + fn else 0:.0%}  (blocks good work)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
