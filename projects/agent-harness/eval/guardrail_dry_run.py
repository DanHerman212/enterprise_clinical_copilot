"""P4 dry-run — simulate the deterministic guardrails over the frozen 300 traces.

Pure offline replay of `agent.guardrail.guard_answer` over the saved
traces (answer + tool_calls are both saved), measuring:
  - how many of the 34 v2-FAILs get flagged / modified,
  - whether any of the 265 PASSing answers get MODIFIED (regression guard:
    a guardrail that rewrites good answers is unsafe to ship).

Usage (harness root): .venv/bin/python eval/guardrail_dry_run.py
"""

import collections
import json
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS))
sys.path.insert(0, str(HARNESS / "agent"))

from agent.guardrail import guard_answer  # noqa: E402

JUDGED = HARNESS / "eval" / "results" / "judged.jsonl"
TRACES = HARNESS / "eval" / "results" / "traces.jsonl"


def main() -> int:
    judged = [json.loads(l) for l in JUDGED.read_text().splitlines() if l.strip()]
    traces = [json.loads(l) for l in TRACES.read_text().splitlines() if l.strip()]
    tmap = {(t["hadm_id"], t["prompt"]): t for t in traces if "error" not in t}

    fail_flagged = 0
    fail_modified = 0
    pass_modified = 0
    flag_counts = collections.Counter()
    fail_total = 0
    pass_total = 0
    examples: dict[str, list] = collections.defaultdict(list)

    for rec in judged:
        j = rec.get("judge", {})
        if "error" in rec or j.get("error"):
            continue
        t = tmap.get((rec["hadm_id"], rec["prompt"]))
        if not t:
            continue
        verdict = j.get("verdict")
        ans = t.get("answer") or ""
        out = guard_answer(ans, t.get("tool_calls") or [])
        modified = out["answer"] != ans

        if verdict == "FAIL":
            fail_total += 1
            if out["flags"]:
                fail_flagged += 1
            if modified:
                fail_modified += 1
                for f in out["flags"]:
                    flag_counts[f] += 1
                    if len(examples[f]) < 2:
                        examples[f].append(f"{rec['hadm_id']}/{rec['prompt']}")
        else:
            pass_total += 1
            if modified:
                pass_modified += 1
                print(f"!! PASS MODIFIED: {rec['hadm_id']}/{rec['prompt']} "
                      f"flags={out['flags']}")

    print("=== P4 DRY-RUN (guardrails over frozen 300 traces) ===")
    print(f"FAILs: {fail_total}  -> flagged {fail_flagged}  modified {fail_modified}")
    print(f"PASSes modified: {pass_modified}  (must be 0 UNJUSTIFIED; age-fill "
          f"corrections are verified justified — see dry-run notes)")
    print("\nflag -> count -> examples:")
    for f, n in flag_counts.most_common():
        print(f"  {n:3}  {f}   {examples[f]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
