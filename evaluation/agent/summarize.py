"""Snapshot summary of judged.jsonl (works on partial or full results).

Reads judged.jsonl and prints overall + per-prompt verdict pass rates, per-
dimension pass rates and mean scores, agent-error count, and the most common
judge flags. Safe to run on a partial judged.jsonl while the judge is still
running.

Usage (harness root): .venv/bin/python eval/summarize.py
"""

import collections
import json
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
JUDGED = HARNESS / "eval" / "results" / "judged.jsonl"

DIMS = ["faithfulness", "groundedness", "citation", "clinical", "safety"]


def main() -> int:
    rows = [json.loads(l) for l in JUDGED.read_text().splitlines() if l.strip()]
    if not rows:
        print("no judged rows yet")
        return 0

    verdict = collections.Counter()
    by_prompt = collections.Counter()
    agent_errors = 0
    dims = {d: {"pass": 0, "total": 0, "score_sum": 0} for d in DIMS}
    flags = collections.Counter()

    for r in rows:
        j = r.get("judge", {})
        if "error" in r or j.get("error"):
            agent_errors += 1
            continue
        v = j.get("verdict")
        verdict[v] += 1
        by_prompt[(r.get("prompt"), v)] += 1
        for d in DIMS:
            s = (j.get("dimensions") or {}).get(d)
            if isinstance(s, int):
                dims[d]["total"] += 1
                dims[d]["score_sum"] += s
                if s >= 2:
                    dims[d]["pass"] += 1
        for f in (j.get("flags") or []):
            flags[f] += 1

    scored = verdict["PASS"] + verdict["FAIL"]
    print(f"rows={len(rows)}  scored={scored}  agent_errors={agent_errors}")
    print(f"verdict: PASS={verdict['PASS']}  FAIL={verdict['FAIL']}  "
          f"pass_rate={verdict['PASS']/scored:.1%}" if scored else "")
    print("\nby prompt:")
    for p in ("risk", "meds", "summarize"):
        print(f"  {p:10} PASS={by_prompt[(p,'PASS')]}  FAIL={by_prompt[(p,'FAIL')]}")
    print("\ndimensions (pass rate | mean score /3):")
    for d in DIMS:
        dd = dims[d]
        mean = dd["score_sum"] / dd["total"] if dd["total"] else 0
        print(f"  {d:13} pass={dd['pass']:3}/{dd['total']:3} "
              f"({dd['pass']/dd['total']:.0%} | mean {mean:.2f})"
              if dd["total"] else f"  {d:13} n/a")
    print("\ntop flags:")
    for f, n in flags.most_common(8):
        print(f"  {n:3}  {f[:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
