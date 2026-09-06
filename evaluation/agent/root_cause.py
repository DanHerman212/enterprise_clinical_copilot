"""P3 — enumerate + group the remaining FAILs under the fixed (v2) judge.

Groups the 34 remaining failures by flag theme, and prints each with its
dimension scores and the offending answer excerpt, so each class can be
attributed to a layer (model/prompt vs tool/guardrail vs judge).

Usage (harness root): .venv/bin/python eval/root_cause.py
"""

import collections
import json
import re
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
JUDGED = HARNESS / "eval" / "results" / "judged.jsonl"
TRACES = HARNESS / "eval" / "results" / "traces.jsonl"

THEMES = [
    ("age", r"age"),
    ("frequency", r"frequen|daily|bid|tid|twice"),
    ("timeframe", r"timeframe|follow.?up|month|week"),
    ("redaction", r"redact|___|omitted field"),
    ("citation", r"citat"),
    ("contradiction", r"contradict"),
    ("medication", r"medication|dose|dosage|drug|Lantus|Cipro|Naproxen|Megestrol|Simvastatin"),
    ("invented/ungrounded", r"invent|ungrounded|not present|not in the source|fabricat|hallucin"),
]


def _theme(flags: list[str]) -> str:
    blob = " ".join(flags).lower()
    for name, pat in THEMES:
        if re.search(pat, blob):
            return name
    return "other"


def main() -> int:
    judged = [json.loads(l) for l in JUDGED.read_text().splitlines() if l.strip()]
    traces = [json.loads(l) for l in TRACES.read_text().splitlines() if l.strip()]
    tmap = {(t["hadm_id"], t["prompt"]): t for t in traces if "error" not in t}

    fails = []
    for rec in judged:
        j = rec.get("judge", {})
        if "error" in rec or j.get("error"):
            continue
        if j.get("verdict") != "FAIL":
            continue
        t = tmap.get((rec["hadm_id"], rec["prompt"]), {})
        fails.append((rec, j, t))

    groups = collections.defaultdict(list)
    for rec, j, t in fails:
        groups[_theme(j.get("flags") or [])].append((rec, j, t))

    print(f"TOTAL FAILs: {len(fails)}\n")
    for theme in sorted(groups, key=lambda k: -len(groups[k])):
        print(f"### {theme}: {len(groups[theme])}")
        for rec, j, t in groups[theme][:6]:
            dims = j.get("dimensions")
            low = [d for d, v in (dims or {}).items() if isinstance(v, int) and v < 2]
            print(f"  - {rec['hadm_id']}/{rec['prompt']}  low_dims={low}")
            for f in (j.get("flags") or [])[:2]:
                print(f"      flag: {f[:150]}")
            ans = (t.get("answer") or "").replace("\n", " ")
            # print the sentence containing the flag-ish term, else first 200 chars
            print(f"      answer: {ans[:220]}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
