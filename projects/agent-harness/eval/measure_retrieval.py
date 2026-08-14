"""True retrieval-quality measurement over the 300 traces.

A redacted template (header fields + '___') has no sentence-length prose. A
passage is 'usable' only if it contains at least one real sentence (>8 words in
a single line). This separates template/redacted passages from real clinical
text — the distinction that matters for whether meds/summarize can ever be
grounded.

Usage (harness root): .venv/bin/python eval/measure_retrieval.py
"""

import collections
import json
import re
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
TRACES = HARNESS / "eval" / "results" / "traces.jsonl"

PLACEHOLDER = re.compile(r"_{3,}|<REDACTED>|\[\*\*.*?\*\*\]")


def _usable(text: str) -> bool:
    """True if the passage has at least one real prose sentence."""
    cleaned = PLACEHOLDER.sub("", text)
    for line in cleaned.splitlines():
        words = [w for w in re.split(r"\s+", line) if re.search(r"[A-Za-z]", w)]
        if len(words) >= 8:
            return True
    return False


def main() -> int:
    rows = [json.loads(l) for l in TRACES.read_text().splitlines() if l.strip()]
    per_prompt: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    passage_total = collections.Counter()
    all_thin = 0
    any_thin = 0

    for r in rows:
        if "error" in r:
            continue
        texts = []
        for tc in r.get("tool_calls") or []:
            if tc.get("name") not in ("rag_search", "rag_search_sections"):
                continue
            for p in (tc.get("response") or {}).get("passages") or []:
                texts.append(p.get("text") or "")
        if not texts:
            per_prompt[r["prompt"]]["empty"] += 1
            continue
        usable = [t for t in texts if _usable(t)]
        if not usable:
            per_prompt[r["prompt"]]["all_template_only"] += 1
            all_thin += 1
        elif len(usable) < len(texts):
            per_prompt[r["prompt"]]["some_usable"] += 1
            any_thin += 1
        else:
            per_prompt[r["prompt"]]["all_usable"] += 1
        for t in texts:
            passage_total["usable" if _usable(t) else "template_only"] += 1

    print("Per prompt (of 299 scored rows):")
    for p in ("risk", "meds", "summarize"):
        print(f"  {p:10}", dict(per_prompt[p]))
    print("\nAcross all retrieved passages:")
    print(" ", dict(passage_total))
    print(f"\nRows with zero usable passages: {all_thin}")
    print(f"Rows with mixed (some usable):  {any_thin}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
