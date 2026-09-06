"""Post-mortem: categorize the 300 judged traces into failure classes.

Reads judged.jsonl + traces.jsonl and buckets each FAIL by its judge flags and
dimension scores into failure classes, cross-tabs against retrieval quality
(substantive / thin / empty), and prints representative examples per class so
the fix-and-retest targets the real failure modes.

Usage (harness root): .venv/bin/python eval/post_mortem.py
"""

import collections
import json
import re
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
JUDGED = HARNESS / "eval" / "results" / "judged.jsonl"
TRACES = HARNESS / "eval" / "results" / "traces.jsonl"

PLACEHOLDER = re.compile(r"_{3,}|<REDACTED>|\[\*\*.*?\*\*\]")


def _passage_texts(trace: dict) -> list[dict]:
    texts = []
    for tc in trace.get("tool_calls") or []:
        if tc.get("name") not in ("rag_search", "rag_search_sections"):
            continue
        for p in (tc.get("response") or {}).get("passages") or []:
            texts.append({"section": p.get("section"), "text": p.get("text") or ""})
    return texts


def _thin(text: str) -> bool:
    body = re.sub(r"\s+", " ", PLACEHOLDER.sub("", text)).strip()
    return len(body) < 140


def _retrieval_class(texts: list[dict]) -> str:
    if not texts:
        return "empty"
    if all(_thin(p["text"]) for p in texts):
        return "thin"
    return "substantive"


def _categorize(rec: dict) -> str:
    j = rec.get("judge", {})
    flags = " ".join(j.get("flags") or []).lower()
    dims = j.get("dimensions", {})
    if "medication" in flags or "meds" in flags:
        return "invented_meds"
    if any(w in flags for w in ("fabricat", "hallucin", "invented", "made up", "not present in")):
        return "fabricated_content"
    if (dims.get("citation") or 0) < 2 and "citat" in flags:
        return "citation_error"
    if (dims.get("groundedness") or 0) < 2:
        return "ungrounded"
    return "other"


def main() -> int:
    judged = [json.loads(l) for l in JUDGED.read_text().splitlines() if l.strip()]
    traces = [json.loads(l) for l in TRACES.read_text().splitlines() if l.strip()]
    tmap = {(t["hadm_id"], t["prompt"]): t for t in traces if "error" not in t}

    classes = collections.Counter()
    cross = collections.Counter()  # (class, retrieval_class)
    examples: dict[str, list] = collections.defaultdict(list)

    for rec in judged:
        j = rec.get("judge", {})
        if "error" in rec or j.get("error"):
            continue
        if j.get("verdict") != "FAIL":
            continue
        cls = _categorize(rec)
        classes[cls] += 1
        t = tmap.get((rec["hadm_id"], rec["prompt"]), {})
        rcls = _retrieval_class(_passage_texts(t))
        cross[(cls, rcls)] += 1
        if len(examples[cls]) < 2:
            examples[cls].append((rec, t, rcls))

    print("=== FAILURE CLASSES (299 scored, 280 FAIL) ===")
    for cls, n in classes.most_common():
        print(f"\n{cls}: {n}")
        print(f"   by retrieval: " + ", ".join(
            f"{k[1]}={v}" for k, v in sorted(cross.items()) if k[0] == cls))

    for cls in classes:
        for rec, t, rcls in examples.get(cls, [])[:1]:
            j = rec["judge"]
            print(f"\n--- example: {cls} | {rec['hadm_id']}/{rec['prompt']} "
                  f"| retrieval={rcls} | dims={j['dimensions']}")
            print(f"   flags: {j.get('flags')}")
            ans = (t.get("answer") or "")
            print(f"   answer: {ans[:260].strip()!r}")
            texts = _passage_texts(t)
            if texts:
                print(f"   passage[{texts[0]['section']}]: {texts[0]['text'][:160]!r}")
            else:
                print("   passages: NONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
