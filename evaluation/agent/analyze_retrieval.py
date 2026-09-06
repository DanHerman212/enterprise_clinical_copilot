"""Quantify retrieval quality across collected traces for the Phase-5 review.

For each trace, classify the RAG retrieval a trace received:
  - empty:       rag called, 0 passages returned
  - template:    passages exist but are mostly redaction placeholders ("___")
                 -> no substantive clinical text for the agent to cite
  - substantive: at least one passage with real clinical text
Also report the fraction of answers that are long + detailed (hallucination
risk) on template-only retrieval.

Usage (harness root): .venv/bin/python eval/analyze_retrieval.py
"""

import json
import re
from collections import Counter
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
TRACES = HARNESS / "eval" / "results" / "traces.jsonl"

PLACEHOLDER = re.compile(r"_{3,}|<REDACTED>|\[\*\*.*?\*\*\]")


def _passage_texts(trace: dict) -> list[dict]:
    texts = []
    for tc in trace.get("tool_calls") or []:
        if tc.get("name") not in ("rag_search", "rag_search_sections"):
            continue
        for p in (tc.get("response") or {}).get("passages") or []:
            t = p.get("text") or ""
            texts.append({"section": p.get("section"), "text": t})
    return texts


def _is_template(text: str) -> bool:
    stripped = PLACEHOLDER.sub("", text)
    body = re.sub(r"\s+", " ", stripped).strip()
    # A header + empty fields has little real prose; <90 chars of non-redacted
    # content is treated as "no substantive clinical text".
    return len(body) < 90


def _classify(texts: list[dict]) -> str:
    if not texts:
        return "empty"
    if all(_is_template(p["text"]) for p in texts):
        return "template"
    return "substantive"


def main() -> int:
    rows = [json.loads(l) for l in TRACES.read_text().splitlines() if l.strip()]
    by_prompt: dict[str, Counter] = {}
    halluc_risk = Counter()
    examples: dict[str, list[str]] = {k: [] for k in ("empty", "template", "substantive")}

    for r in rows:
        if "error" in r:
            continue
        texts = _passage_texts(r)
        cls = _classify(texts)
        by_prompt.setdefault(r["prompt"], Counter())[cls] += 1

        total_chars = sum(len(p["text"]) for p in texts)
        answer = r.get("answer") or ""
        if cls == "template" and len(answer) > 400 and total_chars < 600:
            halluc_risk[cls] += 1
        if len(examples[cls]) < 2:
            examples[cls].append(f"{r['hadm_id']}/{r['prompt']}")

    print("Retrieval quality by prompt (excludes agent-error rows):")
    for p in ("risk", "meds", "summarize"):
        print(f"  {p:10}", dict(by_prompt.get(p, Counter())))
    print("\nOverall:", dict(sum(by_prompt.values(), Counter())))
    print("\nLong-answer-on-template-only (hallucination-risk) rows:", sum(halluc_risk.values()))
    print("Example hadm/prompt per class:")
    for k, v in examples.items():
        print(f"  {k:12} {v}")

    # Show one template-only answer's retrieval byte size to make the finding concrete.
    for r in rows:
        texts = _passage_texts(r)
        if _classify(texts) == "template" and (r.get("answer") or "") and len(r["answer"]) > 400:
            chars = sum(len(p["text"]) for p in texts)
            print(f"\n[example] {r['hadm_id']}/{r['prompt']}: answer={len(r['answer'])} chars "
                  f"vs retrieved={chars} chars over {len(texts)} passages")
            print("  answer head:", (r["answer"] or "").strip()[:160].replace("\n", " "))
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
