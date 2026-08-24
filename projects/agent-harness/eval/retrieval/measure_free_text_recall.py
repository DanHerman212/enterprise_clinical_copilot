"""measure_free_text_recall — index-path retrieval accuracy over the cohort.

The deterministic `_search_sections` path is used for summaries; free-text
`rag_search` still goes through the embedding index. This measures whether the
INDEX returns the RIGHT SECTION for the chip intents:

  - meds   query "medications"        expected discharge_medications (else
                                      discharge_instructions when the note has
                                      no meds section)
  - course query "brief hospital course" expected brief_hospital_course

Metrics per intent over patients whose note HAS the expected section:
  recall@1  — top passage's section == expected
  recall@5  — expected section appears anywhere in the top-k

Also reports the wrong-section rate (top passage is a different section — the
exact symptom reported), and prints a few representative failures.

Requires the RAG index endpoint deployed.

Usage (from projects/agent-harness):
    ../../.venv/bin/python eval/retrieval/measure_free_text_recall.py \
        --ground-truth /tmp/ground_truth.json
"""

import argparse
import asyncio
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
rag_search = rs.rag_search

INTENTS = (
    ("meds", "medications", "discharge_medications", "discharge_instructions"),
    ("course", "brief hospital course", "brief_hospital_course", None),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    rows = json.load(open(args.ground_truth))

    async def run() -> None:
        for intent, query, *expected in INTENTS:
            expected_primary = expected[0]
            expected_alt = expected[1]
            has = 0          # patients whose note has the expected section(s)
            hit1 = 0         # recall@1
            hitk = 0         # recall@k
            top_other: Counter[str] = Counter()  # top passage when != expected
            wrong_section: Counter[str] = Counter()
            examples: list[str] = []
            for row in rows:
                truth = set(row["sections"])
                if expected_primary not in truth:
                    if not (expected_alt and expected_alt in truth):
                        continue  # honest "not available" — not a recall case
                has += 1
                res = await rag_search(row["hadm_id"], query, top_k=args.top_k)
                if res.get("error"):
                    continue
                passages = res.get("passages") or []
                if not passages:
                    continue
                sections = [p["section"] for p in passages]
                expected_set = {expected_primary} | ({expected_alt} if expected_alt else set())
                if sections and sections[0] in expected_set:
                    hit1 += 1
                elif sections:
                    top_other[sections[0]] += 1
                if expected_set & set(sections):
                    hitk += 1
                else:
                    wrong_section[sections[0] if sections else "none"] += 1
                    if len(examples) < 5:
                        examples.append(
                            f"{row['hadm_id']} {row.get('display_name','')}: "
                            f"expected={expected_primary}{'|'+expected_alt if expected_alt else ''} "
                            f"got={sections}")

            denom = has or 1
            print(f"[{intent}] query={query!r} expected={expected_primary}"
                  f"{'|'+expected_alt if expected_alt else ''}")
            print(f"  eligible patients (note has expected section): {has}")
            print(f"  recall@1 = {hit1 / denom:.1%}   recall@{args.top_k} = {hitk / denom:.1%}")
            print(f"  top-passage other-section rate: "
                  f"{sum(top_other.values()) / denom:.1%}  breakdown={dict(top_other)}")
            print(f"  expected section ABSENT from top-k rate: "
                  f"{sum(wrong_section.values()) / denom:.1%}  breakdown={dict(wrong_section)}")
            for e in examples:
                print(f"    e.g. {e}")

    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
