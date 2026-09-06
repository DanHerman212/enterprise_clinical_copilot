"""Show a compact, reviewable view of one pilot case for hand-labeling.

Usage (harness root): .venv/bin/python eval/show_case.py 3
"""

import json
import random
import re
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
TRACES = HARNESS / "eval" / "results" / "traces.jsonl"
JUDGED = HARNESS / "eval" / "results" / "judged.jsonl"
SEED = 20260814
N_PER_CLASS = 6

# Section headers found in MIMIC discharge notes.
_SECTIONS = [
    "History of Present Illness", "Past Medical History", "Physical Exam",
    "Pertinent Results", "Brief Hospital Course", "Medications on Admission",
    "Discharge Medications", "Discharge Disposition", "Discharge Diagnosis",
    "Discharge Condition", "Discharge Instructions", "Family History",
    "Social History", "Chief Complaint",
]
_SECTION_RE = re.compile(r"(?m)^\s*(" + "|".join(_SECTIONS) + r")\s*:\s*$")

# Which sections matter per prompt type (compact view).
_RELEVANT = {
    "risk": ["Brief Hospital Course", "Discharge Diagnosis", "Discharge Condition",
             "History of Present Illness", "Discharge Instructions"],
    "meds": ["Discharge Medications", "Medications on Admission",
             "Discharge Instructions", "Brief Hospital Course"],
    "summarize": ["Brief Hospital Course", "Discharge Diagnosis", "Discharge Condition",
                  "Discharge Instructions", "History of Present Illness",
                  "Discharge Medications"],
}


def _passages(trace: dict) -> list[dict]:
    out = []
    for tc in trace.get("tool_calls") or []:
        if tc.get("name") not in ("rag_search", "rag_search_sections"):
            continue
        for p in (tc.get("response") or {}).get("passages") or []:
            out.append({"section": p.get("section"), "text": p.get("text") or ""})
    return out


def _split_sections(text: str) -> dict[str, str]:
    """Split a note into {Section: text} using known section headers."""
    matches = list(_SECTION_RE.finditer(text))
    result = {}
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        result[title] = text[start:end].strip()
    return result


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


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    picked = _pick_cases()
    if not (1 <= n <= len(picked)):
        print(f"case must be 1..{len(picked)}")
        return 1
    rec, j, t = picked[n - 1]
    print(f"=== Case {n} — hadm {rec['hadm_id']} · {rec['prompt']} ===")
    print(f"\nQUESTION:\n{t['question']}\n")

    # For risk questions, show the predict tool output so faithfulness can be judged.
    for tc in t.get("tool_calls") or []:
        if tc.get("name") == "predict_readmission":
            r = tc.get("response") or {}
            factors = r.get("top_factors")
            if isinstance(factors, list):
                factors = [f.get("feature") for f in factors[:5]]
            print("MODEL OUTPUT (predict_readmission):")
            print(f"  probability={r.get('probability')}  threshold={r.get('threshold')}  "
                  f"decision={r.get('decision')}  top_factors={factors}")

    print("\nANSWER:")
    print(t["answer"])
    ps = _passages(t)
    if not ps:
        print("\nRETRIEVED: (none)")
    else:
        print("\nRETRIEVED (relevant sections only):")
        want = _RELEVANT.get(t["prompt"], _SECTIONS)
        for p in ps:
            parts = _split_sections(p["text"])
            shown = {k: v for k, v in parts.items() if k in want}
            if not shown:
                shown = {"<no matching section>": p["text"][:300]}
            for k, v in shown.items():
                v = re.sub(r"_{3,}", "___", v)
                print(f"\n--- {k} ---")
                print(v[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
