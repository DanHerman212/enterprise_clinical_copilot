"""P1a — Retrieval content audit (offline, uses saved traces).

Establishes ground truth about the RAG index CONTENT for the 100 sampled
patients: do the retrieved passages contain usable clinical prose, or are they
mostly PII-redaction artifacts? This decides whether the failure is a
data/index-layer problem (no usable evidence to retrieve) or an agent-behavior
problem.

Outputs:
  - summary stats to stdout
  - a full passage dump (stratified, incl. judged PASS/FAIL) to
    eval/results/retrieval_audit_dump.txt for manual eyeballing

Usage (harness root): .venv/bin/python eval/audit_retrieval.py
"""

import collections
import json
import re
import statistics
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
TRACES = HARNESS / "eval" / "results" / "traces.jsonl"
JUDGED = HARNESS / "eval" / "results" / "judged.jsonl"
DUMP = HARNESS / "eval" / "results" / "retrieval_audit_dump.txt"

# Header boilerplate fields seen at the start of passages. Redaction placeholders.
PLACEHOLDER = re.compile(r"_{3,}|<REDACTED>|\[\*\*.*?\*\*\]")
HEADER_FIELDS = re.compile(
    r"Name|Unit No|Admission Date|Discharge Date|Date of Birth|Sex|Service|"
    r"Allergies|No Known Allergie|MRN|DOB|Attending|Admission Diagnosis",
    re.IGNORECASE,
)
# Clinical-content signals that indicate a passage actually contains note prose.
CLINICAL_WORDS = re.compile(
    r"\b(medication|mg|mcg|diagnosis|hospital course|discharge|admitted|complaint|"
    r"history|exam|instructions|treatment|surgery|dose|BID|TID|Q8H|Q12H|IV|PO)\b",
    re.IGNORECASE,
)


def _passages(trace: dict) -> list[dict]:
    out = []
    for tc in trace.get("tool_calls") or []:
        if tc.get("name") not in ("rag_search", "rag_search_sections"):
            continue
        for p in (tc.get("response") or {}).get("passages") or []:
            out.append({"section": p.get("section"), "text": p.get("text") or ""})
    return out


def _prose(text: str) -> str:
    """Text minus redaction placeholders and header field labels."""
    cleaned = PLACEHOLDER.sub(" ", text)
    cleaned = HEADER_FIELDS.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _has_long_line(text: str) -> bool:
    for line in text.splitlines():
        words = [w for w in re.split(r"\s+", line) if re.search(r"[A-Za-z]", w)]
        if len(words) >= 8:
            return True
    return False


def main() -> int:
    traces = [json.loads(l) for l in TRACES.read_text().splitlines() if l.strip()]
    judged = {}
    if JUDGED.exists():
        for l in JUDGED.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                judged[(r.get("hadm_id"), r.get("prompt"))] = r.get("judge", {})

    lengths, prose_lens, clin = [], [], []
    per_section = collections.Counter()
    usable = 0
    total_passages = 0

    dump_lines = []
    stratified = []

    # Stratify: for each prompt, pick one PASS + one FAIL case (if available).
    for p in ("risk", "meds", "summarize"):
        for verdict in ("PASS", "FAIL"):
            for r in traces:
                if r.get("prompt") != p:
                    continue
                if "error" in r:
                    continue
                j = judged.get((r["hadm_id"], r["prompt"]), {})
                if j.get("verdict") != verdict:
                    continue
                ps = _passages(r)
                if ps:
                    stratified.append((r, ps, verdict))
                    break

    for r in traces:
        if "error" in r:
            continue
        for p in _passages(r):
            total_passages += 1
            per_section[p["section"]] += 1
            lengths.append(len(p["text"]))
            prose = _prose(p["text"])
            prose_lens.append(len(prose))
            if CLINICAL_WORDS.search(prose):
                clin.append(True)
            else:
                clin.append(False)
            if _has_long_line(p["text"]):
                usable += 1

    n = len(lengths)
    print("=== RETRIEVAL CONTENT AUDIT (P1a) — offline over 300 saved traces ===")
    print(f"total passages retrieved: {total_passages}")
    print(f"per section: {dict(per_section)}")
    print(f"passage length   chars: median {statistics.median(lengths):.0f}  "
          f"mean {statistics.mean(lengths):.0f}  max {max(lengths)}")
    print(f"prose (no redact/header) chars: median {statistics.median(prose_lens):.0f}  "
          f"mean {statistics.mean(prose_lens):.0f}")
    print(f"passages with a >=8-word line (sentence-like): {usable}/{n}")
    print(f"passages containing clinical tokens: {sum(clin)}/{n}")

    # Write the stratified full dump for manual eyeballing.
    with DUMP.open("w") as fh:
        for r, ps, verdict in stratified:
            fh.write("=" * 78 + "\n")
            fh.write(f"{r['hadm_id']}/{r['prompt']}  (judge: {verdict})\n")
            fh.write(f"QUESTION: {r['question']}\n")
            fh.write(f"ANSWER (first 300): {(r.get('answer') or '')[:300]}\n")
            for p in ps:
                fh.write(f"\n--- [{p['section']}] ({len(p['text'])} chars) ---\n")
                fh.write(p["text"] + "\n")
    print(f"\nStratified full dump written: {DUMP} ({len(stratified)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
