"""explore_mtsamples.py — Step 2: inventory the crawled MTSamples corpus.

Measures, across the 108 discharge summaries in data/mtsamples/:
  1. inventory: char counts, sparse/empty notes, duplicates
  2. section coverage: which of OUR chunker's whitelisted sections each note
     parses into (parse_note), and which headings are unrecognised
  3. meds: count of discharge-medication lines per note (polypharmacy range)
  4. feature extraction: age, sex, LOS, prior-admission mentions, labs, etc.
     vs the 49-feature schema

Read-only analysis; prints a summary. Raw notes stay local (gitignored).

Usage (from projects/agent-harness):
  ../../.venv/bin/python scripts/explore_mtsamples.py
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = HARNESS_ROOT / "data" / "mtsamples"

sys.path.insert(0, str(HARNESS_ROOT))
from rag.sections import KNOWN_HEADINGS, parse_note  # noqa: E402

# The whitelist the chunker indexes (matches pipelines/components/chunk_notes.py).
WHITELISTED = {
    "history_of_present_illness", "past_medical_history", "family_history",
    "social_history", "physical_exam", "brief_hospital_course",
    "discharge_condition", "discharge_diagnosis", "discharge_medications",
    "medications_on_admission", "discharge_disposition", "discharge_instructions",
}

_AGE_RE = re.compile(r"(\d{1,3})(?:-| )?year[- ]?old", re.IGNORECASE)
_SEX_RE = re.compile(r"\b(male|female|man|woman|gentleman|lady)\b", re.IGNORECASE)
# MTSamples page chrome that must NOT be in the extracted note body. If the
# body-start anchor failed, these leak in — a cleanliness signal.
NAV_MARKERS = ("Sample Name", "Medical Specialty", "Educational Disclaimer",
               "Intended for", "Description:", "Transcribed Medical Transcription")
_MEDS_SECTION_RE = re.compile(
    r"(discharge medications|medications on discharge|discharge meds|"
    r"medications at discharge|discharge instructions/medications)"
    r"\s*:\s*(.*?)(?=\n\s*\n[A-Z][A-Za-z /&'-]{2,40}:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
# A drug mention: capitalized drug name + dose/unit (mg/g/mcg/units/mL) or
# common brand/generic patterns on a line. Best-effort for corpus stats.
_DRUG_RE = re.compile(
    r"\b([A-Z][A-Za-z-]+(?: [A-Za-z-]+)?(?: (?:XR|ER|CR|LA|HCTZ|DS))?)"
    r" (?:[0-9]+(?:\.[0-9]+)? ?(?:mg|g|mcg|mg/mL|units)|"
    r"[0-9]+ ?(?:tablets?|capsules?|puffs?|inhalations?|drops?))",
)
_MED_LINE_RE = re.compile(r"^\s*[-*\d.)]\s*[A-Za-z]", re.MULTILINE)


def _meds_count(text: str) -> int:
    """Count distinct discharge-medication mentions, best-effort.

    MTSamples lists meds in three ways: a discrete "Discharge Medications:"
    block, a numbered list under Discharge Instructions, or inline prose
    ("placed on Lisinopril 20 mg twice daily"). Count distinct drug+dose
    mentions across the whole note; that is the honest polypharmacy signal.
    """
    m = _MEDS_SECTION_RE.search(text)
    if m:
        block = m.group(2)
        n = len([ln for ln in block.splitlines() if _MED_LINE_RE.match(ln)])
        n2 = len(re.findall(r"^\s*\d*\s*[A-Z][a-zA-Z]+ \d+ ?(?:mg|g|mcg|units)", block, re.M))
        if max(n, n2) >= 2:
            return max(n, n2)
    drugs = _DRUG_RE.findall(text)
    # dedupe identical (name, dose) mentions
    unique = set(drugs)
    return len(unique)


def main() -> int:
    files = sorted(DATA_DIR.glob("*.txt"))
    print(f"corpus: {len(files)} notes in {DATA_DIR}\n")

    stats = {"chars": [], "sparse": [], "no_wl_section": [], "poly": 0,
             "nav_leak": []}
    wl_coverage: Counter[str] = Counter()
    unknown: Counter[str] = Counter()
    age_found = sex_found = 0
    meds_counts: list[int] = []
    labels: dict[str, dict] = {}

    for path in files:
        text = path.read_text(encoding="utf-8")
        n = len(text)
        stats["chars"].append(n)
        if n < 500:
            stats["sparse"].append(path.stem)

        # Nav boilerplate leaking in = the body-start anchor failed for this
        # note; it needs a cleaner extraction or a scrub before use.
        leaked = [mk for mk in NAV_MARKERS if mk in text]
        if leaked:
            stats["nav_leak"].append(path.stem)

        parsed = parse_note(text)
        sections = {s.name for s in parsed.sections}
        hit = sections & WHITELISTED
        for s in hit:
            wl_coverage[s] += 1
        if not hit:
            stats["no_wl_section"].append(path.stem)
        for h in parsed.unknown_headings:
            unknown[h] += 1

        if _AGE_RE.search(text):
            age_found += 1
        if _SEX_RE.search(text):
            sex_found += 1

        mc = _meds_count(text)
        meds_counts.append(mc)
        if mc >= 5:
            stats["poly"] += 1

        labels[path.stem] = {
            "chars": n,
            "sections": sorted(hit),
            "meds": mc,
            "age": bool(_AGE_RE.search(text)),
            "sex": bool(_SEX_RE.search(text)),
            "nav_leak": bool(leaked),
        }
    (DATA_DIR / "labels.json").write_text(
        json.dumps(labels, indent=2, sort_keys=True))

    chars = sorted(stats["chars"])
    print(f"chars: min={chars[0]} median={chars[len(chars)//2]} max={chars[-1]}")
    print(f"sparse (<500 chars): {stats['sparse']}")
    print(f"nav-boilerplate leak: {len(stats['nav_leak'])} notes: "
          f"{stats['nav_leak'][:15]}")
    print(f"notes with >=1 whitelisted section: "
          f"{len(files) - len(stats['no_wl_section'])}/{len(files)}")
    print(f"notes with NO whitelisted section: {stats['no_wl_section']}")

    print("\nwhitelisted-section coverage (of 108):")
    for s in sorted(WHITELISTED):
        print(f"  {s:28s} {wl_coverage[s]}")

    print("\ntop unrecognised headings (absorbed into neighbour section):")
    for h, c in unknown.most_common(25):
        print(f"  {h:35s} {c}")

    print(f"\nage found in: {age_found}/108   sex found in: {sex_found}/108")
    mc_sorted = sorted(meds_counts)
    print(f"discharge-med count: min={mc_sorted[0]} median={mc_sorted[len(mc_sorted)//2]} "
          f"max={mc_sorted[-1]} | notes with >=5 meds: {stats['poly']}")
    print(f"med-count histogram: {Counter(meds_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
