"""audit_features.py — which of the model's 49 features are in-text vs absent.

For each MTSamples note, scan for lightweight *presence signals* per feature
family (age, gender, race, insurance, prior admissions, LOS, ED visits,
procedures, medications, labs, discharge location, admission type, oncology).
This is an extractability audit, NOT extraction: it answers "can this note
anchor feature X?" so Step 7 can plan the fill strategy and Step 8 can pick
prototype patients whose story is coherent with the filled vector.

Output: stdout table + data/mtsamples/feature_audit.json
  { sample_id: { family: true/false, ... } }

Heuristics are deliberately loose (a single signal flags the family) and only
count presence, never absence. Downstream, a feature family with 0% is a pure
fill; 100% is parse-anchored; in between is hybrid.
"""

import json
import re
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = HARNESS_ROOT / "data" / "mtsamples"

SIGNALS: dict[str, list[re.Pattern[str]]] = {
    "age": [
        re.compile(r"\b\d{2,3}\s*(?:-year|-yr|-y\.o\.)\s*(?:old|female|male|woman|man|infant|child|patient|girl|boy)?", re.I),
        re.compile(r"\bage(?: of)?\s+\d{1,3}\b", re.I),
        re.compile(r"\b\d{1,3}\s*(?:yrs?|years?)\s+old\b", re.I),
    ],
    "gender": [
        re.compile(r"\b(?:female|male)\b", re.I),
        re.compile(r"\b(?:woman|man)\b", re.I),
        re.compile(r"\b(?:she|he)\b", re.I),  # weak, but a narrative pronoun is a strong anchor
    ],
    "race": [
        re.compile(r"\b(caucasian|white|african[- ]american|black|hispanic|latino?|asian|american indian|native american|pacific islander)\b", re.I),
    ],
    "insurance": [
        re.compile(r"\b(medicare|medicaid|insurance|blue cross|health plan|managed care)\b", re.I),
    ],
    "prior_admission": [
        re.compile(r"\b(prior|previous|earlier)\s+(?:hospital|inpatient|admission|admissions|icu|stay|stays)\b", re.I),
        re.compile(r"\b(?:re[- ]?admission|re[- ]?hospitalization)\b", re.I),
        re.compile(r"\bwas (?:previously|prior) (?:admitted|hospitalized)\b", re.I),
        re.compile(r"\bhistory of multiple admissions\b", re.I),
    ],
    "los": [
        re.compile(r"\blength of stay\b", re.I),
        re.compile(r"\b(?:hospitalized|admitted|hospital) for\s+\d+\s+days?\b", re.I),
        re.compile(r"\bhospital day[s]? \d+\b", re.I),
        re.compile(r"\b\d+[- ]day (?:hospital|admission|stay)\b", re.I),
    ],
    "ed_visits": [
        re.compile(r"\b(?:emergency room|emergency department|e\.?r\.?)\b", re.I),
        re.compile(r"\b(?:ed|er)\s+visit", re.I),
        re.compile(r"\bseen in (?:the )?(?:emergency|er)\b", re.I),
    ],
    "procedure": [
        re.compile(r"\b(?:underwent|underwent a|underwent an)\b", re.I),
        re.compile(r"\b(surgery|operation|operative|procedure|procedures)\b", re.I),
        re.compile(r"\b(?:exploratory|resection|bypass|angioplasty|stent|catheterization|endoscopy|arthroplasty|discectomy|hysterectomy|cholecystectomy)\b", re.I),
    ],
    "medication": [
        re.compile(r"\b(?:medications?|meds?|prescriptions?|drugs?)\b", re.I),
        re.compile(r"\b\d+\s*(?:mg|mcg|g|units?)\b", re.I),
        re.compile(r"\b(?:take|taking|continue|continue[sd]?|discontinue|prescribed)\b", re.I),
    ],
    "labs": [
        re.compile(r"\b(?:hemoglobin|haemoglobin|hgb|hematocrit|hct|white (?:blood )?count|wbc|sodium|creatinine|bun|potassium|platelet|bmp|cbc|a1c|glucose|bilirubin)\b", re.I),
        re.compile(r"\b(?:lab|labs|laboratory)\b", re.I),
    ],
    "discharge_location": [
        re.compile(r"\b(?:discharged?|sent|transferred|went)\s+to\s+(?:home|a? ?skilled nursing|snf|rehab|nursing home|hospice|assisted living|long[- ]term care)\b", re.I),
        re.compile(r"\b(?:home with|home health|home care)\b", re.I),
        re.compile(r"\b(?:disposition|discharge disposition)\b", re.I),
    ],
    "admission_type": [
        re.compile(r"\b(?:admitted?|came|presented|brought)\s+(?:through|to|from|via)\s+(?:the\s+)?(?:emergency|er|ed|clinic|direct)\b", re.I),
        re.compile(r"\b(?:transfer(?:red)? from|direct admission|ambulatory)\b", re.I),
        re.compile(r"\b(?:elective|urgent|emergency)\s+(?:admission|surgery|procedure)\b", re.I),
    ],
    "oncology": [
        re.compile(r"\b(?:oncology|oncological|chemotherapy|chemoradiation|malignancy|malignant|metastatic|metastasis|tumor|neoplasm|cancer)\b", re.I),
    ],
}


def audit(text: str) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for family, patterns in SIGNALS.items():
        out[family] = any(p.search(text) for p in patterns)
    return out


def main() -> None:
    results: dict[str, dict[str, bool]] = {}
    for path in sorted(DATA_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        results[path.stem] = audit(text)

    DATA_DIR.joinpath("feature_audit.json").write_text(
        json.dumps(results, indent=1), encoding="utf-8"
    )

    families = list(SIGNALS)
    header = "note      " + "".join(f"{f:>12}" for f in families) + "   in-text"
    print(header)
    totals = {f: 0 for f in families}
    for sid in sorted(results, key=int):
        r = results[sid]
        count = sum(r.values())
        row = f"{sid:>9} " + "".join("   X" if r[f] else "    ." for f in families) + f"   {count:2d}"
        print(row)
        for f in families:
            totals[f] += int(r[f])

    n = len(results)
    print("\nExtractability (of %d notes):" % n)
    for f in families:
        print(f"  {f:>18}  {totals[f]:3d}  ({totals[f] * 100 // n}%)")


if __name__ == "__main__":
    main()
