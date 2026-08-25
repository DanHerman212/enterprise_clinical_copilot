"""coherence_scan.py — flag demo-cohort patients whose note content conflicts
with the patient's displayed demographics, or contains redaction artifacts.

Catches the bug class seen 2026-08-25: a neonatal note whose age was filled
from the MOTHER's age (e.g. "born to a 26-year-old … lady") produced a patient
displayed as 26F with a newborn's note (Alicia Kowalski). Also flags bare
placeholder tokens like "Dr. X" (follow-up redaction artifacts).

Checks per patient:
  REMOVE  — cohort-inclusion violation: the risk model was trained on an
         adult-only cohort (all patients >= 18). A note with neonate/
         obstetric-birth markers is a neonate admission (the age in it is a
         relative's), and any filled age < 18 also violates the criterion.
         Such patients must be REMOVED from the demo cohort, not re-aged.
  SEX  — the note's dominant gendered pronoun conflicts with the gender
         feature (only flagged when one side dominates ~2:1 or more).
  REDACTION — bare placeholder tokens (Dr./Mr./Mrs./Ms. X, runs of X, etc.).

Reads the hybrid-108 corpus (eval/results/hybrid_notes.json + hybrid_cohort.json).
Outputs one line per flagged patient + a summary. Exit 0 (a report, not a gate).

Usage (from projects/agent-harness):
  ../../.venv/bin/python scripts/coherence_scan.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
NOTES = HARNESS / "eval" / "results" / "hybrid_notes.json"
COHORT = HARNESS / "eval" / "results" / "hybrid_cohort.json"

# Note-subject markers indicating the note's patient is a neonate / infant /
# toddler. Deliberately SUBJECT-anchored: generic obstetric-history words
# (gravida, para, gestation, "delivered by ambulance") appear in adult notes
# and must not trigger a removal.
NEONATE_MARKERS = re.compile(
    r"\b\d+-?(?:day|week|month)s?-old\b|"           # 5-day-old, 14-month-old
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    r"\s*(?:plus-)?(?:day|week|month)s?-old\b|"     # one-plus-month-old
    r"\b\d+-?(?:year|yr)s?-old\s+(?:boy|girl|child)\b|"  # 3-year-old boy
    r"\b(?:newborn|neonate|neonatal)\b|"
    r"\bex-\d+-?(?:weeks?)?\s+preemie\b|"           # ex-34-week preemie
    r"\bpremature\s+(?:infant|baby|newborn|neonate)\b|"
    r"\b\d+-?(?:pound|lb)s?\s*(?:female|male)?\s*(?:infant|baby|newborn)\b|"
    r"\bborn to a\b|"
    r"\bbaby\s+(?:boy|girl)\b",
    re.I,
)
# A filled age this high cannot be the patient when a neonate marker is present.
MIN_NEONATE_AGE = 2

# Gendered pronouns (neonate female notes use she/her).
MALE_PRONOUNS = re.compile(r"\b(he|him|his|himself)\b", re.I)
FEMALE_PRONOUNS = re.compile(r"\b(she|her|hers|herself)\b", re.I)

# Redaction artifacts — bare placeholder tokens.
REDACTION = re.compile(
    r"\b(?:Dr|Mr|Mrs|Ms|Mx)\.?\s+X\b|"
    r"\b(?<![\w.])[X]{3,}\b",
    re.I,
)

# ---- Training-cohort exclusion criteria (2026-08-25, per user) ----
# The risk model was trained on a cohort that EXCLUDED:
#   (a) patients discharged against medical advice (AMA), and
#   (b) elective admissions where the patient had a planned return to
#       hospital by appointment (planned readmissions, not unplanned ones).

# AMA — the feature already encodes it (discharge_location_ama). Text used as
# a cross-check to catch any note that reads AMA even if the feature missed it.
AMA_TEXT = re.compile(
    r"\b(?:against medical advice|discharged ama|left ama|"
    r"signed out ama|left against medical advice)\b",
    re.I,
)

# Elective + planned-return. The note must contain BOTH an elective/scheduled
# admission signal AND an explicit planned hospital return / re-admission.
# Deliberately EXCLUDES the standard "return to the ER/clinic IF …" discharge
# precaution and non-hospital "return to …" phrases (diet, work), which are
# NOT planned readmissions.
ELECTIVE = re.compile(r"\belective\b|\bplanned\b|\bscheduled\b", re.I)
PLANNED_RETURN = re.compile(
    r"\bre-?admission\b|\bre-?admitted\b|\bre-?admit\b|"      # re-admit
    r"\bwill\s+be\s+(?:re-?)?admitted\b|"
    r"\breturn\s+to\s+the\s+hospital\b|"
    r"\breturn\s+for\s+(?:a|an)\s+(?:scheduled|planned)\b",
    re.I,
)


def _age_check(text: str, age: float) -> list[str]:
    """Cohort-inclusion check: neonate notes and under-18 ages are REMOVE
    candidates (the trained model cohort is adult-only, >= 18)."""
    flags = []
    if NEONATE_MARKERS.search(text):
        flags.append(
            "REMOVE neonate/obstetric-birth note — model cohort is adult-only "
            f"(>=18); age={age:g} is a relative's, not the patient's"
        )
    elif age < 18:
        flags.append(
            f"REMOVE age={age:g} < 18 — violates cohort inclusion criterion "
            "(model trained on >=18)"
        )
    return flags


def _sex_check(text: str, gender: float) -> list[str]:
    m = len(MALE_PRONOUNS.findall(text))
    f = len(FEMALE_PRONOUNS.findall(text))
    if not (m or f):
        return []
    dominant = "M" if m >= f else "F"
    # Only flag when one side dominates clearly (avoids mixed-pronoun noise).
    if (m >= 2 * f and dominant == "M" and gender != 1.0) or (
        f >= 2 * m and dominant == "F" and gender != 0.0
    ):
        return [f"SEX gender={'M' if gender == 1 else 'F'} but note "
                f"dominant pronoun is {dominant} ({m}M/{f}F)"]
    return []


def _redaction_check(text: str) -> list[str]:
    flags = []
    for m in REDACTION.finditer(text):
        flags.append(f"REDACTION {m.group(0)!r} at …{text[max(0, m.start()-40):m.end()+20]!r}")
    return flags


def _ama_check(features: dict, text: str) -> list[str]:
    """AMA exclusion: discharged against medical advice."""
    flags = []
    if features.get("discharge_location_ama") == 1:
        flags.append("REMOVE AMA discharge — excluded from training cohort")
    elif AMA_TEXT.search(text):
        m = AMA_TEXT.search(text)
        flags.append(
            f"REMOVE AMA (text) …{text[max(0, m.start()-40):m.end()+20]!r}"
        )
    return flags


def _hospice_check(features: dict) -> list[str]:
    """Hospice exclusion: discharged to hospice — terminal care, the patient
    is not a readmission candidate ("they are not coming back")."""
    if features.get("discharge_location_hospice") == 1:
        return [
            "REMOVE hospice discharge — excluded (not a readmission patient)"
        ]
    return []


def _planned_return_check(text: str) -> list[str]:
    """Elective-with-planned-return exclusion: elective admission where the
    patient plans to return to hospital by appointment (a planned
    readmission, not an unplanned one)."""
    if ELECTIVE.search(text) and PLANNED_RETURN.search(text):
        m = PLANNED_RETURN.search(text)
        return [
            f"REMOVE elective + planned hospital return …"
            f"{text[max(0, m.start()-60):m.end()+40]!r}"
        ]
    return []


def main() -> int:
    notes = json.loads(NOTES.read_text())
    cohort = json.loads(COHORT.read_text())
    by_hadm = {p["hadm_id"]: p for p in cohort["patients"]}

    counts = {"REMOVE": 0, "SEX": 0, "REDACTION": 0}
    n_flagged = 0
    for n in notes["patients"]:
        hadm = n["hadm_id"]
        c = by_hadm.get(hadm)
        if c is None:
            continue
        feats = c["features"]
        age = feats.get("age")
        gender = feats.get("gender")
        text = n["note"]
        arch = n["archetype"]

        issues = []
        issues += _age_check(text, age)
        if gender is not None:
            issues += _sex_check(text, gender)
        issues += _redaction_check(text)
        issues += _ama_check(feats, text)
        issues += _hospice_check(feats)
        issues += _planned_return_check(text)

        if issues:
            n_flagged += 1
            print(f"\n{hadm}  {arch}  band={c['band']:<10} "
                  f"age={age:g} gender={'M' if gender == 1 else 'F'}")
            for i in issues:
                print(f"    - {i}")
                counts[i.split()[0]] += 1

    print(f"\n=== summary: {n_flagged}/{len(notes['patients'])} patients "
          f"flagged; {dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
