"""audit_cohort_phenotypes — resolve every demo admission to a clinical phenotype.

Why this exists
---------------
Phase 4B pairs the patient's own 30-day readmission estimate with published cohort
rates for *their* condition. Doing that needs one thing the estimate does not carry: a
deterministic answer to "what was this admission for?". The directive calls for the
resolution to be deterministic and reviewable, so this script is the reviewable half —
it classifies every admission in the authorization boundary, prints the evidence behind
each verdict, and writes a proposal file. The served map is a separate, curated module
(`services/agent/admission_phenotypes.py`) that is written from an APPROVED proposal and
is the only thing the runtime reads. A script cannot reclassify a patient silently.

What it reads
-------------
The signal is the discharge note, because the served feature row has no diagnosis,
procedure, or text column at all — it is demographics, counts, admission/discharge flags
and eight laboratory values. Anything resolved from that row alone would be a guess. The
`archetype` field some artifacts carry (e.g. `mtsamples:2788`) is NOT served: it lives in
`evaluation/agent/results/hybrid_cohort.json`, so it is not evidence about what the demo
serves and must not be used as one.

Sections are parsed with `services/mcp/retrieval/sections.py` rather than by a second
splitter, so the headings this reads are the headings the retrieval index was built from.

Two caveats this reports rather than hides
------------------------------------------
1. A large minority of these notes have a DISCHARGE DIAGNOSES heading with nothing under
   it, so the section the directive names as the input is often empty. Classification
   falls back to the procedure block, the chief complaint and the history of present
   illness, and the proposal records which section decided each row.
2. The cohort is a heterogeneous MTSamples-derived mixture that is wider than any seven
   buckets — it includes oncology, obstetric, psychiatric and purely outpatient-flavoured
   admissions (a BRCA-2 carrier, migraine, depression with psychotic features). A 30-day
   readmission benchmark is not a meaningful comparison for all of them, which is why a
   row can be classified as a phenotype and still be marked not matrix-eligible.

    .venv/bin/python scripts/agent/audit_cohort_phenotypes.py
    .venv/bin/python scripts/agent/audit_cohort_phenotypes.py --json-out /tmp/proposal.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from google.cloud import bigquery

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.mcp.config import (  # noqa: E402
    COHORT_TABLE,
    DISCHARGE_TABLE,
    ENTITY_ID_COLUMN,
    PROJECT,
    TABLE,
)
from services.mcp.retrieval import sections as sect  # noqa: E402

# --- the phenotype buckets ---------------------------------------------------
#
# Ordered, and the order is the rule. First match wins, and the surgical and
# procedure-defined buckets come before the medical ones because a discharge note for an
# operation lists the comorbidities underneath it — a knee replacement for a patient with
# COPD is an orthopedic admission, not a respiratory one.
#
# `matrix_eligible` is the clinical judgement, and it is deliberately not the same for
# every bucket. Pairing a 30-day readmission rate with an admission that has no
# readmission literature to speak of produces exactly the mismatch this phase exists to
# remove, so those admissions render the literature list and no comparison.

ORTHOPEDIC = "orthopedic_arthroplasty"
SPINE = "spine_neurosurgery"
GI_HEPATIC = "gi_hepatobiliary_surgery"
SEPSIS = "sepsis_infectious"
COPD = "copd_respiratory"
CARDIAC = "heart_failure_cardiorenal"
RENAL_UROLOGIC = "renal_urologic"
ENDOCRINE = "endocrine_surgery"
ONCOLOGY = "oncology"
OBGYN = "obstetric_gynecologic"
PSYCHIATRIC = "psychiatric"
NEURO_MEDICAL = "neurological_medical"
GENERAL = "general_medicine_metabolic"


@dataclass(frozen=True)
class Rule:
    phenotype: str
    patterns: tuple[str, ...]
    #: Whether a 30-day readmission cohort comparison is clinically meaningful here.
    matrix_eligible: bool = True
    label: str = ""


RULES: tuple[Rule, ...] = (
    Rule(ORTHOPEDIC, (
        r"total knee (?:replacement|arthroplasty)", r"total hip (?:replacement|arthroplasty)",
        r"knee (?:replacement|arthroplasty)", r"hip (?:replacement|arthroplasty)",
        r"\barthroplasty\b", r"\bhemiarthroplasty\b",
        r"tibial plateau fracture", r"femoral neck fracture", r"hip fracture",
        r"open reduction internal fixation", r"\bORIF\b",
        r"femoral head", r"rotator cuff", r"shoulder arthroscopy",
        r"degenerative joint disease", r"osteoarthritis",
    ), label="Orthopedic surgery / arthroplasty"),
    Rule(SPINE, (
        r"lumbar stenosis", r"spinal (?:fusion|stenosis|instability)", r"laminectomy",
        r"discectomy", r"spondylolisthesis", r"spondylolysis",
        r"herniated nucle[ui]s pulposus", r"cervical spondylosis", r"radiculopathy",
        r"craniectomy", r"craniotomy", r"subdural", r"brain tumor", r"brain tumour",
    ), label="Spine / neurosurgery"),
    Rule(GI_HEPATIC, (
        r"cholecystitis", r"cholelithiasis", r"choledocholithiasis", r"cholecystectomy",
        r"pancreatitis", r"gastrointestinal bleed", r"\bGI bleed\b",
        r"upper gastrointestinal bleed", r"bowel obstruction", r"colectomy",
        r"appendicitis", r"diverticulitis", r"\bhernias?\b", r"cirrhosis",
        r"hepatic", r"roux-en-y", r"gastritis", r"esophagitis",
        # General abdominal surgery, which the directive's bucket names explicitly.
        r"gastric bypass", r"bariatric", r"morbid obesity",
    ), label="GI / hepatobiliary surgery"),
    Rule(SEPSIS, (
        r"\bsepsis\b", r"septic", r"bacteremia", r"bacteraemia", r"cellulitis",
        # No trailing boundary: the corpus says "chronic abscesses", and `\babscess\b`
        # missed the only admission this bucket has of that kind.
        r"\babscess", r"osteomyelitis", r"pneumonia", r"urinary tract infection",
        r"\bUTI\b", r"pyelonephritis", r"cytomegalovirus", r"\bCMV\b",
        r"fever of unknown origin", r"influenza", r"clostridium difficile",
    ), label="Sepsis / infection"),
    Rule(COPD, (
        r"\bCOPD\b", r"chronic obstructive pulmonary", r"respiratory failure",
        r"respiratory arrest", r"asthma", r"bronchitis", r"bronchiolitis",
        r"hypox(?:ia|emic)", r"hypercapnic", r"pleural effusion", r"pneumothorax",
    ), label="COPD / respiratory"),
    Rule(CARDIAC, (
        r"heart failure", r"congestive heart", r"\bCHF\b", r"cardiomyopathy",
        r"cardiorenal", r"myocardial infarction", r"coronary artery",
        r"atrial fibrillation", r"cardiac arrest", r"angina", r"pacemaker",
    ), label="Heart failure / cardiorenal"),
    Rule(RENAL_UROLOGIC, (
        r"renal (?:failure|insufficiency|cell carcinoma|cyst)", r"acute kidney injury",
        r"\bAKI\b", r"nephrectomy", r"hematuria", r"haematuria",
        r"benign prostatic hyperplasia", r"\bBPH\b", r"transurethral resection",
        r"hydronephrosis", r"nephrolithiasis", r"kidney stone", r"prostatectomy",
    ), label="Renal / urologic"),
    Rule(ENDOCRINE, (
        r"thyroidectomy", r"\bgoiters?\b", r"\bthyroid", r"parathyroid",
        r"adrenalectomy", r"pheochromocytoma", r"insulinoma",
    ), matrix_eligible=False, label="Endocrine surgery"),
    Rule(ONCOLOGY, (
        r"carcinoma", r"lymphoma", r"leukemia", r"leukaemia", r"melanoma",
        r"metastatic", r"metastases", r"chemotherapy", r"neoplasm", r"malignancy",
        r"BRCA", r"mastectomy", r"lumpectomy", r"tumor lysis", r"sarcoma",
        r"mesothelioma", r"pleural mass",
    ), matrix_eligible=False, label="Oncology"),
    Rule(OBGYN, (
        r"menometrorrhagia", r"menorrhagia", r"ectopic pregnancy", r"hysterectomy",
        r"uterine", r"ovarian", r"endometriosis", r"cervix", r"cesarean",
        r"caesarean", r"pregnan", r"dysmenorrhea", r"fibroids", r"myomectomy",
        r"oophorectomy", r"cervical carcinoma", r"placenta",
    ), matrix_eligible=False, label="Obstetric / gynecologic"),
    Rule(PSYCHIATRIC, (
        r"AXIS I", r"\bADHD\b", r"bipolar", r"depression", r"depressive",
        r"psychotic", r"schizophren", r"suicidal", r"anxiety", r"substance abuse",
        r"alcohol (?:withdrawal|dependence|abuse)", r"detox",
    ), matrix_eligible=False, label="Psychiatric"),
    Rule(NEURO_MEDICAL, (
        r"migraine", r"seizure", r"epilep", r"stroke", r"cerebrovascular",
        r"transient ischemic", r"\bTIA\b", r"syncope", r"traumatic brain injury",
        r"\bTBI\b", r"concussion", r"cervical musculoskeletal strain",
        r"multiple sclerosis", r"myasthenia", r"neuropathy", r"meningitis",
    ), label="Neurological (medical)"),
    Rule(GENERAL, (
        r"diabetes", r"hyperglyce", r"hypoglyce", r"hypertension", r"falls?\b",
        r"hypothermia", r"volume depletion", r"dehydration", r"anemia",
        r"anaemia", r"hypercholesterolemia", r"electrolyte", r"hypokalemia",
    ), label="General medicine / metabolic"),
)

#: Notes that are not hospital admissions at all. The corpus contains outpatient
#: rehabilitation and progress notes (skilled speech therapy after a stroke, a therapy
#: session count, an outpatient referral), which reached it through the MTSamples source
#: rather than through an admission. A readmission rate has no meaning for them, so they
#: are flagged and excluded from the matrix regardless of what they classify as.
OUTPATIENT = re.compile(
    r"outpatient rehabilitation|skilled speech therapy|number of sessions completed"
    r"|referred to .{0,40}outpatient",
    re.IGNORECASE,
)

#: Which section decided a row, in the order they are tried. Discharge diagnoses first —
#: it is the field that means "what this admission was" — then the procedure block, then
#: the two narrative openings, because a note whose diagnoses block is empty almost always
#: states the reason for admission in its opening sentence.
SIGNAL_SECTIONS = (
    sect.DISCHARGE_DIAGNOSIS,
    sect.MAJOR_PROCEDURE,
    sect.CHIEF_COMPLAINT,
    sect.HISTORY_OF_PRESENT_ILLNESS,
)


@dataclass
class Verdict:
    hadm_id: int
    phenotype: str = ""
    label: str = ""
    decided_by: str = ""
    matched: str = ""
    evidence: str = ""
    flags: list[str] = field(default_factory=list)
    matrix_eligible: bool = False


def _first_match(text: str):
    """The first rule whose pattern matches, with the pattern and the match itself."""
    for rule in RULES:
        for pattern in rule.patterns:
            found = re.search(pattern, text, re.IGNORECASE)
            if found:
                return rule, pattern, found
    return None


def _record(verdict: Verdict, rule: Rule, decided_by: str, pattern: str,
            body: str, found) -> None:
    verdict.phenotype = rule.phenotype
    verdict.label = rule.label
    verdict.decided_by = decided_by
    verdict.matched = pattern
    start = max(0, found.start() - 40)
    verdict.evidence = body[start:found.end() + 40]
    # A note that is not an admission is never matrix-eligible, whatever it classified as.
    verdict.matrix_eligible = rule.matrix_eligible and "outpatient_note" not in verdict.flags


def classify(parsed, hadm_id: int, note_text: str = "") -> Verdict:
    """Resolve one admission, recording which section and which pattern decided it."""
    verdict = Verdict(hadm_id=hadm_id)

    if parsed is None:
        verdict.flags.append("no_note")
        return verdict
    if parsed.coverage < 0.5:
        verdict.flags.append(f"low_coverage:{parsed.coverage:.2f}")
    if not parsed.body(sect.DISCHARGE_DIAGNOSIS).strip():
        verdict.flags.append("empty_discharge_diagnosis")

    flat = " ".join(note_text.split())
    if OUTPATIENT.search(flat):
        verdict.flags.append("outpatient_note")

    for section_name in SIGNAL_SECTIONS:
        body = " ".join(parsed.body(section_name).split())
        if not body:
            continue
        hit = _first_match(body)
        if hit:
            _record(verdict, hit[0], section_name, hit[1], body, hit[2])
            return verdict

    # Fallback: the whole note. These notes DO state the reason for admission — under a
    # heading the retrieval parser does not map to a diagnosis section ("ADMITTING
    # DIAGNOSIS", "FINAL DIAGNOSIS/REASON FOR ADMISSION", "POSTOPERATIVE DIAGNOSIS",
    # "PRIMARY DISCHARGE DIAGNOSES", "TRANSFER DIAGNOSES"), or in a narrative with no
    # headings at all. Scanning the note keeps that content reachable WITHOUT touching
    # `sections.py`: its heading vocabulary is baked into the retrieval index and into
    # every chunk id in it, so widening it here would invalidate the index to fix a
    # labelling problem. The flag records that a row needed the fallback, so a reviewer
    # can see which verdicts rest on the procedure block or the opening narrative.
    hit = _first_match(flat)
    if hit:
        verdict.flags.append("matched_whole_note")
        _record(verdict, hit[0], "whole_note", hit[1], flat, hit[2])
        return verdict

    verdict.flags.append("unclassified")
    return verdict


def _markdown(verdicts: list[Verdict], served: set[int], noted: set[int]) -> str:
    """The review document for the map.

    Every row carries the section and the pattern that decided it, and the text around
    the match. The runtime will read the curated map and nothing else, so this is the
    only place a misclassification is visible before it becomes a patient's phenotype —
    which means the evidence column is the point of the document, not decoration.
    """
    lines = [
        "# 04B cohort phenotypes — for review",
        "",
        "Generated by `scripts/agent/audit_cohort_phenotypes.py`. The runtime does not",
        "read this file: it reads the curated map in",
        "`services/agent/admission_phenotypes.py`, which is written from an APPROVED",
        "version of this table. A script cannot reclassify a patient silently.",
        "",
        "Signal is the discharge note. The served feature row carries no diagnosis,",
        "procedure or text column, so there is nothing in it to classify from — the",
        "`archetype` field some artifacts carry is not served and is not evidence.",
        "",
        "`matched_whole_note` marks a row whose diagnosis section the retrieval parser",
        "does not map (ADMITTING DIAGNOSIS, FINAL DIAGNOSIS/REASON FOR ADMISSION,",
        "POSTOPERATIVE DIAGNOSIS...), so the verdict rests on the whole note instead.",
        "`outpatient_note` marks a note that is not a hospital admission at all.",
        "",
        f"Boundary: {len(verdicts)} admissions. "
        f"Feature rows: {len(served)}. Notes: {len(noted)}.",
        "",
        "## Rows",
        "",
        "| hadm_id | phenotype | matrix-eligible | decided by | flags | evidence |",
        "|---|---|---|---|---|---|",
    ]
    for v in verdicts:
        evidence = v.evidence.replace("|", "\\|")[:90]
        flags = ", ".join(v.flags) or "-"
        lines.append(
            f"| {v.hadm_id} | {v.phenotype or '**(unclassified)**'} | "
            f"{'yes' if v.matrix_eligible else 'no'} | {v.decided_by or '-'} | "
            f"{flags} | {evidence} |"
        )
    lines += ["", "## Distribution", ""]
    counts = Counter(v.phenotype or "(unclassified)" for v in verdicts)
    for key, n in counts.most_common():
        lines.append(f"- {key}: {n}")
    lines += ["", "## Exclusions, and why", ""]
    for v in verdicts:
        if v.matrix_eligible:
            continue
        if "no_note" in v.flags:
            reason = "no note and no feature row — the record does not exist"
        elif "outpatient_note" in v.flags:
            reason = "not a hospital admission — an outpatient therapy/progress note"
        else:
            # Deliberately NOT phrased as "no literature exists": this is the bucket's
            # eligibility flag, which is a policy choice, and readmission literature for
            # psychiatric and obstetric admissions plainly does exist. Saying otherwise
            # would be inventing a finding to justify a default.
            reason = (f"bucket `{v.phenotype}` is marked not matrix-eligible "
                      f"(a policy flag, not a finding that no study exists)")
        lines.append(f"- {v.hadm_id} ({v.phenotype or 'unclassified'}): {reason}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=None,
                        help="Write the proposal map here for review.")
    parser.add_argument("--markdown-out", type=Path, default=None,
                        help="Write the review document here.")
    parser.add_argument("--show-evidence", action="store_true",
                        help="Print the matched text for every row.")
    args = parser.parse_args()

    client = bigquery.Client(project=PROJECT)
    ids = [r[ENTITY_ID_COLUMN] for r in client.query(
        f"SELECT {ENTITY_ID_COLUMN} FROM `{PROJECT}.{COHORT_TABLE}` ORDER BY 1"
    )]
    served = {r[ENTITY_ID_COLUMN] for r in client.query(
        f"SELECT {ENTITY_ID_COLUMN} FROM `{PROJECT}.{TABLE}`"
    )}
    notes = {r.hadm_id: r.text for r in client.query(
        f"SELECT hadm_id, text FROM `{DISCHARGE_TABLE}` WHERE hadm_id IN UNNEST(@ids)",
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ArrayQueryParameter("ids", "INT64", ids)]),
    )}

    verdicts = [
        classify(sect.parse_note(notes[hid]) if hid in notes else None, hid,
                 notes.get(hid) or "")
        for hid in ids
    ]

    print(f"authorization boundary : {len(ids)} admissions")
    print(f"feature rows           : {len(served)}")
    print(f"notes                  : {len(notes)}")
    print()

    orphans = [v.hadm_id for v in verdicts if v.hadm_id not in served]
    print("BOUNDARY / DATA MISMATCH")
    print(f"  in the boundary with no feature row: {orphans or 'none'}")
    print(f"  in the boundary with no note       : "
          f"{[v.hadm_id for v in verdicts if 'no_note' in v.flags] or 'none'}")
    print()

    print("PHENOTYPE DISTRIBUTION")
    counts = Counter(v.phenotype or "(unclassified)" for v in verdicts)
    for key, n in counts.most_common():
        rule = next((r for r in RULES if r.phenotype == key), None)
        mark = "" if not rule or rule.matrix_eligible else "   [not matrix-eligible]"
        print(f"  {key:28} {n:3}{mark}")
    eligible = sum(1 for v in verdicts if v.matrix_eligible)
    print(f"  {'':28} ---")
    print(f"  {'matrix-eligible rows':28} {eligible:3}")
    print()

    print("SIGNAL QUALITY")
    for flag in ("empty_discharge_diagnosis", "matched_whole_note", "outpatient_note"):
        n = sum(1 for v in verdicts if flag in v.flags)
        print(f"  {flag:28} {n:3}")
    low = [v for v in verdicts if any(f.startswith("low_coverage") for f in v.flags)]
    print(f"  {'low_coverage(<0.5)':28} {len(low):3}")
    print()

    blocked = [v for v in verdicts if not v.phenotype]
    if blocked:
        print("UNCLASSIFIED — each needs either a rule or a decision to exclude")
        for v in blocked:
            print(f"  {v.hadm_id}  {v.flags}")
        print()

    print("ROWS [hadm_id  phenotype  decided_by  matrix_eligible]")
    for v in verdicts:
        print(f"  {v.hadm_id}  {v.phenotype or '(unclassified)':28} "
              f"{v.decided_by or '-':26} {v.matrix_eligible}")
        if args.show_evidence and v.evidence:
            print(f"      matched {v.matched!r}: ...{v.evidence}...")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({
            "boundary_size": len(ids),
            "proposal": [
                {"hadm_id": v.hadm_id, "phenotype": v.phenotype,
                 "matrix_eligible": v.matrix_eligible, "decided_by": v.decided_by,
                 "matched": v.matched, "evidence": v.evidence, "flags": v.flags}
                for v in verdicts
            ],
        }, indent=2) + "\n")
        print(f"\nproposal written to {args.json_out}")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(_markdown(verdicts, served, set(notes)))
        print(f"review document written to {args.markdown_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
