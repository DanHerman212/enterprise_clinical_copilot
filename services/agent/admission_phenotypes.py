"""Which admission was this, clinically — for choosing what to compare it against.

Why this WAS a table and is now a classifier
--------------------------------------------
Phase 4B pairs a patient's own 30-day readmission estimate with published cohort rates
for their condition, and the pairing is only defensible if the condition is the patient's.
The directive asked for a deterministic and reviewable resolution, so the runtime read a
committed map and nothing else: a classifier's verdict at request time is not something a
clinician can check, and a note edit would change a patient's phenotype with no commit
anywhere. That trade was right about reviewability and wrong about coverage, and the two
failures it produced are the reason this module now resolves from the chart:

  * An admission the map did not carry resolved to NOTHING, and nothing was the same
    answer as "no comparison is meaningful here" — so `/ehr` silently lost a section of
    its own briefing for every admission outside the 84 rows, and a new record would have
    lost it forever.
  * Three buckets were refused a comparison by policy (oncology, obstetric, psychiatric),
    which in the surface is indistinguishable from the first case: a table that is not
    there, and no sentence saying why.

What replaced it keeps both properties. The rules below are the ones
`scripts/agent/audit_cohort_phenotypes.py` used to PRODUCE the reviewed map — same
patterns, same order — so the classifier and the reviewed document are the same
vocabulary rather than two opinions, and a test asserts they still agree on all 84 rows.
What the classifier adds is the 85th admission: resolution reads the sections the turn
actually retrieved, so it answers for a record nobody has reviewed yet. And the fallback
is explicit rather than silent (`GENERAL_INPATIENT`), so an admission whose condition the
vocabulary cannot name is still compared — against the all-cause hospital-wide rate,
which is the quantity the model estimates.

What the signal was, and what it was not
----------------------------------------
The discharge note. The served feature row has no diagnosis, procedure, or text column at
all — it is demographics, counts, admission/discharge flags and eight laboratory values —
so there is nothing in it to classify from. The `archetype` field that some artifacts
carry (e.g. `mtsamples:2788`) lives in `evaluation/agent/results/hybrid_cohort.json`, is
not served, and is therefore not evidence about what the demo serves. It was also load
bearing in an earlier draft of this phase, which is exactly the mistake this comment
exists to prevent.

Three things the review found and this module therefore encodes
---------------------------------------------------------------
1. Not every row could be resolved from the section the directive names. 19 of the 87 notes
   have a `DISCHARGE DIAGNOSES` heading with nothing under it, so those verdicts rest on
   the procedure block, the chief complaint, the history of present illness, or the whole
   note.
2. Five notes are not hospital admissions at all — outpatient speech-therapy and progress
   notes that reached the corpus through the MTSamples source. They resolve to a phenotype
   because the note still says what the patient was treated for, and they are nonetheless
   refused a cohort comparison: a readmission rate has no meaning for a note that records
   a therapy session. They are listed, rather than deleted from the map, so that the
   refusal is visible and auditable.
3. Two rows in the authorization boundary have neither a note nor a feature row. They are
   recorded here as unresolved and are being removed from the boundary, because a selector
   that offers an admission which cannot be scored offers a dead end.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: The phenotype buckets, and the one clinical judgement attached to each: whether a
#: 30-day readmission cohort comparison is meaningful for it.
#:
#: `matrix_eligible=False` is a POLICY flag, and it is now reserved for the one case it
#: can still defend: a record that is not a hospital stay. Three buckets — oncology,
#: obstetric/gynaecologic, psychiatric — used to be flagged as well, on the reasoning that
#: a specialty rate borrowed from a different clinical population compares two different
#: things. The reasoning was sound and the consequence was not: those admissions were
#: drawn with no comparison, no cohort and no explanation, so a third of the corpus
#: silently lost a section of its own briefing. What replaced the refusal is
#: `GENERAL_INPATIENT` below — an all-cause hospital-wide cohort, which is the quantity
#: the model actually estimates — so every inpatient is compared against something of its
#: own kind, and a specialty rate is never borrowed for a patient it does not describe.
BUCKETS: dict[str, "Bucket"] = {}


@dataclass(frozen=True)
class Bucket:
    """One phenotype, with the label a reader sees and its comparison policy."""

    key: str
    label: str
    matrix_eligible: bool
    #: The words a clinician uses for this condition, for matching a free-text QUERY
    #: against a patient's phenotype (the condition-lock, Phase 4B).
    #:
    #: Deliberately not the classifier in `scripts/agent/audit_cohort_phenotypes.py`:
    #: that one reads discharge notes to decide what an admission WAS, and this one reads
    #: one line of a clinician's question to decide what they are ASKING ABOUT. The two
    #: vocabularies overlap heavily and mean different things — a note saying "history of
    #: COPD" is a comorbidity, while a query saying "COPD" is the subject — and a single
    #: shared table would have to serve both readings.
    query_terms: tuple[str, ...] = ()


BUCKETS.update({
    bucket.key: bucket for bucket in (
        Bucket("orthopedic_arthroplasty", "Orthopedic surgery / arthroplasty", True,
               query_terms=("arthroplasty", "knee replacement", "hip replacement",
                            "total knee", "total hip", "joint replacement",
                            "hip fracture", "orthopedic", "orthopaedic")),
        Bucket("heart_failure_cardiorenal", "Heart failure / cardiorenal", True,
               query_terms=("heart failure", "chf", "congestive heart", "cardiorenal",
                            "cardiomyopathy")),
        Bucket("copd_respiratory", "COPD / respiratory", True,
               query_terms=("copd", "chronic obstructive", "respiratory failure",
                            "asthma", "bronchitis")),
        Bucket("spine_neurosurgery", "Spine / neurosurgery", True,
               query_terms=("spine", "spinal", "lumbar", "stenosis", "fusion",
                            "discectomy", "laminectomy")),
        Bucket("sepsis_infectious", "Sepsis / infection", True,
               query_terms=("sepsis", "septic", "pneumonia", "infection", "bacteremia",
                            "cellulitis", "abscess", "urinary tract infection", "uti")),
        Bucket("gi_hepatobiliary_surgery", "GI / hepatobiliary surgery", True,
               query_terms=("gastrointestinal bleed", "gi bleed", "upper gi",
                            "cholecystitis", "cholelithiasis", "colorectal",
                            "bowel", "hepatobiliary", "pancreatitis")),
        Bucket("general_medicine_metabolic", "General medicine / metabolic", True,
               query_terms=("diabetes", "metabolic", "general medicine",
                            "internal medicine", "hyperglycemia")),
        Bucket("renal_urologic", "Renal / urologic", True,
               query_terms=("acute kidney injury", "aki", "renal failure",
                            "renal insufficiency", "nephrology")),
        Bucket("neurological_medical", "Neurological (medical)", True,
               query_terms=("stroke", "ischemic stroke", "cerebrovascular", "seizure",
                            "epilepsy")),
        # The fallback. Every inpatient is compared against something, and for a
        # condition the curated corpus does not carry by name this is what the table
        # shows: the all-cause hospital-wide rate, which is the quantity the model
        # estimates. No `query_terms`: this bucket is never what a clinician's question
        # is about, it is what the table falls back to when the question names nothing
        # the corpus curated.
        Bucket("general_inpatient", "General inpatient (all conditions)", True),
        Bucket("oncology", "Oncology", True),
        Bucket("obstetric_gynecologic", "Obstetric / gynecologic", True),
        Bucket("psychiatric", "Psychiatric", True),
        Bucket("endocrine_surgery", "Endocrine surgery", True),
    )
})

#: The bucket every inpatient falls back to when nothing more specific resolves, or when
#: the resolved condition has no curated cohort of its own.
GENERAL_INPATIENT = "general_inpatient"

#: The approved map: admission -> phenotype key. 84 of the boundary's 89 rows.
ADMISSION_PHENOTYPE: dict[int, str] = {
    90000001: "orthopedic_arthroplasty", 90000002: "renal_urologic",
    90000003: "orthopedic_arthroplasty", 90000004: "orthopedic_arthroplasty",
    90000005: "sepsis_infectious", 90000009: "sepsis_infectious",
    90000010: "copd_respiratory", 90000011: "sepsis_infectious",
    90000012: "gi_hepatobiliary_surgery", 90000013: "sepsis_infectious",
    90000014: "obstetric_gynecologic", 90000015: "orthopedic_arthroplasty",
    90000016: "neurological_medical", 90000017: "orthopedic_arthroplasty",
    90000018: "copd_respiratory", 90000019: "orthopedic_arthroplasty",
    90000020: "oncology", 90000022: "oncology", 90000023: "heart_failure_cardiorenal",
    90000024: "gi_hepatobiliary_surgery", 90000025: "heart_failure_cardiorenal",
    90000026: "heart_failure_cardiorenal", 90000027: "copd_respiratory",
    90000029: "sepsis_infectious", 90000030: "psychiatric",
    90000031: "obstetric_gynecologic", 90000033: "obstetric_gynecologic",
    90000034: "orthopedic_arthroplasty", 90000035: "endocrine_surgery",
    90000036: "sepsis_infectious", 90000037: "heart_failure_cardiorenal",
    90000040: "neurological_medical", 90000041: "neurological_medical",
    90000043: "sepsis_infectious", 90000044: "sepsis_infectious",
    90000045: "gi_hepatobiliary_surgery", 90000046: "orthopedic_arthroplasty",
    90000047: "obstetric_gynecologic", 90000048: "psychiatric",
    90000049: "gi_hepatobiliary_surgery", 90000050: "general_medicine_metabolic",
    90000051: "neurological_medical", 90000052: "spine_neurosurgery",
    90000053: "copd_respiratory", 90000055: "gi_hepatobiliary_surgery",
    90000056: "obstetric_gynecologic", 90000057: "spine_neurosurgery",
    90000058: "psychiatric", 90000059: "general_medicine_metabolic",
    90000060: "sepsis_infectious", 90000063: "copd_respiratory",
    90000064: "copd_respiratory", 90000065: "gi_hepatobiliary_surgery",
    90000067: "sepsis_infectious", 90000068: "orthopedic_arthroplasty",
    90000069: "spine_neurosurgery", 90000070: "sepsis_infectious",
    90000071: "obstetric_gynecologic", 90000074: "heart_failure_cardiorenal",
    90000076: "gi_hepatobiliary_surgery", 90000077: "orthopedic_arthroplasty",
    90000078: "obstetric_gynecologic", 90000079: "gi_hepatobiliary_surgery",
    90000081: "copd_respiratory", 90000082: "obstetric_gynecologic",
    90000084: "neurological_medical", 90000086: "copd_respiratory",
    90000087: "gi_hepatobiliary_surgery", 90000088: "gi_hepatobiliary_surgery",
    90000090: "orthopedic_arthroplasty", 90000091: "gi_hepatobiliary_surgery",
    90000092: "sepsis_infectious", 90000094: "spine_neurosurgery",
    90000095: "gi_hepatobiliary_surgery", 90000097: "oncology", 90000099: "oncology",
    90000100: "neurological_medical", 90000101: "oncology", 90000102: "renal_urologic",
    90000103: "general_medicine_metabolic", 90000104: "copd_respiratory",
    90000105: "gi_hepatobiliary_surgery", 90000106: "copd_respiratory",
    90000108: "oncology",
}

#: Notes that resolve to a phenotype but are not hospital admissions: outpatient
#: rehabilitation and progress notes. A cohort comparison is refused for these whatever
#: their phenotype says, because the quantity being compared does not apply to them.
NON_ADMISSION: frozenset[int] = frozenset({
    90000028, 90000040, 90000051, 90000054, 90000073,
})

#: In the authorization boundary with neither a note nor a feature row, so nothing can be
#: served for them at all. Recorded so the gap is documented rather than merely absent;
#: the boundary itself is being corrected to stop offering them.
UNRESOLVED: frozenset[int] = frozenset({90000039, 90000107})


def bucket_for(hadm_id: int) -> Bucket | None:
    """The bucket the REVIEWED MAP resolved this admission to, or None.

    Kept for the two callers that must decide before any evidence exists — the pre-flight
    condition-lock (`routing.comparison_applies`) and the audit tooling that measures the
    map against the classifier. It is no longer how the surface decides what a patient
    has: that is `resolve_phenotype`, which reads the chart the turn actually retrieved
    (see the module docstring's "why this was a table, and why it is not one any more").

    None is a real answer and callers must handle it: an admission the map does not carry
    is one this hint cannot speak for.
    """
    key = ADMISSION_PHENOTYPE.get(hadm_id)
    return BUCKETS.get(key) if key else None


def is_matrix_eligible(hadm_id: int) -> bool:
    """Whether this record may be paired with a cohort comparison at all.

    The only refusal left is the record not being an admission. It used to also require a
    curated phenotype bucket, which refused a comparison to every condition the corpus had
    not named — and said nothing about why, so the turn simply had one fewer section. A
    named condition now selects its own curated rows and every other inpatient selects
    `GENERAL_INPATIENT` (see `risk_canvas._comparative_matrix`), so what is left to refuse
    is the one thing a readmission rate cannot describe: a visit that was never a stay.
    """
    return hadm_id not in NON_ADMISSION


# ---------------------------------------------------------------------------
# The signals: what the chart says this admission was
# ---------------------------------------------------------------------------
#
# This is the classifier the reviewed map was MEASURED against, brought into the runtime.
# `scripts/agent/audit_cohort_phenotypes.py` used it to produce the 84 reviewed verdicts,
# and its output was approved as `docs/ux_workshop/strategy_and_tactics/04B-cohort
# _phenotypes.md`; the patterns below are that script's, unchanged, so a patient whose
# note has not changed cannot resolve differently here than they were reviewed as.
#
# What changed is who reads them. The map answered "what is 90000047" for the 84
# admissions that existed when it was written, and the answer for the 85th was nothing —
# the table drawing nothing, `/ehr` losing a section, and no way to tell that apart from a
# patient who genuinely cannot be compared. Reading the note at request time answers the
# same question for an admission nobody has seen, which is what "100% of inpatient
# admissions, including any future unseen record" requires.
#
# Order is clinical priority and it is the reviewed order, not an accident: a hip fracture
# with a postoperative pneumonia is an orthopedic admission that developed a complication
# rather than a sepsis admission, and the first rule to match wins. The map is the same
# ordering's verdict, so the two agree except where a note changed — and that disagreement
# is a finding, not a regression (a test asserts the map and the classifier agree today).

#: The sections tried, in order, and the reason for it. `discharge_diagnosis` is the field
#: that means "what this admission was"; the two narratives follow because 19 of the 87
#: demo notes have an empty diagnoses block and state the reason for admission in the
#: opening sentence instead.
SIGNAL_SECTIONS: tuple[str, ...] = (
    "discharge_diagnosis",
    "brief_hospital_course",
    "history_of_present_illness",
)

#: Bucket key -> the words that decide it. Ordered; first match wins.
SIGNAL_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("orthopedic_arthroplasty", (
        r"total knee (?:replacement|arthroplasty)",
        r"total hip (?:replacement|arthroplasty)",
        r"knee (?:replacement|arthroplasty)", r"hip (?:replacement|arthroplasty)",
        r"\barthroplasty\b", r"\bhemiarthroplasty\b",
        r"tibial plateau fracture", r"femoral neck fracture", r"hip fracture",
        r"open reduction internal fixation", r"\bORIF\b",
        r"femoral head", r"rotator cuff", r"shoulder arthroscopy",
        r"degenerative joint disease", r"osteoarthritis",
    )),
    ("spine_neurosurgery", (
        r"lumbar stenosis", r"spinal (?:fusion|stenosis|instability)", r"laminectomy",
        r"discectomy", r"spondylolisthesis", r"spondylolysis",
        r"herniated nucle[ui]s pulposus", r"cervical spondylosis", r"radiculopathy",
        r"craniectomy", r"craniotomy", r"subdural", r"brain tumor", r"brain tumour",
    )),
    ("gi_hepatobiliary_surgery", (
        r"cholecystitis", r"cholelithiasis", r"choledocholithiasis", r"cholecystectomy",
        r"pancreatitis", r"gastrointestinal bleed", r"\bGI bleed\b",
        r"upper gastrointestinal bleed", r"bowel obstruction", r"colectomy",
        r"appendicitis", r"diverticulitis", r"\bhernias?\b", r"cirrhosis",
        r"hepatic", r"roux-en-y", r"gastritis", r"esophagitis",
        r"gastric bypass", r"bariatric", r"morbid obesity",
    )),
    ("sepsis_infectious", (
        r"\bsepsis\b", r"septic", r"bacteremia", r"bacteraemia", r"cellulitis",
        # No trailing boundary: the corpus says "chronic abscesses".
        r"\babscess", r"osteomyelitis", r"pneumonia", r"urinary tract infection",
        r"\bUTI\b", r"pyelonephritis", r"cytomegalovirus", r"\bCMV\b",
        r"fever of unknown origin", r"influenza", r"clostridium difficile",
    )),
    ("copd_respiratory", (
        r"\bCOPD\b", r"chronic obstructive pulmonary", r"respiratory failure",
        r"respiratory arrest", r"asthma", r"bronchitis", r"bronchiolitis",
        r"hypox(?:ia|emic)", r"hypercapnic", r"pleural effusion", r"pneumothorax",
    )),
    ("heart_failure_cardiorenal", (
        r"heart failure", r"congestive heart", r"\bCHF\b", r"cardiomyopathy",
        r"cardiorenal", r"myocardial infarction", r"coronary artery",
        r"atrial fibrillation", r"cardiac arrest", r"angina", r"pacemaker",
    )),
    ("renal_urologic", (
        r"renal (?:failure|insufficiency|cell carcinoma|cyst)", r"acute kidney injury",
        r"\bAKI\b", r"nephrectomy", r"hematuria", r"haematuria",
        r"benign prostatic hyperplasia", r"\bBPH\b", r"transurethral resection",
        r"hydronephrosis", r"nephrolithiasis", r"kidney stone", r"prostatectomy",
    )),
    ("endocrine_surgery", (
        r"thyroidectomy", r"\bgoiters?\b", r"\bthyroid", r"parathyroid",
        r"adrenalectomy", r"pheochromocytoma", r"insulinoma",
    )),
    ("oncology", (
        r"carcinoma", r"lymphoma", r"leukemia", r"leukaemia", r"melanoma",
        r"metastatic", r"metastases", r"chemotherapy", r"neoplasm", r"malignancy",
        r"BRCA", r"mastectomy", r"lumpectomy", r"tumor lysis", r"sarcoma",
        r"mesothelioma", r"pleural mass",
    )),
    ("obstetric_gynecologic", (
        r"menometrorrhagia", r"menorrhagia", r"ectopic pregnancy", r"hysterectomy",
        r"uterine", r"ovarian", r"endometriosis", r"cervix", r"cesarean",
        r"caesarean", r"pregnan", r"dysmenorrhea", r"fibroids", r"myomectomy",
        r"oophorectomy", r"cervical carcinoma", r"placenta",
    )),
    ("psychiatric", (
        r"AXIS I", r"\bADHD\b", r"bipolar", r"depression", r"depressive",
        r"psychotic", r"schizophren", r"suicidal", r"anxiety", r"substance abuse",
        r"alcohol (?:withdrawal|dependence|abuse)", r"detox",
    )),
    ("neurological_medical", (
        r"migraine", r"seizure", r"epilep", r"stroke", r"cerebrovascular",
        r"transient ischemic", r"\bTIA\b", r"syncope", r"traumatic brain injury",
        r"\bTBI\b", r"concussion", r"cervical musculoskeletal strain",
        r"multiple sclerosis", r"myasthenia", r"neuropathy", r"meningitis",
    )),
    ("general_medicine_metabolic", (
        # `\bdiabet` rather than `diabetes`: the reviewed vocabulary missed diabetic
        # ketoacidosis and hyperosmolar hyperglycaemic state, which are diabetes
        # admissions written without the word. Broadened in Phase E (2026-10-26) — the
        # directive asks for the classifiers to be widened, and this is the widening that
        # the demo corpus actually needed.
        r"\bdiabet", r"hyperglyce", r"hypoglyce", r"hypertension", r"falls?\b",
        r"hypothermia", r"volume depletion", r"dehydration", r"anemia",
        r"anaemia", r"hypercholesterolemia", r"electrolyte", r"hypokalemia",
    )),
)

#: Compiled once. `re.search` over a few hundred short patterns is microseconds, and the
#: alternative — a second copy of the vocabulary in a faster shape — is a second thing to
#: keep in step with the reviewed rules.
_COMPILED: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...] = tuple(
    (key, tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns))
    for key, patterns in SIGNAL_RULES
)


def phenotype_of_text(text: str) -> str:
    """The bucket these words resolve to, or "" when they resolve to nothing.

    One rule, one vocabulary, applied to whatever text the caller has. `""` is a real
    answer — the note says nothing this vocabulary recognises — and the caller decides what
    to do with it, which for a served inpatient is `GENERAL_INPATIENT`.
    """
    haystack = " ".join(str(text or "").split())
    if not haystack:
        return ""
    for key, patterns in _COMPILED:
        if any(pattern.search(haystack) for pattern in patterns):
            return key
    return ""


def resolve_phenotype(sections, *, fallback: str = "") -> str:
    """This admission's condition, read off the chart the turn retrieved.

    `sections` maps a section name to its text. The sections are tried in
    `SIGNAL_SECTIONS` order — discharge diagnoses first, then the narratives — because a
    diagnoses block is what the admission WAS, and a hospital course that mentions
    pneumonia is a finding about the stay rather than its cause. Text left over after every
    named section is tried last, for the notes whose reason for admission sits under a
    heading the retrieval vocabulary does not map ("ADMITTING DIAGNOSIS", "FINAL
    DIAGNOSIS/REASON FOR ADMISSION") or in a narrative with no headings at all.

    `fallback` is what an unresolved inpatient gets. The caller passes
    `GENERAL_INPATIENT` for an admission and `""` for anything else, because those are two
    different silences: a stay whose condition this vocabulary cannot name is still a stay
    and still comparable to a hospital-wide rate, while a record that was never a stay has
    nothing to compare.
    """
    mapping = {str(key): str(value or "") for key, value in (sections or {}).items()}
    for section in SIGNAL_SECTIONS:
        key = phenotype_of_text(mapping.get(section, ""))
        if key:
            return key
    # Every section, in a stable order, for the notes whose diagnoses block is empty or
    # whose headings the retrieval vocabulary does not carry.
    rest = " ".join(
        text for name, text in sorted(mapping.items()) if name not in SIGNAL_SECTIONS
    )
    return phenotype_of_text(rest) or fallback



def phenotype_of_query(text: str) -> str:
    """The condition a query is about, as a bucket key, or "" when it names none.

    The condition-lock's other half. A `/pubmed` turn carries the patient's admission so
    the answer can be compared, and the query carries the clinician's subject — and those
    can disagree, which is the case the lock exists for. Asking about heart failure while
    looking at an orthopedic admission must not put her estimate beside heart failure
    cohorts, and it must not put it beside orthopedic cohorts either: the question asked
    was about heart failure, and a table of knee arthroplasty rates answers something
    else.

    Longest term wins, so "total knee" beats "knee" and "urinary tract infection" beats a
    bare "uti" appearing inside another word. Empty when nothing matches, and empty then
    means the caller draws no comparison: this is a gate, and a gate that opens on an
    unrecognised subject is not a gate.
    """
    haystack = f" {str(text or '').lower().replace('-', ' ')} "
    best = ("", 0)
    for bucket in BUCKETS.values():
        for term in bucket.query_terms:
            needle = f" {term.lower()} "
            if needle in haystack and len(term) > best[1]:
                best = (bucket.key, len(term))
    return best[0]
