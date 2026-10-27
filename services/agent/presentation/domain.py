"""Feature label map — the single source of truth for how model features are
named in the UI.

The risk canvas's FactorBars and the agent's prose must name a feature the
same way — a reviewer reading "medication order count" on the bars but
`medication_order_count` in the prose sees a bug. Every feature that can
appear as a top factor gets a human label here; the canvas attaches it to
each factor and the agent prompt embeds the same map so the live agent's
words match the UI.

This module lives with the agent (not the site) because the canvas is
composed here — presentation of model output belongs to the service that
owns the model contract.

Keys are the aggregated PARENT features returned by predict_readmission
(one-hot groups summed to their parent — see mlops/src/encoding.py groups())."""
from __future__ import annotations

import re

# The risk score is shown to two decimals everywhere it is shown: the widget, the
# component payload, and the prose. Six decimals — what the service returns —
# implies a precision the model does not have, and it made the same score read as
# two different numbers depending on which surface you looked at (found live
# 2026-10-12: prose said 0.09, the card said 0.090459).
#
# This is the same hazard the label map exists to prevent, one field over: not two
# NAMES for one factor, but two NUMBERS for one score. It lives beside the map for
# that reason.
RISK_DECIMALS = 2


def display_probability(value):
    """The risk score at the precision it is displayed and described at.

    Returns None for anything that is not a number, so a caller holding no
    estimate renders "unavailable" rather than 0.0. A rounded absence would be
    the most confident wrong number in the payload.

    Callers round for DISPLAY only. A threshold decision reads the served value:
    rounding first would let presentation precision move a clinical call, and it
    would move it in the direction of looking safer at the boundary.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(float(value), RISK_DECIMALS)


FEATURE_LABELS: dict[str, str] = {
    "age": "age",
    "gender": "sex",
    "has_procedure": "procedure performed",
    "procedure_count": "procedures",
    "oncology_flag": "oncology history",
    "medication_count": "medication count",
    "medication_order_count": "medication order count",
    "prior_inpatient_days": "prior inpatient days",
    "prior_admission_count": "prior admissions",
    "index_los_days": "length of stay",
    "recent_ed_visits": "recent ED visits",
    "hemoglobin_min": "lowest hemoglobin",
    "sodium_min": "lowest sodium",
    "sodium_max": "highest sodium",
    "sodium_last": "recent sodium",
    "rbc_min": "lowest red blood cell count",
    "rbc_last": "recent red blood cell count",
    "rdw_max": "red cell distribution width (RDW)",
    "monocytes_min": "lowest monocyte count",
    "race": "race",
    "admission_type": "admission type",
    "discharge_location": "discharge destination",
    "insurance": "insurance type",
}


def label_for(feature: str) -> str:
    """Human label for a feature key; falls back to the raw key if unknown."""
    return FEATURE_LABELS.get(feature, feature)


def humanize(feature: str) -> str:
    """Best human label for a feature key, for visual consumption.

    1. Known parent feature -> curated label (FEATURE_LABELS).
    2. Otherwise normalize the raw key to Title Case, handling both
       camelCase and snake_case: medicationOrderCount and
       medication_order_count both become "Medication Order Count". This is
       the safety net for any feature not in the curated map.
    """
    known = FEATURE_LABELS.get(feature)
    if known is not None:
        return known
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", feature)
    words = [w for w in re.split(r"[\s_\-]+", spaced) if w]
    return " ".join(w.capitalize() for w in words) if words else feature


# ===========================================================================
# Consolidated from `admission_phenotypes.py` (Phase 7).
#
# Which admission was this, clinically — for choosing what to compare it against.
#
# Why this WAS a table and is now a classifier
# --------------------------------------------
# Phase 4B pairs a patient's own 30-day readmission estimate with published cohort rates
# for their condition, and the pairing is only defensible if the condition is the patient's.
# The directive asked for a deterministic and reviewable resolution, so the runtime read a
# committed map and nothing else: a classifier's verdict at request time is not something a
# clinician can check, and a note edit would change a patient's phenotype with no commit
# anywhere. That trade was right about reviewability and wrong about coverage, and the two
# failures it produced are the reason this module now resolves from the chart:
#
#   * An admission the map did not carry resolved to NOTHING, and nothing was the same
#     answer as "no comparison is meaningful here" — so `/ehr` silently lost a section of
#     its own briefing for every admission outside the 84 rows, and a new record would have
#     lost it forever.
#   * Three buckets were refused a comparison by policy (oncology, obstetric, psychiatric),
#     which in the surface is indistinguishable from the first case: a table that is not
#     there, and no sentence saying why.
#
# What replaced it keeps both properties. The rules below are the ones
# `scripts/agent/audit_cohort_phenotypes.py` used to PRODUCE the reviewed map — same
# patterns, same order — so the classifier and the reviewed document are the same
# vocabulary rather than two opinions, and a test asserts they still agree on all 84 rows.
# What the classifier adds is the 85th admission: resolution reads the sections the turn
# actually retrieved, so it answers for a record nobody has reviewed yet. And the fallback
# is explicit rather than silent (`GENERAL_INPATIENT`), so an admission whose condition the
# vocabulary cannot name is still compared — against the all-cause hospital-wide rate,
# which is the quantity the model estimates.
#
# What the signal was, and what it was not
# ----------------------------------------
# The discharge note. The served feature row has no diagnosis, procedure, or text column at
# all — it is demographics, counts, admission/discharge flags and eight laboratory values —
# so there is nothing in it to classify from. The `archetype` field that some artifacts
# carry (e.g. `mtsamples:2788`) lives in `evaluation/agent/results/hybrid_cohort.json`, is
# not served, and is therefore not evidence about what the demo serves. It was also load
# bearing in an earlier draft of this phase, which is exactly the mistake this comment
# exists to prevent.
#
# Three things the review found and this module therefore encodes
# ---------------------------------------------------------------
# 1. Not every row could be resolved from the section the directive names. 19 of the 87 notes
#    have a `DISCHARGE DIAGNOSES` heading with nothing under it, so those verdicts rest on
#    the procedure block, the chief complaint, the history of present illness, or the whole
#    note.
# 2. Five notes are not hospital admissions at all — outpatient speech-therapy and progress
#    notes that reached the corpus through the MTSamples source. They resolve to a phenotype
#    because the note still says what the patient was treated for, and they are nonetheless
#    refused a cohort comparison: a readmission rate has no meaning for a note that records
#    a therapy session. They are listed, rather than deleted from the map, so that the
#    refusal is visible and auditable.
# 3. Two rows in the authorization boundary have neither a note nor a feature row. They are
#    recorded here as unresolved and are being removed from the boundary, because a selector
#    that offers an admission which cannot be scored offers a dead end.
# ===========================================================================


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


# ===========================================================================
# Consolidated from `literature_benchmarks.py` (Phase 7).
#
# Curated cohort benchmarks for the comparative readmission matrix.
#
# `/pubmed` answers a question with the records PubMed returned. The matrix answers a
# different question — how does THIS patient's 30-day readmission estimate compare with
# published cohorts — and the two are not the same kind of thing. A retrieved record is
# whatever the search returned; a benchmark is a cohort with a rate a reader can hold
# against the model's, which most records in a result set do not have. Of the thirty
# records captured for the three demo conditions, **four** state an absolute readmission
# rate at a horizon the model's estimate can be compared with. The rest report a relative
# measure (RR, OR, HR), or a rate at five years, or no rate at all.
#
# So this is a curation and not a parse, and that is why it is written out rather than
# derived. There is no extraction step in the repository to drift, no regex over an
# abstract that can silently stop matching, and no value here a script produced. Each
# entry names the PMID its figures come from and quotes, in the comment directly above
# it, the sentence the figures were read off — so the number can be checked without
# re-reading the paper, and a wrong one has somewhere to point.
#
# Three rules the entries hold to, in the order they are likely to be tempted:
#
#   * A row is included only when its horizon is the model's. A five-year composite beside
#     a 30-day probability is not a comparison, it is two unrelated numbers in adjacent
#     cells with a subtraction waiting to happen. `PARTNER 3` is the case that makes the
#     rule concrete rather than theoretical: both its papers state a rate, and neither
#     belongs in this table.
#   * The rate is quoted as the record states it. "≈20%" rather than "20%", because the
#     abstract says "close to 20%" and the qualifier is part of the figure — dropping it
#     would turn a hedged sentence into a measured one.
#   * A row is one record, not one cohort-mate. Where a study compares two groups (the
#     durotomy readmissions below), the second group is named in `cohort_summary` rather
#     than given a row of its own, so the table cannot be read as two independent findings.
#
# Keyed by PHENOTYPE, since Phase 4B, and the change is the point of that phase. Rows used
# to be looked up by matching the PMIDs a live search returned, which meant the matrix drew
# its rows only when a search happened to return a record that had been curated — measured
# at the time of the change: no live search ever did, so the matrix rendered nothing, and
# the one verification where it appeared did so by luck. A patient's condition is not a
# property of what a search returned, so the rows are now selected by the admission's own
# resolved phenotype (`admission_phenotypes`), which is reviewed, committed, and the same
# table every time. The cost of the old keying is therefore removed rather than documented:
# the matrix renders for any admission whose phenotype has rows below.
#
# What the rows are NOT keyed by is the retrieval, and that is deliberate in the other
# direction too: `/pubmed` on a condition the patient does not have still shows the records
# PubMed returned, as evidence cards badged general — it simply does not borrow the
# patient's own estimate to compare them against, because the two are not the same thing.
#
# Quoting the rate's own sentence
# -------------------------------
# Each row now carries `rate_sentence`: the sentence from the abstract that states the
# figure, reproduced verbatim. It is what the matrix shows, so the number in the table and
# the sentence it came from are the same sentence, and a reader checking the figure does not
# have to take the curation's word for it. A summary written here could be checked only
# against the paper; the paper's own sentence is checkable in the table.
#
# `matched_features` is deliberately near-empty, and the emptiness is the honest answer.
# It names THIS patient's drivers that the cohort itself identified as risk factors for
# readmission. Where a record names none of them the tuple is empty, because a blank cell
# says the literature is silent — which is a fact about the literature and the reader is
# owed it. Filling the column from cohort inclusion criteria would put a claim in the
# record's mouth that the record did not make.
#
# Feature keys, not labels: the pill under a cohort and the bar beside the patient have to
# name one factor one way, and the only way to guarantee that is to look the name up in
# the same map the bars use (`feature_labels.humanize`).
# ===========================================================================

from dataclasses import dataclass


#: The horizon the model's estimate is made at, in days.
#:
#: The matrix compares only rows at this horizon, and it is declared here rather than
#: assumed so that a row at another horizon cannot join the comparison by being added
#: carefully. If the model is ever retrained at a different horizon, this is the one
#: number that has to move with it, and every row below stops qualifying until it does.
MODEL_HORIZON_DAYS = 30


@dataclass(frozen=True)
class CohortBenchmark:
    """One published cohort, as a row the matrix can compare a patient against."""

    pmid: str
    #: The phenotype this row may be shown for, spelled as `admission_phenotypes` spells
    #: it. A row is only ever rendered for a patient resolved to this bucket, which is
    #: what stops an orthopedic patient's estimate meeting a heart failure cohort.
    phenotype: str
    #: The rate, as the record states it. A string, because the qualifier is a digit
    #: or it is not: "≈20%" and "20%" are different claims and one type cannot hold both.
    reported_rate: str
    cohort_summary: str
    evidence_tier: str
    horizon_days: int
    #: The sentence the rate was read from, verbatim. Shown in the matrix, so the figure
    #: and its provenance are the same text.
    rate_sentence: str
    #: Citation parts. Carried here because there is no retrieved record to take them
    #: from now that selection is by phenotype — and a second copy that could disagree
    #: with the search is exactly what this replaces, so these are the only copy.
    journal: str = ""
    year: str = ""
    #: The article's own title, and its first three authors. Read off the record
    #: once (NCBI esummary) and carried here so the references drawer shows the
    #: paper rather than a journal name — the row a reader opens is the paper.
    title: str = ""
    authors: str = ""
    #: This patient's drivers, as feature keys, that the cohort itself named.
    matched_features: tuple[str, ...] = ()


BENCHMARKS: dict[str, CohortBenchmark] = {
    # "Of the 1.5 million individuals at risk, 13.2% (95% CI: 10.5%-16.1%) were
    # readmitted within 30 days and 35.7% (95% CI: 27.1%-44.9%) within 1 year."
    # — "Global Comparison of Readmission Rates for Patients With Heart Failure",
    # Journal of the American College of Cardiology, 2023. The 30-day figure is the
    # one taken; the same paper reports the 1-year rate and it does not qualify.
    "37495280": CohortBenchmark(
        pmid="37495280",
        phenotype="heart_failure_cardiorenal",
        reported_rate="13.2%",
        journal="Journal of the American College of Cardiology",
        year="2023",
        title=(
            "Global Comparison of Readmission Rates for Patients With Heart "
            "Failure."
        ),
        authors="Foroutan F, Rayner DG, Ross HJ, et al.",
        rate_sentence=(
            "Of the 1.5 million individuals at risk, 13.2% (95% CI: 10.5%-16.1%) were "
            "readmitted within 30 days and 35.7% (95% CI: 27.1%-44.9%) within 1 year."
        ),
        cohort_summary=(
            "Hospitalised heart failure, pooled across 24 observational reports "
            "and ~1.5 million patients, with no country or care setting excluded"
        ),
        evidence_tier="Meta-Analysis",
        horizon_days=30,
        # The cohort reports the rate globally and identifies no feature-level driver,
        # so there is nothing here that names any of this patient's factors.
        matched_features=(),
    ),
    # "The pooled 30-, 60-, 90-, 180-, and 365-day readmission rates were 11%, 17%,
    # 17%, 30%, and 37%, respectively." — "Readmission rate for acute exacerbation of
    # chronic obstructive pulmonary disease: A systematic review and meta-analysis",
    # Respiratory Medicine, 2023. 46 studies; the 30-day pooled rate is the one taken.
    "36528962": CohortBenchmark(
        pmid="36528962",
        phenotype="copd_respiratory",
        reported_rate="11%",
        journal="Respiratory Medicine",
        year="2023",
        title=(
            "Readmission rate for acute exacerbation of chronic obstructive "
            "pulmonary disease: A systematic review and meta-analysis."
        ),
        authors="Ruan H, Zhang H, Wang J, et al.",
        rate_sentence=(
            "The pooled 30-, 60-, 90-, 180-, and 365-day readmission rates were 11%, "
            "17%, 17%, 30%, and 37%, respectively."
        ),
        cohort_summary=(
            "Acute exacerbation of COPD, pooled across 24 of 46 included studies "
            "that reported a 30-day rate"
        ),
        evidence_tier="Meta-Analysis",
        horizon_days=30,
        # "We also identified potential risk factors for COPD readmission, including
        # male sex, number of hospitalizations in the previous year, LOS, and
        # comorbidities such as heart failure, tumor or cancer, and diabetes."
        #
        # "Number of hospitalizations in the previous year" is the one phrase in the
        # whole capture that names a factor this patient also has, and it is prior
        # utilisation however it is counted — so the pill carries the patient's own
        # name for that factor rather than a second name for it.
        matched_features=("prior_inpatient_days",),
    ),
    # "Importantly, close to 20% of patients are readmitted within 30 days after
    # hospital discharge, with great cost to the person and society." — "Differential
    # Diagnosis of Suspected Chronic Obstructive Pulmonary Disease Exacerbations in
    # the Acute Care Setting: Best Practice", American Journal of Respiratory and
    # Critical Care Medicine, 2023.
    "36701677": CohortBenchmark(
        pmid="36701677",
        phenotype="copd_respiratory",
        reported_rate="≈20%",
        journal="American Journal of Respiratory and Critical Care Medicine",
        year="2023",
        title=(
            "Differential Diagnosis of Suspected Chronic Obstructive Pulmonary "
            "Disease Exacerbations in the Acute Care Setting: Best Practice."
        ),
        authors="Celli BR, Fabbri LM, Aaron SD, et al.",
        rate_sentence=(
            "Importantly, close to 20% of patients are readmitted within 30 days "
            "after hospital discharge, with great cost to the person and society."
        ),
        cohort_summary=(
            "Hospital discharge after an exacerbation of COPD, as the prevalence "
            "quoted in a clinical practice review of the acute care setting"
        ),
        evidence_tier="Narrative Review",
        horizon_days=30,
        matched_features=(),
    ),
    # "Of the 52 patients discharged home same day, 2 were readmitted (3.8%); of the
    # 59 patients admitted postoperatively, 5 were readmitted (8.5%)." — "Readmission
    # rates after intraoperative durotomy and repair in outpatient lumbar spinal
    # surgery", North American Spine Society Journal, 2026. The same-day group is the
    # row; the admitted group is named in the summary rather than given a row.
    "42004460": CohortBenchmark(
        pmid="42004460",
        phenotype="spine_neurosurgery",
        reported_rate="3.8%",
        journal="North American Spine Society Journal",
        year="2026",
        title=(
            "Readmission rates after intraoperative durotomy and repair in "
            "outpatient lumbar spinal surgery."
        ),
        authors="Proal JD, Moon AS, LaHaise K, et al.",
        rate_sentence=(
            "Of the 52 patients discharged home same day, 2 were readmitted (3.8%); "
            "of the 59 patients admitted postoperatively, 5 were readmitted (8.5%)."
        ),
        cohort_summary=(
            "Outpatient lumbar spine surgery complicated by an intraoperative "
            "durotomy, repaired and discharged the same day (n=52; 8.5% in the "
            "postoperatively admitted comparator, n=59)"
        ),
        evidence_tier="Observational Cohort",
        horizon_days=30,
        matched_features=(),
    ),
    # "The 30-day readmission rate was 3.2% and 3.4% and the CMS-defined complication
    # rate was 3.8% and 2.9% for THA and TKA, respectively." — "The Impact of
    # Pre-Operative Healthcare Utilization on Complications, Readmissions, and
    # Post-Operative Healthcare Utilization Following Total Joint Arthroplasty",
    # The Journal of Arthroplasty, 2022.
    #
    # The knee figure is the row and the hip figure is named in the summary, for the
    # same reason the durotomy comparator is: one study is one row, so a reader cannot
    # total two numbers that a paper reported as alternatives.
    "34793857": CohortBenchmark(
        pmid="34793857",
        phenotype="orthopedic_arthroplasty",
        reported_rate="3.4%",
        journal="The Journal of Arthroplasty",
        year="2022",
        title=(
            "The Impact of Pre-Operative Healthcare Utilization on Complications, "
            "Readmissions, and Post-Operative Healthcare Utilization Following "
            "Total Joint Arthroplasty."
        ),
        authors="Creager AE, Kleven AD, Kesimoglu ZN, et al.",
        rate_sentence=(
            "The 30-day readmission rate was 3.2% and 3.4% and the CMS-defined "
            "complication rate was 3.8% and 2.9% for THA and TKA, respectively."
        ),
        cohort_summary=(
            "Elective primary total knee arthroplasty (n=2,159; 3.2% in the total hip "
            "comparator, n=1,785), single institution, 2015-2019"
        ),
        evidence_tier="Retrospective Cohort",
        horizon_days=30,
        # The cohort's own finding is about prior NON-ELECTIVE utilisation: "presence
        # of any pre-HU was associated with increased risk of 30-day readmission (OR
        # 2.85)" for THA and "≥2 pre-HU events" (OR 3.52) for TKA, where pre-HU is
        # emergency department, urgent care, observation or inpatient admission in the
        # 90 days before surgery. Two of this patient's features measure that concept,
        # so they are named here. The windows need not match for the pill to be honest —
        # it says the cohort identified prior utilisation as a driver, which it did —
        # but the pill is not a claim that the two are computed the same way.
        matched_features=("recent_ed_visits", "prior_admission_count"),
    ),
    # "The overall 30-day unplanned readmission rate of these 33,840 patients was
    # 5.5%." — "Prolonged Length of Stay and Risk of Unplanned 30-Day Readmission After
    # Elective Spine Surgery: Propensity Score-Matched Analysis of 33,840 Patients",
    # Spine, 2020.
    "32341301": CohortBenchmark(
        pmid="32341301",
        phenotype="spine_neurosurgery",
        reported_rate="5.5%",
        journal="Spine",
        year="2020",
        title=(
            "Prolonged Length of Stay and Risk of Unplanned 30-Day Readmission "
            "After Elective Spine Surgery: Propensity Score-Matched Analysis of "
            "33,840 Patients."
        ),
        authors="Kim RB, Wilkerson C, Karsy M, et al.",
        rate_sentence=(
            "The overall 30-day unplanned readmission rate of these 33,840 patients "
            "was 5.5%."
        ),
        cohort_summary=(
            "Elective spine surgery, ACS-NSQIP, 16,920 propensity score-matched pairs "
            "drawn from 99,575 patients"
        ),
        evidence_tier="Observational Cohort",
        horizon_days=30,
        # "Multivariate analysis of the propensity score-matched population ... showed
        # pLOS was associated with an increased risk of 30-day unplanned readmission
        # (odds ratio [OR] 1.423 ...)", where pLOS is a stay of 4 days or more. Length
        # of stay is a feature this patient has, so the pill names it.
        matched_features=("index_los_days",),
    ),
    # "UGIB had a 30-day all-cause readmission rate of 17.4% (95% confidence interval
    # [CI] 16.7-18.2%), stratification revealed a higher rate for variceal UGIB [19.6%
    # (95% CI 17.6-21.5%)] than non-variceal [16.8% (95% CI 16.0-17.5%)]." — "30-day
    # readmission rates among upper gastrointestinal bleeds: A systematic review and
    # meta-analysis", Journal of Gastroenterology and Hepatology, 2023.
    #
    # The overall pooled rate is the row; the variceal split is named in the summary
    # rather than given rows, because the strata are the same 70 studies.
    "36808766": CohortBenchmark(
        pmid="36808766",
        phenotype="gi_hepatobiliary_surgery",
        reported_rate="17.4%",
        journal="Journal of Gastroenterology and Hepatology",
        year="2023",
        title=(
            "30-day readmission rates among upper gastrointestinal bleeds: A "
            "systematic review and meta-analysis."
        ),
        authors="Dunne CL, Kaur S, Delacruz B, et al.",
        rate_sentence=(
            "UGIB had a 30-day all-cause readmission rate of 17.4% (95% confidence "
            "interval [CI] 16.7-18.2%), stratification revealed a higher rate for "
            "variceal UGIB [19.6% (95% CI 17.6-21.5%)] than non-variceal [16.8% "
            "(95% CI 16.0-17.5%)]."
        ),
        cohort_summary=(
            "Upper gastrointestinal bleeding, pooled across 70 studies and 4,292,714 "
            "patients; certainty of evidence low to very low for all outcomes"
        ),
        evidence_tier="Meta-Analysis",
        horizon_days=30,
        matched_features=(),
    ),
    # "The 30-day readmission rate was 18.4% (5645 patients), the 30-day death rate was
    # 3.7% (1146 patients), and the rate of either outcome was 19.7% (6066 patients)."
    # — "Variation in Acute Care Rehabilitation and 30-Day Hospital Readmission or
    # Mortality in Adult Patients With Pneumonia", JAMA Network Open, 2020.
    #
    # A cohort description inside a study whose question is about therapy, which is why
    # the tier says cohort and not trial: the rate is the cohort's own, and the
    # intervention arm it was collected for is not what this row claims.
    "32886119": CohortBenchmark(
        pmid="32886119",
        phenotype="sepsis_infectious",
        reported_rate="18.4%",
        journal="JAMA Network Open",
        year="2020",
        title=(
            "Variation in Acute Care Rehabilitation and 30-Day Hospital "
            "Readmission or Mortality in Adult Patients With Pneumonia."
        ),
        authors="Freburger JK, Chou A, Euloth T, et al.",
        rate_sentence=(
            "The 30-day readmission rate was 18.4% (5645 patients), the 30-day death "
            "rate was 3.7% (1146 patients), and the rate of either outcome was 19.7% "
            "(6066 patients)."
        ),
        cohort_summary=(
            "Adults discharged alive with a primary or secondary diagnosis of "
            "pneumonia or an influenza-related condition, 30,746 patients across 12 "
            "acute care hospitals"
        ),
        evidence_tier="Observational Cohort",
        horizon_days=30,
        matched_features=(),
    ),
    # The fallback cohort (Phase E, 2026-10-26). Every inpatient gets a comparison, and
    # this is the one an admission gets when its own condition has no curated row — or
    # when the chart does not name a condition this corpus carries at all. It is the
    # all-cause, hospital-wide figure, which is the quantity the model estimates: a
    # readmission of any kind within 30 days of discharge, across every condition.
    #
    # The rate is quoted from the abstract's RESULTS sentence, and the qualifier is part
    # of the figure — "almost one fifth (19.6%)" is what the paper says, and printing
    # 19.6% alone would turn a hedged sentence into a measured one.
    "19339721": CohortBenchmark(
        pmid="19339721",
        phenotype=GENERAL_INPATIENT,
        reported_rate="19.6%",
        journal="The New England Journal of Medicine",
        year="2009",
        title=(
            "Rehospitalizations among patients in the Medicare fee-for-service "
            "program."
        ),
        authors="Jencks SF, Williams MV, Coleman EA.",
        rate_sentence=(
            "Almost one fifth (19.6%) of the 11,855,702 Medicare beneficiaries who had "
            "been discharged from a hospital were rehospitalized within 30 days, and "
            "34.0% were rehospitalized within 90 days."
        ),
        cohort_summary=(
            "Hospital-wide, all conditions — Medicare fee-for-service claims for "
            "11,855,702 beneficiaries discharged from a hospital, 2003-2004, national "
            "analysis"
        ),
        # An observational analysis of a national claims cohort. Named as what it is
        # rather than promoted to a tier it does not hold: this row is a fallback, and a
        # fallback that overstates its own strength is worse than no row.
        evidence_tier="Observational Cohort",
        horizon_days=30,
        # The paper names demographic and hospital characteristics as correlates, not
        # this patient's own drivers, so the column stays empty rather than borrowing
        # cohort composition for a claim the record did not make.
        matched_features=(),
    ),
}


def _year(pub_date: str) -> str:
    """The four-digit year a PubMed date states, or an empty string.

    PubMed dates are not one format — "2023", "2023 Aug 01", "2022 Sep-Oct 01" — so
    this reads the year off rather than parsing a date. Nothing here needs a day.
    """
    for token in str(pub_date or "").replace("-", " ").split():
        if len(token) == 4 and token.isdigit():
            return token
    return ""


#: Evidence tiers, strongest first, for the order the matrix draws them in. The tiers are
#: free text on each row rather than an enum, so a rank that is missing degrades to the
#: end of the table instead of dropping the row: an unfamiliar tier should still be shown,
#: just not promoted above a meta-analysis.
_TIER_RANK: dict[str, int] = {
    "Meta-Analysis": 0,
    "Systematic Review": 1,
    "Randomized Controlled Trial": 2,
    "Observational Cohort": 3,
    "Retrospective Cohort": 4,
    "Narrative Review": 5,
}


def benchmarks_for_phenotype(phenotype: str | None) -> list[dict]:
    """The curated rows for one resolved phenotype, strongest evidence first.

    Selected by the patient's condition rather than by what a search returned, which is
    the Phase 4B change and the reason the matrix now renders at all: a search's PMIDs
    are an artifact of a query, and the patient's condition is not. A phenotype with no
    rows returns an empty list, and the caller is expected to draw the estimate alone
    rather than reach for a bucket that does have rows — the one failure this whole phase
    exists to remove is a patient's figure being compared against the wrong cohort.

    Ordering is by evidence tier first, so a pooled estimate leads and a single-centre
    series follows it, with the PMID breaking ties. The order is stable because the table
    is a constant; a test holds it, since a comparison table that reorders itself between
    turns is one a reader cannot check against a previous reading.

    The citation is built here from the row's own journal and year, because selection by
    phenotype means there is no retrieved record to take them from any more. The URL is
    derived from the PMID rather than stored, so a row can never point at a record other
    than the one it cites.
    """
    if not phenotype:
        return []
    rows: list[dict] = []
    for benchmark in BENCHMARKS.values():
        if benchmark.phenotype != phenotype:
            continue
        if benchmark.horizon_days != MODEL_HORIZON_DAYS:
            # Unreachable with today's rows, and it stays because the rule is the point:
            # the horizon filter is applied at selection, not trusted to curation.
            continue
        rows.append({
            "pmid": benchmark.pmid,
            "source": " · ".join(
                part for part in (benchmark.journal, benchmark.year) if part
            ),
            # The paper itself, so the row and the card can name it. `source` is the
            # citation tram: a journal and a year locate the record, and the reference
            # that a reader opens is the article.
            "title": benchmark.title,
            "authors": benchmark.authors,
            # The parts `source` joins, kept as themselves: a citation line reads
            # `Journal. Year. Authors.` and a pre-joined string cannot be punctuated
            # back into that without guessing where the join was.
            "journal": benchmark.journal,
            "year": benchmark.year,
            "reported_rate": benchmark.reported_rate,
            "cohort_summary": benchmark.cohort_summary,
            "matched_features": list(benchmark.matched_features),
            "evidence_tier": benchmark.evidence_tier,
            "horizon_days": benchmark.horizon_days,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{benchmark.pmid}/",
            # The sentence the rate was read from, which is what the matrix shows: the
            # figure and its provenance in the same text, so a reader can check one
            # against the other without leaving the page.
            "excerpt": benchmark.rate_sentence,
        })
    rows.sort(key=lambda row: (_TIER_RANK.get(row["evidence_tier"], len(_TIER_RANK)),
                               row["pmid"]))
    return rows


# ===========================================================================
# Moved here from `a2ui.py` (Phase 7).
#
# `canvas.py` needs `_usable_articles`, and `a2ui.py` imports `canvas.py`, so
# leaving it in the composer makes the pair circular. This module is the one both
# already import.
# ===========================================================================

_LITERATURE_FIELDS = ("pmid", "title", "journal", "pub_date", "authors", "abstract",
                      "url")



def _usable_articles(literature) -> list[dict]:
    """The articles from a literature payload, each shaped for the canvas.

    A record with no pmid is dropped: it cannot be cited or linked, which is the
    whole use of it. The remaining fields are coerced to strings rather than
    passed through, because the console's component schema is strict — one
    missing key drops the entire card, and a card that fails to draw is worse
    than a card with a blank journal.

    An empty list is the answer for three different situations — no search was
    run, the search failed, or it matched nothing — which the canvas treats the
    same and the answer text and provenance caption distinguish between.
    """
    if not isinstance(literature, dict):
        return []
    articles = literature.get("articles")
    if not isinstance(articles, list):
        return []
    usable = []
    for article in articles:
        if not isinstance(article, dict) or not article.get("pmid"):
            continue
        usable.append({
            field: article.get(field) if isinstance(article.get(field), str) else ""
            for field in _LITERATURE_FIELDS
        })
    return usable

