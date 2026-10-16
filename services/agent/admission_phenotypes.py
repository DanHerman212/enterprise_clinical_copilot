"""Which admission was this, clinically — for choosing what to compare it against.

Why this is a table and not a classifier
----------------------------------------
Phase 4B pairs a patient's own 30-day readmission estimate with published cohort rates
for their condition, and the pairing is only defensible if the condition is the patient's.
The directive asks for the resolution to be deterministic and reviewable, so the runtime
reads this committed map and nothing else: a classifier's verdict at request time is not
something a clinician can check, and a note edit would change a patient's phenotype with
no commit anywhere.

The map was produced by `scripts/agent/audit_cohort_phenotypes.py`, whose output was
reviewed and approved as `docs/ux_workshop/strategy_and_tactics/04B-cohort_phenotypes.md`
(2026-10-16). That document carries, for every row, the section and the pattern that
decided it and the text around the match. To change a row, change the reviewed document
and regenerate; do NOT hand-edit an entry here, because the map's whole value is that it
is the reviewed thing.

What the signal was, and what it was not
----------------------------------------
The discharge note. The served feature row has no diagnosis, procedure, or text column at
all — it is demographics, counts, admission/discharge flags and eight laboratory values —
so there is nothing in it to classify from. The `archetype` field that some artifacts
carry (e.g. `mtsamples:2788`) lives in `evaluation/agent/results/hybrid_cohort.json`, is
not served, and is therefore not evidence about what the demo serves. It was also load
bearing in an earlier draft of this phase, which is exactly the mistake this comment
exists to prevent.

Three things the review found and this map therefore encodes
------------------------------------------------------------
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

from dataclasses import dataclass

#: The phenotype buckets, and the one clinical judgement attached to each: whether a
#: 30-day readmission cohort comparison is meaningful for it.
#:
#: `matrix_eligible=False` is a POLICY flag, not a finding that no study exists. Psychiatric
#: and obstetric readmission literature plainly does exist; those admissions are refused a
#: comparison because this demo's model predicts all-cause 30-day readmission for the
#: cohort it was trained on, and a rate borrowed from a different clinical population would
#: read as authoritative while comparing two different things. The directive's clinical
#: review can change any of these; the review document states the same for each excluded row.
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
        Bucket("oncology", "Oncology", False),
        Bucket("obstetric_gynecologic", "Obstetric / gynecologic", False),
        Bucket("psychiatric", "Psychiatric", False),
        Bucket("endocrine_surgery", "Endocrine surgery", False),
    )
})

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
    """The bucket this admission was resolved to, or None when it was not.

    None is a real answer and callers must handle it: an admission the map does not carry
    is one whose condition is unknown, and an unknown condition is the one thing a
    condition-matched comparison must not guess at.
    """
    key = ADMISSION_PHENOTYPE.get(hadm_id)
    return BUCKETS.get(key) if key else None


def is_matrix_eligible(hadm_id: int) -> bool:
    """Whether this admission may be paired with a cohort comparison.

    Two refusals, kept separate because they are different facts: the phenotype's policy
    flag, and the note not being an admission. Eligibility here does not promise the
    curated literature has a row to show — that is `literature_benchmarks`' answer, and a
    bucket can be eligible with nothing yet curated for it.
    """
    if hadm_id in NON_ADMISSION:
        return False
    bucket = bucket_for(hadm_id)
    return bool(bucket and bucket.matrix_eligible)


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
