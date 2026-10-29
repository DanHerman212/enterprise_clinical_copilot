"""Every inpatient gets a comparison; the record has to say what it was.

Three admissions drove this (2026-10-26): `#90000047` (obstetric), `#90000020` (oncology)
and `#90000058` (psychiatric). All three are hospital admissions, all three have a note,
and all three were drawn with no `ComparativeRiskMatrix` — because the reviewed map
resolved their phenotype to a bucket whose policy flag refused the comparison, and a
refused comparison is invisible: the table is not there and nothing says why.

Two changes, and the second is what makes the first safe:

  * Resolution reads the chart. The classifier that PRODUCED the reviewed map is the one
    the runtime now uses, so an admission nobody has reviewed resolves exactly as the
    reviewed ones do.
  * When the resolved condition has no curated cohort, the table falls back to the
    all-cause hospital-wide rate — the quantity the model actually estimates — rather than
    returning nothing. So coverage is a property of the code, not of the curation.

What must not change: the five outpatient notes stay refused, because a readmission rate
does not describe a therapy session. Offline — the note text is supplied, not fetched.
"""

from __future__ import annotations

import pytest

from services.agent.presentation.domain import (
    ADMISSION_PHENOTYPE,
    BUCKETS,
    GENERAL_INPATIENT,
    NON_ADMISSION,
    SIGNAL_RULES,
    is_matrix_eligible,
    resolve_phenotype,
)
from services.agent.presentation.a2ui import compose_presentation, turn_phenotype
from services.agent.presentation.domain import benchmarks_for_phenotype
from services.agent.ingress.contracts import SERVED_HADM_IDS
from services.agent.routing import encounter


def _classify(hadm_id: int) -> None:
    """What the pre-flight records before the turn composes anything.

    `is_matrix_eligible` reads the turn's encounter state, which the pre-flight sets from
    the record's `encounter_type` column. These tests compose without running a turn, so
    the value is set here; `NON_ADMISSION` is the map's own record of the same fact and is
    asserted to agree with it in `test_tier1_cohort_sweep`.
    """
    encounter.set_current(
        encounter.OUTPATIENT if hadm_id in NON_ADMISSION else encounter.INPATIENT
    )

#: One discharge diagnosis per bucket: the words a note uses for that admission. Written
#: as a discharge summary writes them, because that is what the classifier is reading.
DIAGNOSIS = {
    "orthopedic_arthroplasty": "Discharge Diagnosis: Osteoarthritis of the right knee, "
                              "status post total knee arthroplasty",
    "spine_neurosurgery": "Discharge Diagnosis: Lumbar stenosis with neurogenic "
                          "claudication, status post laminectomy",
    "gi_hepatobiliary_surgery": "Discharge Diagnosis: Acute cholecystitis",
    "sepsis_infectious": "Discharge Diagnosis: Sepsis secondary to pneumonia",
    "copd_respiratory": "Discharge Diagnosis: Acute exacerbation of COPD with "
                        "respiratory failure",
    "heart_failure_cardiorenal": "Discharge Diagnosis: Acute decompensated heart failure",
    "renal_urologic": "Discharge Diagnosis: Acute kidney injury, prerenal",
    "endocrine_surgery": "Discharge Diagnosis: Multinodular goiter, thyroidectomy",
    "oncology": "Discharge Diagnosis: Metastatic carcinoma of the lung",
    "obstetric_gynecologic": "Discharge Diagnosis: Menorrhagia with uterine fibroids",
    "psychiatric": "Discharge Diagnosis: Bipolar disorder, manic phase",
    "neurological_medical": "Discharge Diagnosis: Acute ischemic stroke",
    "general_medicine_metabolic": "Discharge Diagnosis: Diabetic ketoacidosis",
}

#: A note that names no condition this vocabulary carries. Not a made-up clinical
#: scenario — this is what an uncurated admission looks like to a classifier, and it is
#: the case that used to produce no table at all.
UNCLASSIFIABLE = ("Discharge Diagnosis: Failure to thrive with generalised weakness "
                  "and poor oral intake")


def _sections(text: str, section: str = "discharge_diagnosis") -> dict[str, str]:
    return {section: text}


def _predict(hadm_id: int) -> dict:
    return {"name": "predict_readmission", "args": {"hadm_id": hadm_id},
            "response": {"hadm_id": hadm_id, "probability": 0.2, "threshold": 0.11,
                         "decision": 1, "model_version": "test-v1"}}


def _ehr_turn(hadm_id: int, diagnosis: str) -> dict:
    """One `/ehr` turn: the note sections it read, and the estimate beside them."""
    _classify(hadm_id)
    sections = [{"name": "rag_search_sections", "args": {"hadm_id": hadm_id},
                 "response": {"query": "sections", "returned": 1, "passages": [{
                     "id": "p1", "section": "discharge_diagnosis",
                     "text": diagnosis}]}}]
    return compose_presentation(
        "/ehr", "Summary of the admission.^[1]", [*sections, _predict(hadm_id)],
        command="summarize",
        # `compose_presentation` no longer reads the encounter for itself (`00d7` Row 11);
        # `turn.py` passes it, so the harness does too.
        is_outpatient=encounter.current_is_outpatient())


def _matrix(out: dict) -> dict | None:
    for component in out["a2ui"]["messages"][1]["updateComponents"]["components"]:
        if component["component"] == "ComparativeRiskMatrix":
            return component
    return None


# --- the signals ---------------------------------------------------------------

class TestTheChartDecides:
    @pytest.mark.parametrize("bucket", sorted(DIAGNOSIS))
    def test_a_note_resolves_to_its_own_bucket(self, bucket):
        assert resolve_phenotype(_sections(DIAGNOSIS[bucket])) == bucket

    def test_an_unrecognised_note_falls_back_rather_than_resolving_nothing(self):
        """The distinction that matters: the vocabulary failing to name a condition is not
        the same finding as a record that cannot be compared."""
        assert resolve_phenotype(_sections(UNCLASSIFIABLE), fallback=GENERAL_INPATIENT) \
            == GENERAL_INPATIENT
        assert resolve_phenotype(_sections(UNCLASSIFIABLE)) == ""

    def test_the_discharge_diagnosis_outranks_the_hospital_course(self):
        """A hip fracture with a postoperative pneumonia is an orthopedic admission that
        developed a complication, not a sepsis admission — which is the reviewed order."""
        assert resolve_phenotype({
            "discharge_diagnosis": "Discharge Diagnosis: Femoral neck fracture",
            "brief_hospital_course": "Brief Hospital Course: postoperative pneumonia "
                                     "treated with antibiotics",
        }) == "orthopedic_arthroplasty"

    def test_a_note_whose_diagnoses_block_is_empty_still_resolves(self):
        """19 of the 87 demo notes have an empty diagnoses block and say the reason for
        admission in the opening narrative instead."""
        assert resolve_phenotype({
            "discharge_diagnosis": "",
            "brief_hospital_course": "Brief Hospital Course: admitted with acute "
                                     "cholecystitis and taken to theatre",
        }) == "gi_hepatobiliary_surgery"

    def test_a_heading_the_retrieval_vocabulary_does_not_carry_is_still_read(self):
        """`ADMITTING DIAGNOSIS` is not a section the index knows, so it arrives as a
        whole-note chunk. The classifier reads whatever text it is given."""
        assert resolve_phenotype({
            "discharge_summary": "ADMITTING DIAGNOSIS: Acute pancreatitis",
        }) == "gi_hepatobiliary_surgery"

    def test_every_bucket_the_reviewed_map_uses_has_a_rule(self):
        """The drift guard. A bucket named by the map but absent from the classifier
        could be reviewed and then never resolved again."""
        rules = {key for key, _patterns in SIGNAL_RULES}
        assert set(ADMISSION_PHENOTYPE.values()) <= rules
        assert set(rules) <= set(BUCKETS)


# --- coverage ------------------------------------------------------------------

class TestEveryInpatientIsCompared:
    def test_every_bucket_resolves_to_a_cohort(self):
        """By construction: a bucket with rows of its own uses them, and a bucket without
        uses the hospital-wide cohort. No bucket can end at an empty table."""
        for bucket in BUCKETS:
            rows = benchmarks_for_phenotype(bucket) or benchmarks_for_phenotype(
                GENERAL_INPATIENT)
            assert rows, f"{bucket} has no cohort and no fallback"

    def test_the_fallback_cohort_is_a_real_paper(self):
        """A fallback that overstates itself is worse than no row: this one has to carry
        the identifier a reader can open and the paper's own sentence."""
        (row,) = benchmarks_for_phenotype(GENERAL_INPATIENT)
        assert row["pmid"] == "19339721"
        assert row["title"].startswith("Rehospitalizations among patients")
        assert row["authors"]
        assert "19.6%" in row["reported_rate"]
        assert "11,855,702" in row["excerpt"]
        assert row["cohort_summary"].startswith("Hospital-wide, all conditions")

    def test_a_condition_with_no_curated_rows_draws_the_hospital_wide_cohort(self):
        """`#90000047`, `#90000020` and `#90000058` in one assertion: the specialty table
        they were refused is replaced by one they can actually be read against."""
        out = _ehr_turn(90000047, DIAGNOSIS["obstetric_gynecologic"])

        matrix = _matrix(out)
        assert matrix is not None, "an inpatient admission drew no comparison"
        assert matrix["rows"][0]["pmid"] == "19339721"

    def test_a_curated_condition_keeps_its_own_cohorts(self):
        """The fallback must never displace curation: an orthopedic admission is compared
        against joint arthroplasty cohorts, not against the hospital-wide rate."""
        out = _ehr_turn(90000015, DIAGNOSIS["orthopedic_arthroplasty"])

        matrix = _matrix(out)
        assert matrix is not None
        assert matrix["rows"][0]["pmid"] == "34793857"
        assert all(row["pmid"] != "19339721" for row in matrix["rows"])

    def test_an_unseen_admission_resolves_and_draws(self):
        """The 85th admission. No map entry, no review, a note — and a table."""
        out = _ehr_turn(99999999, DIAGNOSIS["heart_failure_cardiorenal"])

        matrix = _matrix(out)
        assert matrix is not None
        assert matrix["rows"][0]["pmid"] == "37495280"

    def test_an_unseen_admission_with_an_unclassifiable_note_still_draws(self):
        out = _ehr_turn(99999999, UNCLASSIFIABLE)

        matrix = _matrix(out)
        assert matrix is not None
        assert matrix["rows"][0]["pmid"] == "19339721"

    def test_all_82_served_inpatients_draw_the_table(self):
        """The gate itself, over the whole served cohort.

        The note is the same unclassifiable one for every admission, because what is being
        checked is the pipeline and not the classifier (the per-bucket cases above cover
        the classifier): for every admission this deployment answers for, `/ehr` produces
        a matrix whose rows carry a pmid, a title and authors.
        """
        inpatients = sorted(SERVED_HADM_IDS - NON_ADMISSION)
        assert len(inpatients) == 82

        for hadm_id in inpatients:
            matrix = _matrix(_ehr_turn(hadm_id, UNCLASSIFIABLE))
            assert matrix is not None, f"{hadm_id} drew no comparison"
            assert matrix["rows"], f"{hadm_id} drew a table with no cohorts"
            for row in matrix["rows"]:
                assert row["pmid"], f"{hadm_id}: a cohort with no pmid"
                assert row["title"], f"{hadm_id}: a cohort with no title"
                assert row["authors"], f"{hadm_id}: a cohort with no authors"

    def test_an_outpatient_note_is_still_refused(self):
        """The one refusal that survives, and the reason it does: there was no stay to be
        readmitted from.

        The encounter is passed in since 2026-10-29 (`00d7` Row 11): the predicate no
        longer reads `routing.encounter`'s `ContextVar` for itself.
        """
        for hadm_id in sorted(NON_ADMISSION):
            _classify(hadm_id)
            assert is_matrix_eligible(hadm_id, is_outpatient=True) is False
            assert is_matrix_eligible(hadm_id, is_outpatient=False) is True
            out = _ehr_turn(hadm_id, DIAGNOSIS["neurological_medical"])
            assert _matrix(out) is None, f"{hadm_id} is not an admission"


# --- what the composer does with it ---------------------------------------------

class TestTheComposerReadsTheChart:
    def test_an_admission_with_no_chart_still_compares(self):
        """A stay whose condition nobody read is still a stay, and the all-cause
        hospital-wide rate is still the right thing to hold it against. This is the
        difference the fallback makes, and it is why no inpatient can end at no table."""
        predict = _predict(90000023)
        assert turn_phenotype([predict], [predict]) == GENERAL_INPATIENT

    def test_a_turn_about_no_admission_resolves_nothing(self):
        """A condition-only search has no stay to describe, so it gets no comparison —
        and this is the case where an empty answer is the correct one."""
        literature = [{"name": "search_literature", "args": {"query": "heart failure"},
                       "response": {"query": "heart failure", "returned": 0,
                                    "articles": []}}]
        assert turn_phenotype(literature, literature) == ""

    def test_a_chart_in_the_replay_resolves_the_condition(self):
        """A `/pubmed` turn reads no note of its own. The condition it compares against is
        the one the conversation established."""
        notes = [{"name": "rag_search_sections", "args": {"hadm_id": 90000023},
                  "response": {"passages": [{
                      "section": "discharge_diagnosis",
                      "text": DIAGNOSIS["heart_failure_cardiorenal"]}]}}]
        literature = [{"name": "search_literature", "args": {"query": "heart failure"},
                       "response": {"query": "heart failure", "returned": 0,
                                    "articles": []}}]

        assert turn_phenotype(literature, [*notes, *literature]) \
            == "heart_failure_cardiorenal"

    def test_an_unclassifiable_chart_in_an_inpatient_still_compares(self):
        notes = [{"name": "rag_search_sections", "args": {"hadm_id": 90000023},
                  "response": {"passages": [{"section": "discharge_diagnosis",
                                             "text": UNCLASSIFIABLE}]}}]
        assert turn_phenotype(notes, notes) == GENERAL_INPATIENT
