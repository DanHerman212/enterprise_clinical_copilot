"""The clinician-UX remediation of 2026-11-08, pinned at each seam.

Four defects were reported against the served product, and this module holds one focused
test per layer the fix touched:

  1. a paragraph about one note section wore the same pill on every sentence and the
     marker's hover card opened the whole 400-word section. The collapse that answered the
     pills was withdrawn in Stage 1 (2026-11-08) — per-sentence attribution is the prompt's
     own contract, and each removal raised a visible safety swap over untouched prose — so
     what this module pins here is the hover card;
  2. `/ehr`'s literature comparison matched a population on condition alone, so a rate
     published for octogenarians sat beside a 41-year-old's estimate;
  3. `/meds` opened with a statement about the note's headings and listed IV morphine, an
     inpatient Decadron course and the take-home prescriptions in one table;
  4. a free-form follow-up on turn 2 could be answered only by restating the record, and
     an auxiliary empty retrieval replaced a grounded answer with a fixed absence.

Each test is narrow on purpose: the rule is the one the layer owns, and the assertions are
the observable consequence a reader would see.
"""

from __future__ import annotations

import pytest

from services.agent.engine import deterministic
from services.agent.engine import prompts
from services.agent.engine.deterministic import parse_medications
from services.agent.postprocess import answer as answer_text
from services.agent.postprocess import citations
from services.agent.postprocess.evidence_state import SessionEvidenceState

HADM = 90000069


# --------------------------------------------------------------------------------------
# 1. Every sentence keeps its own pill, and a hover card that shows the claim.
# --------------------------------------------------------------------------------------


def test_four_sentences_on_one_passage_keep_their_own_markers():
    """One paragraph, one section, four claims: four pills, and no guardrail flag.

    The paragraph-scoped collapse of 2026-11-08 removed three of the four markers and
    reported each removal as `citation_repeat_collapsed`, which the BFF reads as withdrawn
    prose — a "Reconciled with medical record" badge over an untouched answer. What this
    module keeps from that report is the hover card: the badge opens the claim, not the
    section (Stage 1, 2026-11-08).
    """
    answer = (
        "She was admitted with cervical spondylosis ^[4]. "
        "An anterior fusion was performed the next morning ^[4]. "
        "The postoperative course was uncomplicated ^[4]. "
        "She was discharged home on postoperative day two ^[4]."
    )

    collapsed, flags = citations.collapse_repeated_citations(answer)

    assert collapsed == answer
    assert flags == []


def test_a_paragraph_break_still_opens_a_new_citation():
    """Two paragraphs are two claims, however recently the passage was cited."""
    answer = ("Her course was uncomplicated ^[2].\n\n"
              "The discharge diagnosis was cervical spondylosis ^[2].")

    collapsed, flags = citations.collapse_repeated_citations(answer)

    assert collapsed == answer
    assert flags == []


#: A hospital course long enough to be the wall the hover card showed.
_LONG_COURSE = " ".join(
    f"Sentence {index} records an uneventful postoperative day with stable vitals and no "
    f"new complaint raised by the patient or the nursing staff on that shift."
    for index in range(1, 21)
) + " The patient was discharged home on postoperative day two on a tapering steroid dose."


def test_a_focused_excerpt_is_bounded_and_centred_on_the_claim():
    from services.agent.presentation.a2ui import _focused_note_excerpt

    assert len(_LONG_COURSE) > 2000
    claim = "She was discharged home on postoperative day two on a tapering steroid dose."

    excerpt = _focused_note_excerpt(_LONG_COURSE, claim)

    assert 0 < len(excerpt) <= 420
    # The sentence the claim was read from is IN the excerpt, which is the point of the
    # window: a hover card that opens the first 400 characters of a course shows the
    # admission rather than the discharge the reader asked about.
    assert "tapering steroid dose" in excerpt
    assert excerpt.startswith("…") or excerpt.endswith("…")


def test_an_excerpt_with_no_claim_context_is_still_bounded():
    from services.agent.presentation.a2ui import _focused_note_excerpt

    excerpt = _focused_note_excerpt(_LONG_COURSE, "")

    assert 0 < len(excerpt) <= 420
    # It stops at a sentence end or is marked as cut; either way it is an opening.
    assert excerpt.endswith("…") or excerpt.endswith(".")


def test_a_short_section_is_returned_whole():
    from services.agent.presentation.a2ui import _focused_note_excerpt

    assert _focused_note_excerpt("Discharge diagnosis: cervical spondylosis.", "anything") == (
        "Discharge diagnosis: cervical spondylosis.")


# --------------------------------------------------------------------------------------
# 2. The comparison is a population match, not only a condition match.
# --------------------------------------------------------------------------------------


def test_an_elderly_subgroup_is_refused_for_a_patient_under_65():
    assert deterministic._cohort_matches_patient_age(
        "Elderly patients (age >=65 years)", "", 41) is False


def test_the_same_subgroup_is_a_valid_benchmark_for_an_82_year_old():
    assert deterministic._cohort_matches_patient_age(
        "Elderly patients (age >=65 years)", "", 82) is True


def test_a_general_adult_cohort_is_accepted_for_either_age():
    cohort = "1,204 patients treated 2010-2017 at a single institution."

    assert deterministic._cohort_matches_patient_age(cohort, "", 41) is True
    assert deterministic._cohort_matches_patient_age(cohort, "", 82) is True


def test_an_unknown_age_filters_nothing():
    """A test that guessed would trade a wrong cohort for no cohort at all."""
    assert deterministic._cohort_matches_patient_age(
        "Elderly patients (age >=65 years)", "", None) is True


def test_a_young_adult_cohort_is_refused_for_an_older_patient():
    assert deterministic._cohort_matches_patient_age(
        "Patients aged 18 to 40 years", "", 78) is False


ACDF_ARTICLE = {
    "pmid": "27246481",
    "title": "Thirty-day readmission after anterior cervical discectomy and fusion.",
    "journal": "Spine",
    "pub_date": "2016 May",
    "authors": "Nakamura B, Osei K.",
    "abstract": (
        "BACKGROUND: Anterior cervical discectomy and fusion is increasingly performed in "
        "older adults. RESULTS: Elderly patients (age >=65 years) had a 30-day readmission "
        "rate of 9.2%. CONCLUSIONS: Age was the strongest predictor."
    ),
    "url": "https://pubmed.ncbi.nlm.nih.gov/27246481/",
}

ACDF_ROW = {
    "pmid": "27246481",
    "clinical_match": True,
    "rationale": "Same operation: anterior cervical discectomy and fusion.",
    "horizon_days": 30,
    "reported_rate": "9.2%",
    "rate_sentence": "Elderly patients (age >=65 years) had a 30-day readmission rate of 9.2%.",
    "cohort_summary": "Elderly patients (age >=65 years) undergoing ACDF.",
    "evidence_tier": "Retrospective Cohort",
}

ACDF_41 = deterministic.ClinicalTarget(
    condition="cervical spondylotic myelopathy",
    procedure="anterior cervical discectomy and fusion",
    anchors=("cervical spondylotic myelopathy", "anterior cervical discectomy and fusion"),
    patient_age=41,
    patient_sex="female",
)

ACDF_82 = deterministic.ClinicalTarget(
    condition="cervical spondylotic myelopathy",
    procedure="anterior cervical discectomy and fusion",
    anchors=("cervical spondylotic myelopathy", "anterior cervical discectomy and fusion"),
    patient_age=82,
    patient_sex="female",
)


@pytest.mark.parametrize("pmid", ["27246481"])
def test_pmid_27246481_is_rejected_for_a_41_year_old(pmid):
    """The reported case: an elderly-only rate beside a 41-year-old's estimate."""
    verified, reason = deterministic.verify_comparator(
        ACDF_ROW, {pmid: ACDF_ARTICLE}, ACDF_41)

    assert verified is None
    assert "age-restricted" in reason


def test_the_same_row_is_verified_for_an_82_year_old():
    verified, reason = deterministic.verify_comparator(
        ACDF_ROW, {"27246481": ACDF_ARTICLE}, ACDF_82)

    assert reason == ""
    assert verified is not None
    assert verified["reported_rate"] == "9.2%"


def test_the_extraction_prompt_is_told_the_patients_demographics():
    text = prompts.literature_comparator_input(ACDF_41, [])

    assert "INDEX PATIENT DEMOGRAPHICS: age=41, sex=female" in text


def test_the_extraction_prompt_states_the_bracket_rule():
    prompt = " ".join(prompts.LITERATURE_COMPARATOR_PROMPT.split())

    assert "AGE-BRACKET COMPATIBILITY" in prompt
    assert "when this patient is UNDER 65, refuse cohorts of elderly" in prompt
    assert "PATHOLOGY OR PROCEDURE ALIGNMENT" in prompt


def test_demographics_are_read_from_the_manifest_first():
    context = "<admission_metadata>\n- age: 41\n- gender: Female\n</admission_metadata>"
    sections = {"brief_hospital_course": "A 78-year-old man was admitted."}

    assert deterministic.patient_demographics(context, sections) == (41, "female")


def test_demographics_fall_back_to_the_note_when_the_manifest_is_absent():
    sections = {"brief_hospital_course": "She is a 41-year-old female with neck pain."}

    assert deterministic.patient_demographics("", sections) == (41, "female")


def test_a_chart_that_states_neither_yields_an_unknown_age():
    assert deterministic.patient_demographics("", {"brief_hospital_course": "No age."}) == (
        None, "")


# --------------------------------------------------------------------------------------
# 3. `/meds` on a narrative note: direct prose, and no ward administration.
# --------------------------------------------------------------------------------------


#: The hospital course the reported `/meds` turn read, abridged to the medication clauses.
COURSE = (
    "Postoperatively she was managed with IV morphine for pain and an inpatient Decadron "
    "course for airway swelling. She was sent home with a Medrol Dosepak, and continued "
    "pain medication coverage with Darvocet and Flexeril for muscle spasms."
)


def test_the_fallback_instruction_asks_for_take_home_drugs_first():
    lowered = " ".join(prompts.MEDS_FALLBACK_FORMATTING.split()).lower()

    assert "going home on" in lowered
    # The exclusion is stated as a prohibition rather than left to inference.
    assert "do not include an inpatient-only administration" in lowered
    assert "iv morphine" in lowered and "decadron" in lowered
    # And the inpatient list is still reachable when the record holds nothing else.
    assert "only if the passage names no discharge or take-home medication" in lowered


def test_the_fallback_narrative_states_the_absence_and_names_the_drugs():
    """Reversed 2026-11-10: the absence IS stated, and the drugs are attributed by setting.

    The earlier wording avoided naming the document's headings on the grounds that it read
    as an apology. The medical-risk defect that replaced it was worse: a reader told only
    "documented during the hospital course" could not tell a take-home prescription from a
    ward administration, and an extractor that labelled every row the same way filled the
    home grid with IV morphine. The sentence now says the record has no dedicated section
    and separates what continues at discharge from what was only given during the stay.
    """
    rows = [{"medication": name, "dosage": "", "frequency": "", "route": "",
             "setting": "discharge"}
            for name in ("Medrol", "Darvocet", "Flexeril")]

    text = answer_text.medication_narrative(
        rows, 1, sections=("brief_hospital_course",))

    assert text == (
        "No dedicated discharge medications section is documented in the record. "
        "Documented to continue at discharge: Medrol, Darvocet, Flexeril ^[1].")
    assert "does not contain a dedicated discharge medications section" not in text
    assert "rather than a formal discharge medication list" not in text


def test_a_stated_route_survives_verification_and_an_invented_one_does_not():
    payload = (
        '{"medications": ['
        '{"medication": "morphine", "dosage": "", "frequency": "", "route": "IV"},'
        '{"medication": "Medrol", "dosage": "", "frequency": "", "route": "subcutaneous"}'
        "]}"
    )

    verified, _dropped = parse_medications(payload, COURSE)

    routes = {row["medication"]: row["route"] for row in verified}
    assert routes["morphine"] == "IV"
    assert routes["Medrol"] == "", "a route the passage never states is not carried"


def test_the_dedicated_path_is_unchanged():
    rows = [{"medication": "Celebrex", "dosage": "200 mg", "frequency": "daily"}]

    text = answer_text.medication_narrative(
        rows, 1, sections=("discharge_medications",))

    assert text.startswith("1 discharge medication, as recorded in the discharge note")


# --------------------------------------------------------------------------------------
# 4. A follow-up may explain, and an empty auxiliary search must not overwrite it.
# --------------------------------------------------------------------------------------


def test_the_hub_permits_grounded_clinical_explanation():
    hub = " ".join(prompts.CORE_HUB_PROMPT.split())

    assert "You DO explain" in hub
    assert "restating the numbers answers nothing" in hub
    # The two limits are still stated, because they are the safety boundary.
    assert "every PATIENT fact" in hub
    assert "never as a finding about this patient" in hub


def test_an_empty_auxiliary_search_does_not_replace_a_grounded_answer():
    """The interpretive follow-up: the record is held, the search adds nothing."""
    record = SessionEvidenceState(hadm_id=HADM)
    record.ingest_tool_output("get_lab_trends", {"hadm_id": HADM}, {
        "hadm_id": HADM,
        "synthetic": True,
        "trends": [{
            "key": "rdw", "label": "RDW", "unit": "%", "decimals": 1,
            "series": [{"day": 0.0, "value": 12.0, "flag": "normal"},
                       {"day": 3.0, "value": 15.6, "flag": "high"}],
        }],
    })
    empty = {"name": "rag_search", "args": {}, "response": {"returned": 0, "passages": []}}

    verdict = answer_text.no_evidence_answer(
        "Which of these labs is most concerning?",
        [empty], evidence_state=record)

    assert verdict is None, (
        "a session holding a panel has an answer to give; the empty search says only "
        "that the note corpus had no matching passage")


def test_a_declined_model_still_gets_the_fixed_sentence():
    record = SessionEvidenceState(hadm_id=HADM)
    record.ingest_tool_output("rag_search", {"hadm_id": HADM}, {
        "hadm_id": HADM, "returned": 1,
        "passages": [{"id": "p1", "section": "chief_complaint", "text": "Pain."}],
    })
    empty = {"name": "rag_search", "args": {}, "response": {"returned": 0, "passages": []}}

    verdict = answer_text.no_evidence_answer(
        "Was she intubated?", [empty], declined=True, evidence_state=record)

    assert verdict is not None
    assert "no information in the patient's chart" in verdict


def test_a_driver_analyte_is_not_reported_as_unread():
    """`rdw_max` is one of the estimate's own inputs, so the panel is not "unread"."""
    from services.agent.engine import graph

    record = SessionEvidenceState(hadm_id=HADM)
    record.ingest_tool_output("predict_readmission", {"hadm_id": HADM}, {
        "hadm_id": HADM,
        "probability_display": "17.6%",
        "decision": 1,
        "top_factors": [{"feature": "rdw_max", "direction": "increases"}],
    })
    state = {"question": "Why does her RDW increase readmission risk?",
             "allowed_tools": frozenset({"get_lab_trends", "rag_search"}),
             "evidence_state": record}

    assert graph._unqueried_analytes(state) == []


def test_an_analyte_the_session_never_touched_is_still_reported():
    from services.agent.engine import graph

    record = SessionEvidenceState(hadm_id=HADM)
    record.ingest_tool_output("predict_readmission", {"hadm_id": HADM}, {
        "hadm_id": HADM,
        "probability_display": "17.6%",
        "decision": 1,
        "top_factors": [{"feature": "rdw_max", "direction": "increases"}],
    })
    state = {"question": "What were her hemoglobin and creatinine trends?",
             "allowed_tools": frozenset({"get_lab_trends", "rag_search"}),
             "evidence_state": record}

    assert graph._unqueried_analytes(state) == ["hemoglobin", "creatinine"]
