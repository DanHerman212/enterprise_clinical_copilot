"""Tests for the citation layer: renumbering, remapping, intent resolution,
and section extraction.

These are the deterministic evidence semantics of the answer contract. They
live with the agent — the layer where the answer, the guardrails, and the
retrieved evidence meet — and the BFF passes the results through unchanged.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.agent.citations import (  # noqa: E402
    SECTION_ALIASES,
    citation_remap,
    extract_section,
    first_citation,
    intent_sections,
    renumber_citations,
)
from services.mcp.retrieval.sections import KNOWN_HEADINGS  # noqa: E402


class TestAliasConsolidation:
    """The section vocabulary derives from the canonical KNOWN_HEADINGS used
    by index build and serving — one vocabulary, one drift surface (S7-02)."""

    def test_aliases_derive_from_the_canonical_headings(self):
        for section in KNOWN_HEADINGS["brief_hospital_course"]:
            assert section in SECTION_ALIASES["brief_hospital_course"]

    def test_presentation_extras_are_merged(self):
        assert "Activity" in SECTION_ALIASES["activity"]
        assert "MRN" in SECTION_ALIASES["unit_no"]

    def test_long_form_instructions_header_is_known(self):
        assert (
            "Instructions Given to the Patient at the Time of Discharge"
            in SECTION_ALIASES["discharge_instructions"]
        )


class TestFirstCitation:
    def test_returns_the_first_marker_number(self):
        assert first_citation("A claim^[3] and another^[1]") == 3

    def test_defaults_to_one_without_markers(self):
        assert first_citation("no citations here") == 1
        assert first_citation("") == 1


class TestRenumberCitations:
    def test_meds_only_answer_renumbers_to_first(self):
        """A meds-only answer cites ^[3] (discharge_medications is the 3rd
        section in rag_search_sections order) — it must read as ^[1]."""
        assert renumber_citations(
            "The patient was discharged on the following medications^[3]:"
        ) == "The patient was discharged on the following medications^[1]:"

    def test_renumbers_by_order_of_first_appearance(self):
        assert renumber_citations("A^[2] and B^[1] and C^[3]") == (
            "A^[1] and B^[2] and C^[3]")

    def test_no_markers_is_unchanged(self):
        assert renumber_citations("no citations here") == "no citations here"

    def test_stacked_citations_collapse_to_one_marker(self):
        assert renumber_citations(
            "discharged on pain medication ^[1]^[2]^[3]^[4]^[5]."
        ) == "discharged on pain medication ^[1]."
        assert renumber_citations(
            "discharged on pain medication ^[1] ^[2] ^[3]."
        ) == "discharged on pain medication ^[1]."


class TestCitationRemap:
    def test_maps_renumbered_numbers_back_to_original_passages(self):
        # 'A^[2] and B^[1] and C^[3]' renumbers to 1,2,3 in appearance order,
        # so the map must translate back to the original positions (2,1,3).
        assert citation_remap("A^[2] and B^[1] and C^[3]") == {
            "1": 2, "2": 1, "3": 3}

    def test_no_markers_is_an_empty_map(self):
        assert citation_remap("no citations here") == {}


class TestIntentSections:
    def test_maps_medication_questions_to_the_meds_intent_set(self):
        assert intent_sections(
            "What medications was this patient discharged on?"
        ) == ("discharge_medications", "discharge_instructions")

    def test_maps_instructions_questions(self):
        assert intent_sections(
            "What were her discharge instructions? For admission 90000015."
        ) == ("discharge_instructions",)

    def test_maps_diagnosis_questions(self):
        assert intent_sections("list her diagnoses") == ("discharge_diagnosis",)

    def test_maps_hospital_course_questions(self):
        assert intent_sections("summarize the hospital course") == (
            "brief_hospital_course",)

    def test_summarize_and_risk_have_no_single_section_intent(self):
        assert intent_sections(
            "Summarize the recent discharge notes for this patient.") == ()
        assert intent_sections(
            "Assess the 30-day readmission risk for this patient.") == ()

    def test_empty_question_has_no_intent(self):
        assert intent_sections(None) == ()
        assert intent_sections("") == ()


class TestExtractSection:
    def test_bounds_at_allergies_and_activity_headers(self):
        """The meds source must not swallow trailing headers (Allergies,
        Activity) that the bounding vocabulary previously did not recognize."""
        note = ("DISCHARGE MEDICATIONS: Tylenol 650 mg q.6h., Lasix 80 mg "
                "daily.\n\n"
                "ALLERGIES: None.\n\n"
                "ACTIVITY: Per PT.\n\n"
                "FOLLOWUP INSTRUCTIONS: Call the office.")
        meds = extract_section(note, "discharge_medications")
        assert "Tylenol" in meds
        assert "ALLERGIES" not in meds
        assert "ACTIVITY" not in meds

    def test_handles_mtsamples_headers(self):
        """MTSamples notes use different headers than MIMIC canon ("HOSPITAL
        COURSE:", "DISCHARGE DIAGNOSES:"), and the SourceCard body must come
        from the cited section — not the whole note."""
        note = (
            "CHIEF COMPLAINT: Knee pain.\n\n"
            "HISTORY OF PRESENT ILLNESS: The patient is a 61-year-old female.\n\n"
            "HOSPITAL COURSE: She underwent a right total knee replacement and "
            "recovered well.\n\n"
            "DISCHARGE DIAGNOSES: 1. S/p right TKA.\n\n"
            "MEDICATIONS: Celebrex 200 mg daily.\n\n"
            "INSTRUCTIONS GIVEN TO THE PATIENT AT THE TIME OF DISCHARGE: "
            "Continue Celebrex for one month."
        )
        course = extract_section(note, "brief_hospital_course")
        assert course is not None
        assert "right total knee replacement" in course
        assert "CHIEF COMPLAINT" not in course
        assert "DISCHARGE DIAGNOSES" not in course

        dx = extract_section(note, "discharge_diagnosis")
        assert dx is not None
        assert "S/p right TKA" in dx
        assert "HOSPITAL COURSE" not in dx

        meds = extract_section(note, "discharge_medications")
        assert meds is not None
        assert "Celebrex" in meds

        instr = extract_section(note, "discharge_instructions")
        assert instr is not None
        assert "for one month" in instr
        assert "Celebrex 200 mg daily." not in instr

    def test_missing_section_returns_none(self):
        assert extract_section("CHIEF COMPLAINT: Knee pain.",
                               "discharge_medications") is None
