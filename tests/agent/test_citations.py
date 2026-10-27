"""Tests for the citation layer: renumbering, remapping, intent resolution,
and section extraction.

These are the deterministic evidence semantics of the answer contract. They
live with the agent — the layer where the answer, the guardrails, and the
retrieved evidence meet — and the BFF passes the results through unchanged.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.agent.postprocess.citations import (  # noqa: E402
    SECTION_ALIASES,
    citation_remap,
    cited_numbers,
    dedupe_list_citations,
    extract_section,
    first_citation,
    intent_sections,
    renumber_citations,
)
from services.mcp.retrieval.sections import KNOWN_HEADINGS  # noqa: E402


class TestLiteratureCitations:
    r"""`^[PMID: 31234567]` must be invisible to the note-citation pipeline.

    Two citation schemes travel in one answer: note passages by number, and
    published articles by PMID badge. The renumberer rewrites `^[n]` to
    first-appearance order against the resolved note sources, and a PMID badge is
    not one of those.

    What keeps them apart is the prefix, not luck: `_CITATION_RE` is
    `\^\[(\d+...)\]`, so it requires a DIGIT where a badge has a letter. The tests
    below pin both halves — that a badge is skipped, and that the skipping is the
    prefix's doing rather than an accident of the digits involved.

    The failure this guards against is quiet and convincing: a renumbered or
    stripped PMID still renders as a plausible citation, and it points at the wrong
    paper rather than at nothing at all.
    """

    def test_a_pmid_badge_survives_renumbering(self):
        answer = "Ceftriaxone remains first line.^[PMID: 31234567]"

        assert renumber_citations(answer) == answer

    def test_several_pmids_in_one_badge_are_also_skipped(self):
        answer = "Two trials support this.^[PMID: 31234567, 39156338]"

        assert renumber_citations(answer) == answer

    def test_a_section_name_the_model_invented_is_removed_rather_than_printed(self):
        """`^[brief_hospital_course]` is not a citation, and it must not be shown.

        Seen on the demo screen 2026-10-17: the model cited a passage by its section
        slug instead of its number, no pass matched it, and the reader got the raw
        braces mid-sentence. A missing citation is recoverable; markup in prose reads
        as corruption.
        """
        answer = (
            "She used the CPM machine for longer sessions ^{0}[brief_hospital_course]. "
            "Overall she progressed well.^[3]"
        ).format("^")

        cleaned = renumber_citations(answer)

        assert "brief_hospital_course" not in cleaned
        assert "^[3]" in cleaned or "^[1]" in cleaned

    def test_a_real_pmid_badge_survives_the_unknown_marker_filter(self):
        """The filter keeps PMID badges; only inventions are dropped."""
        answer = "A trial agrees.^[PMID: 34793857] Another claim.^[made_up_section]"

        cleaned = renumber_citations(answer)

        assert "^[PMID: 34793857]" in cleaned
        assert "made_up_section" not in cleaned

    def test_a_pmid_badge_contributes_no_note_citation(self):
        answer = "The note documents this^[1] and a trial agrees.^[PMID: 31234567]"

        assert cited_numbers(answer) == [1]
        assert citation_remap(answer) == {"1": 1}

    def test_a_pmid_badge_does_not_disturb_note_numbering(self):
        """The note citation keeps its own sequence; the badge is not in it.

        The `^[3]` here belongs to a note passage the model numbered third, so it
        renumbers to `^[1]` on its own terms. If the badge were being counted, the
        numbering would come out differently — which is the point of asserting the
        exact string rather than just "the badge survived".
        """
        answer = "Published work^[PMID: 99] and the note^[3]."

        assert renumber_citations(answer) == (
            "Published work^[PMID: 99] and the note^[1]."
        )

    def test_a_bare_pmid_number_would_be_mistaken_for_a_note_citation(self):
        """Why the prefix is load-bearing rather than decorative.

        Drop `PMID:` and the digits parse as an n-numbered note citation, to be
        renumbered against passages that do not support them. This is the failing
        case the prompt rule exists to prevent, asserted so the rule is not
        mistaken for a style preference.
        """
        assert cited_numbers("See ^[31234567].") == [31234567]


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


class TestListCitationDeduplication:
    r"""One list, one marker.

    A ten-item list of discharge diagnoses arrived from the model with `^[3]` at the end
    of every item. The claim the marker supports is the LIST; eleven identical badges
    read as eleven sources, and a reader cannot tell that list from one where the model
    stopped checking. `dedupe_list_citations` moves the marker to the sentence that
    introduces the list and takes it off the items.

    Deterministic rather than prompted, because it is a shape and not a judgement: a
    contiguous list whose items all close with the SAME marker is a list attributed to one
    passage, and the move is the same every time. Mixed-source lists are left alone — that
    is a list making a per-item claim, and hoisting one marker would leave the rest
    unattributed.
    """

    def test_a_repeated_marker_is_hoisted_onto_the_lead_in(self):
        lead_in = "The discharge diagnoses recorded for this admission are:"
        items = "\n".join(f"{position}. Diagnosis {position} ^[3]"
                          for position in range(1, 11))
        answer = lead_in + "\n" + items

        lines = dedupe_list_citations(answer).splitlines()

        assert lines[0] == (
            "The discharge diagnoses recorded for this admission are ^[3]:"
        )
        assert all("^[" not in line for line in lines[1:]), "\n".join(lines)
        # The list itself is untouched apart from the markers.
        assert lines[1] == "1. Diagnosis 1"
        assert lines[10] == "10. Diagnosis 10"

    def test_with_no_lead_in_the_marker_stays_on_the_last_item(self):
        answer = "- Item one ^[2]\n- Item two ^[2]\n- Item three ^[2]"

        assert dedupe_list_citations(answer).splitlines() == [
            "- Item one", "- Item two", "- Item three ^[2]",
        ]

    def test_a_lead_in_that_already_cites_keeps_its_marker(self):
        answer = "Findings are summarised below ^[2]:\n- A ^[2]\n- B ^[2]"

        assert dedupe_list_citations(answer).splitlines() == [
            "Findings are summarised below ^[2]:", "- A", "- B",
        ]

    def test_a_list_citing_different_passages_is_left_alone(self):
        answer = "Findings:\n- A ^[1]\n- B ^[3]\n- C ^[1]"

        assert dedupe_list_citations(answer) == answer

    def test_a_list_whose_items_are_not_all_cited_is_left_alone(self):
        answer = "Findings:\n- A ^[3]\n- B"

        assert dedupe_list_citations(answer) == answer

    def test_a_single_item_list_is_not_a_list(self):
        answer = "The diagnosis:\n1. Sepsis ^[3]"

        assert dedupe_list_citations(answer) == answer

    def test_the_hoisted_marker_is_then_renumbered_and_remapped(self):
        """The seam the passes share: dedupe first, then renumber, then remap.

        A marker taken off the list is a marker that no longer takes a number, so a remap
        built from the un-deduped answer would shift every later citation by one.
        """
        answer = (
            "The findings are:\n"
            "- A ^[3]\n"
            "- B ^[3]\n"
            "The impression follows ^[7]."
        )

        renumbered = renumber_citations(answer).splitlines()
        assert renumbered[0] == "The findings are ^[1]:"
        # `^[7]` is the second distinct passage the answer cites, so it becomes `2`, and
        # the remap says which original passage each renumbered marker addresses.
        assert citation_remap(answer) == {"1": 3, "2": 7}
