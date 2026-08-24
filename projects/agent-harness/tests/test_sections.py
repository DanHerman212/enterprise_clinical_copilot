"""Tests for discharge-note section parsing.

The regression test in here (`test_social_history_is_not_truncated_by_subfields`)
is the reason this module was written. An earlier regex extracted Social History
by stopping at the next line matching `[A-Z][A-Za-z ]{2,40}:`, which MIMIC's own
sub-fields - "Marital status:", "Tobacco:" - satisfy. Sections silently
truncated to their first line, and the resulting redaction rate was both
confidently reported and wrong. A wrong number that looks fine is this project's
recurring failure mode; the point of these tests is to make that particular one
impossible to reintroduce.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag import sections  # noqa: E402
from rag.sections import (  # noqa: E402
    BRIEF_HOSPITAL_COURSE,
    DISCHARGE_CONDITION,
    DISCHARGE_INSTRUCTIONS,
    DISCHARGE_MEDICATIONS,
    FAMILY_HISTORY,
    FOLLOWUP_INSTRUCTIONS,
    PHYSICAL_EXAM,
    SOCIAL_HISTORY,
    parse_note,
    redaction_profile,
)

# A note shaped like the real thing, including the traps: a Social History whose
# first line is a redaction placeholder followed by real content, sub-fields that
# look like headings, a numbered medication list, and a repeated Physical Exam.
NOTE = """
Name:  ___                     Unit No:   ___

Admission Date:  ___              Discharge Date:   ___

Date of Birth:  ___             Sex:   M

Service: MEDICINE

Allergies:
Penicillin

Attending: ___.

Chief Complaint:
Shortness of breath

Major Surgical or Invasive Procedure:
None

History of Present Illness:
Mr. ___ is a ___ year old man with CHF who presents with dyspnea.

Past Medical History:
CHF, diabetes mellitus type 2

Social History:
___
Marital status:     Widowed
Tobacco:            Current smoker, 1 ppd
Alcohol:            Denies
Occupation:         Retired
Lives alone with no caregiver at home.

Family History:
Father had CHF.

Physical Exam:
Vitals: T 98.6, BP 120/70
General: Well appearing

Pertinent Results:
___ 07:15AM BLOOD WBC-8.0

Brief Hospital Course:
Patient was diuresed. He missed several outpatient appointments prior to
admission and I have some concern about his ability to manage at home.

Discharge Exam:
Vitals: T 98.4, BP 118/72

Discharge Medications:
1. Furosemide 40 mg PO daily
2. Metoprolol 25 mg PO BID

Discharge Disposition:
Home

Discharge Diagnosis:
Acute on chronic systolic heart failure

Discharge Condition:
Mental Status: Clear and coherent.
Level of Consciousness: Alert and interactive.
Activity Status: Ambulatory - requires assistance.

Discharge Instructions:
Weigh yourself every morning. Call your doctor if you gain 3 lbs.

Followup Instructions:
___
"""


# --- the regression that motivated the module --------------------------------


def test_social_history_is_not_truncated_by_subfields():
    """Sub-field lines must stay inside the section body, not end it."""
    body = parse_note(NOTE).body(SOCIAL_HISTORY)

    # The old regex returned only the placeholder, or only "Marital status:".
    assert "Tobacco:" in body
    assert "Current smoker" in body
    assert "Occupation:" in body
    assert "Lives alone with no caregiver at home." in body


def test_social_history_with_placeholder_first_line_is_not_called_redacted():
    """The exact misclassification behind the bad corpus-wide redaction rate."""
    body = parse_note(NOTE).body(SOCIAL_HISTORY)
    profile = redaction_profile(body)

    assert profile.placeholders >= 1  # the leading ___ is still there
    assert not profile.is_placeholder_only  # but real content survives
    assert profile.informative_chars > 50


def test_discharge_condition_subfields_are_also_preserved():
    """Same trap, different section - Mental Status:, Activity Status:, etc."""
    body = parse_note(NOTE).body(DISCHARGE_CONDITION)

    assert "Clear and coherent" in body
    assert "Ambulatory - requires assistance" in body


# --- ordinary parsing --------------------------------------------------------


def test_expected_sections_are_found():
    parsed = parse_note(NOTE)
    found = {s.name for s in parsed.sections}

    for expected in (
        SOCIAL_HISTORY,
        FAMILY_HISTORY,
        BRIEF_HOSPITAL_COURSE,
        DISCHARGE_CONDITION,
        DISCHARGE_INSTRUCTIONS,
        FOLLOWUP_INSTRUCTIONS,
    ):
        assert expected in found


def test_section_bodies_stop_at_the_next_known_heading():
    parsed = parse_note(NOTE)

    family = parsed.body(FAMILY_HISTORY)
    assert family == "Father had CHF."
    assert "Vitals" not in family  # did not run on into Physical Exam


def test_numbered_lines_are_not_treated_as_headings():
    body = parse_note(NOTE).body(DISCHARGE_MEDICATIONS)

    assert "Furosemide" in body
    assert "Metoprolol" in body  # section did not end at the first list item


def test_repeated_headings_are_all_kept():
    """A note carries both an admission and a discharge exam."""
    exams = parse_note(NOTE).get_all(PHYSICAL_EXAM)

    assert len(exams) == 2
    assert "T 98.6" in exams[0].body
    assert "T 98.4" in exams[1].body


def test_get_returns_the_first_match():
    parsed = parse_note(NOTE)

    assert parsed.get(PHYSICAL_EXAM) is parsed.get_all(PHYSICAL_EXAM)[0]


def test_unknown_headings_are_reported_not_acted_on():
    parsed = parse_note(NOTE)
    unknown = {h.lower() for h in parsed.unknown_headings}

    # Recorded, so a genuinely missing section header is discoverable...
    assert "marital status" in unknown
    assert "tobacco" in unknown
    assert "vitals" in unknown
    # ...but they did not create sections.
    assert all(s.name in sections.KNOWN_HEADINGS for s in parsed.sections)


def test_coverage_is_high_for_a_well_formed_note():
    parsed = parse_note(NOTE)

    assert parsed.coverage > 0.95
    assert parsed.length == len(NOTE)


def test_missing_section_is_reported_as_absent():
    parsed = parse_note("Chief Complaint:\nChest pain\n")

    assert parsed.get(SOCIAL_HISTORY) is None
    assert parsed.body(SOCIAL_HISTORY) == ""
    assert parsed.body(SOCIAL_HISTORY, default="(absent)") == "(absent)"
    assert SOCIAL_HISTORY not in parsed


# --- heading normalisation ---------------------------------------------------


@pytest.mark.parametrize(
    "heading",
    ["Followup Instructions", "Follow-up Instructions", "FOLLOW-UP INSTRUCTIONS"],
)
def test_heading_aliases_fold_to_one_canonical_name(heading):
    parsed = parse_note(f"{heading}:\nSee your PCP in 1 week.\n")

    assert parsed.body(FOLLOWUP_INSTRUCTIONS) == "See your PCP in 1 week."


def test_long_form_discharge_instructions_header_folds_to_canonical():
    """MTSamples long-form header (hadm 90000015) must parse as discharge
    instructions, not bury the block inside discharge_diagnosis."""
    note = ("DISCHARGE MEDICATIONS:\nTylenol 650 mg q.6h.\n\n"
            "INSTRUCTIONS GIVEN TO THE PATIENT AT THE TIME OF DISCHARGE:\n"
            "Continue Tylenol and follow up in one week.\n")
    parsed = parse_note(note)
    assert parsed.body(DISCHARGE_INSTRUCTIONS).startswith("Continue Tylenol")
    assert "Tylenol 650 mg" not in parsed.body(DISCHARGE_INSTRUCTIONS)


def test_leading_whitespace_before_a_heading_is_tolerated():
    parsed = parse_note("   Social History:\nLives alone.\n")

    assert parsed.body(SOCIAL_HISTORY) == "Lives alone."


def test_duplicate_alias_is_rejected_at_build_time(monkeypatch):
    """Two sections claiming one heading would make parsing order-dependent."""
    monkeypatch.setattr(
        sections,
        "KNOWN_HEADINGS",
        {"first": ("Social History",), "second": ("social  history",)},
    )

    with pytest.raises(ValueError, match="claimed by both"):
        sections._build_lookup()


# --- degenerate input --------------------------------------------------------


def test_empty_note():
    parsed = parse_note("")

    assert parsed.sections == ()
    assert parsed.coverage == 0.0
    assert parsed.length == 0


def test_note_with_no_recognised_headings_reports_zero_coverage():
    """A parse failure has to be visible, not silent."""
    parsed = parse_note("Some free text with no structure at all.\n")

    assert parsed.sections == ()
    assert parsed.coverage == 0.0


def test_crlf_line_endings_are_normalised():
    parsed = parse_note("Social History:\r\nLives alone.\r\n\r\nFamily History:\r\nNC\r\n")

    assert parsed.body(SOCIAL_HISTORY) == "Lives alone."
    assert parsed.body(FAMILY_HISTORY) == "NC"


# --- redaction measurement ---------------------------------------------------


def test_redaction_profile_of_placeholder_only_text():
    profile = redaction_profile("___")

    assert profile.is_placeholder_only
    assert not profile.is_empty
    assert profile.informative_chars == 0


def test_redaction_profile_of_absent_text():
    profile = redaction_profile("   \n  ")

    assert profile.is_empty
    assert not profile.is_placeholder_only


def test_redaction_profile_of_mixed_text():
    profile = redaction_profile("Marital status: Married\nTobacco: ___")

    assert not profile.is_placeholder_only
    assert profile.placeholders == 1
    assert profile.informative_chars > 0
