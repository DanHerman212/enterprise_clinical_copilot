"""Section parsing for MIMIC-IV discharge notes.

Why this module exists
----------------------
Discharge notes are semi-structured: a sequence of named sections
("Social History:", "Brief Hospital Course:", ...) whose bodies are free text.
Most of what follows depends on knowing where those boundaries fall.

  * The redaction probe (A1) needs per-section content, so it can report which
    sections are actually usable instead of one meaningless whole-note number.
  * Concept tagging (A3) scopes some concepts to specific sections - clinician
    hedging only counts in Discharge Condition and Brief Hospital Course, where
    a clinician is assessing the patient, not in Discharge Instructions, which
    are saturated with conditional boilerplate aimed at the patient.
  * Chunking (guide section 4) uses sections as the natural chunk boundary, so a
    retrieved passage can be cited as "Social History" rather than as
    "characters 8100-8600".

The bug this module is designed not to repeat
---------------------------------------------
The obvious implementation is one regex per section: start at the header, stop
at the next line that *looks like* a header.

    r"(?s)\\nSocial History:\\n(.*?)\\n[A-Z][A-Za-z ]{2,40}:"

That is wrong, and it was wrong here. MIMIC's Social History contains sub-fields
on their own lines - "Marital status:", "Tobacco:", "Alcohol:", "Occupation:" -
and every one of them matches the "looks like a header" shape. The section
truncates at the first sub-field, so a note whose first line is a redaction
placeholder but which has real content below reads as fully redacted. It
produced a confident, wrong, and conveniently alarming redaction rate.

The fix is to split only on headers we recognise. `KNOWN_HEADINGS` is an
explicit allowlist. A line that matches the generic heading shape but is absent
from that list is *recorded* and otherwise ignored, so it stays part of the
enclosing section's body where it belongs.

That trades one failure mode for a milder one: a real section header we forgot
to list gets absorbed into the preceding section rather than splitting it. So
`ParsedNote` reports `unknown_headings` and `coverage`, and the A1 probe
aggregates both across the corpus. A forgotten header then shows up as a ranked,
countable item instead of as silence. Under-splitting is also the safer error
for a gate: it dilutes sections rather than truncating them.

Dependencies
------------
Pure standard library, deliberately. This module is imported both by the harness
venv (numpy 1.26, xgboost) and by the medspaCy venv (numpy 2.x), so it must not
drag either one in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --- canonical section names -------------------------------------------------
# Call sites refer to sections by these constants, never by raw heading text, so
# that adding an alias never means touching a call site.

NAME = "name"
UNIT_NO = "unit_no"
ADMISSION_DATE = "admission_date"
DISCHARGE_DATE = "discharge_date"
DATE_OF_BIRTH = "date_of_birth"
SEX = "sex"
SERVICE = "service"
ALLERGIES = "allergies"
ATTENDING = "attending"
CHIEF_COMPLAINT = "chief_complaint"
MAJOR_PROCEDURE = "major_procedure"
HISTORY_OF_PRESENT_ILLNESS = "history_of_present_illness"
REVIEW_OF_SYSTEMS = "review_of_systems"
PAST_MEDICAL_HISTORY = "past_medical_history"
PAST_SURGICAL_HISTORY = "past_surgical_history"
SOCIAL_HISTORY = "social_history"
FAMILY_HISTORY = "family_history"
PHYSICAL_EXAM = "physical_exam"
PERTINENT_RESULTS = "pertinent_results"
BRIEF_HOSPITAL_COURSE = "brief_hospital_course"
MEDICATIONS_ON_ADMISSION = "medications_on_admission"
DISCHARGE_MEDICATIONS = "discharge_medications"
DISCHARGE_DISPOSITION = "discharge_disposition"
DISCHARGE_DIAGNOSIS = "discharge_diagnosis"
DISCHARGE_CONDITION = "discharge_condition"
DISCHARGE_INSTRUCTIONS = "discharge_instructions"
DISCHARGE_SUMMARY = "discharge_summary"
FOLLOWUP_INSTRUCTIONS = "followup_instructions"
FACILITY = "facility"

# Headings that introduce each section. Written as they read; normalisation
# below makes matching case- and punctuation-insensitive, so "Follow-Up
# Instructions" and "followup instructions" both land on FOLLOWUP_INSTRUCTIONS.
KNOWN_HEADINGS: dict[str, tuple[str, ...]] = {
    NAME: ("Name",),
    UNIT_NO: ("Unit No",),
    ADMISSION_DATE: ("Admission Date",),
    DISCHARGE_DATE: ("Discharge Date",),
    DATE_OF_BIRTH: ("Date of Birth",),
    SEX: ("Sex",),
    SERVICE: ("Service",),
    ALLERGIES: ("Allergies",),
    ATTENDING: ("Attending",),
    CHIEF_COMPLAINT: (
        "Chief Complaint",
        # MTSamples uses the reason for admission as its opening statement.
        "Reason for Admission",
    ),
    MAJOR_PROCEDURE: (
        "Major Surgical or Invasive Procedure",
        "Major Surgical or Invasive Procedures",
        # MTSamples procedures sections (deliberate: a procedures block is the
        # closest thing these notes have to the major-procedure line).
        "Procedure",
        "Procedures",
        "Procedures Performed",
        "Operations Performed",
        "Principal Procedure",
        "Principal Procedures",
        "Procedure Performed During This Hospitalization",
        "Procedures During This Hospitalization",
        "Procedures During Hospitalization",
        "Operations and Procedures",
    ),
    HISTORY_OF_PRESENT_ILLNESS: (
        "History of Present Illness", "HPI",
        # MTSamples opens its narrative with a bare "History:"/"Brief History:"
        # (or a plain "History of Illness:"). Deliberate: the body is the HPI
        # narrative, not a structured past-medical-history block.
        "History",
        "History of Illness",
        "Brief History",
        "Brief History of Present Illness",
        "Current History",
    ),
    REVIEW_OF_SYSTEMS: ("Review of Systems", "ROS"),
    PAST_MEDICAL_HISTORY: (
        "Past Medical History", "PMH",
        "Past History",
        "Past Medical/Family/Social History",
        "Past Medical, Family, Social History",
    ),
    PAST_SURGICAL_HISTORY: (
        "Past Surgical History",
        "Surgical History",
    ),
    SOCIAL_HISTORY: ("Social History",),
    FAMILY_HISTORY: ("Family History",),
    PHYSICAL_EXAM: (
        "Physical Exam",
        "Physical Examination",
        "Admission Exam",
        "Admission Physical Exam",
        "Discharge Exam",
        "Discharge Physical Exam",
        # MTSamples discharge-focused physical exam headings.
        "Discharge Physical Examination",
        "Physical Examination at the Time of Discharge",
        "Physical Examination on Discharge",
    ),
    PERTINENT_RESULTS: (
        "Pertinent Results", "Pertinent Labs",
        # MTSamples laboratory blocks are the pertinent-results equivalent.
        "Laboratory Data",
        "Laboratory Studies",
        "Laboratory",
        "Pertinent Laboratories",
        "Discharge Labs",
        "Laboratories on Admission",
        "Significant Labs and X-Rays",
        "Additional Laboratory Studies",
    ),
    BRIEF_HOSPITAL_COURSE: (
        "Brief Hospital Course", "Hospital Course",
        "Course in the Hospital",
        "History and Hospital Course",
        "Brief Hospital Course Summary",
        "Brief Summary of Hospital Course",
        # MTSamples: "Course on Admission" is the admission hospital course.
        "Course on Admission",
    ),
    MEDICATIONS_ON_ADMISSION: ("Medications on Admission",),
    DISCHARGE_MEDICATIONS: (
        "Discharge Medications",
        # MTSamples: a bare "Medications:" in a discharge summary is the
        # discharge list (admission meds are labelled "Medications on
        # Admission:" and hit MEDICATIONS_ON_ADMISSION instead).
        "Medications",
        "Medications on Discharge",
        "Home Medications",
        "New Medications",
        "Current Medications",
        "Medications and Advice on Discharge",
        "Discharge Medications/Instructions",
    ),
    DISCHARGE_DISPOSITION: (
        "Discharge Disposition",
        # MTSamples heading for where the patient went on discharge.
        "Disposition",
    ),
    DISCHARGE_DIAGNOSIS: (
        "Discharge Diagnosis", "Discharge Diagnoses",
        # MTSamples lists admission/admitting/secondary/final diagnoses; these
        # all feed the same diagnosis picture for our purposes.
        "Admission Diagnosis",
        "Admission Diagnoses",
        "Admitting Diagnosis",
        "Admitting Diagnoses",
        "Secondary Diagnosis",
        "Secondary Diagnoses",
        "Diagnoses on Admission",
        "Diagnoses on Discharge",
        "Primary Diagnoses",
        "Final Diagnosis",
        "Final Diagnoses",
        # MTSamples: "Diagnosis at Admission" and a bare "Diagnoses" block.
        "Diagnosis at Admission",
        "Diagnoses",
    ),
    DISCHARGE_CONDITION: (
        "Discharge Condition",
        "Condition",
        "Condition on Discharge",
        "Conditions on Discharge",
        "Condition Upon Discharge",
        "Condition at Discharge",
        "Condition of Patient on Discharge",
        "Condition of the Patient at Discharge",
    ),
    DISCHARGE_INSTRUCTIONS: (
        "Discharge Instructions",
        # MTSamples discharge plan/instructions equivalents. Deliberate: a
        # "Discharge Plan:" block is a plan of care (meds, diet, activity,
        # followup) — instructions content, not a condition statement.
        "Discharge Plan",
        "Additional Instructions",
        "Special Instructions",
        "Instructions to Patient",
        "Discharge Diet",
        "Discharge Activities",
        "Physical Activity",
        "Recommendations",
        "Discharge Instructions/Medications",
        # MTSamples 2788-style long-form header (site alias already knows it;
        # adopting it here means parse_note/chunking treat it as discharge
        # instructions instead of burying it inside another section).
        "Instructions Given to the Patient at the Time of Discharge",
    ),
    DISCHARGE_SUMMARY: (
        "Discharge Summary",
        "Discharge Summaries",
    ),
    FOLLOWUP_INSTRUCTIONS: (
        "Followup Instructions", "Follow-up Instructions",
        # MTSamples followup headings (plural/abbreviated forms).
        "Followup",
        "Follow Up",
        "Follow-Up",
        "Followup Appointments",
        "Instructions for Followup",
    ),
    FACILITY: ("Facility",),
}

# A line that could be a heading: starts a line, a handful of words, then a
# colon. Matching this is necessary but not sufficient - see module docstring.
# The first character must be a letter, which excludes numbered list items
# ("1. Diabetes: controlled") without needing a special case.
_HEADING_RE = re.compile(
    r"^[ \t]*([A-Za-z][A-Za-z0-9 /&'\-]{1,60}?)[ \t]*:",
    re.MULTILINE,
)

# A full line that is exactly a known heading with NO trailing colon. Used for
# the MTSamples numbered-list format ("DISCHARGE DIAGNOSES\n1. ...") where a
# recognised heading carries no colon and _HEADING_RE would miss it.
_BARE_HEADING_RE = re.compile(
    r"^[ \t]*([A-Za-z][A-Za-z0-9 /&'\-]{1,60})[ \t]*$",
    re.MULTILINE,
)

# Runs of underscores are MIMIC's de-identification placeholder.
_PLACEHOLDER_RE = re.compile(r"_{2,}")

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _normalise_heading(heading: str) -> str:
    """Fold a raw heading to its lookup key: lowercase, alphanumerics only.

    "Follow-Up  Instructions" and "follow up instructions" both fold to
    "follow up instructions".
    """
    return _NON_ALNUM_RE.sub(" ", heading.lower()).strip()


def _build_lookup() -> dict[str, str]:
    """Invert KNOWN_HEADINGS into {normalised heading: canonical name}.

    Raises on a duplicate alias. Two sections claiming one heading would make
    parsing depend on dict ordering, which is exactly the kind of quiet
    arbitrariness this module exists to avoid.
    """
    lookup: dict[str, str] = {}
    for canonical, headings in KNOWN_HEADINGS.items():
        for heading in headings:
            key = _normalise_heading(heading)
            if key in lookup and lookup[key] != canonical:
                raise ValueError(
                    f"heading {heading!r} is claimed by both "
                    f"{lookup[key]!r} and {canonical!r}"
                )
            lookup[key] = canonical
    return lookup


_HEADING_LOOKUP = _build_lookup()


@dataclass(frozen=True)
class Section:
    """One parsed section.

    `start` and `end` are offsets into the *normalised* note text (line endings
    folded to "\\n"), spanning the heading through the end of the body.
    """

    name: str
    heading: str
    body: str
    start: int
    end: int


@dataclass(frozen=True)
class ParsedNote:
    """The result of parsing one note.

    `unknown_headings` and `coverage` exist so that a parse failure is loud.
    A note that comes back with no sections and 0.0 coverage is visibly broken;
    a corpus-wide tally of unknown headings tells us which real headings are
    missing from KNOWN_HEADINGS.
    """

    sections: tuple[Section, ...]
    unknown_headings: tuple[str, ...]
    coverage: float
    length: int

    def get(self, name: str) -> Section | None:
        """First section with this canonical name, or None.

        "First" matters for headings that legitimately repeat: a note often
        carries both an admission and a discharge physical exam. Use `get_all`
        when the distinction matters.
        """
        for section in self.sections:
            if section.name == name:
                return section
        return None

    def get_all(self, name: str) -> tuple[Section, ...]:
        return tuple(s for s in self.sections if s.name == name)

    def body(self, name: str, default: str = "") -> str:
        """Body text of the first matching section, or `default` if absent."""
        section = self.get(name)
        return section.body if section is not None else default

    def __contains__(self, name: object) -> bool:
        return any(s.name == name for s in self.sections)


def parse_note(text: str) -> ParsedNote:
    """Split a discharge note into its named sections.

    Only headings in KNOWN_HEADINGS split the text. Heading-shaped lines that
    are not recognised are collected into `unknown_headings` and left inside the
    body of the section that contains them.
    """
    if not text:
        return ParsedNote(sections=(), unknown_headings=(), coverage=0.0, length=0)

    # Fold line endings first so that offsets and the "^" anchor agree.
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")

    known: list[tuple[re.Match[str], str]] = []
    unknown: list[str] = []
    seen_spans: list[tuple[int, int]] = []
    for match in _HEADING_RE.finditer(normalised):
        raw = match.group(1).strip()
        canonical = _HEADING_LOOKUP.get(_normalise_heading(raw))
        if canonical is None:
            unknown.append(raw)
        else:
            known.append((match, canonical))
            seen_spans.append((match.start(), match.end()))

    # Second pass for the numbered-list format: a full line that is exactly a
    # known heading (no colon) is a heading. Unknown bare lines stay in the
    # surrounding body, preserving the allowlist design.
    for match in _BARE_HEADING_RE.finditer(normalised):
        if any(match.start() < end and match.end() > start
               for start, end in seen_spans):
            continue
        canonical = _HEADING_LOOKUP.get(_normalise_heading(match.group(1).strip()))
        if canonical is None:
            continue
        known.append((match, canonical))
        seen_spans.append((match.start(), match.end()))
    known.sort(key=lambda mc: mc[0].start())

    sections: list[Section] = []
    for index, (match, canonical) in enumerate(known):
        body_start = match.end()
        # The section runs to the next *recognised* heading, not the next
        # heading-shaped line. This is the whole point of the module.
        if index + 1 < len(known):
            body_end = known[index + 1][0].start()
        else:
            body_end = len(normalised)
        sections.append(
            Section(
                name=canonical,
                heading=match.group(1).strip(),
                body=normalised[body_start:body_end].strip(),
                start=match.start(),
                end=body_end,
            )
        )

    covered = sum(s.end - s.start for s in sections)
    coverage = covered / len(normalised) if normalised else 0.0

    return ParsedNote(
        sections=tuple(sections),
        unknown_headings=tuple(unknown),
        coverage=coverage,
        length=len(normalised),
    )


@dataclass(frozen=True)
class RedactionProfile:
    """How much real content survives de-identification in a piece of text.

    MIMIC replaces identifiers with runs of underscores. A section can be
    entirely placeholders ("Social History:\\n___"), entirely absent, or - the
    case the old regex missed - mostly placeholders with real content around
    them ("Marital status: Married\\nTobacco: ___").

    `informative_chars` counts alphanumeric characters that remain once the
    placeholder runs are removed, which is the only one of the three numbers
    that tells us whether the text is worth embedding.
    """

    chars: int
    placeholders: int
    informative_chars: int

    @property
    def is_empty(self) -> bool:
        return self.chars == 0

    @property
    def is_placeholder_only(self) -> bool:
        """Text exists, but every scrap of content in it was redacted."""
        return self.chars > 0 and self.informative_chars == 0


def redaction_profile(text: str) -> RedactionProfile:
    """Measure surviving content in `text`."""
    stripped = text.strip()
    if not stripped:
        return RedactionProfile(chars=0, placeholders=0, informative_chars=0)

    placeholders = len(_PLACEHOLDER_RE.findall(stripped))
    without_placeholders = _PLACEHOLDER_RE.sub(" ", stripped)
    informative = sum(1 for ch in without_placeholders if ch.isalnum())

    return RedactionProfile(
        chars=len(stripped),
        placeholders=placeholders,
        informative_chars=informative,
    )
