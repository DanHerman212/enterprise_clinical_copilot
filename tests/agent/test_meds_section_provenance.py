"""`/meds` reports what the record has.

One defect from the 2026-11-04 run, and it is about claiming more than the retrieval
supports.

`dev_t1_04` (admission 90000106) has no medication section at all. The `discharge_medications`
tool fell back to top-k retrieval and returned the hospital course, the extraction named two
drugs mentioned there, and the narrative called them "2 discharge medications, as recorded in
the discharge note" — a list the record does not contain.

The rule this pins: a formal list is only claimed when a formal list was returned.

Not pinned here, deliberately: which drugs an arrival list names. `MEDS_FORMATTING` states
that `/meds` extracts the regimen to continue at home and excludes drugs the patient merely
arrived on, and that rule is clinically correct. A regex that re-read the arrival clause to
put those names back was removed on 2026-11-04 for exactly that reason — the conflict was in
an assertion, not in the extraction, and the assertion was the thing to change.
"""

from __future__ import annotations

import pytest

from services.agent.engine.deterministic import _medication_sections
from services.agent.postprocess import answer as answer_text


def _rows(*names: str) -> list[dict[str, str]]:
    return [{"medication": name, "dosage": "", "frequency": ""} for name in names]


def test_a_dedicated_section_is_still_counted_as_a_discharge_medication_list():
    text = answer_text.medication_narrative(
        _rows("Celebrex", "Coumadin"), 1, sections=("discharge_medications",))

    assert text.startswith("2 discharge medications, as recorded in the discharge note")
    assert " ^[1]" in text


def test_a_fallback_section_is_never_called_a_discharge_medication_list():
    """`dev_t1_04`: two drugs from the hospital course are not a discharge list.

    The provenance rule still holds — the sentence attributes the list to the section it
    was read from — but it now says so in clinical words. The old wording opened with a
    statement about the document's headings ("does not contain a dedicated discharge
    medications section"), which read as an apology in an answer about drugs the record
    does state (2026-11-08).
    """
    text = answer_text.medication_narrative(
        _rows("Zithromax", "albuterol"), 1, sections=("brief_hospital_course",))

    assert "as recorded in the discharge note" not in text
    assert "does not contain a dedicated discharge medications section" not in text
    assert text == (
            "No dedicated discharge medications section is documented in the record. "
            "Documented to continue at discharge: Zithromax, albuterol ^[1].")

def test_the_sections_are_read_from_the_retrieval_not_guessed():
    recorded = [
        {"name": "discharge_medications",
         "args": {"hadm_id": 90000106},
         "response": {"passages": [
             {"section": "brief_hospital_course", "text": "The patient was admitted."}]}},
    ]
    assert _medication_sections(recorded) == ("brief_hospital_course",)


def test_an_empty_retrieval_still_says_it_has_nothing():
    text = answer_text.medication_narrative([], 0)
    assert text == (
        "No specific discharge medication information is available for this admission.")


@pytest.mark.parametrize("section", ["discharge_medications", "brief_hospital_course"])
def test_the_marker_is_present_whenever_a_passage_exists(section):
    text = answer_text.medication_narrative(_rows("Celebrex"), 1, sections=(section,))
    assert " ^[1]" in text
