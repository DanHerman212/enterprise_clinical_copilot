"""`/meds` reports what the record has, and names what the section it read says.

Two defects from the 2026-11-04 forensic run, and both are about claiming more than the
retrieval supports.

`dev_t1_04` (admission 90000106) has no medication section at all. The `discharge_medications`
tool fell back to top-k retrieval and returned the hospital course, the extraction named two
drugs mentioned there, and the narrative called them "2 discharge medications, as recorded in
the discharge note" — a list the record does not contain.

`dev_t1_03` (admission 90000015) has a real `discharge_medications` section whose first
sentence is an arrival list — "On transfer, Celebrex, Coumadin, Colace, Synthroid, Lovenox,
Percocet, Toprol XL, niacin, and trazodone." The extraction names only the regimen to
continue, which is the right call: an arrival list is not a discharge prescription. But two
drugs the section documents then vanished from the answer entirely.

The rules these tests pin: a formal list is only claimed when a formal list was returned, and
a drug the section documents on arrival is reported as exactly that.
"""

from __future__ import annotations

import pytest

from services.agent.engine.deterministic import _arrival_list_names, _medication_sections
from services.agent.postprocess import answer as answer_text

#: The section that made `dev_t1_03` fail, verbatim from the retrieval payload.
TRANSFER_SECTION = (
    "On transfer, Celebrex, Coumadin, Colace, Synthroid, Lovenox, Percocet, Toprol XL, "
    "niacin, and trazodone.\n\n"
    "The patient is advised to continue taking the following medications: Celebrex 200 mg "
    "daily, for one month, Colace 100 mg b.i.d. for one month, Protonix 40 mg b.i.d. for "
    "one month, Synthroid 137 mcg daily, Diprosone cream 0.05% cream b.i.d. to the right "
    "arm and right abdomen, oxycodone SR 20 mg p.o. q.12h. for five days, then decrease to "
    "oxycodone SR 10 mg p.o. q.12h. for five days, Percocet 5/325 mg one to two tablets "
    "q.6h. to be used p.r.n. for breakthrough pain, trazodone 50 mg p.o. at bedtime p.r.n. "
    "for two weeks, Ativan 0.25 mg b.i.d. for two weeks, and Toprol-XL 50 mg daily."
)


def _rows(*names: str) -> list[dict[str, str]]:
    return [{"medication": name, "dosage": "", "frequency": ""} for name in names]


def test_a_dedicated_section_is_still_counted_as_a_discharge_medication_list():
    text = answer_text.medication_narrative(
        _rows("Celebrex", "Coumadin"), 1, sections=("discharge_medications",))

    assert text.startswith("2 discharge medications, as recorded in the discharge note")
    assert " ^[1]" in text


def test_a_fallback_section_is_never_called_a_discharge_medication_list():
    """`dev_t1_04`: two drugs from the hospital course are not a discharge list."""
    text = answer_text.medication_narrative(
        _rows("Zithromax", "albuterol"), 1, sections=("brief_hospital_course",))

    assert "as recorded in the discharge note" not in text
    assert "does not contain a dedicated discharge medications section" in text
    assert "brief hospital course" in text
    assert "Zithromax" in text and "albuterol" in text


def test_a_drug_the_section_names_on_arrival_is_reported_as_arriving_on_it():
    """`dev_t1_03`: Lovenox is in the section, and must not disappear from the answer."""
    names = _arrival_list_names(TRANSFER_SECTION, frozenset({"celebrex", "coumadin"}))
    assert "Lovenox" in names
    assert "niacin" in names
    # A drug the discharge regimen already reported is not repeated as an arrival entry.
    assert "Celebrex" not in names

    text = answer_text.medication_narrative(
        _rows("Celebrex", "Coumadin"), 1,
        sections=("discharge_medications",), arrival_names=tuple(names))

    assert "Lovenox" in text
    assert "on transfer" in text
    # It is not counted among the discharge medications.
    assert text.startswith("2 discharge medications")


def test_a_regimen_drug_is_not_repeated_as_an_arrival_entry():
    text = answer_text.medication_narrative(
        _rows("Celebrex", "Coumadin"), 1,
        sections=("discharge_medications",), arrival_names=("Coumadin",))

    assert text.count("Coumadin") == 1


def test_prose_that_merely_mentions_arriving_is_not_parsed_for_drug_names():
    """"On admission she was taking aspirin 81 mg daily" is not a list.

    Guessing drug names out of prose is the parser this module must not grow; only a
    comma list of three or more items is read.
    """
    assert _arrival_list_names("On admission she was taking aspirin 81 mg daily.") == []


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
