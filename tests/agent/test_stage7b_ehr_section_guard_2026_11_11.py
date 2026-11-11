"""Stage 7B: the `/ehr` discharge section may only rest on a discharge section.

Four answers across two days reported a hospital-course drug as a home prescription under
`**Discharge plan & medications.**`. The prompt forbade it three times; the fourth answer
did it anyway. So the rule moved into the settlement pass, where the marker has already
been bound to a passage and its section is a fact rather than an intention.

Two properties matter and both are tested here.

It removes only what the section cannot support. Each of the thirteen real discharge
paragraphs from run `pilot-verify-20261111-001447` is a fixture, verbatim as served: the
three from admission `90000023` lose one sentence each, and the other ten come back
byte-identical. A filter that also shortened the ten would be deleting a clinician's
discharge plan to protect a rule about one drug.

And it runs at the right point. Markers are pre-compaction ledger slots when this sees
them, so a paragraph it shortens still compacts to `1..K` with no gap — the second test
drives the whole settlement to prove it, rather than the filter alone.

`MEDS_SPOKE` gained the other half of the same confusion: a cohort that is told a drug is
adjusted by laboratory results is not being told it has a taper.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.agent.engine import prompts
from services.agent.postprocess import answer as answer_text
from services.agent.postprocess import citations, discharge_section

FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "discharge_paragraphs_2026_11_11.json")
    .read_text())

#: The three items whose discharge paragraph cited the hospital course. One admission, so
#: the same note text behind all three.
DROPPING = {"dev_t2_02", "dev_t2_05", "dev_t4_02"}


def _units(fixture: dict) -> list[dict]:
    """The array a marker number indexes, position `n` at `[n - 1]`.

    The fixtures carry the SERVED paragraph, whose markers are page numbers, and the map
    from each page number to the section it resolved to. So position `n` is reconstructed
    from that map rather than from the payload order, and the text comes from the passage
    the section names — which is what the arrival-list rule reads.
    """
    markers = {int(number): section for number, section in fixture["markers"].items()}
    text_by_section: dict[str, str] = {}
    for unit in fixture["units"]:
        text_by_section.setdefault(unit["section"], unit["text"])
    highest = max(markers) if markers else 0
    return [{"section": markers.get(position, ""),
             "text": text_by_section.get(markers.get(position, ""), "")}
            for position in range(1, highest + 1)]


@pytest.mark.parametrize("fixture", FIXTURES, ids=[row["item"] for row in FIXTURES])
def test_only_the_hospital_course_sentence_leaves_the_discharge_section(fixture):
    units = _units(fixture)
    text, flags = discharge_section.filter_discharge_section(
        fixture["paragraph"], passages=units, units=units)

    if fixture["item"] not in DROPPING:
        assert text == fixture["paragraph"], (
            "a discharge paragraph resting only on discharge sections was altered")
        assert flags == ()
        return

    assert flags == ("discharge_section_dropped:brief_hospital_course",)
    markers = {int(number): section for number, section in fixture["markers"].items()}
    dropped_marker = next(number for number, section in markers.items()
                          if section == "brief_hospital_course")
    body = fixture["paragraph"][len(discharge_section.DISCHARGE_LABEL):]
    for sentence in discharge_section.split_sentences(body):
        if f"^[{dropped_marker}]" in sentence:
            continue
        assert sentence in text, f"a supported sentence was dropped: {sentence!r}"
    assert f"^[{dropped_marker}]" not in text


def test_the_dropped_sentence_is_the_one_that_claimed_a_prescription():
    """`dev_t4_02` turn 1, verbatim: the sentence that opened the failing section."""
    fixture = next(row for row in FIXTURES if row["item"] == "dev_t4_02")
    units = _units(fixture)

    text, _flags = discharge_section.filter_discharge_section(
        fixture["paragraph"], passages=units, units=units)

    assert "was prescribed" not in text
    body = fixture["paragraph"][len(discharge_section.DISCHARGE_LABEL):]
    claim = next(sentence for sentence in discharge_section.split_sentences(body)
                 if "^[3]" in sentence)
    assert claim.startswith("The patient was prescribed")
    assert claim not in text
    assert "discharged home with instructions" in text


def test_the_surviving_markers_compact_to_one_to_k():
    """The filter runs before the page numbering, so nothing is left pointing at a gap."""
    fixture = next(row for row in FIXTURES if row["item"] == "dev_t4_02")
    units = _units(fixture)

    filtered, _flags = discharge_section.filter_discharge_section(
        fixture["paragraph"], passages=units, units=units)
    compacted, display = citations.compact_display_numbers(filtered)

    assert citations.cited_numbers(compacted) == [1]
    assert display == {"1": 5}


def test_settlement_applies_the_filter_only_to_a_summary_turn():
    """The seam, end to end: a `/ehr` answer is filtered, a free-text answer is not."""
    fixture = next(row for row in FIXTURES if row["item"] == "dev_t4_02")
    units = _units(fixture)
    call = {"name": "rag_search_sections", "args": {"hadm_id": 90000023},
            "response": {"returned": len(units), "passages": units}}
    prose = fixture["paragraph"]

    summarised = answer_text.settle_answer(
        prose=prose, deterministic=None, tool_calls=[call], replayed_tool_calls=[],
        command="summarize")
    free_text = answer_text.settle_answer(
        prose=prose, deterministic=None, tool_calls=[call], replayed_tool_calls=[])

    assert "was prescribed" not in summarised.prose
    assert "discharge_section_dropped:brief_hospital_course" in summarised.flags
    assert citations.cited_numbers(summarised.prose) == [1]

    # A free-text turn is settled too, so its numbers compact. What must not happen is the
    # filter: the sentence comes through, and no drop is reported.
    assert "was prescribed" in free_text.prose
    assert not any(flag.startswith("discharge_section")
                   for flag in free_text.flags)


# --- the arrival list is not a discharge regimen -------------------------------------

ARRIVAL_LIST = ("On transfer, Celebrex, Coumadin, Colace, Synthroid, Lovenox, Percocet, "
                "Toprol XL, niacin, and trazodone.")


def _paragraph(marker: int) -> str:
    return (f"{discharge_section.DISCHARGE_LABEL} The patient was prescribed a drug "
            f"^[{marker}]. He was advised to keep every appointment ^[2].")


def test_an_arrival_list_cannot_state_a_discharge_prescription():
    """`medications_on_transfer` is a medication section whose content is the wrong visit.

    In a transitional-care note the heading carries the acute hospital's list, and a drug
    on it may have been stopped before discharge. The section name would otherwise pass the
    filter, which is why the opener is read and not just the name.
    """
    units = [{"section": "discharge_medications", "text": ARRIVAL_LIST},
             {"section": "discharge_instructions", "text": "He was advised to follow up."}]

    text, flags = discharge_section.filter_discharge_section(
        _paragraph(1), passages=units, units=units)

    assert "was prescribed" not in text
    assert flags == ("discharge_section_dropped:discharge_medications",)
    assert "keep every appointment" in text


def test_a_discharge_medications_list_that_is_not_an_arrival_list_survives():
    units = [{"section": "discharge_medications", "text": "Celebrex 200 mg daily."},
             {"section": "discharge_instructions", "text": "Follow up in two weeks."}]
    prose = _paragraph(1)

    text, flags = discharge_section.filter_discharge_section(
        prose, passages=units, units=units)

    assert text == prose
    assert flags == ()


def test_a_note_with_no_discharge_section_is_left_alone():
    """`90000106` holds only the hospital course. There is nothing to place the drug against.

    Deleting the one sentence that mentions a medication would leave the reader with less
    than the record says, and the record's own hospital-course instruction is real content.
    """
    units = [{"section": "brief_hospital_course",
              "text": "He was given a course of antibiotics."}]
    prose = _paragraph(1)

    text, flags = discharge_section.filter_discharge_section(
        prose, passages=units, units=units)

    assert text == prose
    assert flags == ()


def test_a_paragraph_with_no_markers_is_never_touched():
    units = [{"section": "brief_hospital_course", "text": "Any text."},
             {"section": "discharge_instructions", "text": "Any text."}]
    prose = f"{discharge_section.DISCHARGE_LABEL} The patient was discharged home."

    text, flags = discharge_section.filter_discharge_section(
        prose, passages=units, units=units)

    assert text == prose
    assert flags == ()


# --- the sentence splitter ------------------------------------------------------------

def test_the_splitter_keeps_clinical_shorthand_inside_one_sentence():
    """A splitter that breaks on `p.o.` hands this module half a sentence and deletes it."""
    text = ("Take 20 mg p.o. q.12h. for five days, then 10 mg p.o. b.i.d. ^[3]. "
            "Dr. X reviewed the 0.05% cream on day 2 ^[4]. Mr. ABC was seen by Dr. Y ^[5].")

    sentences = discharge_section.split_sentences(text)

    assert len(sentences) == 3
    assert sentences[0].startswith("Take 20 mg p.o. q.12h.")
    assert sentences[2].startswith("Mr. ABC was seen by Dr. Y")


# --- the medication spoke's taper rule ------------------------------------------------

def test_a_taper_is_a_schedule_and_not_a_lab_guided_adjustment():
    text = " ".join(prompts.MEDS_SPOKE.split())

    assert "A TAPER is a scheduled step-down across defined intervals" in text
    assert "is dose adjustment, not a taper" in text
    assert "name only the scheduled step-down" in text
    # The pinned example that taught the distinction stays.
    assert "the dose will be adjusted according to the INRs" in text
    assert "state BOTH" in text


def test_the_spoke_budgets_hold():
    assert len(prompts.CORE_HUB_PROMPT.split()) <= 1500
    for name in ("CHART_SPOKE", "MEDS_SPOKE", "LABS_SPOKE", "RISK_SPOKE",
                 "LITERATURE_SPOKE"):
        assert len(getattr(prompts, name).split()) <= 650, name


def test_no_prompt_names_a_dev_set_literal():
    """Stage 6F's hygiene rule, re-asserted beside the literals this stage could reintroduce."""
    for name in ("CORE_HUB_PROMPT", "CHART_SPOKE", "MEDS_SPOKE", "LABS_SPOKE",
                 "RISK_SPOKE", "LITERATURE_SPOKE", "EHR_FORMATTING",
                 "LABS_FORMATTING", "MEDS_FORMATTING", "LITERATURE_FORMATTING",
                 "FORMATTER_PREAMBLE", "SYSTEM_PROMPT"):
        text = getattr(prompts, name)
        for literal in ("Plavix", "Lovenox", "on no meds", "14.0 to 13.4",
                        "08/30/2007", "09/21/2007", "dev_t", "holdout_"):
            assert literal not in text, f"{name} contains {literal!r}"
