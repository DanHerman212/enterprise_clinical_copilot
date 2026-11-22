"""The prompt and binding conflicts the calibration removed (2026-11-21).

Each one is a place where the agent was told two things that could not both be obeyed, or
where the harness and the product disagreed about what an answer should contain. The
dataset and rubric halves are asserted against the files themselves, because those are the
contracts the judge and the critic read.
"""

from __future__ import annotations

import pathlib

from services.agent.engine import deterministic, graph, prompts
from services.agent.routing import routing

REPO = pathlib.Path(__file__).resolve().parents[2]
RUBRIC = (REPO / "evaluation" / "rubric.md").read_text(encoding="utf-8")
SLICE = (REPO / "evaluation" / "datasets" / "selection_slice.jsonl").read_text(
    encoding="utf-8")


def _flat(text: str) -> str:
    return " ".join(text.split())


# --- a literature turn answers from its records only ----------------------------------

def test_a_literature_turn_is_not_given_the_threads_chart_state():
    """The session block is the one route by which risk state reaches a literature answer.

    Given it, a `/pubmed` turn compared this patient against the cohorts it had just
    retrieved, read an earlier turn's drivers back as its own, and called a male patient
    "her" (`v2_d3_slash_01`).
    """
    source = pathlib.Path(deterministic.__file__).read_text(encoding="utf-8")
    assert 'if command.wire != "literature" else ""' in source


def test_the_literature_instruction_describes_no_patient():
    text = _flat(prompts.LITERATURE_FORMATTING)

    assert "claim about her" not in text
    assert "her drivers" not in text
    assert "no prognosis, no comparison against this patient's own risk" in text


def test_one_badge_per_claim_and_none_for_a_tangential_record():
    text = _flat(prompts.LITERATURE_FORMATTING)

    assert "immediately after the specific claim or sentence" in text
    assert "Never attach two badges to one sentence" in text
    assert "a tangential record left uncited is correct" in text


# --- a frequency is a schedule, and a list is a list ----------------------------------

def test_the_medication_fallback_keeps_route_out_of_frequency():
    text = _flat(prompts.MEDS_FALLBACK_FORMATTING)

    assert '"frequency" holds a dosing schedule or timing and nothing else' in text
    assert 'it goes in "route", never in "frequency"' in text


def test_the_medication_fallback_forbids_slicing_a_parallel_list():
    text = _flat(prompts.MEDS_FALLBACK_FORMATTING)

    assert "is four items" in text
    assert 'a leading verb ("Resume", "Continue") is not a frequency' in text


# --- the open driver question keeps the series ----------------------------------------

OPEN = "What is driving this patient's readmission risk, and does the chart support it?"
SCOPED = "Why does prior inpatient days increase this patient's readmission risk?"
BAR = "How does prior inpatient days affect this patient's readmission risk?"


def test_only_a_question_that_names_a_factor_is_scoped():
    assert routing.names_one_driver(SCOPED)
    assert routing.names_one_driver(BAR)
    assert routing.names_one_driver("Why is age a readmission risk driver?")
    assert not routing.names_one_driver(OPEN)


def test_the_open_driver_question_may_read_a_laboratory_series():
    allowed = graph._freetext_allowed(True, False, False, question=OPEN)

    assert "get_lab_trends" in allowed
    assert "predict_readmission" in allowed


def test_a_scoped_driver_question_still_loses_the_series():
    for question in (SCOPED, BAR):
        allowed = graph._freetext_allowed(True, False, False, question=question)
        assert "get_lab_trends" not in allowed, question


def test_a_declared_driver_bar_loses_the_series_whatever_its_wording():
    allowed = graph._freetext_allowed(True, False, False, question=OPEN,
                                      intent="risk_driver")
    assert "get_lab_trends" not in allowed


def test_the_risk_domain_keeps_the_laboratory_tool():
    """Otherwise the domain narrowing takes back what the binder just allowed."""
    assert "get_lab_trends" in routing.DOMAIN_TOOLS[routing.DOMAIN_RISK]


def test_the_risk_spoke_leads_with_the_score_when_the_thread_has_not():
    text = _flat(prompts.RISK_SPOKE)

    assert "LEAD WITH THE SCORE WHEN THE THREAD HAS NOT STATED IT" in text


# --- the rubric agrees with the harness -----------------------------------------------

def test_the_rubric_states_the_conditional_headings():
    text = _flat(RUBRIC)

    assert "*Discharge plan & medications* needs at least one of `discharge_medications`" \
        in text
    assert "omitting *Discharge plan & medications* is **correct**" in text


def test_the_rubric_states_the_medication_contract():
    text = _flat(RUBRIC)

    assert "`RENDERED MEDICATION TABLE`" in text
    assert "A blank cell is the product saying the note gives nothing" in text
    assert "Equipment and supplies are separated from drugs by the product" in text


# --- the slice accepts the display form the product writes ----------------------------

def test_the_slice_accepts_one_decimal_probabilities():
    for display in ("7.5% (0.0751)", "9.5% (0.0948)", "11.9% (0.1193)",
                    "21.8% (0.2176)", "11.0% (0.11)"):
        assert display in SLICE, display


def test_the_slice_no_longer_asks_the_prose_to_name_a_category():
    assert "Where the record names a category rather than a product it says so" not in SLICE
    # What the row carries, rather than what a component is called: the guidance names the
    # items, and an undocumented dose or frequency is left empty.
    assert "leaving every undocumented dosage and frequency empty" in SLICE
