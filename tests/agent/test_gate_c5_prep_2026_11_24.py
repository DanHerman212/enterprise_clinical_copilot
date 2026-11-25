"""Gate C5-preparation: the five dev failures, repaired where each one originated.

The 40-scenario dev suite at `8587a05` failed five items and every one of them was traced to
a place in the product rather than to a place in the model's judgement:

1. `dev_d5_fup_01` turn 3 — the `/labs` instruction asked for "the largest movement" on a
   panel where nothing was abnormal, and nothing computed which movement that was.
2. `dev_d3_slash_01` turn 1 — a rate of `10.66% ± 6.87%` reached a briefing as "the 6.87%
   reported", and the comparator verifier's membership check could not tell the standard
   deviation from the rate.
3. `dev_d5_free_01` turn 1 — a five-passage `rag_search` returned `physical_exam` and
   `discharge_instructions` above `past_surgical_history` and `major_procedure`, so the
   answer said the record documented no other operations.
4. `dev_d5_free_01` turn 5 — the structured panel and the hospital course state different
   creatinine courses for one admission, and no rule told the answer to say which was which.
5. `dev_d2_slash_01` turn 1 — a sentence spelling out findings a course passage only defers
   to ("revealed the above findings") carried one marker instead of both.

This file pins the repairs. The dataset keys for the two contradictory scenarios live in
`evaluation/datasets/dev_suite.jsonl`, and are checked here too so a later re-sync cannot
quietly restore a requirement the product is built to refuse.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.agent.engine import deterministic
from services.agent.engine import prompts
from services.agent.routing import admission_context  # noqa: F401  (import side effect)

#: The panel `90000103` is served: four timepoints over a 1.2-day stay, every reading inside
#: its reference interval, anchored on nothing (`"anchored": []`). Copied from the recorded
#: payload of `dev_d5_free_01` turn 5 rather than invented, because the numbers are the
#: subject of the test: creatinine 0.65 -> 0.98 is the largest proportional change on it.
_PANEL_90000103 = [{"name": "get_lab_trends", "args": {"hadm_id": 90000103}, "response": {
    "hadm_id": 90000103, "returned": 9, "trends": [
        {"key": "rdw", "label": "RDW", "unit": "%", "decimals": 1,
         "reference_low": 11.5, "reference_high": 14.5, "anchored": [],
         "series": [{"day": 0.0, "value": 12.3, "flag": "normal"},
                    {"day": 0.4, "value": 12.9, "flag": "normal"},
                    {"day": 0.8, "value": 13.4, "flag": "normal"},
                    {"day": 1.2, "value": 13.9, "flag": "normal"}]},
        {"key": "hemoglobin", "label": "Hemoglobin", "unit": "g/dL", "decimals": 1,
         "reference_low": 12.0, "reference_high": 15.5, "anchored": [],
         "series": [{"day": 0.0, "value": 13.9, "flag": "normal"},
                    {"day": 0.4, "value": 12.7, "flag": "normal"},
                    {"day": 0.8, "value": 13.2, "flag": "normal"},
                    {"day": 1.2, "value": 13.4, "flag": "normal"}]},
        {"key": "creatinine", "label": "Creatinine", "unit": "mg/dL", "decimals": 2,
         "reference_low": 0.5, "reference_high": 1.1, "anchored": [],
         "series": [{"day": 0.0, "value": 0.65, "flag": "normal"},
                    {"day": 0.4, "value": 0.75, "flag": "normal"},
                    {"day": 0.8, "value": 0.89, "flag": "normal"},
                    {"day": 1.2, "value": 0.98, "flag": "normal"}]},
    ]}}]


# ------------------------------------------------- 1. the in-range panel names its own shift


def test_the_labs_instruction_no_longer_asks_for_the_largest_movement():
    """"Largest" was the word the judge scored as unsupported, so it is gone from the rule.

    The instruction now points at the `<lab_flags>` line, which states which analyte moved
    most in proportion, and forbids the superlative. Both halves are asserted: a rule that
    merely stopped mentioning the superlative while still inviting one would pass a test
    that only searched for the old phrase.
    """
    text = " ".join(prompts.LABS_FORMATTING.split())

    assert "largest movement" not in text
    assert "Notable in-range shift" in text or "notable in-range shift" in text
    assert "Never rank it against the other analytes" in text


def test_an_in_range_panel_gets_the_top_relative_shift_named_neutrally():
    """`dev_d5_fup_01` turn 3: creatinine +50.8%, eGFR -13.2%, white cells +45.3%.

    Creatinine is the largest proportional change, so it is the one named — and the line
    states the two values and the interval, with no adjective comparing it to anything.
    """
    annotation = deterministic.lab_flag_annotation(_PANEL_90000103)

    assert "Notable in-range shift: Creatinine 0.65 -> 0.98 mg/dL" in annotation
    assert "reference 0.50\u20131.10 mg/dL" in annotation
    for word in ("largest", "biggest", "most significant"):
        assert word not in annotation.lower()
    # An in-range panel states no critical-flag sentence: that one belongs to a panel with
    # abnormal readings and none critical.
    assert "carries a critical flag" not in annotation


def test_a_panel_with_an_abnormal_reading_gets_no_in_range_shift_line():
    """The two sentences are alternatives. A panel with a flagged reading has its own."""
    panel = [{"name": "get_lab_trends", "response": {"trends": [
        {"key": "hemoglobin", "label": "Hemoglobin", "unit": "g/dL", "decimals": 1,
         "reference_low": 12.0, "reference_high": 15.5,
         "series": [{"day": 0.0, "value": 8.5, "flag": "low"},
                    {"day": 3.0, "value": 10.7, "flag": "low"}]},
        {"key": "creatinine", "label": "Creatinine", "unit": "mg/dL", "decimals": 2,
         "reference_low": 0.5, "reference_high": 1.1,
         "series": [{"day": 0.0, "value": 0.65, "flag": "normal"},
                    {"day": 3.0, "value": 0.98, "flag": "normal"}]},
    ]}}]

    annotation = deterministic.lab_flag_annotation(panel)

    assert "Notable in-range shift" not in annotation
    assert "No reading in this panel carries a critical flag." in annotation


# ------------------------------------------------------ 2. a dispersion is not a cohort rate

_DISPERSE_SENTENCE = (
    "The 30-day readmission rates of ischemic stroke survivors ranged from 1.41% to "
    "27.64%, with a mean value of 10.66%\u00b16.87%."
)


def _target():
    return deterministic.ClinicalTarget(
        condition="ischemic stroke", procedure="", anchors=(), patient_age=None,
        patient_sex="",
    )


def _article(abstract: str = _DISPERSE_SENTENCE) -> dict:
    return {
        "pmid": "34763470",
        "title": "Risk factors for 30-day readmission in patients with ischemic stroke.",
        "journal": "Annals of palliative medicine",
        "journal_abbrev": "Ann Palliat Med",
        "pub_date": "2021 Oct",
        "authors": "Deng Z, Wu X, Hu L, et al.",
        "abstract": abstract,
        "url": "https://pubmed.ncbi.nlm.nih.gov/34763470/",
    }


def _row(rate: str) -> dict:
    return {
        "pmid": "34763470",
        "clinical_match": True,
        "rationale": "adult ischemic stroke cohort",
        "horizon_days": 30,
        "reported_rate": rate,
        "rate_sentence": _DISPERSE_SENTENCE,
        "cohort_summary": "ischemic stroke survivors, n=1,829,964, 17 studies",
        "evidence_tier": "systematic review",
    }


def test_the_standard_deviation_is_refused_and_the_mean_is_kept():
    """`10.66%±6.87%` states one rate. Both figures are printed, so membership is not enough."""
    articles = {"34763470": _article()}

    kept, reason = deterministic.verify_comparator(_row("6.87%"), articles, _target())
    assert kept is None, "the standard deviation is not a cohort rate"
    assert "6.87%" in reason and "dispersion" in reason

    kept_mean, reason_mean = deterministic.verify_comparator(
        _row("10.66%"), articles, _target())
    assert reason_mean == ""
    assert kept_mean is not None
    assert kept_mean["reported_rate"] == "10.66%"


@pytest.mark.parametrize("written", [
    "10.66 % +/- 6.87 %",          # a spaced plus-or-minus
    "10.66% plus or minus 6.87%",  # spelled out
    "10.66% (SD 6.87%)",           # labelled after the figure
    "10.66% \u00b1 6.87%",         # the same sign with whitespace
    "10.66% (SE 6.87%)",           # standard error
    "10.66% (SEM 6.87%)",          # standard error of mean
    "10.66% (IQR 6.87%)",          # interquartile range
    "10.66%, standard error of 6.87%",
])
def test_every_spelling_of_a_dispersion_is_refused(written):
    articles = {"34763470": _article(written)}
    row = {**_row("6.87%"), "rate_sentence": written}

    kept, reason = deterministic.verify_comparator(row, articles, _target())

    assert kept is None, written
    assert "dispersion" in reason


def test_a_two_figure_rate_carrying_a_dispersion_term_is_refused():
    sentence = "The 30-day readmission rate was 10.66% \u00b1 6.87% across cohorts."
    articles = {"34763470": _article(sentence)}
    row = {**_row("10.66% to 6.87%"), "rate_sentence": sentence}

    kept, reason = deterministic.verify_comparator(row, articles, _target())

    assert kept is None
    assert "dispersion" in reason


def test_a_rate_the_sentence_really_reports_is_still_kept():
    """The check refuses a figure only when EVERY place it appears is a spread."""
    sentence = ("The rate fell from 6.87% in the historical cohort to 4.1% after the "
                "programme, a difference of 2.77 percentage points.")
    articles = {"34763470": _article(sentence)}
    row = {**_row("6.87%"), "rate_sentence": sentence}

    kept, reason = deterministic.verify_comparator(row, articles, _target())

    assert reason == "", reason
    assert kept is not None


def test_the_extraction_prompt_says_the_dispersion_is_not_the_rate():
    text = " ".join(prompts.LITERATURE_COMPARATOR_PROMPT.split())

    assert "A MEAN WITH A DISPERSION IS ONE RATE, NOT TWO" in text
    assert "10.66% \u00b1 6.87%" in text
    assert "never `6.87%`" in text


# ------------------------------------------- 3. the retrieval workaround is gone for good


def test_no_retrieval_workaround_is_left_in_the_tool():
    """The index carries the sections now, so the ranking hack had to go.

    A temporary repair lived here for one gate: a `top_k` of 8, a keyword boost that pulled
    four sections to the front, and a BigQuery read for the sections the index did not
    carry. All three existed because `past_surgical_history` was not indexed. It is indexed
    as of 2026-11-24, and a boost layered on top of a complete index would re-order answers
    against names nobody chose.
    """
    import inspect

    from services.mcp.tools import retrieval as tool

    source = inspect.getsource(tool)
    for name in ("_HISTORY_CUES", "SECTION_BOOST_SLACK", "HISTORY_SECTIONS",
                 "_parsed_history_passages", "_preferred_sections", "DEFAULT_TOP_K"):
        assert name not in source, f"{name} is the workaround coming back"
    assert inspect.signature(tool.rag_search).parameters["top_k"].default == 5


def test_the_sections_the_workaround_served_are_in_the_index_itself():
    """The fix is the chunk set, so the chunk set is where the assertion belongs."""
    from services.mcp.retrieval.chunking import INDEX_SECTIONS

    assert "past_surgical_history" in INDEX_SECTIONS
    assert "major_procedure" in INDEX_SECTIONS


# ------------------------------------------------- 4. the panel and the note are both named


def test_the_hub_says_which_course_is_the_panels_and_which_the_notes():
    """When structured lab series are unanchored synthetic curves, say which is which."""
    text = " ".join(prompts.LABS_SPOKE.split())

    assert "AND WHEN THE THREAD ALREADY QUOTED A DIFFERENT NUMBER, SAY WHICH IS WHICH" in text
    assert "unanchored_analytes" in text
    # And the authority rule it qualifies is still there, unweakened.
    assert "STRUCTURED DATA OUTRANKS NOTE PROSE" in text
    assert "A note's laboratory number never overrides the series" in text


# ------------------------------------------------------- 5. the clinic names, verbatim each


def test_the_chart_spoke_forbids_a_hybrid_clinic_name():
    text = " ".join(prompts.CHART_SPOKE.split())

    assert "TWO CLINIC NAMES THAT SHARE WORDS ARE STILL TWO CLINICS" in text
    assert '"Transplant Clinic at ABCD"' in text
    assert '"Renal Transplant Clinic"' in text
    assert "dev_d5_fup_01" in text


def test_the_briefing_rule_requires_both_markers_on_a_deferred_finding():
    """A course passage that defers to diagnosis findings carries both markers."""
    text = " ".join(prompts.EHR_FORMATTING.split())

    assert "Prefer one passage per sentence" in text
    assert "attach BOTH markers before the period" in text
    assert "^[1]^[3]" in text


# --------------------------------------------- 6. the two answer keys match the product rules


def _dataset() -> dict[str, dict]:
    path = Path(__file__).resolve().parents[2] / "evaluation/datasets/dev_suite.jsonl"
    items = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            items[item["id"]] = item
    return items


def test_the_meds_key_records_both_discharge_and_inpatient_treatments():
    """Narrative fallback extracts both take-home and inpatient treatments, separated by setting."""
    turn_2 = _dataset()["dev_d3_slash_01"]["expected_output"]["turn_2"]["facts"]

    assert turn_2["inpatient_treatment_named"] == ["Plavix", "baby aspirin", "Xopenex"]
    assert turn_2["discharge_medication"] == [
        "albuterol metered dose inhaler",
        "two remaining Zithromax doses to finish at home",
    ]


def test_the_drain_key_asks_only_what_the_question_asked():
    """Turn 3 asks whether there was a bile leak. The fluid and the two drains were not asked."""
    expected = _dataset()["dev_d5_adv_02"]["expected_output"]

    assert set(expected["turn_3"]["facts"]) == {"leak"}
    assert "not failed for it" in expected["behaviour"]
