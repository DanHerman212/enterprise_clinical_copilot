"""Stage 4b — the three horizon checks that refused records the abstracts support.

Every case here is a real refusal from one live `/ehr` turn on `90000069`
(2026-11-08, trace `eb43e055`): twenty abstracts retrieved, ten aligned, two proposed,
zero verified. The refusals were correct-looking and wrong:

  * `33049410` states "unplanned readmissions (4.96% vs 3.72%)" in Results and declares
    30-day outcomes elsewhere in the paper — the horizon was a property of the study, and
    the check only looked 240 characters back;
  * `31528438` states "5.1% (30 days) to 7.7% (90 days postoperatively)" — a plural that
    `\\bday\\b` cannot match, so a sentence naming 30 days was refused for not naming it;
  * `38504222` states that NO patient was readmitted within 30 days — a true zero written
    in words, which no numeral-matching check can confirm.

The negative cases are kept as tight as the positive ones: a rate that belongs to another
horizon must still be refused, because accepting it puts a 90-day figure beside a 30-day
estimate.
"""

from __future__ import annotations

import logging

import pytest

from services.agent.engine.deterministic import (
    ClinicalTarget,
    _reply_rows,
    verify_comparator,
)

CERVICAL = ClinicalTarget(
    condition="cervical spondylosis",
    procedure="anterior cervical discectomy and fusion",
    anchors=("cervical spondylosis",),
    patient_age=41,
    patient_sex="female",
)


def _article(pmid: str, title: str, abstract: str) -> dict:
    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "journal": "Spine",
        "journal_abbrev": "Spine",
        "pub_date": "2019 Mar",
        "authors": "Author A, Author B",
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    }


def _row(pmid: str, rate: str, sentence: str, *, horizon: int = 30) -> dict:
    return {
        "pmid": pmid,
        "clinical_match": True,
        "horizon_days": horizon,
        "reported_rate": rate,
        "rate_sentence": sentence,
        "cohort_summary": "adults undergoing anterior cervical discectomy and fusion, "
                          "n=197, single centre",
        "evidence_tier": "retrospective cohort",
    }


# --------------------------------------------------------------------------------------
# 1. The horizon is a property of the STUDY (33049410)
# --------------------------------------------------------------------------------------

#: The horizon is stated AFTER the quoted sentence, so a window of preceding text cannot
#: reach it. This is the shape the old check missed: Methods declares "thirty days", the
#: Results sentence reports the figures, and nothing between them repeats the horizon.
SENTENCE_33049410 = (
    "Unplanned readmissions occurred in 4.96% of the registry cohort and 3.72% of the "
    "comparison cohort."
)
ABSTRACT_33049410 = (
    "We reviewed consecutive adult patients undergoing anterior cervical discectomy and "
    "fusion at a single institution and compared them with a national registry sample. "
    + SENTENCE_33049410
    + " Outcomes were assessed over the thirty days after discharge for every patient in "
      "the registry."
)
ARTICLE_33049410 = _article(
    "33049410",
    "Readmission after anterior cervical discectomy and fusion: a registry comparison",
    ABSTRACT_33049410,
)


def test_a_horizon_stated_once_in_the_study_verifies_a_results_sentence():
    kept, reason = verify_comparator(
        _row("33049410", "4.96%", SENTENCE_33049410), {"33049410": ARTICLE_33049410},
        CERVICAL,
    )

    assert reason == ""
    assert kept is not None
    assert kept["reported_rate"] == "4.96%"
    assert kept["horizon_days"] == 30


def test_a_study_without_a_thirty_day_horizon_anywhere_is_still_refused():
    """The widening is to the abstract, not to "any horizon will do"."""
    abstract = ABSTRACT_33049410.replace(
        "over the thirty days after discharge",
        "over the ninety days after discharge",
    )
    kept, reason = verify_comparator(
        _row("33049410", "4.96%", SENTENCE_33049410),
        {"33049410": _article("33049410", "Readmission after ACDF", abstract)},
        CERVICAL,
    )

    assert kept is None
    assert "no 30-day horizon" in reason


# --------------------------------------------------------------------------------------
# 2. A sentence that names two horizons, and the figure each one owns (31528438)
# --------------------------------------------------------------------------------------

SENTENCE_31528438 = (
    "Readmission rates for ACDF ranged from 5.1% (30 days) to 7.7% (90 days "
    "postoperatively)."
)
ARTICLE_31528438 = _article(
    "31528438",
    "Readmission after anterior cervical discectomy and fusion at one institution",
    "A consecutive series of adult patients was reviewed. " + SENTENCE_31528438
    + " All patients were followed for at least one year.",
)


def test_a_plural_horizon_inside_the_quote_verifies():
    """"30 days" is a 30-day horizon; `\\bday\\b` could never match it."""
    kept, reason = verify_comparator(
        _row("31528438", "5.1%", SENTENCE_31528438), {"31528438": ARTICLE_31528438},
        CERVICAL,
    )

    assert reason == ""
    assert kept is not None
    assert kept["reported_rate"] == "5.1%"


def test_the_ninety_day_figure_of_a_range_sentence_is_refused():
    """The row publishes one number, and it has to be the 30-day one."""
    kept, reason = verify_comparator(
        _row("31528438", "7.7%", SENTENCE_31528438), {"31528438": ARTICLE_31528438},
        CERVICAL,
    )

    assert kept is None
    assert "30-day horizon states (5.1%)" in reason


def test_a_true_ninety_day_sentence_is_still_refused():
    """A record that only reports a 90-day rate is not this patient's comparison."""
    sentence = "The 90-day readmission rate was 12.4% in this cohort."
    kept, reason = verify_comparator(
        _row("30000001", "12.4%", sentence, horizon=30),
        {"30000001": _article("30000001", "Readmission after ACDF", sentence)},
        CERVICAL,
    )

    assert kept is None
    assert "different horizon" in reason


def test_a_wrong_declared_horizon_is_still_refused():
    sentence = "The 30-day readmission rate was 6.2%."
    kept, reason = verify_comparator(
        _row("30000002", "6.2%", sentence, horizon=90),
        {"30000002": _article("30000002", "Readmission after ACDF", sentence)},
        CERVICAL,
    )

    assert kept is None
    assert "horizon is 90 days, not 30" in reason


def test_a_fabricated_rate_is_still_refused():
    kept, reason = verify_comparator(
        _row("31528438", "9.9%", SENTENCE_31528438), {"31528438": ARTICLE_31528438},
        CERVICAL,
    )

    assert kept is None
    assert "not the figure its own sentence states" in reason


# --------------------------------------------------------------------------------------
# 3. A zero stated in words (38504222)
# --------------------------------------------------------------------------------------

SENTENCE_38504222 = (
    "In this cohort no patient was readmitted within 30 days after discharge; however, "
    "an incisional hematoma requiring evacuation occurred in two patients."
)
ARTICLE_38504222 = _article(
    "38504222",
    "Complications after anterior cervical discectomy and fusion",
    "A consecutive series of adult patients was reviewed. " + SENTENCE_38504222,
)


def test_a_verbal_zero_readmission_verifies():
    kept, reason = verify_comparator(
        _row("38504222", "0%", SENTENCE_38504222), {"38504222": ARTICLE_38504222},
        CERVICAL,
    )

    assert reason == ""
    assert kept is not None
    assert kept["reported_rate"] == "0%"


def test_a_verbal_zero_needs_the_negation_to_be_about_readmission():
    """"no difference in readmission" is not a zero rate."""
    sentence = ("There was no difference in readmission between the two approaches "
                "within 30 days.")
    kept, reason = verify_comparator(
        _row("30000003", "0%", sentence),
        {"30000003": _article("30000003", "Readmission after ACDF", sentence)},
        CERVICAL,
    )

    assert kept is None
    assert "not the figure its own sentence states" in reason


def test_a_zero_claim_with_no_verbal_support_is_still_refused():
    sentence = "The 30-day readmission rate was 4.1% in this cohort."
    kept, reason = verify_comparator(
        _row("30000004", "0%", sentence),
        {"30000004": _article("30000004", "Readmission after ACDF", sentence)},
        CERVICAL,
    )

    assert kept is None
    assert "not the figure its own sentence states" in reason


# --------------------------------------------------------------------------------------
# 4. The refusal carries the sentence it is about
# --------------------------------------------------------------------------------------


def test_a_refusal_records_the_sentence_it_refused(caplog):
    caplog.set_level(logging.INFO, logger="services.agent.engine.deterministic")

    _reply_rows(
        {"comparators": [_row("31528438", "7.7%", SENTENCE_31528438)]},
        [ARTICLE_31528438],
        CERVICAL,
        [],
    )

    assert "refused 31528438" in caplog.text
    assert "30-day horizon states (5.1%)" in caplog.text
    assert "quoted='Readmission rates for ACDF ranged from 5.1% (30 days)" in caplog.text


@pytest.mark.parametrize("sentence", [
    "Readmission rates for ACDF ranged from 5.1% (30 days) to 7.7% (90 days postoperatively).",
    "in this cohort no patient was readmitted within 30 days after discharge",
])
def test_the_horizon_patterns_accept_the_forms_abstracts_use(sentence):
    from services.agent.engine.deterministic import _THIRTY_DAY

    assert _THIRTY_DAY.search(sentence)
