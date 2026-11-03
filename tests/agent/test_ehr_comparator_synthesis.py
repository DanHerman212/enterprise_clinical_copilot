"""The briefing's risk section has to place the estimate beside the published cohorts.

UX gap 3, reported 2026-11-02. `/ehr` drew three things in one column — the narrative, the
risk widget, and the comparison table — and the narrative knew about two of them. The
verified cohorts were reached by retrieval, checked against their abstracts, drawn by the
table and numbered in the references list, and never mentioned in the prose that explains
the comparison; the reader got "her readmission risk is 21.9%" directly above a table
comparing it against 3.4% and 11.9%, and nothing joined the two.

Three things had to be true for the sentence to exist and to resolve:

* the formatter has to be TOLD what the rows are — they are not a tool result, because no
  tool returned them;
* the prompt has to know that the block may be absent, and what to write then;
* a `^[PMID: …]` badge naming a row has to survive `enforce_citations`, which deletes any
  identifier the turn did not verify.
"""

from __future__ import annotations

from services.agent.engine import deterministic
from services.agent.engine import prompts
from services.agent.postprocess import citations
from services.agent.postprocess.guardrail import guard_answer

COMPARATOR = {
    "pmid": "32742739",
    "title": "Readmission after total hip arthroplasty in patients over 80.",
    "authors": "Creager AE, Kleven AD, et al.",
    "journal": "Geriatric orthopaedic surgery & rehabilitation",
    "pub_date": "2020 Sep",
    "year": "2020",
    "reported_rate": "4.9%",
    "rate_kind": "point",
    "cohort_summary": "Patients over 80 years old undergoing THA, n=7730, NSQIP.",
    "evidence_tier": "Retrospective database review",
    "rate_sentence": "376 (4.9%) patients experienced an unplanned readmission.",
    "url": "https://pubmed.ncbi.nlm.nih.gov/32742739/",
}

TWO_ARM = dict(
    COMPARATOR,
    pmid="28142261",
    journal="Neurosurgical focus",
    year="2017",
    reported_rate="2.6% vs. 0.4%",
    rate_kind="comparison",
    cohort_summary="5590 ACDF and 487 CDR patients, ACS NSQIP.",
)


# --- the block the formatter reads ---------------------------------------------

def test_the_block_names_the_pmid_the_rate_and_the_cohort():
    block = deterministic.comparator_block([COMPARATOR])

    assert block.startswith("<verified_comparators>")
    assert block.rstrip().endswith("</verified_comparators>")
    assert "PMID 32742739" in block
    assert "Geriatric orthopaedic surgery & rehabilitation · 2020" in block
    assert "4.9% (a single figure)" in block
    assert "cohort: Patients over 80 years old" in block
    # The citation form is stated, because it is the form enforcement accepts.
    assert "^[PMID: <pmid>]" in block


def test_a_two_arm_rate_is_flagged_as_two_figures():
    block = deterministic.comparator_block([TWO_ARM])

    assert "2.6% vs. 0.4% (two arms, published as two figures — quote both)" in block


def test_the_rate_sentence_stays_out_of_the_block():
    """It is the evidence the row was verified against, not prose to quote at a reader."""
    assert "376 (4.9%) patients experienced" not in deterministic.comparator_block(
        [COMPARATOR])


def test_no_rows_appends_nothing_at_all():
    assert deterministic.comparator_block([]) == ""
    assert deterministic.comparator_block(None) == ""


def test_the_formatter_input_carries_the_block_after_the_evidence():
    from langchain_core.messages import HumanMessage

    text = deterministic._formatter_input(
        [HumanMessage(content="note passages here")], False, "/ehr",
        [COMPARATOR])

    assert text.startswith("Question asked: /ehr")
    assert text.index("note passages here") < text.index("<verified_comparators>")


# --- the badge survives only when the cohort was verified ----------------------

def test_a_verified_comparator_pmid_may_be_cited():
    answer = "Her estimate of 21.9% sits above the 4.9% published for octogenarian THA ^[PMID: 32742739]."

    guarded = guard_answer(answer, [], state={
        "comparators": [COMPARATOR],
        "messages": [],
    })

    assert "^[PMID: 32742739]" in guarded["answer"]
    assert not [f for f in guarded["flags"] if "citation" in f]


def test_a_pmid_the_table_never_verified_is_still_deleted():
    """The pool holds twenty-five candidates; the table draws at most two.

    Allowing the pool would let the answer cite a cohort the table refused — a pill with
    no row behind it, which is the defect the numbering exists to prevent.
    """
    answer = "This sits above the 9.9% published elsewhere ^[PMID: 99999999]."

    guarded = guard_answer(answer, [], state={"comparators": [COMPARATOR], "messages": []})

    assert "99999999" not in guarded["answer"]


def test_the_allowance_is_the_drawn_rows_not_the_pool():
    assert citations.comparator_pmids([COMPARATOR, TWO_ARM]) == frozenset(
        {"32742739", "28142261"})
    assert citations.comparator_pmids([]) == frozenset()
    assert citations.comparator_pmids(None) == frozenset()


# --- what the prompt says -----------------------------------------------------

def _ehr_prompt() -> str:
    return " ".join(prompts.EHR_FORMATTING.split())


def test_the_prompt_places_the_estimate_against_the_cohorts():
    text = _ehr_prompt()

    assert "<verified_comparators> block" in text
    assert "population benchmark and not a prediction about this patient" in text
    assert "^[PMID: 35186616]" in text


def test_every_cohort_rate_carries_its_own_badge_and_never_a_combined_one():
    """The risk paragraph lost both badges when two cohorts shared a sentence.

    Measured 2026-11-03: the placement rule ("before the full stop") and the
    one-number-per-claim rule were written for note passages, and the model applied them
    to published cohorts too — either dropping the badges or merging two records into
    `^[PMID: 111, 222]`, which settlement could not draw. The instruction now scopes the
    old rules to note passages and states the comparator rule on its own.
    """
    text = " ".join(_ehr_prompt().split())

    assert "EVERY published rate you state carries its own badge" in text
    assert "This rule and the one-number-per-claim rule above are about note passages" in text
    assert "never `^[PMID: 35186616, 37028984]`" in text


def test_the_prompt_forbids_a_comparison_when_there_is_no_block():
    text = " ".join(_ehr_prompt().split())

    assert "Where an input block of published cohorts is NOT given" in text
    assert "do not write a sentence about the comparison at all" in text
    assert "Do not reach for a rate you know from training" in text


def test_the_prompt_keeps_a_two_arm_row_two_figures():
    text = _ehr_prompt()

    assert "A two-arm row is two figures" in text
    assert "never average them" in text
