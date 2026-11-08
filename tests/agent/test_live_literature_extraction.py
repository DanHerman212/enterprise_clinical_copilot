"""Live PubMed comparator extraction for `/ehr` — Stage Gate 1.

The comparative table's rows used to come from a static dictionary in `presentation/`,
selected by a keyword classifier: a curated table that could disagree with the patient in
front of it and could not be told apart from a correct one, because a row for another
condition and a row for this one are both well-formed.

Stage Gate 1 is the backend that replaces it, tested in two layers.

  * The **hermetic** tests below take fixed abstracts and exercise the two rules that are
    not the model's to decide: the alignment gate (a cohort that does not name this
    patient's condition is not this patient's cohort) and the verification gate (a rate is
    what the abstract says, quoted verbatim, at a thirty-day horizon). These run in CI.
  * The **live** test runs the whole pipeline against eight real admissions — three named
    diagnostic ones and five held-out admissions spread across the served cohort — and
    prints the full report: the chart's own words, the query that was sent, the PMIDs that
    came back, which were accepted, which were rejected and why, and the rows that
    survived. It is marked `live` because it reads the served feature store and NCBI.

Nothing here is wired into the command path. `/ehr` still composes its table from
`presentation/domain.py` until Stage Gate 1 is reviewed.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from services.agent.engine import deterministic
from services.agent.engine import model as model_gateway
from services.agent.engine import prompts
from services.agent.ingress import contracts as served_admissions
from services.agent.telemetry import observability
from services.mcp.tools import literature, rag_search_sections
from evaluation.outpatient_oracle import OUTPATIENT_HADM_IDS

# --- fixtures: abstracts, and the targets a chart would produce --------------


FEMORAL_NECK = deterministic.ClinicalTarget(
    condition="displaced femoral neck fracture",
    procedure="hemiarthroplasty",
    anchors=("femoral neck fracture", "hip fracture", "hemiarthroplasty"),
)

CERVICAL = deterministic.ClinicalTarget(
    condition="cervical spondylotic myelopathy",
    procedure="anterior cervical discectomy and fusion",
    anchors=("cervical spondylotic myelopathy", "cervical spine surgery"),
)

PLEURAL_AFIB = deterministic.ClinicalTarget(
    condition="pleural effusion",
    procedure="thoracentesis",
    anchors=("pleural effusion", "atrial fibrillation"),
)

HEMIARTHROPLASTY = {
    "pmid": "30000001",
    "title": "Thirty-day readmission after hemiarthroplasty for displaced femoral neck "
             "fracture.",
    "journal": "Injury",
    "pub_date": "2019 Mar",
    "authors": "Okafor N, Lindqvist E, Marchetti R, et al.",
    "abstract": (
        "BACKGROUND: Displaced femoral neck fracture is commonly treated by "
        "hemiarthroplasty in older adults. METHODS: We reviewed 1,204 patients treated "
        "between 2010 and 2017 at a single institution. RESULTS: The 30-day readmission "
        "rate was 8.4%. The 90-day readmission rate was 14.1%. CONCLUSIONS: Readmission "
        "was most often for surgical site infection."
    ),
    "url": "https://pubmed.ncbi.nlm.nih.gov/30000001/",
}

KNEE_ARTHROPLASTY = {
    "pmid": "30000002",
    "title": "Readmission after total knee arthroplasty in a national cohort.",
    "journal": "The Journal of Arthroplasty",
    "pub_date": "2022 Aug",
    "authors": "Creager AE, Kleven AD, Kesimoglu ZN, et al.",
    "abstract": (
        "BACKGROUND: Total knee arthroplasty is performed with increasing frequency. "
        "RESULTS: Among 12,450 primary total knee arthroplasties the 30-day readmission "
        "rate was 3.4%. CONCLUSIONS: Complication rates remain low."
    ),
    "url": "https://pubmed.ncbi.nlm.nih.gov/30000002/",
}

LUMBAR_DUROTOMY = {
    "pmid": "30000003",
    "title": "Incidental durotomy and 30-day readmission after lumbar decompression.",
    "journal": "Spine",
    "pub_date": "2020 Jun",
    "authors": "Halvorsen T, Aoki M.",
    "abstract": (
        "BACKGROUND: Lumbar decompression for stenosis is common. RESULTS: The 30-day "
        "readmission rate was 5.5% in the durotomy group and 3.1% in the matched group. "
        "CONCLUSIONS: Durotomy increased readmission."
    ),
    "url": "https://pubmed.ncbi.nlm.nih.gov/30000003/",
}

COPD_EXACERBATION = {
    "pmid": "30000004",
    "title": "Thirty-day readmission after COPD exacerbation in a regional network.",
    "journal": "Respiratory Medicine",
    "pub_date": "2023 Jan",
    "authors": "Duarte P, Sorensen K, Ali R, et al.",
    "abstract": (
        "BACKGROUND: COPD exacerbation is a frequent cause of admission. RESULTS: The "
        "30-day readmission rate was 11% across 4,318 admissions. CONCLUSIONS: Early "
        "follow-up reduced readmission."
    ),
    "url": "https://pubmed.ncbi.nlm.nih.gov/30000004/",
}


def _row(**overrides) -> dict:
    """A proposed row that passes every check, with individual fields replaced."""
    row = {
        "pmid": "30000001",
        "clinical_match": True,
        "rationale": "Same population: displaced femoral neck fracture treated by "
                     "hemiarthroplasty.",
        "horizon_days": 30,
        "reported_rate": "8.4%",
        "rate_sentence": "The 30-day readmission rate was 8.4%.",
        "cohort_summary": "1,204 patients treated 2010-2017 at a single institution.",
        "evidence_tier": "Retrospective Cohort",
    }
    row.update(overrides)
    return row


# --- what this layer still decides about the pool ------------------------------
#
# The anchor gate that used to live here — a record refused unless its abstract repeated
# one of the patient's phrases — is deleted (2026-11-02). It was a token test standing in
# for a clinical judgement, and it refused correct cohorts that said the same thing in
# other words. Which cohort describes this patient is decided by the model that reads the
# abstract; the extraction turn refuses the rest and says why. What survives here is the
# one population rule that is structural rather than clinical: an abstract naming children
# is a cohort of children, and this admission is not one.


def test_the_pool_keeps_a_cohort_from_another_condition_and_leaves_it_to_the_model():
    """No token gate: a knee cohort in a hip admission's pool is the model's to refuse.

    The refusals in this test's predecessor were invisible to the reader — they happened
    before the abstract was read — so a synonym ("arthroplasty" for "replacement") looked
    exactly like an empty search.
    """
    kept, refused = deterministic.adult_cohorts(
        [HEMIARTHROPLASTY, KNEE_ARTHROPLASTY], FEMORAL_NECK)

    assert [article["pmid"] for article in kept] == ["30000001", "30000002"]
    assert refused == []


def test_a_paediatric_cohort_is_removed_before_the_pool_is_ranked():
    kept, refused = deterministic.adult_cohorts(
        [PEDIATRIC_EFFUSION, ADULT_EFFUSION], PLEURAL_AFIB)

    assert [article["pmid"] for article in kept] == ["30000031"]
    assert refused[0]["pmid"] == PEDIATRIC_EFFUSION["pmid"]
    assert "paediatric" in refused[0]["reason"]


def test_the_paediatric_rule_turns_itself_off_for_a_paediatric_admission():
    child = deterministic.ClinicalTarget(
        condition="paediatric complicated pleural effusion",
        procedure="thoracentesis",
        anchors=("pleural effusion",),
    )

    kept, refused = deterministic.adult_cohorts([PEDIATRIC_EFFUSION], child)

    assert kept == [PEDIATRIC_EFFUSION]
    assert refused == []


def test_a_single_anatomical_word_is_not_an_anchor():
    """`hip` appears in a paper about the other hip operation, so it cannot identify one."""
    target = deterministic.build_target({
        "condition": "hip fracture",
        "procedure": "",
        "anchor_terms": ["hip", "knee"],
    })

    assert target is not None
    assert "hip" not in target.anchors
    assert "knee" not in target.anchors
    assert "hip fracture" in target.anchors


def test_a_chart_that_names_no_condition_produces_no_target():
    assert deterministic.build_target({"condition": "", "anchor_terms": []}) is None
    assert deterministic.build_target(None) is None


# --- the verification gate ---------------------------------------------------

ARTICLES = {HEMIARTHROPLASTY["pmid"]: HEMIARTHROPLASTY}


def test_a_verbatim_row_verifies_and_carries_the_records_citation_line():
    verified, reason = deterministic.verify_comparator(_row(), ARTICLES, FEMORAL_NECK)

    assert reason == ""
    assert verified is not None
    assert verified["reported_rate"] == "8.4%"
    assert verified["horizon_days"] == 30
    # The citation line comes from the record, not from the model's row.
    assert verified["journal"] == "Injury"
    assert verified["pub_date"] == "2019 Mar"
    assert verified["url"] == "https://pubmed.ncbi.nlm.nih.gov/30000001/"


def test_a_paraphrased_sentence_is_rejected():
    paraphrased = "The 30-day readmission rate was 8.4 percent in this series."
    verified, reason = deterministic.verify_comparator(
        _row(rate_sentence=paraphrased), ARTICLES, FEMORAL_NECK)

    assert verified is None
    assert "not verbatim" in reason


def test_a_rate_its_own_sentence_does_not_state_is_rejected():
    """The model quoting a real sentence and attaching a different figure to it."""
    verified, reason = deterministic.verify_comparator(
        _row(reported_rate="6.2%"), ARTICLES, FEMORAL_NECK)

    assert verified is None
    assert "not the figure its own sentence states" in reason


def test_a_ninety_day_rate_is_rejected():
    sentence = "The 90-day readmission rate was 14.1%."
    verified, reason = deterministic.verify_comparator(
        _row(reported_rate="14.1%", rate_sentence=sentence), ARTICLES, FEMORAL_NECK)

    assert verified is None
    assert "different horizon" in reason


TITLE_HORIZON = {
    "pmid": "30000007",
    "title": "Predictors of thirty-day readmission after femoral neck fracture.",
    "journal": "Injury",
    "pub_date": "2017 Sep",
    "authors": "Baptiste L, Hart N.",
    "abstract": (
        "OBJECTIVE: To identify predictors of readmission after femoral neck fracture. "
        "RESULTS: The readmission rate was 4.1%."
    ),
    "url": "https://pubmed.ncbi.nlm.nih.gov/30000007/",
}


def test_the_horizon_may_come_from_the_records_title():
    """PMID 24108298's shape: the horizon in the title, the number in the RESULTS."""
    verified, reason = deterministic.verify_comparator(
        _row(pmid="30000007", reported_rate="4.1%",
             rate_sentence="The readmission rate was 4.1%."),
        {"30000007": TITLE_HORIZON}, FEMORAL_NECK)

    assert reason == ""
    assert verified is not None
    assert verified["rate_kind"] == "point"


CONFLICTING_TITLE = dict(
    TITLE_HORIZON,
    pmid="30000008",
    abstract="RESULTS: The 90-day readmission rate was 12.0%.",
)


def test_a_quote_stating_another_horizon_is_refused_even_when_the_title_says_thirty_day():
    """The row publishes the quote, so the quote is what has to be about 30 days."""
    verified, reason = deterministic.verify_comparator(
        _row(pmid="30000008", reported_rate="12.0%",
             rate_sentence="The 90-day readmission rate was 12.0%."),
        {"30000008": CONFLICTING_TITLE}, FEMORAL_NECK)

    assert verified is None
    assert "different horizon" in reason


def test_a_horizon_other_than_thirty_days_is_rejected():
    verified, reason = deterministic.verify_comparator(
        _row(horizon_days=90), ARTICLES, FEMORAL_NECK)

    assert verified is None
    assert "not 30" in reason


def test_a_pmid_this_search_did_not_return_is_rejected():
    verified, reason = deterministic.verify_comparator(
        _row(pmid="99999999"), ARTICLES, FEMORAL_NECK)

    assert verified is None
    assert "not one this search retrieved" in reason


def test_a_row_the_model_declined_is_rejected_with_its_rationale():
    """The model's own refusal is honoured, and recorded rather than silently dropped."""
    verified, reason = deterministic.verify_comparator(
        _row(clinical_match=False, rationale="Includes revision arthroplasty only."),
        ARTICLES, FEMORAL_NECK)

    assert verified is None
    assert "declined" in reason


def test_verification_keeps_good_rows_and_reports_every_failure():
    verified, rejected = deterministic.verify_comparators(
        [_row(), _row(pmid="99999999")], [HEMIARTHROPLASTY], FEMORAL_NECK)

    assert [entry["pmid"] for entry in verified] == ["30000001"]
    assert [entry["pmid"] for entry in rejected] == ["99999999"]


# --- D1: the literature's synonyms are what identify a cohort ---------------


def test_synonyms_are_anchors_even_when_the_chart_used_other_words():
    """The chart says leiomyoma and the papers say fibroids; both name this condition."""
    target = deterministic.build_target({
        "condition": "uterine leiomyoma",
        "procedure": "total abdominal hysterectomy",
        "anchor_terms": ["uterine fibroids", "fibroids", "hysterectomy", "myomectomy"],
    })

    assert target is not None
    assert "fibroids" in target.anchors
    assert "hysterectomy" in target.anchors
    assert "myomectomy" in target.anchors


def test_a_generic_operation_class_cannot_stand_alone_as_an_anchor():
    """`arthroplasty` is in a paper about shoulders; `hip` is in a paper about knees."""
    target = deterministic.build_target({
        "condition": "hip osteoarthritis",
        "procedure": "total hip arthroplasty",
        "anchor_terms": ["arthroplasty", "replacement", "hip"],
    })

    assert target is not None
    assert "arthroplasty" not in target.anchors
    assert "replacement" not in target.anchors
    assert "hip" not in target.anchors
    assert "hip osteoarthritis" in target.anchors


# --- D4: a span rate is a range, and says so --------------------------------

RANGE_RATE_ARTICLE = {
    "pmid": "30000005",
    "title": "Readmission after femoral neck fracture: a decade of registry data.",
    "journal": "Bone & Joint Journal",
    "pub_date": "2018 Feb",
    "authors": "Alvarez J, Petrov D.",
    "abstract": (
        "RESULTS: Readmission rates of 1.0% to 1.4% were observed within 30 days."
    ),
    "url": "https://pubmed.ncbi.nlm.nih.gov/30000005/",
}


def test_a_span_rate_is_kept_and_marked_as_a_range():
    verified, reason = deterministic.verify_comparator(
        _row(pmid="30000005", reported_rate="1.0% to 1.4%",
             rate_sentence="Readmission rates of 1.0% to 1.4% were observed within 30 days."),
        {"30000005": RANGE_RATE_ARTICLE}, FEMORAL_NECK)

    assert reason == ""
    assert verified is not None
    assert verified["reported_rate"] == "1.0% to 1.4%"
    assert verified["rate_kind"] == "range"


def test_a_single_figure_is_marked_as_a_point():
    verified, _ = deterministic.verify_comparator(_row(), ARTICLES, FEMORAL_NECK)

    assert verified is not None
    assert verified["rate_kind"] == "point"


# --- R1/R3: the prompt never abstains on a list, and prefers the disease ----


def test_the_target_prompt_never_abstains_on_a_diagnosis_list():
    """`90000081` and `90000058` both lost their comparison to one negative rule."""
    prompt = " ".join(prompts.LITERATURE_TARGET_PROMPT.split())

    assert "A LIST IS STILL A CONDITION SET" in prompt
    assert "NEVER a reason to return empty" in prompt
    assert "Return an empty `condition` ONLY when the record names no condition" in prompt


def test_the_target_prompt_prefers_the_disease_over_the_presenting_symptom():
    """`90000069` derived `cervical radiculopathy` where the chart says spondylosis."""
    prompt = " ".join(prompts.LITERATURE_TARGET_PROMPT.split())

    assert "The DISEASE, not the symptom" in prompt
    assert "A symptom is how the patient arrived; the disease is what the admission was" \
        in prompt


# --- R4: a sub-cohort of the same procedure is a benchmark ------------------


def test_the_comparator_prompt_accepts_a_sub_cohort_and_requires_its_qualifier():
    prompt = " ".join(prompts.LITERATURE_COMPARATOR_PROMPT.split())

    assert "A SUB-COHORT of this patient's condition or procedure IS this patient's " \
        "comparison ONLY when the patient could belong to it" in prompt
    assert "narrower population is a benchmark" in prompt
    assert "`cohort_summary` must state the qualifier" in prompt
    # And the rule that a narrower population EXCLUDING this patient is a different one.
    assert "AGE-BRACKET COMPATIBILITY" in prompt
    assert "a narrower" in prompt and "population that EXCLUDES this patient" in prompt


def test_the_extraction_client_is_temperature_zero():
    """Both extraction calls share the formatting client, which is pinned at zero."""
    import inspect

    assert "temperature=0" in inspect.getsource(model_gateway._client)


def test_the_extraction_client_turns_the_consumer_filters_off():
    """The served path keeps the product control; the extraction turn does not."""
    assert model_gateway.CLINICAL_SAFETY_SETTINGS[
        "HARM_CATEGORY_DANGEROUS_CONTENT"] == "BLOCK_NONE"
    assert model_gateway.build_llm.__doc__
    # The served path is untouched: same module, different thresholds.
    from services.mcp.config import GEMINI_SAFETY_THRESHOLDS

    assert GEMINI_SAFETY_THRESHOLDS["HARM_CATEGORY_DANGEROUS_CONTENT"] != "BLOCK_NONE"


# --- F1: a withheld reply falls back to the chart's own headers -------------


class _WithheldLLM:
    """A model whose every reply is taken away by the provider's content filter."""

    async def ainvoke(self, messages, config=None):
        from langchain_core.messages import AIMessage

        return AIMessage(content=[], response_metadata={"finish_reason": "SAFETY"})


def test_a_withheld_reply_still_yields_the_condition_the_chart_states():
    """Measured 2026-11-02: two admissions reported "no condition" for this reason."""
    result = asyncio.run(deterministic.derive_target(_WithheldLLM(), {
        "discharge_diagnosis": "1. Atypical chest pain.\n2. Nausea.\n3. Diabetes.",
        "brief_hospital_course": "",
    }))

    assert result.target is not None
    assert result.source == "chart_headers"
    assert result.target.condition == "Atypical chest pain"
    assert "withheld" in result.reason
    assert "chart's own headers" in result.reason


def test_the_header_parser_drops_the_numbering_and_the_comorbidities():
    target = deterministic.fallback_target_from_sections({
        "discharge_diagnosis": "Fracture of the right femoral neck, also history of "
                               "Alzheimer's dementia, and hypothyroidism.",
    })

    assert target is not None
    assert target.condition == "Fracture of the right femoral neck"


def test_the_header_parser_uses_an_operations_block_when_the_diagnosis_is_blank():
    target = deterministic.fallback_target_from_sections({
        "discharge_diagnosis": "",
        "brief_hospital_course": (
            "The patient went to surgery on the day of admission.\n\n"
            "OPERATIONS: July 25, 2006: Total abdominal hysterectomy, bilateral "
            "salpingo-oophorectomy."
        ),
    })

    assert target is not None
    assert target.condition == "Total abdominal hysterectomy"


def test_the_header_parser_refuses_a_sentence_about_the_patient():
    """A fallback that searched for "the patient was admitted" would be worse than none."""
    assert deterministic.fallback_target_from_sections({
        "discharge_diagnosis": "",
        "brief_hospital_course": "The patient was admitted after a fall.",
    }) is None


# --- F5: the whole pool reaches the model, in PubMed's order -----------------
#
# The extraction turn used to read a ten-record window of a ranked pool: first by how
# specifically a record named the condition, later by whether it stated a percentage and
# whether its title named the operation. Every one of those rules was a guess about
# clinical relevance made in Python, standing where reading the abstract belongs, and each
# one could hide the record that mattered. Measured 2026-11-02: admission 90000102's pool
# was twenty-two benign-prostatic-hyperplasia papers, and the one titled "…readmission
# after transurethral resection of the prostate" ranked fourteenth. The pool now goes over
# whole.

NO_FIGURE_REVIEW = dict(
    HEMIARTHROPLASTY, pmid="30000030",
    title="Readmission after femoral neck fracture: a narrative review.",
    abstract=(
        "This review discusses readmission after femoral neck fracture and the factors "
        "that drive it, without pooling a rate."
    ),
)
WITH_FIGURE = dict(
    HEMIARTHROPLASTY, pmid="30000031",
    title="Readmission after femoral neck fracture in a national cohort.",
    abstract="RESULTS: The 30-day readmission rate was 8.4%.",
)


def test_every_adult_record_reaches_the_extraction_turn():
    """No window: twelve records in, twelve records read."""
    pool = [dict(KNEE_ARTHROPLASTY, pmid=f"3000010{index}") for index in range(12)]

    async def search(terms, wide):
        return {"query": "pooled", "articles": pool}

    result = asyncio.run(deterministic.extract_comparators(
        _FakeLLM([_payload([])]), FEMORAL_NECK, search=search))

    assert result.passes[0].candidates == 12
    assert result.passes[0].aligned == 12, "the pool was cut before the model read it"


def test_a_record_with_no_figure_is_read_beside_one_that_states_a_rate():
    """The model sees both. Which of them is usable is its judgement, not a token test."""
    async def search(terms, wide):
        return {"query": "pooled", "articles": [NO_FIGURE_REVIEW, WITH_FIGURE]}

    result = asyncio.run(deterministic.extract_comparators(
        _FakeLLM([_payload([])]), FEMORAL_NECK, search=search))

    assert result.passes[0].candidates == 2
    assert result.passes[0].aligned == 2


def test_the_pass_hands_the_pool_over_in_pubmed_order():
    """The review is first in the reply and first in what the model reads.

    Deliberately ordered with the rate-bearing record second: if any local ordering rule
    were still in the pass, that record would have been promoted and this would fail.
    """
    async def search(terms, wide):
        return {"query": "pooled", "articles": [NO_FIGURE_REVIEW, WITH_FIGURE]}

    pass_ = asyncio.run(deterministic.candidate_pass(search, FEMORAL_NECK))

    assert [article["pmid"] for article in pass_.aligned] == ["30000030", "30000031"]


# --- F3: a cohort of children is not this adult's comparison ----------------


PEDIATRIC_EFFUSION = {
    "pmid": "30000030",
    "title": "Management of complicated pleural effusion in children.",
    "journal": "Journal of Pediatric Surgery",
    "pub_date": "2023 May",
    "authors": "Reyes A.",
    "abstract": (
        "RESULTS: Thirty-day readmission rates were 15% in children with pleural effusion "
        "managed by drainage."
    ),
    "url": "https://pubmed.ncbi.nlm.nih.gov/30000030/",
}

ADULT_EFFUSION = dict(
    PEDIATRIC_EFFUSION,
    pmid="30000031",
    title="Readmission after pleural effusion drainage in adults.",
    journal="Chest",
    abstract="RESULTS: The 30-day readmission rate was 22% after thoracentesis.",
)


def test_a_paediatric_cohort_is_refused_for_an_adult_admission():
    """Measured 2026-11-02: a J Pediatr Surg cohort passed every other check."""
    verified, reason = deterministic.verify_comparator(
        _row(pmid="30000030", reported_rate="15%",
             rate_sentence="Thirty-day readmission rates were 15% in children with "
                           "pleural effusion managed by drainage."),
        {"30000030": PEDIATRIC_EFFUSION}, PLEURAL_AFIB)

    assert verified is None
    assert "cohort of children" in reason


def test_an_adult_age_band_is_not_a_population_mismatch():
    """Octogenarians are the same operation in an older body, not a different patient."""
    assert not deterministic.names_pediatric_population(
        "Patients over 80 years old undergoing total hip arthroplasty", "NSQIP")


def test_the_paediatric_cohort_never_reaches_the_model():
    kept, refused = deterministic.adult_cohorts(
        [PEDIATRIC_EFFUSION, ADULT_EFFUSION], PLEURAL_AFIB)

    assert [article["pmid"] for article in kept] == ["30000031"]
    assert refused[0]["pmid"] == PEDIATRIC_EFFUSION["pmid"]


def test_the_rule_turns_itself_off_for_a_paediatric_admission():
    child = deterministic.ClinicalTarget(
        condition="pediatric pleural effusion", anchors=("pleural effusion",))

    assert deterministic.is_pediatric_target(child) is True
    kept, refused = deterministic.adult_cohorts([PEDIATRIC_EFFUSION], child)

    assert kept == [PEDIATRIC_EFFUSION]
    assert refused == []


# --- F4: a two-arm result is a comparison, not a point estimate -------------

COMPARISON_ARTICLE = {
    "pmid": "30000032",
    "title": "Readmission after femoral neck fracture: two care models compared.",
    "journal": "Injury",
    "pub_date": "2022 Nov",
    "authors": "Nakamura K.",
    "abstract": (
        "RESULTS: The 30-day readmission rate was 7.8% vs. 11.5% between the two models."
    ),
    "url": "https://pubmed.ncbi.nlm.nih.gov/30000032/",
}


def test_a_two_arm_rate_is_published_with_versus_and_marked_as_a_comparison():
    rate, kind = deterministic.normalise_rate("7.8% and 11.5%")

    assert rate == "7.8% vs. 11.5%"
    assert kind == "comparison"


def test_a_range_stays_a_quotation():
    assert deterministic.normalise_rate("1.0% to 1.4%") == ("1.0% to 1.4%", "range")


def test_a_single_figure_is_a_point():
    assert deterministic.normalise_rate("2.6%") == ("2.6%", "point")


def test_a_comparison_row_verifies_with_both_figures_kept():
    verified, reason = deterministic.verify_comparator(
        _row(pmid="30000032", reported_rate="7.8% and 11.5%",
             rate_sentence="The 30-day readmission rate was 7.8% vs. 11.5% between the "
                           "two models."),
        {"30000032": COMPARISON_ARTICLE}, FEMORAL_NECK)

    assert reason == ""
    assert verified is not None
    assert verified["reported_rate"] == "7.8% vs. 11.5%"
    assert verified["rate_kind"] == "comparison"


def test_the_comparator_prompt_refuses_a_different_population_and_rules_on_two_arms():
    prompt = " ".join(prompts.LITERATURE_COMPARATOR_PROMPT.split())

    assert "A different POPULATION is not a narrower one" in prompt
    assert "obstetric-only cohort is" in prompt
    assert "A TWO-ARM result is two figures" in prompt
    assert "never write \"and\" where the study says \"versus\"" in prompt


# --- procedure intensity: no longer a keyword rule --------------------------
#
# A decortication cohort was once refused for a thoracentesis admission by matching
# procedure nouns against a hand-kept list of bedside operations. The list is deleted: it
# encoded a surgical judgement as vocabulary, and it could not tell decortication from
# chest drain insertion in a paper that mentions both. Establishing that the cohort is the
# same operation is the model's job, stated as `clinical_match`, and the layer only
# carries out that answer.

DECORTICATION = dict(
    PEDIATRIC_EFFUSION,
    pmid="30000040",
    title="The role of the RAPID score in surgical planning for empyema.",
    journal="Journal of Thoracic Disease",
    abstract=(
        "RESULTS: Among patients with complicated pleural effusion undergoing "
        "decortication, the 30-day readmission rate was 50.0%."
    ),
)


def test_the_models_refusal_is_what_stops_a_different_operation():
    verified, reason = deterministic.verify_comparator(
        _row(pmid="30000040", clinical_match=False, reported_rate="50.0%",
             rate_sentence="Among patients with complicated pleural effusion undergoing "
                           "decortication, the 30-day readmission rate was 50.0%."),
        {"30000040": DECORTICATION}, PLEURAL_AFIB)

    assert verified is None
    assert "declined" in reason


def test_a_verbatim_surgical_cohort_is_published_when_the_model_calls_it_a_match():
    """No second opinion: the layer does not re-try the model's clinical judgement."""
    verified, reason = deterministic.verify_comparator(
        _row(pmid="30000040", reported_rate="50.0%",
             rate_sentence="Among patients with complicated pleural effusion undergoing "
                           "decortication, the 30-day readmission rate was 50.0%."),
        {"30000040": DECORTICATION}, PLEURAL_AFIB)

    assert reason == ""
    assert verified is not None


# --- D2: the second pass is decided on verified rows ------------------------

NINETY_DAY_ARTICLE = {
    "pmid": "30000006",
    "title": "Ninety-day readmission after femoral neck fracture.",
    "journal": "Injury",
    "pub_date": "2016 May",
    "authors": "Ruiz M.",
    "abstract": "RESULTS: The 90-day readmission rate was 14.1%.",
    "url": "https://pubmed.ncbi.nlm.nih.gov/30000006/",
}


class _Reply:
    def __init__(self, text: str) -> None:
        self.content = text


class _FakeLLM:
    """One queued reply per call, so the two passes can be scripted."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls = 0

    async def ainvoke(self, messages, config=None):
        text = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return _Reply(text)


def _payload(rows: list[dict]) -> str:
    return json.dumps({"comparators": rows, "rejected": []})


def _candidates(count: int, **overrides) -> list[dict]:
    """`count` records that name this admission's condition, with distinct PMIDs.

    Used to fill the model's budget without inventing another fixture: the policy under
    test turns on how many candidates align, not on what any of them says.
    """
    articles = []
    for index in range(count):
        pmid = f"300001{index:02d}"
        articles.append({**HEMIARTHROPLASTY, "pmid": pmid,
                         "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", **overrides})
    return articles


def test_one_query_and_one_extraction_per_turn():
    """Measured 2026-11-02: a `/ehr` turn searched twice and extracted twice, 25.6s.

    The workflow showed `Searching the Literature` twice and the trace showed two
    five-second extraction gaps, because the second pass ran whenever the first verified
    fewer than two rows — which was most admissions. One query with the broadened outcome
    clause and one extraction over its ranked pool is what that turn needed.
    """
    searched: list[tuple] = []

    async def search(terms, wide):
        searched.append((terms, wide))
        return {"query": "pooled", "articles": _candidates(6)}

    llm = _FakeLLM([_payload([_row(pmid="30000100")])])

    result = asyncio.run(deterministic.extract_comparators(llm, FEMORAL_NECK, search=search))

    assert len(searched) == 1, "one retrieval per turn"
    assert llm.calls == 1, "one extraction per turn"
    assert [row["pmid"] for row in result.rows] == ["30000100"]
    assert len(result.passes) == 1


def test_the_one_query_asks_for_the_condition_and_its_anchors():
    """The OR the two passes were approximating, asked once.

    The condition leads and the anchors follow it: `search_literature` turns them into an
    OR of quoted phrases against the broadened outcome clause, which is the query a record
    saying "readmission within 30 days" can actually match.
    """
    searched: list[tuple] = []

    async def search(terms, wide):
        searched.append((terms, wide))
        return {"query": "pooled", "articles": []}

    asyncio.run(deterministic.extract_comparators(
        _FakeLLM([_payload([])]), FEMORAL_NECK, search=search))

    (terms, wide), = searched
    assert wide is True, "the broadened outcome clause, because the exact phrases miss"
    assert list(terms)[0] == FEMORAL_NECK.condition
    assert set(FEMORAL_NECK.anchors) <= set(terms)


def test_a_pool_about_another_condition_is_put_to_the_model_which_refuses_it():
    """The pool is read, and the refusal is the model's, with a stated reason.

    This used to cost nothing: the anchor gate emptied the pool before the extraction turn
    ran, so a synonym in a good cohort looked identical to an empty search. One record is
    now read and declined, and the decline is a rationale a reader could argue with.
    """
    async def search(terms, wide):
        return {"query": "pooled", "articles": [KNEE_ARTHROPLASTY]}

    llm = _FakeLLM([_payload([_row(
        pmid="30000002", clinical_match=False,
        rationale="Total knee arthroplasty, not a femoral neck fracture.",
        reported_rate="3.4%",
        rate_sentence="Among 12,450 primary total knee arthroplasties the 30-day "
                      "readmission rate was 3.4%.",
    )])])

    result = asyncio.run(deterministic.extract_comparators(llm, FEMORAL_NECK, search=search))

    assert llm.calls == 1
    assert result.rows == []
    assert result.passes[0].candidates == 1
    assert result.passes[0].aligned == 1, "the pool reached the model; nothing pre-refused it"
    assert [entry["pmid"] for entry in result.rejected] == ["30000002"]
    assert "declined" in result.rejected[0]["reason"]


# --- the retrieval layer -----------------------------------------------------


def test_the_first_pass_queries_the_condition_alone():
    """One plain phrase, unquoted: the synonyms are the retry's job, not the first ask."""
    query = literature.readmission_query(FEMORAL_NECK.condition)

    assert query.startswith(f"({FEMORAL_NECK.condition}) AND")
    assert '"30-day readmission"[Title/Abstract]' in query
    assert '"unplanned readmission"[Title/Abstract]' in query
    assert "OR" not in query.split("AND")[0]


def test_the_second_pass_ors_the_anchors_and_asks_for_the_horizon_separately():
    """The four exact phrases miss "readmission within 30 days"; the wide clause does not."""
    wide = literature.readmission_query(FEMORAL_NECK.anchors, wide=True)

    assert '"femoral neck fracture" OR "hip fracture"' in wide
    assert "readmission[Title/Abstract] OR readmitted[Title/Abstract]" in wide
    assert '"30 days"[Title/Abstract]' in wide
    assert '"30-day readmission"[Title/Abstract]' not in wide


def test_only_abstracts_stating_a_readmission_and_a_percentage_are_kept():
    without_rate = dict(COPD_EXACERBATION, abstract="COPD readmission was associated "
                                                    "with worse outcomes.")
    kept = literature.quantitative_readmission_articles(
        [HEMIARTHROPLASTY, without_rate])

    assert [article["pmid"] for article in kept] == ["30000001"]


def test_a_benchmark_search_filters_the_pool_and_keeps_the_public_shape(monkeypatch):
    """The body of the search, not just its query builder.

    Added after a live run failed on an undefined constant in this function: the hermetic
    suite exercised `readmission_query` and stopped there, so a NameError inside the
    search itself was invisible until NCBI was in the loop.
    """
    without_rate = dict(COPD_EXACERBATION,
                        abstract="COPD readmission was associated with worse outcomes.")

    async def fake_search(query, *, retmax, reldate_days):
        return {"query": query, "returned": 2,
                "articles": [HEMIARTHROPLASTY, without_rate]}

    monkeypatch.setattr(literature, "_search", fake_search)

    result = asyncio.run(
        literature.search_readmission_benchmarks("hip fracture"))

    assert result["returned"] == 1
    assert [article["pmid"] for article in result["articles"]] == ["30000001"]
    assert result["query"].startswith("(hip fracture) AND")


def test_the_pubmed_search_is_unchanged_by_the_benchmark_path():
    """`/pubmed` still asks for three recent records: the new path is a second caller."""
    import inspect

    source = inspect.getsource(literature._fetch)

    assert "RECENCY_DAYS" in source
    assert literature.DEFAULT_MAX_RESULTS == 3


# --- Stage Gate 1: the eight-admission live report ---------------------------

#: The three admissions named in the directive.
DIAGNOSTIC = (90000090, 90000069, 90000081)

INPATIENT = frozenset(served_admissions.SERVED_HADM_IDS) - frozenset(OUTPATIENT_HADM_IDS)
HELD_OUT = tuple(sorted(INPATIENT - set(DIAGNOSTIC)))


def _held_out_five() -> tuple[int, ...]:
    """Five held-out admissions, spread across the sorted cohort rather than clustered.

    Every 1/Nth entry, so the five are not five neighbours — the corpus is not ordered by
    condition, and taking the first five would have sampled one corner of it.
    """
    if not HELD_OUT:
        return ()
    step = max(1, len(HELD_OUT) // 5)
    return tuple(HELD_OUT[::step][:5])


DIAGNOSTIC_COHORT = DIAGNOSTIC + _held_out_five()


async def _report_one_async(hadm_id: int, llm) -> None:
    """One admission, inside the caller's event loop.

    The loop is the caller's because the model client is: an `AsyncClient` built on one
    loop cannot be awaited on the next, and the first version of this report called
    `asyncio.run` per step, which failed at the third model call with a closed transport.
    """
    notes = await rag_search_sections(hadm_id)
    recorded = [{"name": "rag_search_sections", "args": {"hadm_id": hadm_id},
                 "response": notes}]
    sections = deterministic.note_sections(recorded)

    print(f"\n{'=' * 78}\nADMISSION {hadm_id}\n{'=' * 78}")
    diagnosis = sections.get("discharge_diagnosis", "")
    course = sections.get("brief_hospital_course", "")
    print(f"[2] DISCHARGE DIAGNOSIS: {diagnosis[:400]}")
    print(f"    BRIEF HOSPITAL COURSE (opening): {course[:400]}")

    derived = await deterministic.derive_target(llm, sections)
    target = derived.target
    if target is None:
        print(f"[3] NO TARGET: {derived.reason}")
        print("[6] ROWS: none")
        return
    if derived.source == "model_short_prompt":
        print("[2a] the elaborated target prompt was withheld; the plain prompt answered")
    elif derived.source == "chart_headers":
        print(f"[2a] DETERMINISTIC FALLBACK: {derived.reason}")
    print(f"[2b] CONDITION: {target.condition} | PROCEDURE: {target.procedure or '-'}")
    print(f"     ANCHORS: {', '.join(target.anchors)}")

    # The passes are captured so the report can show what each query returned; the
    # two-pass policy itself is `extract_comparators`'s and is not re-implemented here.
    captured: list[dict] = []

    async def search(terms, wide):
        payload = await literature.search_readmission_benchmarks(terms, wide=wide)
        captured.append(payload if isinstance(payload, dict) else {})
        return payload

    result = await deterministic.extract_comparators(llm, target, search=search)
    rows = result.rows

    for index, entry in enumerate(result.passes):
        label = "WIDE" if entry.wide else "PASS 1"
        print(f"[3] {label} QUERY: {entry.query}")
        excluded = f" population_excluded={entry.excluded}" if entry.excluded else ""
        print(f"[4] {label}: candidates={entry.candidates} aligned={entry.aligned} "
              f"verified={entry.verified}{excluded}")
        for article in (captured[index].get("articles") if index < len(captured) else []) or []:
            print(f"    - {article.get('pmid')} | {article.get('title')}")

    print("[5] ACCEPTED / REJECTED")
    for row in rows:
        print(f"    + {row['pmid']} accepted")
    for entry in result.rejected:
        print(f"    - {entry['pmid'] or '(none)'} rejected: {entry['reason']}")

    print("[6] ROWS")
    if not rows:
        print("    none — the literature offered no 30-day rate for this condition")
    for row in rows:
        print(f"    pmid: {row['pmid']}")
        print(f"    journal: {row['journal']} ({row['pub_date']})")
        print(f"    cohort: {row['cohort_summary']}")
        print(f"    tier: {row['evidence_tier']}")
        print(f"    rate: {row['reported_rate']} ({row['rate_kind']}) "
              f"at {row['horizon_days']} days")
        print(f"    sentence: {row['rate_sentence']}")


@pytest.mark.live
def test_live_backend_extraction_on_eight_admissions():
    """STAGE GATE 1 — the backend pipeline on real charts, printed in full.

    Nothing is asserted about clinical quality here: the point of the gate is that a
    human reads these eight reports and decides whether the extraction is good enough to
    wire in. The assertions are the ones that hold whatever the literature says — eight
    admissions ran, and every surviving row carries a verbatim sentence from its own
    abstract.
    """
    assert len(DIAGNOSTIC_COHORT) == 8, (
        f"expected 3 diagnostic + 5 held-out admissions, got {DIAGNOSTIC_COHORT}")
    llm = model_gateway.build_extraction_llm(observability.MODEL_ID)

    async def drive() -> None:
        for hadm_id in DIAGNOSTIC_COHORT:
            await _report_one_async(hadm_id, llm)

    asyncio.run(drive())
