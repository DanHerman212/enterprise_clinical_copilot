"""Stage 9A — one ranker, in the tool, that counts the cohort.

The defect these pin, measured on the deployed build `e60fa3a` (2026-11-09):

* `90000090` (displaced femoral neck fracture) drew an empty comparison matrix over a pool
  that contained two cohorts reporting a 30-day rate. The extraction read ten abstracts in
  PubMed's order; one of them, `39420969`, states that males "were more likely to be
  readmitted at 30 (P < .001)" — a finding without a figure — and the model proposed that
  one row. `34378796` (13.8%) and `36002345` (34.2% vs 7.5%) were in the same pool and were
  never turned into columns. `40664407` ranked eleventh and the cap of ten cut it.
* `90000069` (cervical spondylotic myelopathy) sorted `38504222` — 33 patients, "no patient
  was readmitted within 30 days" — above `31609801` — 1,140 patients from ACS-NSQIP at 3.4%
  — because the ranker was three booleans about vocabulary and had no term for size.

Two changes, and these tests are the contract for both:

1. `services/mcp/tools/literature.py` owns the whole retrieval decision — which query,
   whether to ask a second one, whose population, and in what order a model reads the
   pool — and `score_benchmark_candidate` scores a record on what it actually states,
   including `min(20, 5*log10(N))` for the cohort size.
2. `extract_comparators` spends the prompt on the abstracts that can produce a row, and
   orders the rows it gets back by the same evidence the pool was ranked on.

The abstracts below are verbatim PubMed records fetched through E-utilities on 2026-11-09.
Verbatim matters: every assertion downstream is a regex over prose a real paper wrote, and a
paraphrase here would let a rule pass that the record itself defeats.

Offline: no network call is made by any test in this file.
"""

from __future__ import annotations

import asyncio
import json

from langchain_core.messages import AIMessage

from services.agent.engine import deterministic
from services.mcp.tools import literature

# --------------------------------------------------------------------------------------
# The records, as PubMed returns them
# --------------------------------------------------------------------------------------

RECORDS = {
    # 90000090 — 13.8% at 30 days, and the cohort is named 1,859 patients in Aragon.
    "34378796": {
        "pmid": "34378796",
        "title": ("Differences in health outcomes for high-need high-cost patients across "
                  "high-income countries."),
        "journal": "Health Services Research",
        "journal_abbrev": "Health Serv Res",
        "pub_date": "2021 Dec",
        "authors": "Figueroa JF, et al.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/34378796/",
        "abstract": (
            "OBJECTIVE: This study explores variations in outcomes of care for two types of "
            "patient personas-an older frail person recovering from a hip fracture and a "
            "multimorbid older patient with congestive heart failure (CHF) and diabetes. "
            "PRINCIPAL FINDINGS: The hip sample across ranged from 1859 patients in Aragon, "
            "Spain, to 42,849 in France. Readmission rates for all periods were higher for "
            "the CHF persona than the hip persona. At 30 days, the average readmission rate "
            "for the hip persona was 13.8% and 27.6% for the CHF persona. CONCLUSION: Across "
            "11 countries, there are meaningful differences in health system outcomes for "
            "two types of patients."
        ),
    },
    # 90000090 — 34.2% vs 7.5%, inside a sentence that also reports mortality.
    "36002345": {
        "pmid": "36002345",
        "title": ("Clostridium difficile colitis portends poor outcomes in lower extremity "
                  "orthopaedic trauma surgery."),
        "journal": "Injury",
        "journal_abbrev": "Injury",
        "pub_date": "2022 Oct",
        "authors": "Ganta A, et al.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/36002345/",
        "abstract": (
            "METHODS: The ACS-NSQIP (2015-2019) was queried for patients who underwent "
            "surgical fixation of lower extremity fractures. RESULTS: 95,532 patients were "
            "included, 681 (0.71%) of whom developed CDC. Patients with CDC had "
            "significantly increased 30-day mortality rates (10.6% vs 4.4%; OR 1.80, 95% CI "
            "1.41-2.31), readmission (34.2% vs 7.5%; OR 5.13, 95% CI 4.36-6.05, and length "
            "of stay (7.5 days vs 5.3 days) compared to patients without CDC."
        ),
    },
    # 90000090 — the row the model proposed: a 30-day readmission FINDING with no figure.
    "39420969": {
        "pmid": "39420969",
        "title": ("Hip Fracture Patterns, Hospital Course, and Mortality Differ Between "
                  "Males and Females."),
        "journal": "Geriatric Orthopaedic Surgery & Rehabilitation",
        "journal_abbrev": "Geriatr Orthop Surg Rehabil",
        "pub_date": "2024",
        "authors": "Piple AS, et al.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/39420969/",
        "abstract": (
            "MATERIALS AND METHODS: This retrospective case series was performed in a "
            "metropolitan healthcare system involving 2996 hip fracture patients >59 years "
            "old who underwent surgical management. RESULTS: Males were more likely to be "
            "readmitted at 30 (P < .001) and 90 (P = .015) days after discharge. The 30-day "
            "mortality was higher for males vs females (6.6% vs 4.5%, P = .015)."
        ),
    },
    # 90000090 — eleventh in PubMed's order, and a rate from the national audit.
    "40664407": {
        "pmid": "40664407",
        "title": "Understanding readmission after hip fracture: a mixed methods study "
                 "protocol.",
        "journal": "BMJ Open",
        "journal_abbrev": "BMJ Open",
        "pub_date": "2025 Jul",
        "authors": "Patel R, et al.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/40664407/",
        "abstract": (
            "INTRODUCTION: Around 75 000 people suffer from hip fractures yearly in the "
            "United Kingdom (UK) leading to significant mortality and morbidity. Although "
            "mortality has dropped from 8% to 5% between 2013 and 2023 after hip fractures, "
            "those undergoing surgery for hip fractures have a 30-day readmission rate "
            "which has remained stagnant at around 11% over the same decade in the UK."
        ),
    },
    # 90000069 — 1,140 patients from ACS-NSQIP, 3.4% at 30 days.
    "31609801": {
        "pmid": "31609801",
        "title": ("30-Day Postoperative Morbidity and Readmission Following Revision "
                  "Anterior Cervical Discectomy and Fusion (ACDF)."),
        "journal": "Clinical Spine Surgery",
        "journal_abbrev": "Clin Spine Surg",
        "pub_date": "2019 Dec",
        "authors": "Bovonratwet P, et al.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/31609801/",
        "abstract": (
            "METHODS: The 2012-2016 American College of Surgeons-National Surgical Quality "
            "Improvement Program (ACS-NSQIP) database was queried using Current Procedural "
            "Terminology codes for ACDF. A total of 1140 patients were retrieved for "
            "analysis. RESULTS: A 30-day readmission rate was 3.4% (N=39) after a revision "
            "ACDF."
        ),
    },
    # 90000069 — 33 patients, and the zero it reports is the sentence that was published.
    "38504222": {
        "pmid": "38504222",
        "title": ("Safety and efficacy of day anterior cervical discectomy and fusion "
                  "procedure for degenerative cervical spondylosis: a retrospective "
                  "analysis."),
        "journal": "BMC Musculoskeletal Disorders",
        "journal_abbrev": "BMC Musculoskelet Disord",
        "pub_date": "2024 Mar",
        "authors": "Xu B, et al.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/38504222/",
        "abstract": (
            "RESULTS: All 33 patients (23 women and 10 men) underwent successful surgery and "
            "experienced significant symptomatic and neurological improvements. During the "
            "follow-up period, no patient was readmitted within 30 days after discharge; "
            "however, an incisional hematoma was reported in one patient on the 6th day "
            "after discharge, which was cured by pressure dressing."
        ),
    },
}

HIP = deterministic.ClinicalTarget(
    condition="displaced femoral neck fracture",
    procedure="hemiarthroplasty",
    anchors=("femoral neck fracture", "hip fracture"),
    patient_age=81,
    patient_sex="F",
)

CERVICAL = deterministic.ClinicalTarget(
    condition="cervical spondylotic myelopathy",
    procedure="anterior cervical discectomy and fusion",
    anchors=("cervical spondylotic myelopathy",),
    patient_age=41,
    patient_sex="female",
)


def _article(pmid: str, **overrides) -> dict:
    return {**{"pmid": pmid, **RECORDS[pmid]}, **overrides}


def _score(pmid: str, target: deterministic.ClinicalTarget) -> dict:
    return literature.score_benchmark_candidate(
        _article(pmid), condition=target.condition, anchors=target.anchors)


# --------------------------------------------------------------------------------------
# 1. The ranker scores what the record states, including how many patients it studied
# --------------------------------------------------------------------------------------


def test_the_records_that_state_a_30_day_rate_outrank_the_one_that_states_only_a_finding():
    """`90000090`: `39420969` was the row the extraction proposed, and it has no figure.

    "Males were more likely to be readmitted at 30 (P < .001)" is a true statement about
    readmission and not a benchmark: there is nothing in it to put beside the patient's
    own estimate. The two records that do state a 30-day rate have to rank above it.
    """
    hip = {pmid: _score(pmid, HIP) for pmid in
           ("34378796", "36002345", "39420969", "40664407")}

    assert hip["34378796"]["has_readmission_rate_sentence"] is True
    assert hip["36002345"]["has_readmission_rate_sentence"] is True
    assert hip["39420969"]["has_readmission_rate_sentence"] is False
    assert hip["34378796"]["rank_score"] > hip["39420969"]["rank_score"]
    assert hip["36002345"]["rank_score"] > hip["39420969"]["rank_score"]


def test_a_1_140_patient_cohort_outranks_a_33_patient_series_reporting_zero():
    """`90000069`: the matrix showed the case series and not the registry.

    Both records are about this patient's operation. One studied 33 people and readmitted
    none; the other studied 1,140 from a national database at 3.4%. The regression is
    real and it is the wrong benchmark, so size has to break the tie.
    """
    registry = _score("31609801", CERVICAL)
    series = _score("38504222", CERVICAL)

    assert registry["sample_size_n"] == 1140
    assert series["sample_size_n"] == 33
    assert registry["has_readmission_rate_sentence"] is True
    assert series["has_readmission_rate_sentence"] is False
    assert registry["rank_score"] > series["rank_score"]


def test_the_sample_size_is_read_through_the_words_the_paper_used():
    """Three real phrasings, each of which defeated a naive pattern.

    `39420969` says "2996 hip fracture patients"; `40664407` writes "75 000 people", where
    the digits after the space parsed as a cohort of zero; `38504222` says "All 33
    patients", under the three digits a `\\d{3,}` pattern required.
    """
    assert literature.sample_size(RECORDS["39420969"]["abstract"]) == 2996
    assert literature.sample_size(RECORDS["40664407"]["abstract"]) == 75000
    assert literature.sample_size(RECORDS["38504222"]["abstract"]) == 33


def test_a_complication_reported_in_two_patients_is_not_a_cohort_size():
    sentence = ("An incisional hematoma was reported in 2 patients and required "
                "evacuation in 1 patient.")

    assert literature.sample_size(sentence) is None


def test_a_cohort_named_as_hospitalisations_is_still_a_cohort():
    """`37037836` says "a total of 782 hospitalizations for AECOPD were enrolled".

    Not every paper counts patients. The live run on 2026-11-09 read that abstract with no
    size at all, because only "patients", "admissions" and their siblings were recognised.
    """
    assert literature.sample_size(
        "A total of 782 hospitalizations for AECOPD were enrolled.") == 782
    assert literature.sample_size(
        "We reviewed 1,204 encounters in the emergency department.") == 1204


def test_the_verbal_zero_is_penalised_rather_than_deleted():
    """It is a real finding, and it is the finding of a small cohort.

    `38504222` reads "no patient was readmitted within 30 days". Dropping the record would
    discard a true result; ranking it as if it were a registry would publish a 0% rate from
    33 people beside an 81-year-old's estimate. It stays, at the bottom, with its size on it.
    """
    series = _score("38504222", CERVICAL)

    assert series["has_readmission_rate_sentence"] is False
    assert series["rank_score"] < 0


# --------------------------------------------------------------------------------------
# 2. The tool owns the second query, the cap and the order
# --------------------------------------------------------------------------------------


def _stub_search(monkeypatch, first: list[dict], second: list[dict] | None = None):
    """Stand in for E-utilities, and record the terms the tool asked for."""
    asked: list[str] = []

    async def search(query: str, *, retmax: int, reldate_days: int):
        asked.append(query)
        pool = first if len(asked) == 1 else (second or [])
        return {"query": query, "returned": len(pool), "articles": list(pool)}

    monkeypatch.setattr(literature, "_search", search)
    return asked


def test_the_tool_broadens_a_thin_pool_and_merges_it_by_pmid(monkeypatch):
    """`90000047`: four records is not a pool, and the horizon clause is usually why.

    The second query drops the literal 30-day requirement. Consolidating that here is the
    point of the cutover: the agent used to re-ask, and the two halves drifted.
    """
    strict_only = [_article("39420969"), _article("34378796")]
    broadened = [_article("34378796"), _article("36002345"), _article("40664407")]
    asked = _stub_search(monkeypatch, strict_only, broadened)

    result = asyncio.run(literature.search_readmission_benchmarks(
        ["displaced femoral neck fracture", "femoral neck fracture", "hip fracture"]))

    assert len(asked) == 2, "the tool did not ask the broadened question"
    assert '"30-day readmission"[Title/Abstract]' in asked[0], (
        "the first query is not horizon-pinned")
    assert '"30-day"[Title/Abstract]' not in asked[0]
    assert '"30-day"[Title/Abstract]' in asked[1], "the second query kept the horizon"
    pmids = [article["pmid"] for article in result["articles"]]
    assert pmids.count("34378796") == 1, "the merged pool was not de-duplicated"
    assert set(pmids) == {"39420969", "34378796", "36002345", "40664407"}


def test_a_pool_that_fills_the_threshold_is_not_asked_again(monkeypatch):
    full = [_article(pmid) for pmid in RECORDS]
    full += [{**RECORDS["36002345"], "pmid": f"7000{index:04d}"} for index in range(2)]
    asked = _stub_search(monkeypatch, full)

    asyncio.run(literature.search_readmission_benchmarks(
        ["displaced femoral neck fracture", "hip fracture"]))

    assert len(asked) == 1


def test_the_twelfth_record_is_not_cut_before_it_is_ranked(monkeypatch):
    """`40664407` ranked eleventh on `90000090` and the old cap of ten deleted it.

    The pool the tool returns is ranked, and it is deep enough that a rate-bearing record
    sitting low in PubMed's order still reaches the extraction.
    """
    filler = [
        {**RECORDS["39420969"], "pmid": f"5000{index:04d}"} for index in range(11)
    ]
    asked = _stub_search(monkeypatch, [*filler, _article("40664407"), _article("34378796")])

    result = asyncio.run(literature.search_readmission_benchmarks(
        ["displaced femoral neck fracture", "femoral neck fracture", "hip fracture"]))

    pmids = [article["pmid"] for article in result["articles"]]
    assert asked, "the tool did not search"
    assert "40664407" in pmids
    assert pmids.index("40664407") < pmids.index("50000000"), (
        "a protocol quoting the national rate ranked below a record with no figure")


def test_a_paediatric_cohort_never_takes_a_slot(monkeypatch):
    paediatric = {**RECORDS["36002345"], "pmid": "60000001",
                  "title": "Readmission after paediatric femoral fracture fixation.",
                  "abstract": "A total of 220 children with femoral fractures were reviewed. "
                              "The 30-day readmission rate was 4.1%."}
    _stub_search(monkeypatch, [_article("34378796"), paediatric])

    result = asyncio.run(literature.search_readmission_benchmarks(
        ["displaced femoral neck fracture", "hip fracture"]))

    assert [article["pmid"] for article in result["articles"]] == ["34378796"]


# --------------------------------------------------------------------------------------
# 3. The extraction reads the rate-bearing abstracts first, and orders what it returns
# --------------------------------------------------------------------------------------


class _Reply:
    """A model that answers with one JSON payload and remembers the prompt it was given."""

    def __init__(self, payload):
        self._payload = payload
        self.prompts: list[str] = []

    async def ainvoke(self, messages, config=None):
        self.prompts.append(str(messages[-1].content))
        return AIMessage(content=json.dumps(self._payload))


def _row(pmid: str, rate: str, sentence: str, summary: str) -> dict:
    return {
        "pmid": pmid,
        "clinical_match": True,
        "horizon_days": 30,
        "reported_rate": rate,
        "rate_sentence": sentence,
        "cohort_summary": summary,
        "evidence_tier": "Retrospective cohort",
    }


SENTENCE_31609801 = "A 30-day readmission rate was 3.4% (N=39) after a revision ACDF."
SENTENCE_38504222 = (
    "During the follow-up period, no patient was readmitted within 30 days after "
    "discharge; however, an incisional hematoma was reported in one patient on the 6th day "
    "after discharge, which was cured by pressure dressing."
)


def test_the_extraction_is_given_the_rate_bearing_abstracts_and_the_cohort_sizes():
    """Ten abstracts with one usable cohort in them is a prompt that cannot work."""
    pool = [
        _article(pmid, **literature.score_benchmark_candidate(
            _article(pmid), condition=CERVICAL.condition, anchors=CERVICAL.anchors))
        for pmid in ("38504222", "31609801")
    ]
    llm = _Reply({"comparators": []})

    async def search(terms, wide):
        return {"query": "strict", "articles": pool}

    asyncio.run(deterministic.extract_comparators(llm, CERVICAL, search=search))

    prompt = llm.prompts[0]
    assert prompt.index('pmid="31609801"') < prompt.index('pmid="38504222"'), (
        "the registry was read after the case series")
    assert "cohort size: 1140" in prompt
    assert "cohort size: 33" in prompt


def test_verified_rows_are_ordered_by_evidence_and_not_by_the_models_proposal_order():
    """`MAX_COMPARATORS` is two, so the two that survive are a clinical statement.

    The model proposes the verbal zero first because that is the order it read them in.
    The row that reaches the table first has to be the one with a numeric rate from 1,140
    patients.
    """
    pool = [
        _article(pmid, **literature.score_benchmark_candidate(
            _article(pmid), condition=CERVICAL.condition, anchors=CERVICAL.anchors))
        for pmid in ("38504222", "31609801")
    ]
    llm = _Reply({"comparators": [
        _row("38504222", "0%", SENTENCE_38504222, "Adults undergoing day ACDF"),
        _row("31609801", "3.4%", SENTENCE_31609801, "Adults undergoing revision ACDF"),
    ]})

    async def search(terms, wide):
        return {"query": "strict", "articles": pool}

    result = asyncio.run(deterministic.extract_comparators(llm, CERVICAL, search=search))

    assert [row["pmid"] for row in result.rows] == ["31609801", "38504222"]
    assert result.rows[0]["reported_rate"] == "3.4%"
    assert result.rows[0]["sample_size_n"] == 1140
    assert result.rows[0]["has_readmission_rate_sentence"] is True
    assert result.rows[1]["rate_kind"] == "zero" or result.rows[1]["reported_rate"] == "0%"


# --------------------------------------------------------------------------------------
# 4. The ranking has to survive the wire, or none of the above is live
# --------------------------------------------------------------------------------------


def test_the_published_literature_schema_declares_the_ranking():
    """The defect that made the first two Stage 9A runs pointless.

    The MCP server derives its output schema from `LiteratureResult`, and FastMCP drops
    every key the schema does not declare. So the tool ranked the pool correctly, the agent
    received articles without `rank_score`, without `sample_size_n` and without
    `has_readmission_rate_sentence`, and `candidate_pass` fell through to the three-boolean
    fallback and re-ranked what had already been ranked. Measured 2026-11-09: the tool
    scored `34275708` at `n=17491, score=119.0` and the row the table published carried
    `n=None`.

    Asserted against the schema the server actually builds, not against the TypedDict
    alone: declaring a key in Python and forgetting it is a silent failure, and this is the
    one place it can be caught without a live call.
    """
    from pydantic import TypeAdapter

    from services.mcp.contracts import LiteratureResult

    article = TypeAdapter(LiteratureResult).json_schema()["$defs"]["LiteratureArticle"]

    for field in ("rank_score", "sample_size_n", "has_readmission_rate_sentence"):
        assert field in article["properties"], (
            f"{field} is not declared, so the transport will drop it")
    assert "rank_score" not in article.get("required", ()), (
        "a general `/pubmed` search carries no ranking, so the key cannot be required")
    assert "sample_size_n" not in article.get("required", ())


def test_the_agent_uses_the_tools_ranking_when_it_arrives():
    """The fallback ranker is for test doubles, not for a payload that was already ranked."""
    pool = [
        _article(pmid, **literature.score_benchmark_candidate(
            _article(pmid), condition=CERVICAL.condition, anchors=CERVICAL.anchors))
        for pmid in ("38504222", "31609801")
    ]

    async def search(terms, wide):
        return {"query": "strict", "articles": pool}

    pass_ = asyncio.run(deterministic.candidate_pass(search, CERVICAL))

    assert [article["pmid"] for article in pass_.aligned] == ["31609801", "38504222"]


def test_a_row_whose_record_is_missing_still_verifies_and_sorts_last():
    """Verification is what admits a row; the ordering is not allowed to delete one."""
    rows = [
        {"pmid": "99999999", "reported_rate": "5%", "rate_kind": "numeric"},
        {"pmid": "31609801", "reported_rate": "3.4%", "sample_size_n": 1140,
         "rank_score": 80.28, "has_readmission_rate_sentence": True},
    ]

    ordered = deterministic._by_evidence_scale(rows)

    assert [row["pmid"] for row in ordered] == ["31609801", "99999999"]
