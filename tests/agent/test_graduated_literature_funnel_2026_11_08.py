"""Stage 7 — the two-tier graduated literature funnel for `/ehr`.

The defect this pins: a `/ehr` turn's only route from a PubMed record to the reader was a
VERIFIED 30-day comparator row. Where the abstracts stated no 30-day percentage the turn
published nothing from the literature at all — retrieved, filtered, read, and dropped —
and a reader comparing two admissions saw peer-reviewed research on one and none on the
other (measured 2026-11-08: 7 of 12 phenotypes drew a row, 5 drew nothing, and every one of
the 12 retrieved records).

Four things had to become true:

* a strict query that finds almost nothing is re-asked with the broader outcome clause, so
  the pool is never four abstracts deep;
* when no cohort verifies, the top of the ranked pool is published instead of discarded;
* the prose is given those records with a block that says there is NO benchmark among
  them, and their PMIDs survive `enforce_citations`;
* the comparison's direction is computed, not inferred — the relation is arithmetic.

Offline: search and model are fakes, and no network call is made.
"""

from __future__ import annotations

import asyncio

from services.agent.engine import deterministic
from services.agent.engine import prompts
from services.agent.postprocess import citations
from services.agent.postprocess.guardrail import guard_answer
from services.agent.presentation import a2ui

TARGET = deterministic.ClinicalTarget(
    condition="displaced femoral neck fracture",
    procedure="hemiarthroplasty",
    anchors=("femoral neck fracture", "hip fracture"),
    patient_age=81,
    patient_sex="F",
)


def _article(pmid: str, *, title: str = "", abstract: str = "") -> dict:
    return {
        "pmid": pmid,
        "title": title or f"Readmission after hemiarthroplasty, record {pmid}.",
        "journal": "Bone Joint J",
        "pub_date": "2021 Mar",
        "authors": "Ng K, et al.",
        "abstract": abstract or (
            "Readmission was common, and frailty predicted it independently."),
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    }


POOL = [_article(str(40000000 + index)) for index in range(6)]
STRICT = POOL[:3]


class _FakeLLM:
    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.prompts: list[str] = []

    async def ainvoke(self, messages, config=None):
        self.prompts.append(str(messages[-1].content))
        from langchain_core.messages import AIMessage
        import json
        payload = self._payloads.pop(0) if self._payloads else {"comparators": []}
        return AIMessage(content=json.dumps(payload))


def _row(pmid: str) -> dict:
    return {
        "pmid": pmid,
        "clinical_match": True,
        "horizon_days": 30,
        "reported_rate": "3.8%",
        "rate_sentence": "The 30-day readmission rate was 3.8%.",
        "cohort_summary": "Adults over 65 after hemiarthroplasty, n=1200.",
        "evidence_tier": "Retrospective cohort",
    }


# --- 1. recall: a thin pool is broadened, a healthy one is not -----------------

def test_a_thin_pool_is_re_broadened_and_merged_by_pmid():
    """Four records is not a pool, and the strict clause is usually why.

    `90000047` (salpingectomy) returned four: the horizon clause requires the abstract to
    spell "30-day" or "thirty days", which an ectopic-pregnancy series rarely does. The
    second query drops that requirement and the two pools are merged, de-duplicated.
    """
    searched: list[tuple] = []

    async def search(terms, wide):
        searched.append((terms, wide))
        if wide:
            return {"query": "strict", "articles": STRICT}
        # The broader pass overlaps the strict one by one record.
        return {"query": "broad", "articles": [*POOL[2:], POOL[0]]}

    pass_ = asyncio.run(deterministic.candidate_pass(search, TARGET))

    assert [call[1] for call in searched] == [True, False], (
        "the broadened query is the second pass and it drops the horizon clause")
    assert pass_.candidates == 6, "the merged pool holds each record once"
    pmids = [article["pmid"] for article in pass_.aligned]
    assert len(pmids) == len(set(pmids))


def test_a_healthy_pool_is_asked_once():
    """Twenty records needs no second round trip; the cost is the reader's time."""
    searched: list[tuple] = []

    async def search(terms, wide):
        searched.append((terms, wide))
        return {"query": "strict", "articles": [_article(str(50000000 + i))
                                                for i in range(20)]}

    asyncio.run(deterministic.candidate_pass(search, TARGET))

    assert len(searched) == 1


def test_the_pass_reports_the_pool_before_the_cap():
    """`aligned` is the extraction cap; `pool_size` is what the filter kept.

    Reporting only the cap made "aligned=10, excluded=0" read as "ten matched" when it
    meant "ten were read and nothing was filtered" — the confusion that hid `cr`'s lost
    cohort (2026-11-08).
    """
    async def search(terms, wide):
        return {"query": "strict", "articles": [_article(str(60000000 + i))
                                                for i in range(14)]}

    pass_ = asyncio.run(deterministic.candidate_pass(search, TARGET))
    rows = deterministic._pass_rows([pass_.report(
        verified=0, rejected=pass_.refused, outcome="verified_empty")])

    assert pass_.pool_size == 14
    assert rows[0]["pool_size"] == 14
    assert rows[0]["aligned"] == deterministic.MAX_EXTRACTION_CANDIDATES
    assert rows[0]["candidates"] == 14


# --- 2. Tier 2: no verified cohort still publishes what was read ---------------

def test_a_pass_that_verifies_nothing_still_hands_over_the_pool():
    """The extraction's input travels out with its output.

    A caller that knows only `rows == []` cannot tell "the literature said nothing" from
    "the literature said it without a number"; the pool is what tells them apart.
    """
    async def search(terms, wide):
        return {"query": "strict", "articles": POOL}

    result = asyncio.run(deterministic.extract_comparators(
        _FakeLLM([{"comparators": []}]), TARGET, search=search))

    assert result.rows == []
    assert [article["pmid"] for article in result.pool] == [
        article["pmid"] for article in POOL]


def test_the_top_two_ranked_records_are_published_when_no_row_verified():
    fallback = deterministic._fallback_literature([], POOL)

    assert len(fallback) == deterministic.MAX_FALLBACK_LITERATURE
    assert [article["pmid"] for article in fallback] == [
        POOL[0]["pmid"], POOL[1]["pmid"]]


def test_a_turn_with_a_verified_cohort_publishes_no_fallback():
    """A benchmark and an unquantified paper in one paragraph invite an average."""
    assert deterministic._fallback_literature([_row("40000000")], POOL) == []


# --- 3. the prose is given the records, and told what they are not ------------

def test_the_fallback_block_says_there_is_no_benchmark_in_it():
    block = deterministic.fallback_literature_block(POOL[:2])

    assert block.startswith("<retrieved_readmission_literature>")
    assert block.rstrip().endswith("</retrieved_readmission_literature>")
    assert "NO benchmark figure" in block
    assert "^[PMID: <pmid>]" in block
    assert f"PMID {POOL[0]['pmid']}" in block
    assert "reports: Readmission was common" in block


def test_exactly_one_of_the_two_blocks_reaches_the_formatter():
    from langchain_core.messages import HumanMessage

    messages = [HumanMessage(content="note passages here")]
    with_cohort = deterministic._formatter_input(
        messages, False, "/ehr", [_row("40000000")], fallback_literature=POOL[:2])
    without = deterministic._formatter_input(
        messages, False, "/ehr", [], fallback_literature=POOL[:2])
    neither = deterministic._formatter_input(messages, False, "/ehr", [], None)

    assert "<verified_comparators>" in with_cohort
    assert "<retrieved_readmission_literature>" not in with_cohort
    assert "<retrieved_readmission_literature>" in without
    assert "<verified_comparators>" not in without
    assert "<verified_comparators>" not in neither
    assert "<retrieved_readmission_literature>" not in neither


def test_the_prompt_forbids_a_rate_when_only_the_fallback_is_given():
    """A rate-shaped sentence over a paper that reported no rate is the failure."""
    assert "<retrieved_readmission_literature>" in prompts.EHR_FORMATTING
    assert "never stating a readmission rate of any kind" in prompts.EHR_FORMATTING


# --- 4. the direction of the comparison is computed, not inferred -------------

def test_the_block_states_the_relation_for_a_one_figure_row():
    block = deterministic.comparator_block([_row("40000000")], patient_estimate=0.09)

    assert "relation: patient estimate (9.0%) is higher than the published rate (3.8%)" \
        in block


def test_a_lower_estimate_and_a_span_each_get_their_own_word():
    lower = deterministic.comparator_block([_row("40000000")], patient_estimate=0.012)
    inside = deterministic.comparator_block(
        [dict(_row("40000000"), reported_rate="2.6% vs. 12.0%", rate_kind="comparison")],
        patient_estimate=0.09)
    above = deterministic.comparator_block(
        [dict(_row("40000000"), reported_rate="1.0% to 1.4%", rate_kind="range")],
        patient_estimate=0.09)

    assert "is lower than the published rate (3.8%)" in lower
    assert "is within the published span (2.6% vs. 12.0%)" in inside
    assert "is above the whole published span (1.0% to 1.4%)" in above


def test_no_estimate_means_no_relation_line():
    """A direction needs two figures, and a turn holding one states none."""
    block = deterministic.comparator_block([_row("40000000")], patient_estimate=None)

    assert "relation:" not in block


def test_the_estimate_is_read_from_the_prediction_call():
    calls = [
        {"name": "rag_search_sections", "args": {}, "response": {}},
        {"name": "predict_readmission", "args": {}, "response": {"probability": 0.075}},
    ]

    assert deterministic._patient_estimate(calls) == 0.075
    assert deterministic._patient_estimate([]) is None
    assert deterministic._patient_estimate(
        [{"name": "predict_readmission", "args": {}, "response": {"probability": None}}]
    ) is None


def test_the_prompt_tells_the_formatter_to_use_the_relation_verbatim():
    assert "relation:" in prompts.EHR_FORMATTING
    assert "do not decide the\ndirection yourself" in prompts.EHR_FORMATTING


# --- 5. the badge survives; the drawer gets a row; no LiteratureList ----------

def test_a_fallback_pmid_may_be_cited():
    answer = ("Frailty predicted readmission in this population ^[PMID: 40000000].")

    guarded = guard_answer(answer, [], state={"fallback_literature": POOL[:2]})

    assert "^[PMID: 40000000]" in guarded["answer"]
    assert not [flag for flag in guarded["flags"] if "citation" in flag]


def test_a_pmid_the_fallback_never_held_is_still_deleted():
    answer = "An unrelated rate is reported ^[PMID: 99999999]."

    guarded = guard_answer(answer, [], state={"fallback_literature": POOL[:2]})

    assert "99999999" not in guarded["answer"]


NOTES = [
    {"id": "n1", "section": "brief_hospital_course",
     "text": "She underwent hemiarthroplasty for a displaced femoral neck fracture."},
    {"id": "n2", "section": "discharge_diagnosis",
     "text": "1. Displaced femoral neck fracture."},
]


def _presented(answer: str):
    tool_calls = [
        {"name": "rag_search_sections", "args": {"hadm_id": 90000074},
         "response": {"passages": NOTES}},
        {"name": "predict_readmission", "args": {"hadm_id": 90000074},
         "response": {"probability": 0.09, "probability_display": "9.0%",
                      "threshold": 0.11, "threshold_display": "11.0%"}},
    ]
    return a2ui.compose_presentation(
        "/ehr", answer, tool_calls, None, "summarize",
        fallback_literature=POOL[:2])


def test_a_fallback_record_is_a_pill_a_row_and_a_source():
    answer = ("**Presentation & diagnosis.** She was admitted for a displaced femoral "
              "neck fracture ^[2]. **Readmission risk & trajectory.** Her estimate is "
              "9.0%. The retrieved abstracts report frailty as a driver "
              "^[PMID: 40000000].")

    out = _presented(answer)

    published = [row for row in out["references"] if row.get("pmid")]
    assert [row["pmid"] for row in published] == ["40000000"]
    # Numbered after the note passage the answer shows, in one sequence.
    assert [row["index"] for row in published] == [3]
    assert [source["cite"] for source in out["sources"] if source.get("pmid")] == [3]

    narrative = next(
        component for message in out["a2ui"]["messages"]
        for component in message.get("updateComponents", {}).get("components", [])
        if component["component"] == "CitedNarrative"
    )
    resolving = [entry for entry in narrative["citations"]
                 if str(entry.get("pmid") or "") == "40000000"]
    assert [entry["index"] for entry in resolving] == [3]
    # The badge addresses the number it is published under, not the identifier.
    assert "^[3]" in out["answer"]
    assert "^[PMID:" not in out["answer"]


def test_the_fallback_does_not_turn_the_briefing_into_a_literature_turn():
    """The canvas is unchanged: no LiteratureList, and still a patient-chart answer."""
    out = _presented(
        "**Presentation & diagnosis.** A displaced femoral neck fracture ^[2]. "
        "**Readmission risk & trajectory.** Her estimate is 9.0%, and the retrieved "
        "studies report frailty as its driver ^[PMID: 40000000].")

    components = [
        component.get("component")
        for message in out["a2ui"]["messages"]
        for component in message.get("updateComponents", {}).get("components", [])
    ]
    assert "LiteratureList" not in components
    assert out["literature"] is None
    assert out["scope"] == a2ui.SCOPE_PATIENT


# --- 6. Turn 2 keeps the Tier 2 records --------------------------------------

def test_the_stamped_records_survive_the_replay_prune():
    """`graph.replay_messages` keeps a search by what its payload says was cited.

    The badge in the served answer is a POSITION by then, so the identifiers have to travel
    on the payload; that is the stamp `turn.py` writes from the references it just built.
    """
    payload = {
        "query": "hemiarthroplasty",
        "returned": 6,
        "articles": POOL,
        "cited_pmids": ["40000000", "40000001"],
    }

    kept = citations.cited_articles(payload, answer="")

    assert [article["pmid"] for article in kept] == ["40000000", "40000001"]


def test_an_unstamped_pool_contributes_nothing_to_the_thread():
    """The other half of the rule: a pool the turn did not rest on is not evidence."""
    payload = {"query": "hemiarthroplasty", "returned": 6, "articles": POOL}

    assert citations.cited_articles(payload, answer="") == []
