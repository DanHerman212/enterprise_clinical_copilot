"""Stage 8 — one citation authority, and a structured reply that does not think.

Two production defects, both measured on `agent-00266-kbs` (2026-11-08):

* `/ehr` on `90000069` and `90000020` served a `Readmission risk & trajectory` section with
  NO badges and a reference drawer that listed the two papers. The model had written
  `^[PMID: …]` correctly; settlement deleted it. Settlement, the guardrail and the composer
  each assembled their own badge allowance and the first of them did not know about the
  records a zero-cohort turn publishes.
* The same turns spent 34.7 s and 60.7 s inside `extract_comparators`, on five refused
  attempts and on two replies that burned ~3.9k reasoning tokens each and arrived
  unparseable.

The tests below pin the two repairs at the seams that failed.
"""

from __future__ import annotations

import asyncio
import inspect

from langchain_core.messages import HumanMessage

from services.agent.engine import deterministic
from services.agent.postprocess import answer as answer_text
from services.agent.postprocess import citations
from services.agent.postprocess.guardrail import guard_answer
from services.agent.presentation import a2ui

HADM = 90000069

ARTICLE = {
    "pmid": "31893147",
    "title": "Comparison of 30-day outcome following ACDF with or without instrumentation.",
    "authors": "Lim S, Haider S, Zakaria H, et al.",
    "journal": "Surgical neurology international",
    "journal_abbrev": "Surg Neurol Int",
    "pub_date": "2019",
    "abstract": "Readmission was 4.1% vs. 3.2%; postmatch, 3.9% vs. 3.3%.",
    "url": "https://pubmed.ncbi.nlm.nih.gov/31893147/",
}

SECOND = dict(
    ARTICLE,
    pmid="38504222",
    title="Day-case anterior cervical discectomy for degenerative cervical spondylosis.",
    journal="BMC musculoskeletal disorders",
    journal_abbrev="BMC Musculoskelet Disord",
    pub_date="2024",
    abstract="There was no increase in readmission for same-day discharge.",
)

NOTES = [
    {"id": "n1", "section": "brief_hospital_course",
     "text": "She underwent anterior cervical discectomy and fusion."},
    {"id": "n2", "section": "discharge_diagnosis",
     "text": "1. Cervical spondylosis with herniated nucleus pulposus."},
]

SEARCH = {
    "name": "search_literature",
    "args": {"query": "cervical spondylosis", "scope": "readmission"},
    "response": {"query": "cervical spondylosis", "returned": 2,
                 "articles": [ARTICLE, SECOND],
                 "fallback_pmids": ["31893147", "38504222"]},
}

SECTIONS_CALL = {
    "name": "rag_search_sections", "args": {"hadm_id": HADM},
    "response": {"passages": NOTES},
}

PREDICTION_CALL = {
    "name": "predict_readmission", "args": {"hadm_id": HADM},
    "response": {"probability": 0.075, "probability_display": "7.5%",
                 "threshold": 0.11, "threshold_display": "11.0%"},
}

#: The section as the formatter wrote it: two badges, no comparator rows in the table.
PROSE = (
    "**Presentation & diagnosis.** A 41-year-old female with cervical spondylosis "
    "and herniated nucleus pulposus ^[2].\n\n"
    "**Readmission risk & trajectory.** Her estimated 30-day readmission risk is 7.5%, "
    "which is below the 11.0% threshold. The retrieved literature does not state a 30-day "
    "readmission percentage benchmark for this condition. Studies on anterior cervical "
    "discectomy and fusion examine the impact of instrumentation on 30-day results "
    "^[PMID: 31893147] and the safety of day-case procedures ^[PMID: 38504222]."
)

TURN_CALLS = [SECTIONS_CALL, PREDICTION_CALL, SEARCH]


# --- 1. one authority -------------------------------------------------------

def test_the_allowance_is_the_union_of_every_channel():
    allowed = citations.allowed_literature_pmids(
        TURN_CALLS,
        comparators=[{"pmid": "32423759"}],
        fallback_literature=[ARTICLE],
        thread_literature=[{"pmid": "99999999"}],
    )

    assert {"32423759", "31893147", "99999999"} <= allowed
    # The readmission-scope search contributes its DECLARED records and not its pool: the
    # second article is in `fallback_pmids`, so it is allowed; a pool member that was never
    # published would not be.
    assert "38504222" in allowed


def test_a_pool_member_that_was_never_published_is_not_allowed():
    pool = [ARTICLE, SECOND, dict(ARTICLE, pmid="30000123")]
    call = {"name": "search_literature",
            "args": {"scope": "readmission"},
            "response": {"returned": 3, "articles": pool,
                         "fallback_pmids": ["31893147"]}}

    allowed = citations.allowed_literature_pmids([call])

    assert allowed == frozenset({"31893147"}), (
        "an undeclared candidate is not evidence, however recently it was retrieved")


def test_settlement_keeps_a_fallback_badge_with_no_comparator_rows():
    """The defect: `comparators == []` and the badges died here, before the guardrail."""
    settled = answer_text.settle_answer(
        prose=PROSE,
        deterministic=None,
        tool_calls=TURN_CALLS,
        replayed_tool_calls=[],
        comparators=[],
        fallback_literature=[ARTICLE, SECOND],
    )

    assert "^[PMID: 31893147]" in settled.prose
    assert "^[PMID: 38504222]" in settled.prose
    assert [flag for flag in settled.flags if "citation" in flag] == []


def test_the_guardrail_keeps_them_too():
    guarded = guard_answer(PROSE, TURN_CALLS, [], state={
        "comparators": [], "fallback_literature": [ARTICLE, SECOND]})

    assert "^[PMID: 31893147]" in guarded["answer"]
    assert "^[PMID: 38504222]" in guarded["answer"]
    assert [flag for flag in guarded["flags"] if "citation" in flag] == []


def test_a_pmid_no_channel_holds_is_still_deleted():
    """The allowance was widened, not removed."""
    settled = answer_text.settle_answer(
        prose="An unrelated rate is reported ^[PMID: 99999999].",
        deterministic=None, tool_calls=TURN_CALLS, replayed_tool_calls=[],
        fallback_literature=[ARTICLE],
    )

    assert "99999999" not in settled.prose
    assert settled.flags == ("citation_out_of_range:^99999999",)


def test_drawer_and_prose_name_the_same_records():
    """`cited_in_prose == literature_in_drawer`, which is the invariant a reader sees.

    Settlement runs first, then the guardrail, then composition — so this asserts the whole
    chain: what survives the first two is exactly what the third publishes.
    """
    settled = answer_text.settle_answer(
        prose=PROSE, deterministic=None, tool_calls=TURN_CALLS, replayed_tool_calls=[],
        comparators=[], fallback_literature=[ARTICLE, SECOND])
    guarded = guard_answer(settled.prose, TURN_CALLS, [], state={
        "comparators": [], "fallback_literature": [ARTICLE, SECOND],
        "citation_display_map": dict(settled.display)})

    out = a2ui.compose_presentation(
        "/ehr", guarded["answer"], TURN_CALLS, None, "summarize",
        comparators=[], fallback_literature=[ARTICLE, SECOND])

    in_drawer = {row["pmid"] for row in out["references"] if row.get("pmid")}
    markers = {int(value) for value in
               __import__("re").findall(r"\^\[(\d+)\]", out["answer"])}
    cited = {row["pmid"] for row in out["references"]
             if row.get("pmid") and int(row.get("index") or 0) in markers}

    assert in_drawer == {"31893147", "38504222"}
    assert cited == in_drawer, "every published record is cited in the prose"
    assert "^[PMID:" not in out["answer"], "the badges resolve to numbers"


def test_a_fallback_record_the_answer_does_not_name_is_not_published():
    """Material for a section is not a reference until the section rests on it."""
    prose = PROSE.replace(" ^[PMID: 38504222]", "")
    settled = answer_text.settle_answer(
        prose=prose, deterministic=None, tool_calls=TURN_CALLS, replayed_tool_calls=[],
        comparators=[], fallback_literature=[ARTICLE, SECOND])

    out = a2ui.compose_presentation(
        "/ehr", settled.prose, TURN_CALLS, None, "summarize",
        comparators=[], fallback_literature=[ARTICLE, SECOND])

    assert {row["pmid"] for row in out["references"] if row.get("pmid")} == {"31893147"}


def test_the_comparator_step_declares_what_it_publishes():
    """The channel settlement can actually read: it holds the calls, not the state."""
    class _Box:
        pass

    published = deterministic._fallback_literature([], [ARTICLE, SECOND])
    announced = [str(article.get("pmid")) for article in published]

    call = {"name": "search_literature", "args": {}, "response": {"returned": 2}}
    call["response"] = {**call["response"], "fallback_pmids": announced}

    assert citations.published_literature_pmids([call]) == frozenset(
        {"31893147", "38504222"})
    assert _Box is not None


# --- 2. the structured reply does not think, and gives up fast ---------------

class _RecordingLLM:
    """A model that records the generation parameters it was called under."""

    def __init__(self, payload: str, fail_times: int = 0):
        self.payload = payload
        self.fail_times = fail_times
        self.calls: list[dict] = []
        self.seen: list[dict] = []

    def bind(self, **kwargs):
        self.seen.append(kwargs)
        return self

    async def ainvoke(self, messages, config=None):
        self.calls.append({"messages": messages, "config": config})
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        from langchain_core.messages import AIMessage
        return AIMessage(content=self.payload)


def test_the_extraction_call_forbids_thinking_and_forces_json():
    llm = _RecordingLLM('{"condition": "cervical spondylosis"}')

    payload, why = asyncio.run(deterministic._structured_reply(
        llm, "system", "human", None))

    assert payload == {"condition": "cervical spondylosis"}
    assert why == ""
    bound = llm.seen[-1]
    # `thinking_level`, not `thinking_budget`: Gemini 3 gives the level precedence and
    # discards the budget, so binding the budget is a setting that never applies.
    assert bound["thinking_level"] == "low"
    assert "thinking_budget" not in bound
    assert bound["response_mime_type"] == "application/json"


def test_a_capacity_refusal_costs_two_attempts_and_no_more():
    llm = _RecordingLLM("{}", fail_times=99)

    payload, why = asyncio.run(deterministic._structured_reply(
        llm, "system", "human", None))

    assert payload is None
    assert why == deterministic.REPLY_RATE_LIMITED
    assert len(llm.calls) == deterministic.STRUCTURED_MAX_ATTEMPTS == 2


def test_the_formatting_turn_keeps_its_own_retry_ceiling():
    """The extraction's ceiling is a latency decision and must not shrink the answer's."""
    assert deterministic.STRUCTURED_MAX_ATTEMPTS < deterministic.RETRY_MAX_ATTEMPTS


def test_a_malformed_reply_is_not_retried():
    """Not-JSON is a model answer, not a capacity refusal; asking again buys nothing."""
    llm = _RecordingLLM("I could not find a suitable cohort.")

    payload, why = asyncio.run(deterministic._structured_reply(
        llm, "system", "human", None))

    assert payload is None
    assert why == deterministic.REPLY_NOT_JSON
    assert len(llm.calls) == 1


def test_the_structured_params_never_reach_the_formatting_turn():
    """`_formatting_turn` must keep its thinking: its prose is what the reader reads."""
    assert "effort" in deterministic._STRUCTURED_CALL_PARAMS
    assert "effort" not in inspect.getsource(
        deterministic._formatting_turn)


def test_the_binder_is_optional_for_test_doubles():
    class _Bare:
        pass

    assert deterministic._bounded_call(_Bare(), {"thinking_budget": 0}).__class__ is _Bare
    assert deterministic._bounded_call(
        _RecordingLLM("{}"), {"response_mime_type": "application/json"}) is not None


def test_the_formatter_input_still_carries_the_fallback_block():
    text = deterministic._formatter_input(
        [HumanMessage(content="note passages here")], False, "/ehr", [],
        fallback_literature=[ARTICLE, SECOND])

    assert "<retrieved_readmission_literature>" in text
    assert "<verified_comparators>" not in text
