"""Settlement has to allow the cohorts the comparison table just verified.

`/ehr` makes two kinds of `search_literature` call, and only one of them is the answer's
own evidence. The model's search declares no scope and its records are the answer's pills.
The readmission retrieval declares `scope="readmission"` and its survivors are published
by the comparison table as cohort rows carrying their own PMIDs — which is why
`literature_pmids` deliberately excludes it, and why `/ehr` does not draw itself as a
literature turn with twenty-five candidates.

The exclusion emptied settlement's badge allowance. `^[PMID: 24108298]` in the risk
paragraph therefore named an identifier nobody had "retrieved", and `enforce_citations`
deleted it as `citation_out_of_range` — so the marker never reached `guard_answer`, the
pass that already unions `comparator_pmids` and would have kept it. Measured on the
production UI 2026-11-03: both patients showed their comparators in the REFERENCES drawer
and the comparison table, and none in the prose.

So both passes build the same allowance from the same rows. These tests pin the three
facts that has to mean: a verified badge survives settlement, an unverified one still
dies there, and the surviving badge resolves to a citation in the composed narrative.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from services.agent.engine import graph
from services.agent.postprocess import answer as answer_text
from services.agent.presentation import a2ui

COMPARATOR = {
    "pmid": "24108298",
    "title": "Predictors of thirty-day readmission after anterior cervical fusion.",
    "authors": "Lovecchio F, Hsu WK, Smith TR, et al.",
    "journal": "Spine",
    "year": "2014",
    "reported_rate": "2.5%",
    "rate_kind": "point",
    "cohort_summary": "ACDF and anterior corpectomy and fusion, ACS-NSQIP 2011.",
    "evidence_tier": "Retrospective cohort study",
    "rate_sentence": "The nationwide readmission rate after ACDF surgery was 2.5%.",
    "url": "https://pubmed.ncbi.nlm.nih.gov/24108298/",
}

SECOND_COMPARATOR = dict(
    COMPARATOR,
    pmid="35985540",
    journal="Global spine journal",
    year="2022",
    url="https://pubmed.ncbi.nlm.nih.gov/35985540/",
)

THIRD_COMPARATOR = dict(
    COMPARATOR,
    pmid="33534292",
    journal="Clinical spine surgery",
    year="2021",
    url="https://pubmed.ncbi.nlm.nih.gov/33534292/",
)

#: The retrieval the comparison table reads: a real search, on a scope that says it serves
#: the comparator pipeline rather than the answer's own citation list.
READMISSION_SEARCH = {
    "name": "search_literature",
    "args": {"query": "Anterior cervical discectomy and fusion", "scope": "readmission"},
    "response": {"query": "Anterior cervical discectomy and fusion", "returned": 1,
                 "articles": [{"pmid": "24108298", "title": COMPARATOR["title"],
                               "journal": "Spine", "pub_date": "2014",
                               "abstract": "…", "url": COMPARATOR["url"]}]},
}

PASSAGES = {
    "name": "rag_search_sections",
    "args": {"hadm_id": 90000069},
    "response": {"returned": 1, "passages": [
        {"cite": 1, "section": "brief_hospital_course",
         "text": "The patient underwent anterior cervical discectomy and fusion."},
    ]},
}

PREDICT = {
    "name": "predict_readmission",
    "args": {"hadm_id": 90000069},
    "response": {"probability": 0.075, "threshold": 0.11},
}

RISK_PARAGRAPH = (
    "**Hospital course & procedures.** The patient underwent anterior cervical "
    "discectomy and fusion ^[1].\n\n"
    "**Readmission risk & trajectory.** Her estimated risk is 7.5%, below the 11.0% "
    "threshold. This estimate is higher than the 2.5% reported after anterior cervical "
    "discectomy and fusion procedures ^[PMID: 24108298], though this published cohort is "
    "a population benchmark and not a prediction about this patient."
)


def _state(prose: str, *, comparators, tool_calls=None) -> dict:
    return {
        "messages": [AIMessage(content=prose)],
        "deterministic_answer": None,
        "tool_calls": tool_calls or [PASSAGES, PREDICT, READMISSION_SEARCH],
        "replayed_tool_calls": [],
        "wants_literature": False,
        "comparators": comparators,
    }


def _citation_flags(flags) -> list[str]:
    return [flag for flag in flags if "citation_out_of_range" in flag]


# --- the allowance ------------------------------------------------------------

def test_the_readmission_search_is_not_the_answers_own_evidence():
    """The premise. If this ever becomes true, `/ehr` draws itself as a literature turn."""
    from services.agent.postprocess import citations

    assert citations.literature_pmids([READMISSION_SEARCH], None) == frozenset()


def test_a_verified_comparator_badge_survives_settlement():
    """One cohort, on 90000069's shape: the badge is part of the sentence and stays in it."""
    settled = answer_text.settle_answer(
        prose=RISK_PARAGRAPH,
        deterministic=None,
        tool_calls=_state("", comparators=[COMPARATOR])["tool_calls"],
        replayed_tool_calls=[],
        comparators=[COMPARATOR],
    )

    assert "^[PMID: 24108298]" in settled.prose
    assert _citation_flags(settled.flags) == []


def test_both_comparator_badges_survive_settlement():
    """90000015 names two cohorts in one paragraph, one badge after each study's rate."""
    prose = (
        "**Readmission risk & trajectory.** Her risk is 21.9%. This is above the 4.9% "
        "reported for octogenarian total hip arthroplasty ^[PMID: 35985540], and below "
        "the 26.9% reported for revision arthroplasty ^[PMID: 33534292]."
    )
    comparators = [SECOND_COMPARATOR, THIRD_COMPARATOR]

    settled = answer_text.settle_answer(
        prose=prose,
        deterministic=None,
        tool_calls=[PASSAGES, PREDICT, READMISSION_SEARCH],
        replayed_tool_calls=[],
        comparators=comparators,
    )

    assert "^[PMID: 35985540]" in settled.prose
    assert "^[PMID: 33534292]" in settled.prose
    assert _citation_flags(settled.flags) == []


def test_the_state_adapter_reads_the_same_rows():
    """`sanitize_answer_markers` holds a state, so it must allow what the graph allows."""
    state = _state(RISK_PARAGRAPH, comparators=[COMPARATOR])

    answer_text.sanitize_answer_markers(state)

    assert "^[PMID: 24108298]" in state["messages"][-1].content
    assert "guardrail_flags" not in state


# --- and the identifier nobody verified still dies here -------------------------

def test_a_pmid_the_table_never_verified_is_still_deleted():
    prose = (
        "**Readmission risk & trajectory.** Her risk is 7.5%, far below the 9.9% "
        "published elsewhere ^[PMID: 99999999]."
    )

    settled = answer_text.settle_answer(
        prose=prose,
        deterministic=None,
        tool_calls=[PASSAGES, PREDICT, READMISSION_SEARCH],
        replayed_tool_calls=[],
        comparators=[COMPARATOR],
    )

    assert "99999999" not in settled.prose
    assert settled.flags == ("citation_out_of_range:^99999999",)


def test_no_comparators_means_no_comparator_allowance():
    """The parameter is an allowance, not a blanket permit: absent rows permit nothing."""
    settled = answer_text.settle_answer(
        prose=RISK_PARAGRAPH,
        deterministic=None,
        tool_calls=[PASSAGES, PREDICT, READMISSION_SEARCH],
        replayed_tool_calls=[],
    )

    assert "24108298" not in settled.prose
    assert settled.flags == ("citation_out_of_range:^24108298",)


# --- what the reader is actually handed ---------------------------------------

def _narrative_component(envelope: dict) -> dict:
    for component in a2ui._surface_components(envelope):
        if component.get("component") == "CitedNarrative":
            return component
    raise AssertionError("the turn drew no CitedNarrative")


def test_the_surviving_badge_reaches_the_wire_and_resolves_to_its_row():
    """End to end: `graph._settle` then the composer, which is the payload a reader gets.

    The pill is drawn from the narrative's citation list, so a marker in the text whose
    PMID is in no entry is a dead badge. `add_comparator_citations` appends the table's
    cohorts after the prose is composed, and this is the assertion that the two arrive
    together.
    """
    state = _state(RISK_PARAGRAPH, comparators=[COMPARATOR])

    graph._settle(state, prose=RISK_PARAGRAPH)
    assert "^[PMID: 24108298]" in state["messages"][-1].content

    out = a2ui.compose_presentation(
        "/ehr",
        state["messages"][-1].content,
        state["tool_calls"],
        None,
        "summarize",
        evidence=state["tool_calls"],
        citation_map=state.get("citation_display_map") or {},
        comparators=[COMPARATOR],
    )

    narrative = _narrative_component(out["a2ui"])
    resolving = [entry for entry in narrative["citations"]
                 if str(entry.get("pmid") or "") == "24108298"]
    assert len(resolving) == 1
    assert resolving[0]["kind"] == "literature"
    # The number the drawer lists the record under is the number the marker addresses.
    row = [row for row in out["references"] if row.get("pmid") == "24108298"]
    assert len(row) == 1
    assert row[0]["index"] == resolving[0]["index"]
    # And the marker IS that number. The prose channel draws one number per pill (the
    # browser's marker regex reads a single integer), so a badge left in its identifier
    # form was unrenderable where it stood while the table cell and the drawer row beside
    # it printed the index. Composition translates it, and the answer a caller receives is
    # the same string the component draws (Charter 13 Law 5; Stage 1, 2026-11-08).
    assert f"^[{resolving[0]['index']}]" in narrative["text"]
    assert "^[PMID:" not in narrative["text"]
    assert out["answer"] == narrative["text"]
