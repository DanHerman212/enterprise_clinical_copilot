"""A follow-up turn inherits the thread's evidence, including its published records.

Reported 2026-11-04: an `/ehr` briefing stated the readmission estimate beside a verified
PubMed comparator, and the next typed question — "how does the peer reviewed medical
research for readmission risk compare to the risk model?" — was answered by searching the
discharge note and reporting that the notes document no peer-reviewed research.

Three things had gone wrong together, and these tests pin the repair of each:

  * the replay dropped the thread's `search_literature` call, so the records were not in
    the transcript at all;
  * a typed follow-up carries no `intent`, so `route` forced a note search whose reply
    REPLACED the answer the model had written from the thread;
  * settlement allowed PMIDs only from this turn's own search, so a badge naming the
    thread's benchmark would have been deleted even if the model had written it.

Offline: the model is a stub and the toolbox returns the stored payloads.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from langchain_core.messages import AIMessage

from services.agent.engine import graph
from services.agent.engine import model as model_gateway
from services.agent.engine.mcp_client import MCPToolbox
from services.agent.postprocess import answer as answer_text
from services.agent.postprocess import citations
from services.agent.postprocess.guardrail import guard_answer
from services.agent.presentation import a2ui

HADM = 90000069

ARTICLE = {
    "pmid": "24108298",
    "title": "Predictors of thirty-day readmission after anterior cervical fusion.",
    "authors": "Lovecchio F, Hsu WK, Smith TR, et al.",
    "journal": "Spine",
    "pub_date": "2014 Mar",
    "abstract": "The nationwide readmission rate after ACDF surgery in this study is 2.5%.",
    "url": "https://pubmed.ncbi.nlm.nih.gov/24108298/",
}

UNCITED = dict(ARTICLE, pmid="99999999", title="A candidate this turn never cited.")

#: The pool a comparator search really returns: the two records the table can verify sit
#: among twenty-five candidates, and only the one the answer named may travel.
POOL = [ARTICLE, UNCITED] + [
    dict(ARTICLE, pmid=str(30000000 + index),
         title=f"Candidate {index}: readmission after cervical fusion.")
    for index in range(23)
]

SECTIONS = {
    "hadm_id": HADM,
    "query": "discharge notes",
    "returned": 2,
    "passages": [
        {"id": f"MT-1_brief_hospital_course_1", "section": "brief_hospital_course",
         "text": "She underwent anterior cervical discectomy and fusion."},
        {"id": f"MT-1_discharge_diagnosis_1", "section": "discharge_diagnosis",
         "text": "1. Cervical spondylosis with herniated nucleus pulposus."},
    ],
}

PREDICTION = {
    "hadm_id": HADM,
    "probability": 0.075,
    "threshold": 0.11,
    "probability_display": "7.5%",
    "threshold_display": "11.0%",
    "decision": 0,
    "model_version": "test-v1",
    "top_factors": [{"feature": "rdw_max", "value": "18.2", "direction": "increases"}],
}

#: The comparator search as `/ehr` runs it: a scope that says it serves the comparison
#: table rather than the answer's own citation list.
COMPARATOR_SEARCH = {
    "hadm_id": HADM,
    "query": "Anterior cervical discectomy and fusion",
    "returned": len(POOL),
    "articles": POOL,
}

EHR_TURN = {
    "question": "/ehr For admission 90000069.",
    "answer": (
        "**Readmission risk & trajectory.** Her estimated risk is 7.5%, below the 11.0% "
        "threshold. This estimate is higher than the 2.5% reported after anterior cervical "
        "discectomy and fusion procedures ^[PMID: 24108298], though this published cohort "
        "is a population benchmark."
    ),
    # Every one of these is re-derivable, so the caller stores no payload and the replay
    # resolves it again — the wire the site actually sends.
    "tool_calls": [
        {"name": "rag_search_sections", "args": {"hadm_id": HADM}, "payload": None},
        {"name": "predict_readmission", "args": {"hadm_id": HADM}, "payload": None},
        {"name": "search_literature",
         "args": {"query": "Anterior cervical discectomy and fusion",
                  "scope": "readmission"},
         "payload": None},
    ],
}

#: The reported follow-up, verbatim. No cue in `routing._LITERATURE_CUES` matches it.
FOLLOW_UP = ("how does the peer reviewed medical research for readmission risk compare "
             "to the risk model?")

ANSWER = ("Her estimate sits above the 2.5% published for anterior cervical discectomy "
          "and fusion ^[PMID: 24108298].")


LABS = {
    "hadm_id": HADM,
    "synthetic": True,
    "note": "Values are synthetic.",
    "trends": [
        {"label": "Creatinine", "unit": "mg/dL",
         "points": [{"day": 1, "value": 1.1}, {"day": 3, "value": 1.7}]},
    ],
}


class _AnswersInProse:
    """A model that answers and calls nothing, recording the turn it was given."""

    def __init__(self, answer: str = ANSWER):
        self.answer = answer
        self.turns: list[str | None] = []
        self.seen: list[list] = []

    def bind_tools(self, tools, tool_choice=None):
        return _BoundTurn(self, tool_choice)

    async def ainvoke(self, messages, **kwargs):  # pragma: no cover - always bound
        self.seen.append(list(messages))
        self.turns.append(None)
        return AIMessage(content=self.answer)


class _BoundTurn:
    def __init__(self, owner: _AnswersInProse, choice):
        self._owner = owner
        self._choice = choice

    async def ainvoke(self, messages, **kwargs):
        self._owner.seen.append(list(messages))
        self._owner.turns.append(self._choice)
        return AIMessage(content=self._owner.answer)


class _Tool:
    """Just enough shape for `graph._tools` to advertise it."""

    description = "A tool the audit stubs."
    input_schema = {"type": "object", "properties": {}}


def _box() -> MCPToolbox:
    """A toolbox that re-resolves the stored calls and offers every retrieval tool.

    The tools are registered because `bound_for` falls back to an UNBOUND model when the
    narrowed set matches nothing this server advertises — and an unbound model records no
    `tool_choice`, which would hide whether the forced round happened.
    """
    box = MCPToolbox(session=None)
    box._tools = {
        name: _Tool() for name in (
            "rag_search", "rag_search_sections", "discharge_medications",
            "get_lab_trends", "predict_readmission", "search_literature",
        )
    }
    payloads = {
        "rag_search_sections": SECTIONS,
        "predict_readmission": PREDICTION,
        "search_literature": COMPARATOR_SEARCH,
        "get_lab_trends": LABS,
    }
    box.calls = []

    async def call(name, arguments):
        box.calls.append(name)
        return payloads[name]

    box.call = call
    return box


def _ask(question: str, *, turns, model=None):
    model = model or _AnswersInProse()
    with patch.object(model_gateway, "build_llm", lambda name: model):
        state = asyncio.run(graph.ask(
            _box(), question, hadm_id=HADM, model="test", turns=turns,
        ))
    return state, model


def _transcript_text(model) -> str:
    return "\n".join(str(message.content) for message in model.seen[0])


# --- Step 1: the thread's evidence reaches the follow-up ----------------------

def test_the_follow_up_sees_the_prediction_and_the_comparator_records():
    state, model = _ask(FOLLOW_UP, turns=[EHR_TURN])

    transcript = _transcript_text(model)
    # (a) The scoring payload, projected the way a live call is.
    assert 'name="predict_readmission"' in transcript
    assert "7.5%" in transcript
    # And the record the comparator search returned and Turn 1 rested on.
    assert "Predictors of thirty-day readmission" in transcript
    assert "Spine" in transcript

    # The record the turn rested on is the thread's evidence; the pool is not.
    assert [article["pmid"] for article in state["thread_literature"]] == ["24108298"]
    assert [call["name"] for call in state["replayed_tool_calls"]] == [
        "rag_search_sections", "predict_readmission", "search_literature"]


def test_a_replayed_search_does_not_consume_note_citation_numbers():
    """An article is addressed by its PMID; it takes no position in the passage array."""
    state, _ = _ask(FOLLOW_UP, turns=[EHR_TURN])

    assert citations.citation_bound(
        state["tool_calls"], state["replayed_tool_calls"]) == len(SECTIONS["passages"])


def test_a_replayed_search_is_pruned_to_the_cited_record():
    """Twenty-five candidates were returned; the turn named one, so the thread holds one.

    Replaying the pool put every abstract in front of the next turn, which then quoted
    figures from candidates nothing had displayed and cited them — settlement stripped the
    markers and left the numbers behind (90000069, 2026-11-04).
    """
    state, model = _ask(FOLLOW_UP, turns=[EHR_TURN])

    replayed_search = [call for call in state["replayed_tool_calls"]
                       if call["name"] == "search_literature"]
    assert len(replayed_search) == 1
    assert [a["pmid"] for a in replayed_search[0]["response"]["articles"]] == ["24108298"]
    assert replayed_search[0]["response"]["returned"] == 1

    transcript = _transcript_text(model)
    assert "A candidate this turn never cited" not in transcript
    assert "Candidate 7:" not in transcript


def test_a_search_the_turn_cited_nothing_from_is_not_replayed_at_all():
    turns = [dict(EHR_TURN, answer=(
        "**Readmission risk & trajectory.** Her risk is 7.5%, below the 11.0% threshold."))]

    state, model = _ask(FOLLOW_UP, turns=turns)

    names = [call["name"] for call in state["replayed_tool_calls"]]
    assert "search_literature" not in names
    assert state["thread_literature"] == []
    assert "Predictors of thirty-day readmission" not in _transcript_text(model)


# --- Step 3: a thread with evidence is not sent back for more of it -----------

def test_the_follow_up_is_answered_in_one_pass():
    state, model = _ask(FOLLOW_UP, turns=[EHR_TURN])

    assert model.turns == [None], (
        "the run went back for evidence on a question the thread had already answered")
    assert state["tool_calls"] == []
    assert "deterministic_answer" not in state, (
        "the abstention sentence replaced the answer the model wrote from the thread")


def test_a_first_turn_still_gets_the_forced_retrieval():
    """The 2026-10-15 protection, unchanged where it applies: nothing was retrieved."""
    _, model = _ask(FOLLOW_UP, turns=[])

    assert model.turns == [None, "any"]


def test_a_lab_only_thread_still_gets_the_forced_retrieval():
    """A panel is not the whole note: the exemption is full-chart synthesis.

    A thread whose only replay is a laboratory series holds no account of the admission,
    so a new chart question asked into it still needs the note read. Bypassing the demand
    for ANY replay turned the one protection this path has into a hole (2026-11-04).
    """
    labs_turn = [{
        "question": "/labs",
        "answer": "Creatinine rose across the admission.",
        "tool_calls": [{"name": "get_lab_trends", "args": {"hadm_id": HADM},
                        "payload": LABS}],
    }]

    state, model = _ask("What does the discharge note say about her course?",
                        turns=labs_turn)

    assert model.turns == [None, "any"]
    assert [call["name"] for call in state["replayed_tool_calls"]] == ["get_lab_trends"]


# --- Step 2: the badge survives settlement, the guardrail and composition ------

def test_settlement_keeps_a_badge_naming_the_thread_record():
    state, _ = _ask(FOLLOW_UP, turns=[EHR_TURN])

    settled = answer_text.settle_answer(
        prose=ANSWER,
        deterministic=None,
        tool_calls=state["tool_calls"],
        replayed_tool_calls=state["replayed_tool_calls"],
        thread_literature=state["thread_literature"],
    )

    assert "^[PMID: 24108298]" in settled.prose
    assert [f for f in settled.flags if "citation" in f] == []


def test_the_guardrail_keeps_it_too():
    state, _ = _ask(FOLLOW_UP, turns=[EHR_TURN])

    guarded = guard_answer(ANSWER, state["tool_calls"],
                           state["replayed_tool_calls"], state=state)

    assert "^[PMID: 24108298]" in guarded["answer"]
    assert [f for f in guarded["flags"] if "citation" in f] == []


def test_a_pmid_the_thread_never_cited_is_still_deleted():
    """The pool travelled with the turn; the record it did not rest on did not."""
    state, _ = _ask(FOLLOW_UP, turns=[EHR_TURN])

    settled = answer_text.settle_answer(
        prose="Two cohorts are compared ^[PMID: 99999999].",
        deterministic=None,
        tool_calls=[],
        replayed_tool_calls=state["replayed_tool_calls"],
        thread_literature=state["thread_literature"],
    )

    assert "99999999" not in settled.prose
    assert settled.flags == ("citation_out_of_range:^99999999",)


def _narrative(envelope: dict) -> dict:
    for component in a2ui._surface_components(envelope):
        if component.get("component") == "CitedNarrative":
            return component
    raise AssertionError("the turn drew no CitedNarrative")


def _components(envelope: dict) -> list[str]:
    return [component.get("component")
            for component in a2ui._surface_components(envelope)]


def test_the_badge_resolves_to_a_pill_and_a_reference_row():
    state, _ = _ask(FOLLOW_UP, turns=[EHR_TURN])
    settled = answer_text.settle_answer(
        prose=ANSWER, deterministic=None, tool_calls=[],
        replayed_tool_calls=state["replayed_tool_calls"],
        thread_literature=state["thread_literature"],
    )

    out = a2ui.compose_presentation(
        FOLLOW_UP, settled.prose, state["tool_calls"], None, None,
        evidence=state["replayed_tool_calls"],
        citation_map=settled.display,
        thread_literature=state["thread_literature"],
    )

    narrative = _narrative(out["a2ui"])
    assert "^[PMID: 24108298]" in narrative["text"]
    resolving = [entry for entry in narrative["citations"]
                 if str(entry.get("pmid") or "") == "24108298"]
    assert len(resolving) == 1
    assert resolving[0]["kind"] == "literature"

    rows = [row for row in out["references"] if row.get("pmid") == "24108298"]
    assert len(rows) == 1
    assert rows[0]["index"] == resolving[0]["index"]
    assert "Spine" in rows[0]["subtitle"]


def test_the_follow_up_draws_no_comparison_table():
    """`_draws_matrix` is scoped to `/ehr`: the rows widen the allowance, not the column."""
    state, _ = _ask(FOLLOW_UP, turns=[EHR_TURN])

    out = a2ui.compose_presentation(
        FOLLOW_UP, ANSWER, state["tool_calls"], None, None,
        evidence=state["replayed_tool_calls"],
        thread_literature=state["thread_literature"],
    )

    assert "ComparativeRiskMatrix" not in _components(out["a2ui"])
    assert out["literature"] is None
    assert "LiteratureList" not in _components(out["a2ui"])


def test_an_inherited_record_is_published_only_when_the_answer_names_it():
    """A thread record is not this turn's evidence until the turn rests on it.

    The turn's OWN search keeps the older rule — an uncited result is still published,
    because the turn read it — but an inherited record was read by an earlier turn, and a
    drawer row for it would claim this answer rests on a paper it never mentioned.
    """
    state, _ = _ask(FOLLOW_UP, turns=[EHR_TURN])

    out = a2ui.compose_presentation(
        FOLLOW_UP, "Her estimate sits above the published benchmark.",
        state["tool_calls"], None, None,
        evidence=state["replayed_tool_calls"],
        thread_literature=state["thread_literature"],
    )

    assert out["references"] == []
    assert all(source.get("pmid") != "24108298" for source in out["sources"])


# --- the numbering a chart follow-up keeps ------------------------------------

def test_a_chart_follow_up_numbers_its_passages_from_the_thread():
    """Replaying a search must not shift `^[1..N]`: an article is not a passage.

    The chart turn's first replayed passage is still `^[1]`, and `^[2]` is still the
    second — with a published-records call sitting between them in the replay.
    """
    state, _ = _ask("What did the discharge note say about her diagnosis?",
                    turns=[EHR_TURN])

    assert citations.citation_bound(state["tool_calls"],
                                    state["replayed_tool_calls"]) == 2

    out = a2ui.compose_presentation(
        "What did the discharge note say about her diagnosis?",
        "The diagnosis is listed ^[1], with the operation described ^[2].",
        state["tool_calls"], None, None,
        evidence=state["replayed_tool_calls"],
    )

    assert [source["section"] for source in out["sources"]] == [
        "brief_hospital_course", "discharge_diagnosis"]
