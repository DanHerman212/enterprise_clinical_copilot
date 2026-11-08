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
from services.agent.engine import deterministic
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

    # What the follow-up's model reads is the record overlay `agent_node` injects, plus
    # the question. The replayed turns contribute no tool bodies (Phase 1 Sprint 1.4).
    transcript = _transcript_text(model)
    assert "<session_evidence_state" in transcript
    # (a) The scoring payload, projected the way a live call is: the display string, and
    # no float it could paste.
    assert "probability_display: 7.5%" in transcript
    assert "0.075" not in transcript.replace("7.5%", "")

    # (b) The comparator search's POOL is not the answer's evidence, and never reaches the
    # follow-up — 25 candidates came back and one was used. What DOES reach it, since Seam
    # 1 (2026-11-08), is the record the answer actually cited: `replay_messages` prunes the
    # pool to the cited articles and drops the candidate-pool scope, so those survivors are
    # thread evidence and render under `<literature>`.
    #
    # Before the fix the scope rode along in `args`, the record refused the pruned payload
    # as a pool, and `<literature>` rendered empty — which is why a follow-up asking how
    # this patient compares to "those cohorts" was answered by re-searching PubMed and
    # reporting no matches.
    assert "Predictors of thirty-day readmission" in transcript
    assert "^[PMID: 24108298]" in transcript and "2.5%" in transcript
    # The pool's other twenty-four candidates are still not evidence, and still not shown.
    assert "A candidate this turn never cited" not in transcript
    assert "99999999" not in transcript
    assert "Candidate 3:" not in transcript

    # The record the turn rested on is the thread's evidence; the pool is not. Both reach
    # the follow-up as data — `thread_literature` for the badge allowance and the composer,
    # `replayed_tool_calls` for the guardrail — rather than through the model's window.
    assert [article["pmid"] for article in state["thread_literature"]] == ["24108298"]
    assert [call["name"] for call in state["replayed_tool_calls"]] == [
        "rag_search_sections", "predict_readmission", "search_literature"]
    carried = state["replayed_tool_calls"][-1]["response"]["articles"]
    assert [article["pmid"] for article in carried] == ["24108298"]
    assert "Spine" in carried[0]["journal"]


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


def test_a_lab_only_thread_answers_from_its_own_evidence():
    """A typed chart follow-up over a laboratory session: the record IS the evidence.

    This used to demand a note search, and the exemption was `rag_search_sections` in the
    replay — a guess from the call log about whether the thread had read enough. The
    record answers it directly: a panel is clinical evidence, the model was shown it in
    `<session_evidence_state>`, every uncapped tool was bound, and it chose to answer.
    Sending it back replaced that answer with the forced round's, because only the last
    message is settled.

    What still demands a retrieval is a question whose MODALITY the record does not hold —
    see `test_a_risk_question_over_a_lab_only_thread_still_demands` below.
    """
    labs_turn = [{
        "question": "/labs",
        "answer": "Creatinine rose across the admission.",
        "tool_calls": [{"name": "get_lab_trends", "args": {"hadm_id": HADM},
                        "payload": LABS}],
    }]

    state, model = _ask("What does the discharge note say about her course?",
                        turns=labs_turn)

    assert model.turns == [None], (
        "the turn already held clinical evidence and answered from it")
    assert [call["name"] for call in state["replayed_tool_calls"]] == ["get_lab_trends"]


def test_a_risk_question_over_a_lab_only_thread_still_demands():
    """The record is a set of modalities, and this question needs one it has not got.

    A panel says nothing about the estimate. `_session_holds_evidence` asks the question's
    own mode of the record — `prediction` for a risk question — so an answer written
    without one is still sent back for the tool that can produce it.
    """
    labs_turn = [{
        "question": "/labs",
        "answer": "Creatinine rose across the admission.",
        "tool_calls": [{"name": "get_lab_trends", "args": {"hadm_id": HADM},
                        "payload": LABS}],
    }]

    _, model = _ask("What is this patient's readmission risk score?", turns=labs_turn)

    assert model.turns == [None, "any"]


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
    resolving = [entry for entry in narrative["citations"]
                 if str(entry.get("pmid") or "") == "24108298"]
    assert len(resolving) == 1
    assert resolving[0]["kind"] == "literature"

    rows = [row for row in out["references"] if row.get("pmid") == "24108298"]
    assert len(rows) == 1
    assert rows[0]["index"] == resolving[0]["index"]
    assert "Spine" in rows[0]["subtitle"]
    # The inherited record's badge addresses the index it is published under rather than
    # its identifier, so the pill in the prose, the drawer row and the source entry name
    # the paper with one number (Stage 1, 2026-11-08).
    assert f"^[{resolving[0]['index']}]" in narrative["text"]
    assert "^[PMID:" not in narrative["text"]


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


# --- a mid-thread COMMAND inherits the same thread ----------------------------

#: The tool result a `/meds` turn retrieves, with an id no earlier turn registered — so
#: where it lands in the record says what numbering the command was running under.
MEDS = {
    "hadm_id": HADM,
    "returned": 1,
    "passages": [
        {"id": "MT-2_discharge_medications_1", "section": "discharge_medications",
         "text": "Discharge Medications: aspirin 81 mg daily."},
    ],
}

#: One `/ehr` turn, stored the way the site stores it: the retrieval payload is
#: re-derivable, so the caller kept none of it and the replay resolves it again.
THREAD_TURN = {
    "question": "/ehr For admission 90000069.",
    "answer": "The course was uncomplicated ^[1].",
    "tool_calls": [
        {"name": "rag_search_sections", "args": {"hadm_id": HADM}, "payload": None},
    ],
}


def _numbered_box(payloads: dict) -> MCPToolbox:
    """A toolbox that answers each tool from its own payload, recording what was asked."""
    box = MCPToolbox(session=None)
    box._tools = {name: _Tool() for name in payloads}
    box.calls = []

    async def call(name, arguments):
        box.calls.append(name)
        return payloads[name]

    box.call = call
    return box


def test_a_mid_thread_command_replays_the_thread_before_its_own_tools_run():
    """Stage 3 §2: `/meds` on turn 2 continues the thread instead of starting a new one.

    The command branch of `ask` used to return before `replay_messages`, so a command
    invoked inside a thread began from an EMPTY record. Two things followed from that: the
    command's first unit took position 1 — an integer the earlier turn's passage already
    owned, so a marker written against one addressed the other — and `replayed_tool_calls`
    was absent from the state, so settlement and the guardrail measured the answer against
    a thread that had, in their account, never read the chart (2026-11-08).

    The numbering is asserted through the record itself rather than through a marker: the
    command's own passage must land at `N+1`, where `N` is what the thread had already
    registered. A reset would put it at 1.
    """
    box = _numbered_box({"rag_search_sections": SECTIONS,
                         "discharge_medications": MEDS})
    real = deterministic.run_command
    captured: dict = {}

    async def spy(*args, **kwargs):
        record = kwargs.get("evidence_state")
        captured["replayed_calls"] = list(kwargs.get("replayed_calls") or [])
        captured["record"] = record
        captured["positions_before"] = record.position_count() if record else 0
        return await real(*args, **kwargs)

    with patch.object(model_gateway, "build_llm", lambda name: _AnswersInProse()), \
            patch.object(graph, "run_command", spy):
        state = asyncio.run(graph.ask(
            box, "and the discharge medications?", hadm_id=HADM, model="test",
            turns=[THREAD_TURN], question_kind="meds",
        ))

    # (a) The replay ran, and it is the thread's call that reached `run_command`.
    assert [call["name"] for call in captured["replayed_calls"]] == ["rag_search_sections"]
    # (b) The command was handed the record the replay built, already holding the thread's
    #     two passages — not a fresh empty one it would have to hydrate again.
    assert captured["record"] is not None
    assert captured["positions_before"] == 2
    # (c) So the command's own passage is numbered from N+1, not from 1.
    assert captured["record"].position_of_note_passage(
        "MT-2_discharge_medications_1") == 3
    # (d) And the thread survives on the state every later layer reads.
    assert [call["name"] for call in state["replayed_tool_calls"]] == ["rag_search_sections"]
    assert [call["name"] for call in state["tool_calls"]] == ["discharge_medications"]


# --- Stage 5: literature cited by NUMBER survives the replay ------------------

#: The `/ehr` turn as the site actually stores it since Stage 1: the search response is
#: kept (it is result-kept), and the answer addresses the cohort by its DISPLAY NUMBER —
#: `^[5]` — because composition rewrote the PMID badge. No `PMID:` string survives in the
#: prose, which is exactly what made `cited_articles` decide the turn had cited nothing.
EHR_TURN_BY_NUMBER = {
    "question": "/ehr For admission 90000069.",
    "answer": (
        "**Readmission risk & trajectory.** Her estimated risk is 7.5%, below the 11.0% "
        "threshold. A retrospective cohort reported an ACDF readmission rate of 2.5% "
        "^[5], though that cohort is a population benchmark rather than a prediction for "
        "this patient."
    ),
    "citation_map": {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5},
    "tool_calls": [
        {"name": "rag_search_sections", "args": {"hadm_id": HADM}, "payload": None},
        {"name": "predict_readmission", "args": {"hadm_id": HADM}, "payload": None},
        # Stamped at settlement from the turn's own references — the identifiers the prose
        # no longer contains.
        {"name": "search_literature",
         "args": {"query": "Anterior cervical discectomy and fusion",
                  "scope": "readmission"},
         "payload": dict(COMPARATOR_SEARCH, cited_pmids=[ARTICLE["pmid"]])},
    ],
}


def test_a_search_cited_by_display_number_survives_the_replay():
    """The call is kept, so `replayed_calls` is 3 and the thread keeps its records.

    Reading the citation off PMID badges alone dropped the call (`replayed_calls` 3 -> 2)
    the moment Stage 1 began rewriting the badge to `^[5]`: the search, its records and the
    comparison they exist for all vanished from the thread (2026-11-08).
    """
    state, _ = _ask(FOLLOW_UP, turns=[EHR_TURN_BY_NUMBER])

    assert [call["name"] for call in state["replayed_tool_calls"]] == [
        "rag_search_sections", "predict_readmission", "search_literature"]
    assert [article["pmid"] for article in state["thread_literature"]] == [ARTICLE["pmid"]]


def test_the_replayed_records_reach_the_session_record():
    """The overlay is the second channel, and it fails with the first one when the call is
    dropped — which is why the loss was total rather than partial."""
    state, _ = _ask(FOLLOW_UP, turns=[EHR_TURN_BY_NUMBER])

    rendered = state["evidence_state"].render_for_prompt()
    assert "Predictors of thirty-day readmission" in rendered


def test_a_follow_up_without_a_literature_word_may_still_cite_the_thread():
    """`literature_replay` is a fact about the THREAD, not about the question's wording.

    Gated on wording, a follow-up that quoted the thread had an empty PMID allowance and
    `enforce_citations` deleted a correct citation.
    """
    answer = "That cohort reported 2.5% ^[PMID: 24108298]."

    guarded = guard_answer(
        answer, [], [], literature_replay=True,
        finish_reason=None,
        state={"thread_literature": [ARTICLE]},
    )

    assert "24108298" in guarded["answer"]
    assert "citation_out_of_range" not in " ".join(guarded["flags"])


def test_an_empty_exploratory_search_does_not_hide_the_thread():
    """A fruitless search on a follow-up must not draw over records the thread holds."""
    out = a2ui.compose_presentation(
        "what does the literature say about cervical laminoplasty?",
        "The cohort reported 2.5% ^[PMID: 24108298].",
        [{"name": "search_literature",
          "args": {"query": "cervical laminoplasty"},
          "response": {"query": "cervical laminoplasty", "returned": 0, "articles": []}}],
        None,
        None,
        thread_literature=[ARTICLE],
    )

    assert "LiteratureList" not in _components(out["a2ui"])
    assert any(source.get("pmid") == ARTICLE["pmid"] for source in out["sources"])
