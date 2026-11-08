"""Seam 2 — the latency levers, pinned individually.

Each of the three changes removes a serial cost, and each is the kind of thing that comes
back silently: a reasoning level passed at the call site instead of at construction, a
retry that fires on a failure it cannot repair, a second hydration nobody notices because
it produces the right answer a second time. So each is asserted where a regression would
show up — on the constructed client, on the call count, and on the record handed back.
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage

from services.agent.engine import deterministic
from services.agent.engine import graph
from services.agent.engine import model as model_gateway
from services.agent.postprocess.evidence_state import SessionEvidenceState

HADM = 90000069

CERVICAL = deterministic.ClinicalTarget(
    condition="cervical spondylotic myelopathy",
    procedure="anterior cervical discectomy and fusion",
    anchors=("cervical spondylotic myelopathy",),
    patient_age=41,
    patient_sex="female",
)


# --------------------------------------------------------------------------------------
# 1. The extraction clients think less than the served path.
# --------------------------------------------------------------------------------------


def test_the_extraction_client_runs_at_the_low_effort_level():
    llm = model_gateway.build_extraction_llm(graph.MODEL_ID)

    assert model_gateway.EXTRACTION_REASONING_EFFORT == "low"
    assert getattr(llm, "reasoning_effort", None) == "low"


def test_the_served_path_keeps_the_synthesis_level():
    llm = model_gateway.build_llm(graph.MODEL_ID)

    assert getattr(llm, "reasoning_effort", None) == model_gateway.GEMINI_REASONING_EFFORT


# --------------------------------------------------------------------------------------
# 2. The pool is bounded by input size, and the reply ceiling follows it.
# --------------------------------------------------------------------------------------


def test_the_input_bound_is_ten_and_the_reply_ceiling_is_the_client_default():
    assert deterministic.MAX_EXTRACTION_CANDIDATES == 10
    assert deterministic.COMPARATOR_REPLY_TOKENS == 4096


def test_the_retry_over_halves_is_spent_only_on_a_reply_that_was_cut_off():
    """A withheld reply is the filter acting on the abstracts; halves cannot repair it."""
    llm = _Withheld(["SAFETY", "SAFETY", "SAFETY"])

    rows, rejected, outcome = asyncio.run(deterministic._extract_pass(
        llm, CERVICAL, [_abstract(str(index)) for index in range(4)], None))

    assert llm.calls == 1, "a withheld pool was retried in halves"
    assert rows == []
    assert rejected and "withheld" in rejected[0]["reason"]
    assert outcome == "extraction_degraded"


# --------------------------------------------------------------------------------------
# 3. A chart that states its diagnosis costs one call, not two.
# --------------------------------------------------------------------------------------


def test_a_malformed_reply_falls_back_to_the_chart_headers_before_the_retry():
    llm = _Text(["not json at all", "still not json"])

    result = asyncio.run(deterministic.derive_target(llm, {
        "discharge_diagnosis": "1. Cervical spondylosis with herniated nucleus pulposus.\n"
                               "2. Hypertension.",
    }))

    assert llm.calls == 1, "a second model call was paid for a diagnosis in the header"
    assert result.source == "chart_headers"
    assert result.target is not None
    assert result.target.condition.startswith("Cervical spondylosis")
    assert result.target.patient_age is None, "no manifest was passed in this call"


def test_a_withheld_reply_still_pays_for_the_plainer_prompt():
    """The retry exists for exactly this: a filter acting on the elaborated wording."""
    llm = _Withheld(["SAFETY", "SAFETY"])

    result = asyncio.run(deterministic.derive_target(llm, {
        "discharge_diagnosis": "1. Cervical spondylosis.",
    }))

    assert llm.calls == 2, "the withheld reply skipped the retry it exists for"
    assert result.source == "chart_headers"


# --------------------------------------------------------------------------------------
# 4. The replay resolves once, concurrently, and hands the record back.
# --------------------------------------------------------------------------------------


class _RecordingToolbox:
    """A toolbox that answers from a table and records concurrent calls."""

    def __init__(self, payloads: dict) -> None:
        self._payloads = payloads
        self.asked: list[str] = []
        self.in_flight = 0
        self.peak = 0

    async def call(self, name, arguments):
        self.asked.append(name)
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        await asyncio.sleep(0)
        self.in_flight -= 1
        return self._payloads[name]


def _passage_call(name: str, passage_id: str) -> dict:
    return {"name": name, "args": {"hadm_id": HADM}, "payload": None}


TURN = {
    "question": "/ehr For admission 90000069.",
    "answer": "A course ^[1].",
    "citation_map": {"1": 1},
    "tool_calls": [_passage_call("rag_search_sections", "a"),
                   _passage_call("rag_search", "b")],
}

PAYLOADS = {
    "rag_search_sections": {"hadm_id": HADM, "returned": 1, "passages": [
        {"id": "a", "section": "brief_hospital_course", "text": "A course."}]},
    "rag_search": {"hadm_id": HADM, "returned": 1, "passages": [
        {"id": "b", "section": "discharge_diagnosis", "text": "A diagnosis."}]},
}


def test_missing_payloads_are_resolved_concurrently_and_once_each():
    box = _RecordingToolbox(PAYLOADS)

    _messages, calls, _articles = asyncio.run(graph.replay_messages(
        box, [TURN], hadm_id=HADM))

    assert sorted(box.asked) == ["rag_search", "rag_search_sections"]
    assert box.peak == 2, "the re-resolution was serial"
    assert [call["name"] for call in calls] == ["rag_search_sections", "rag_search"]


def test_the_replay_hands_back_the_record_it_built():
    sink: dict = {}
    box = _RecordingToolbox(PAYLOADS)

    _messages, calls, _articles = asyncio.run(graph.replay_messages(
        box, [TURN], hadm_id=HADM, record_sink=sink))

    session = sink["record"]
    assert isinstance(session, SessionEvidenceState)
    # The same object `ask` used to hydrate a second time, so this is the equality it was
    # paying for on every turn.
    assert session == SessionEvidenceState.hydrate_from_calls(
        HADM, [graph._presented_call(call) for call in calls])


def test_a_turn_with_no_admission_sinks_no_record():
    sink: dict = {}

    asyncio.run(graph.replay_messages(_RecordingToolbox(PAYLOADS), [TURN],
                                      hadm_id=None, record_sink=sink))

    assert sink["record"] is None


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------


def _abstract(pmid: str) -> dict:
    return {
        "pmid": pmid,
        "title": "Readmission after anterior cervical discectomy and fusion.",
        "journal": "Spine",
        "pub_date": "2022 Apr",
        "authors": "Nakamura B.",
        "abstract": "The 30-day readmission rate was 4.1%.",
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    }


class _Text:
    """A model that answers with prose, however many times it is asked."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls = 0

    def bind(self, **kwargs):
        return self

    async def ainvoke(self, messages, config=None, **_kwargs):
        self.calls += 1
        text = self._replies.pop(0) if self._replies else "no json here"
        return AIMessage(content=text, response_metadata={"finish_reason": "STOP"})


class _Withheld:
    """A model whose every reply is taken away by the provider's content filter."""

    def __init__(self, finish_reasons: list[str]) -> None:
        self._reasons = list(finish_reasons)
        self.calls = 0

    def bind(self, **kwargs):
        return self

    async def ainvoke(self, messages, config=None, **_kwargs):
        self.calls += 1
        reason = self._reasons.pop(0) if self._reasons else "SAFETY"
        return AIMessage(content=[], response_metadata={"finish_reason": reason})
