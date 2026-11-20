"""Eval set v7, problem 1: the control flow a turn is allowed to spend.

Four behaviours, each of which was a measured cost rather than a theory:

* a single-domain question binds only that domain's tools, so a medication chip is not
  offered a laboratory panel and a laboratory chip is not offered the discharge
  medication extraction;
* a chip the console declared is answered from the record it was generated from, and a
  retrieval whose result the record already holds is withdrawn rather than refused;
* the console's own driver wording (`"Why does {driver} increase this patient's
  readmission risk?"`) is a driver question, which it was not;
* the pre-flight profile and the replay's passage re-resolution are read through a
  short-lived `hadm_id`-scoped cache, so a six-turn thread pays for them once.

The prompts' own budgets and the model's thinking level are asserted where they live —
`test_prompt_contract.py` and `test_pass_a_spans_profiles_metadata_2026_11_15.py`.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from services.agent.engine import graph
from services.agent.engine import model as model_gateway
from services.agent.engine import prompts
from services.agent.engine.mcp_client import MCPToolbox
from services.agent.postprocess import evidence_state as session_evidence
from services.agent.routing import admission_context
from services.agent.routing import routing

HADM = 90000015


# --------------------------------------------------------------------------------------
# The questions this product actually generates, one per follow-up family.
# --------------------------------------------------------------------------------------

P1_TAPER = "What is the dosing and taper schedule for oxycodone SR at discharge?"
P2_ANTICOAG = "What is the discharge dosing plan for Coumadin?"
P3_PRECAUTIONS = ("What post-discharge activity restrictions, diet instructions, or "
                  "warning signs are documented?")
P4_FOLLOW_UP = ("What outpatient follow-up appointments or clinic visits are scheduled "
                "after discharge?")
P5_REGIMEN = "What are the discharge dosing instructions for Metoprolol and Furosemide?"
P6_DRIVER = "Why does oncology history increase this patient's readmission risk?"
P6_PROTECTIVE = "Why does beta blocker use lower this patient's readmission risk?"
P7_LAB = ("How does the abnormal discharge Potassium (6.0 mmol/L) compare to its "
          "reference range and trend?")
P8_LAB_PANEL = ("What were the discharge values for Sodium and Potassium across the "
                "normal laboratory panels?")
P9_NOTE = ("What were the primary diagnosis and hospital course documented in the "
           "clinical note?")
LITERATURE = "What else does the published literature report on anticoagulation?"
OPEN_ENDED = "What changed overnight?"

MEDICATION_CHIPS = (P1_TAPER, P2_ANTICOAG, P5_REGIMEN)
LAB_CHIPS = (P7_LAB, P8_LAB_PANEL)


# --------------------------------------------------------------------------------------
# 1. The vocabulary: one domain per question, and a union where a question spans two.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("question", MEDICATION_CHIPS)
def test_a_medication_chip_names_one_domain(question):
    assert routing.question_domains(question) == frozenset({routing.DOMAIN_MEDICATIONS})


@pytest.mark.parametrize("question", LAB_CHIPS)
def test_a_laboratory_chip_names_one_domain(question):
    assert routing.question_domains(question) == frozenset({routing.DOMAIN_LABS})


@pytest.mark.parametrize("question", (P3_PRECAUTIONS, P4_FOLLOW_UP))
def test_a_care_plan_chip_names_one_domain(question):
    assert routing.question_domains(question) == frozenset({routing.DOMAIN_CARE_PLAN})


@pytest.mark.parametrize("question", (P6_DRIVER, P6_PROTECTIVE))
def test_a_driver_chip_names_the_risk_domain(question):
    assert routing.question_domains(question) == frozenset({routing.DOMAIN_RISK})


def test_a_note_chip_names_the_chart_domain():
    assert routing.question_domains(P9_NOTE) == frozenset({routing.DOMAIN_CHART})


def test_a_literature_chip_names_the_literature_domain():
    """The subject may also be a drug, and the literature branch narrows on its own.

    `followups._LITERATURE` slots the topic of the search, so "…report on anticoagulation?"
    names a medication as well. The union rule then leaves the set alone — which is right,
    because the literature branch has already reduced it to `LITERATURE_QUESTION_TOOLS`.
    """
    assert routing.DOMAIN_LITERATURE in routing.question_domains(LITERATURE)


def test_a_question_the_vocabulary_does_not_know_names_no_domain():
    """No domain is the safe answer, and it means "bind the standing set"."""
    assert routing.question_domains(OPEN_ENDED) == frozenset()


def test_the_composed_admission_sentence_does_not_make_every_question_a_chart_question():
    """"admission" is a chart cue, and `compose_question` appends it to EVERY question."""
    composed = f"{P2_ANTICOAG} For admission {HADM}."

    assert routing.question_domains(composed) == frozenset({routing.DOMAIN_MEDICATIONS})


def test_a_question_spanning_two_subjects_keeps_both():
    both = "What is the discharge dosing plan, and how did the creatinine trend?"

    assert routing.question_domains(both) == frozenset(
        {routing.DOMAIN_MEDICATIONS, routing.DOMAIN_LABS})


# --------------------------------------------------------------------------------------
# 2. The bound set: a single-domain question loses the other domains' tools.
# --------------------------------------------------------------------------------------


def test_a_medication_question_is_not_offered_the_laboratory_panel():
    allowed = graph._freetext_allowed(False, False, question=P1_TAPER)

    assert "discharge_medications" in allowed
    assert "get_lab_trends" not in allowed
    assert "predict_readmission" not in allowed


def test_a_laboratory_question_is_not_offered_the_medication_extraction():
    allowed = graph._freetext_allowed(False, False, question=P7_LAB)

    assert "get_lab_trends" in allowed
    assert "discharge_medications" not in allowed


def test_a_note_question_is_offered_the_note_tools_and_nothing_else():
    allowed = graph._freetext_allowed(False, False, question=P9_NOTE)

    assert allowed == routing.NOTE_SEARCH_TOOLS


def test_a_driver_question_keeps_the_predictor_and_loses_the_panel():
    allowed = graph._freetext_allowed(True, False, question=P6_DRIVER)

    assert "predict_readmission" in allowed
    assert "get_lab_trends" not in allowed
    assert "discharge_medications" not in allowed


def test_a_driver_that_is_an_analyte_keeps_the_series():
    """The drill-down's own question: the driver IS the value, so the panel is the record."""
    question = ("What in this patient's chart explains red cell distribution width (RDW) "
                "as a readmission risk driver?")

    allowed = graph._freetext_allowed(
        True, False, routing.asks_about_a_lab_analyte(question), question=question)

    assert "get_lab_trends" in allowed
    assert "predict_readmission" in allowed


def test_two_domains_keep_the_union():
    both = "What is the discharge dosing plan, and how did the creatinine trend?"
    allowed = graph._freetext_allowed(False, False, question=both)

    assert "discharge_medications" in allowed
    assert "get_lab_trends" in allowed


def test_a_question_with_no_domain_binds_the_standing_set():
    """The vocabulary failing to recognise a question must cost nothing it did before."""
    assert graph._freetext_allowed(False, False, question=OPEN_ENDED) \
        == routing.FREETEXT_TOOLS
    assert graph._freetext_allowed(False, False) == routing.FREETEXT_TOOLS


def test_a_declared_literature_follow_up_keeps_its_search():
    """`followups._LITERATURE` asks what ELSE the literature reports."""
    allowed = graph._freetext_allowed(
        False, True, declared_follow_up=True, question=LITERATURE)

    assert "search_literature" in allowed


# --------------------------------------------------------------------------------------
# 3. The prompt: only the spokes whose rules can apply.
# --------------------------------------------------------------------------------------


def test_a_medication_turn_is_given_the_medication_rules_and_not_the_laboratory_ones():
    bound = graph._freetext_allowed(False, False, question=P1_TAPER)
    domains = routing.question_domains(P1_TAPER)
    composed = prompts.compose_system_prompt(
        allowed_tools=bound, domains=domains, has_labs=True)

    assert prompts.MEDS_SPOKE.strip() in composed
    assert prompts.LABS_SPOKE.strip() not in composed


def test_a_laboratory_turn_is_given_the_laboratory_rules_and_not_the_medication_ones():
    bound = graph._freetext_allowed(False, False, question=P7_LAB)
    domains = routing.question_domains(P7_LAB)
    composed = prompts.compose_system_prompt(
        allowed_tools=bound, domains=domains, has_labs=True)

    assert prompts.LABS_SPOKE.strip() in composed
    assert prompts.MEDS_SPOKE.strip() not in composed


def test_a_two_domain_turn_keeps_both_spokes():
    both = "What is the discharge dosing plan, and how did the creatinine trend?"
    bound = graph._freetext_allowed(False, False, question=both)
    composed = prompts.compose_system_prompt(
        allowed_tools=bound, domains=routing.question_domains(both))

    assert prompts.MEDS_SPOKE.strip() in composed
    assert prompts.LABS_SPOKE.strip() in composed


def test_the_chart_and_risk_rules_tell_the_model_to_read_the_record_first():
    """The contradiction the third task removes: "you MUST call" over evidence in hand."""
    chart = " ".join(prompts.CHART_SPOKE.split())
    risk = " ".join(prompts.RISK_SPOKE.split())

    assert "CHECK THE SESSION FIRST" in chart
    assert "CHECK THE SESSION FIRST" in risk
    # The rule the qualification must NOT weaken: nothing in hand means a real search.
    assert "you MUST call `rag_search`" in chart


# --------------------------------------------------------------------------------------
# 4. The declared turn: a retrieval the record already answers is withdrawn.
# --------------------------------------------------------------------------------------


def _record(*, medications: bool = False, labs: bool = False,
            notes: bool = False, prediction: bool = False) -> Any:
    """A session record holding exactly the evidence named."""
    record = session_evidence.SessionEvidenceState(hadm_id=HADM)
    if medications:
        record.ingest_tool_output(
            "discharge_medications", {"hadm_id": HADM},
            {"passages": [{"section": "discharge_medications",
                           "text": "Aspirin 81 mg daily."}]})
    if labs:
        record.ingest_tool_output(
            "get_lab_trends", {"hadm_id": HADM},
            {"trends": [{"label": "Potassium", "unit": "mmol/L",
                         "reference_low": 3.5, "reference_high": 5.0,
                         "series": [{"day": 1, "value": 6.0, "flag": "high"}]}]})
    if notes:
        record.ingest_tool_output(
            "rag_search_sections", {"hadm_id": HADM},
            {"passages": [{"section": "brief_hospital_course",
                           "text": "Diuresed with IV furosemide."}]})
    if prediction:
        record.ingest_tool_output(
            "predict_readmission", {"hadm_id": HADM},
            {"probability": 0.62, "threshold": 0.5, "decision": "above",
             "top_factors": [{"feature": "prior_inpatient_days",
                              "direction": "increases", "contribution": 0.31}]})
    return record


def _declared(*, follow_up: bool = False, risk_driver: bool = False,
              record: Any) -> dict:
    return {
        "tool_calls": [],
        "replayed_tool_calls": [],
        "evidence_state": record,
        "follow_up_declared": follow_up,
        "risk_driver_question": risk_driver,
    }


def test_a_declared_medication_chip_is_not_offered_the_extraction_again():
    allowed = graph._freetext_allowed(False, False, question=P1_TAPER)
    state = _declared(follow_up=True, record=_record(medications=True))

    assert "discharge_medications" not in graph._still_bound(allowed, state)


def test_a_declared_driver_chip_is_not_offered_the_predictor_again():
    allowed = graph._freetext_allowed(True, False, question=P6_DRIVER)
    state = _declared(risk_driver=True,
                      record=_record(prediction=True, notes=True))

    bound = graph._still_bound(allowed, state)
    assert "predict_readmission" not in bound
    assert "rag_search" not in bound


def test_a_declared_note_chip_is_not_offered_the_note_search_again():
    allowed = graph._freetext_allowed(False, False, question=P9_NOTE)
    state = _declared(follow_up=True, record=_record(notes=True))

    assert graph._still_bound(allowed, state) == frozenset()


def test_an_empty_record_withdraws_nothing():
    """A declaration over no evidence is not a licence to answer from nothing."""
    allowed = graph._freetext_allowed(False, False, question=P1_TAPER)
    state = _declared(follow_up=True, record=_record())

    assert graph._still_bound(allowed, state) == allowed


def test_an_undeclared_turn_keeps_its_tools():
    allowed = graph._freetext_allowed(False, False, question=P1_TAPER)
    state = _declared(record=_record(medications=True))

    assert graph._still_bound(allowed, state) == allowed


# --------------------------------------------------------------------------------------
# 5. The read cache: one answer per (admission, tool, arguments), for a few minutes.
# --------------------------------------------------------------------------------------


class _ServedBox:
    """A toolbox shaped like the served one: a session, and `call` from the class."""

    def __init__(self, payload: dict) -> None:
        self.session = object()
        self.payload = payload
        self.calls: list[tuple[str, dict]] = []

    async def call(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, dict(arguments)))
        return self.payload


class _StubBox:
    """A test double: `call` is an instance attribute, so the cache must pass it through."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[tuple[str, dict]] = []
        self.call = self._call

    async def _call(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, dict(arguments)))
        return self.payload


PROFILE = {"hadm_id": HADM, "profile": {"admission_type": "Urgent (URGENT)"}}


def test_the_profile_is_read_once_for_two_turns_of_one_admission():
    box = _ServedBox(PROFILE)

    first = asyncio.run(admission_context.hydrate(box, HADM))
    second = asyncio.run(admission_context.hydrate(box, HADM))

    assert first == second
    assert len(box.calls) == 1


def test_another_admission_never_shares_an_entry():
    box = _ServedBox(PROFILE)

    asyncio.run(admission_context.hydrate(box, HADM))
    asyncio.run(admission_context.hydrate(box, 90000069))

    assert [arguments["hadm_id"] for _name, arguments in box.calls] == [HADM, 90000069]


def test_an_expired_entry_is_read_again(monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr(admission_context, "_clock", lambda: clock["now"])
    box = _ServedBox(PROFILE)

    asyncio.run(admission_context.hydrate(box, HADM))
    clock["now"] += admission_context.READ_CACHE_TTL_SECONDS + 1
    asyncio.run(admission_context.hydrate(box, HADM))

    assert len(box.calls) == 2


def test_an_error_is_never_cached():
    """A transport failure pinned for five minutes would outlive the outage."""
    box = _ServedBox({"error": "unknown_patient", "message": "no row"})

    asyncio.run(admission_context.cached_read(
        box, "predict_readmission", {"hadm_id": HADM}, hadm_id=HADM))
    asyncio.run(admission_context.cached_read(
        box, "predict_readmission", {"hadm_id": HADM}, hadm_id=HADM))

    assert len(box.calls) == 2
    assert admission_context.read_cache_size() == 0


def test_a_stub_toolbox_never_touches_the_cache():
    box = _StubBox(PROFILE)

    asyncio.run(admission_context.cached_read(
        box, admission_context.PROFILE_TOOL, {"hadm_id": HADM}, hadm_id=HADM))
    asyncio.run(admission_context.cached_read(
        box, admission_context.PROFILE_TOOL, {"hadm_id": HADM}, hadm_id=HADM))

    assert len(box.calls) == 2
    assert admission_context.read_cache_size() == 0


def test_a_tool_outside_the_cacheable_set_is_always_read():
    box = _ServedBox({"articles": []})

    asyncio.run(admission_context.cached_read(
        box, "search_literature", {"query": "anticoagulation"}, hadm_id=HADM))
    asyncio.run(admission_context.cached_read(
        box, "search_literature", {"query": "anticoagulation"}, hadm_id=HADM))

    assert len(box.calls) == 2


def test_the_cache_is_bounded():
    box = _ServedBox({"passages": []})
    for index in range(admission_context.READ_CACHE_MAX_ENTRIES + 5):
        asyncio.run(admission_context.cached_read(
            box, "rag_search", {"hadm_id": HADM, "query": f"query {index}"},
            hadm_id=HADM))

    assert admission_context.read_cache_size() == admission_context.READ_CACHE_MAX_ENTRIES


# --------------------------------------------------------------------------------------
# 6. End to end: a declared medication chip is answered in one pass with fewer tools.
# --------------------------------------------------------------------------------------


class _Tool:
    description = "A tool the test publishes."
    input_schema = {"type": "object", "properties": {}}


class _RecordingModel:
    """Answers in prose and records the bound set of every pass it is asked for.

    The set is recorded on `ainvoke`, not on `bind_tools`: `build_graph` binds twice before
    the run starts (AUTO and the forced round), so a bind-time counter reports the wiring
    rather than the passes.
    """

    def __init__(self) -> None:
        self.bound: list[list[str]] = []

    def bind_tools(self, tools, tool_choice=None):
        return _BoundPass(self, sorted(tool.name for tool in tools))

    async def ainvoke(self, messages, **kwargs) -> AIMessage:
        self.bound.append([])
        return AIMessage(content="The discharge dosing plan is recorded.")


class _BoundPass:
    def __init__(self, owner: _RecordingModel, names: list[str]) -> None:
        self._owner = owner
        self._names = names

    async def ainvoke(self, messages, **kwargs) -> AIMessage:
        self._owner.bound.append(self._names)
        return AIMessage(content="The discharge dosing plan is recorded.")


def _box(*names: str) -> MCPToolbox:
    box = MCPToolbox(session=None)
    box._tools = {name: _Tool() for name in names}

    async def call(name: str, arguments: dict) -> dict:
        return {"hadm_id": arguments.get("hadm_id"), "passages": []}

    box.call = call
    return box


def _medication_turns(monkeypatch) -> list[dict]:
    model = _RecordingModel()
    monkeypatch.setattr(model_gateway, "build_llm", lambda name, **_kw: model)
    turns = [{
        "question": "/meds",
        "answer": "Discharged on oxycodone SR.",
        "tool_calls": [{
            "name": "discharge_medications",
            "args": {"hadm_id": HADM},
            "payload": {"returned": 1, "passages": [{
                "section": "discharge_medications",
                "text": "Oxycodone SR 20 mg every 12 hours, then decrease to 10 mg."}]},
        }],
    }]

    asyncio.run(graph.ask(
        _box("rag_search", "rag_search_sections", "discharge_medications",
             "get_lab_trends", "predict_readmission"),
        P1_TAPER, hadm_id=HADM, model="test", turns=turns, intent="follow_up"))

    return model.bound


def test_a_declared_medication_chip_is_bound_to_the_note_tools_and_nothing_else(monkeypatch):
    bound = _medication_turns(monkeypatch)

    assert len(bound) == 1, "the run went back for evidence it already had"
    # The extraction is in the replay, and the panel is a different domain's tool.
    assert "discharge_medications" not in bound[0]
    assert "get_lab_trends" not in bound[0]
    assert "rag_search" in bound[0]
