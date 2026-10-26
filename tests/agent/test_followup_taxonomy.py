"""The follow-up taxonomy: every family, from the payload that emits it to the pass that
answers it.

`test_followups.py` asks whether each template fires only when the record carries the
fact it asks about. It does not ask the next question, which is the one the product
turns on: once the chip is on screen and clicked, is it answered in one pass, and can it
go back for evidence it already has?

So each family is driven twice here, from the same table:

  1. **Emits.** A state holding only that family's backing payload must produce that chip
     and nothing else. One case per numbered family, both `P2` wordings, and the
     literature chip.
  2. **Answers.** The same chip is clicked the way the console clicks it — the backing
     turn replayed, `intent="follow_up"` — and the run must make exactly one model turn.
     Two would mean the answer the reader was shown was thrown away and asked for again.

The third question is about what the model is *offered*. `route` tests the model's own
tool calls before any exemption, so a declaration can stop the forced round and cannot
stop a voluntary one. For `P6` and `_LITERATURE` the tool whose result the chip was
generated from used to be bound on the click — `predict_readmission` by the wording
`satisfies routing.asks_why_risk_driver`, `search_literature` by the words "published
literature" that also make the chip routable — so a model that took the offer re-entered
the loop. Those two are now withheld on a declaration, and the last two tests are the
pair: absent when declared, still bound when a person types the same sentence.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

from services.agent import followups, graph
from services.agent.mcp_client import MCPToolbox

ANSWER = "Discharged home in stable condition."

#: Every tool the reasoning loop can be offered, so a withheld one is visibly withheld
#: rather than missing from the stub.
ALL_TOOLS = ("rag_search", "rag_search_sections", "discharge_medications",
             "get_lab_trends", "predict_readmission", "search_literature")


# --------------------------------------------------------------------------- fixtures


def _passages(*rows: tuple[str, str]) -> list[dict[str, str]]:
    return [{"section": section, "text": text} for section, text in rows]


def _plan_state(text: str, *, rows: list[str] | None = None,
                command: str | None = "meds") -> dict[str, Any]:
    return {
        "tool_calls": [{
            "name": "discharge_medications",
            "response": {"returned": 1, "passages": _passages(
                ("discharge_medications", text))},
        }],
        "command": command,
        "command_payload": {
            "medications": [{"medication": name, "dose": "1 mg", "frequency": "daily"}
                            for name in (rows or [])],
            "dropped": [],
            "returned_passages": 1,
        },
    }


def _sections_state(sections: dict[str, str]) -> dict[str, Any]:
    return {
        "tool_calls": [{
            "name": "rag_search_sections",
            "response": {"returned": len(sections),
                         "passages": [{"section": name, "text": text}
                                      for name, text in sections.items()]},
        }],
        "command": None,
    }


def _trend(label: str, value: float, low: float, high: float, *,
           unit: str = "mmol/L", decimals: int = 1) -> dict[str, Any]:
    flag = "high" if value > high else "low" if value < low else "normal"
    return {"label": label, "unit": unit, "reference_low": low, "reference_high": high,
            "decimals": decimals,
            "series": [{"day": 0, "value": low, "flag": "normal"},
                       {"day": 3, "value": value, "flag": flag}]}


def _labs_state(*trends: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool_calls": [{"name": "get_lab_trends",
                        "response": {"returned": len(trends), "trends": list(trends)}}],
        "command": "labs",
    }


def _risk_state() -> dict[str, Any]:
    return {
        "tool_calls": [{
            "name": "predict_readmission",
            "response": {"probability": 0.2, "threshold": 0.11, "decision": "above",
                         "probability_display": "20.0%", "threshold_display": "11.0%",
                         "top_factors": [{"feature": "oncology_flag",
                                          "contribution": 0.31,
                                          "direction": "increases", "value": 1}]},
        }],
        "command": "risk",
    }


def _literature_state(articles: int = 2) -> dict[str, Any]:
    return {
        "tool_calls": [{
            "name": "search_literature",
            "response": {"returned": articles,
                         "articles": [{"pmid": str(1000 + index), "title": f"Study {index}",
                                       "journal": "J Test Med", "year": 2024}
                                      for index in range(articles)]},
        }],
        "command": "literature",
    }


# ------------------------------------------------------------------------------- table


@dataclass(frozen=True)
class _Case:
    """One template family, and everything needed to emit it and click it."""

    id: str
    chip: str
    question: str
    state: dict[str, Any]
    scope: str | None = "patient_chart"
    #: The tool a click must NOT be able to reach: its result is the chip's evidence and
    #: is already in the replay by the time the chip exists.
    crossing: str | None = None
    #: Whether the same sentence, typed by a person with no transcript, still reaches
    #: the crossing tool. True unless the fix detached the wording from its cue test.
    typed_reaches: bool = True


CASES: tuple[_Case, ...] = (
    _Case(
        id="P1-taper",
        chip="What is the dosing and taper schedule for oxycodone SR at discharge?",
        question="/meds",
        state=_plan_state(
            "Discharge medications: Oxycodone SR 20 mg every 12 hours for five days, "
            "then decrease to oxycodone SR 10 mg every 12 hours."),
    ),
    _Case(
        id="P2-anticoagulant",
        chip="What is the discharge dosing plan for Coumadin?",
        question="/meds",
        state=_plan_state("Discharge medications: Coumadin 7 mg daily for 30 days."),
    ),
    _Case(
        id="P2-anticoagulant-monitored",
        chip="What is the discharge dosing and monitoring plan for Coumadin?",
        question="/meds",
        state=_plan_state(
            "Discharge medications: Coumadin 7 mg daily for 30 days. INR every Monday "
            "and Thursday."),
    ),
    _Case(
        id="P3-precautions",
        chip=("What post-discharge activity restrictions, diet instructions, or warning "
              "signs are documented?"),
        question="/ehr",
        state=_plan_state(
            "Discharge instructions: Return for fever or wound drainage. No driving for "
            "one week.", command="summarize"),
    ),
    _Case(
        id="P4-follow-up",
        chip=("What outpatient follow-up appointments or clinic visits are scheduled "
              "after discharge?"),
        question="/ehr",
        state=_plan_state("Discharge instructions: Follow up with Dr. Alvarez in two "
                          "weeks.", command="summarize"),
    ),
    _Case(
        id="P5-regimen",
        chip="What are the discharge dosing instructions for Metoprolol and Furosemide?",
        question="/meds",
        state=_plan_state(
            "Discharge medications: Metoprolol 25 mg twice daily. Furosemide 40 mg daily.",
            rows=["Metoprolol", "Furosemide"]),
    ),
    _Case(
        id="P6-risk-driver",
        chip="Why does oncology history increase this patient's readmission risk?",
        question="/risk",
        state=_risk_state(),
        crossing="predict_readmission",
        # The sentence lost the word "driver", and "driver" was the token
        # `routing.asks_why_risk_driver` needed to bind the predictor on wording alone.
        # That is the accepted cost of the grammar fix: a clinician who TYPES this
        # sentence is answered from the record without the model's attribution, and a
        # clinician who CLICKS it is answered from the prediction the transcript already
        # holds. See `test_the_risk_chip_sentence_typed_reaches_the_record_not_the_model`.
        typed_reaches=False,
    ),
    _Case(
        id="P7-lab-abnormal",
        chip=("How does the abnormal discharge Potassium (6.0 mmol/L) compare to its "
              "reference range and trend?"),
        question="/labs",
        state=_labs_state(_trend("Potassium", 6.0, 3.5, 5.0)),
    ),
    _Case(
        id="P8-lab-normal",
        chip=("What were the discharge values for Sodium and Potassium across the normal "
              "laboratory panels?"),
        question="/labs",
        state=_labs_state(_trend("Sodium", 140.0, 135.0, 145.0),
                          _trend("Potassium", 4.2, 3.5, 5.0)),
    ),
    _Case(
        id="P9-note-overview",
        chip=("What were the primary diagnosis and hospital course documented in the "
              "clinical note?"),
        question="What changed overnight?",
        state=_sections_state({"discharge_diagnosis": "Acute decompensated heart failure.",
                               "brief_hospital_course": "Diuresed with IV furosemide."}),
    ),
    _Case(
        id="literature",
        chip="What else does the published literature report on anticoagulation?",
        question="/pubmed anticoagulation",
        state=_literature_state(),
        scope="literature",
        crossing="search_literature",
    ),
)

IDS = [case.id for case in CASES]


# ------------------------------------------------------------------------------- stubs


class _Tool:
    description = "A tool the audit stubs."
    input_schema = {"type": "object", "properties": {}}


def _box(*names: str) -> MCPToolbox:
    box = MCPToolbox(session=None)
    box._tools = {name: _Tool() for name in (names or ALL_TOOLS)}

    async def call(name: str, arguments: dict) -> dict:
        return {"hadm_id": arguments.get("hadm_id")}

    box.call = call
    return box


class _Model:
    """Answers in prose, and records the tool set every pass was offered.

    `choices` is the whole signal for the loop: `None` is the AUTO binding a first pass
    gets and `"any"` is the forced round `demand_retrieval` produces, so its length is the
    number of model passes the run made. `bound` is what each of those passes could call.
    """

    def __init__(self) -> None:
        self.choices: list[str | None] = []
        self.bound: list[list[str]] = []

    def bind_tools(self, tools, tool_choice=None):
        self.bound.append(sorted(tool.name for tool in tools))
        return _Turn(self, tool_choice)


class _Turn:
    def __init__(self, owner: _Model, choice: str | None) -> None:
        self._owner = owner
        self._choice = choice

    async def ainvoke(self, messages, **kwargs) -> AIMessage:
        self._owner.choices.append(self._choice)
        return AIMessage(content=ANSWER)


def _replay(case: _Case) -> list[dict[str, Any]]:
    """The backing turn, in the shape the site stores and sends back."""
    return [{
        "question": case.question,
        "answer": ANSWER,
        "tool_calls": [{"name": call["name"], "args": {"hadm_id": 90000015},
                        "payload": call["response"]}
                       for call in case.state["tool_calls"]],
    }]


def _run(case: _Case, model: _Model, **kwargs) -> dict:
    with patch.object(graph, "_build_llm", lambda name: model):
        return asyncio.run(graph.ask(_box(), case.chip, hadm_id=90000015,
                                     model="test", **kwargs))


# ------------------------------------------------------------------ 1. the chip exists


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_family_emits_its_chip_and_nothing_else(case: _Case):
    """Each case is a state holding one family's backing payload, so a second chip means
    two families fired off one signal — which would be a trigger defect, not a fixture
    defect, and is worth failing on."""
    chips = followups.suggest(case.question, ANSWER, scope=case.scope, state=case.state)

    assert chips == [case.chip]


# ----------------------------------------------------------- 2. one pass, once declared


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_clicked_chip_is_answered_in_one_pass(case: _Case):
    """The console declares a chip; `route` ends the turn on the declaration. Two passes
    would mean the run went back for evidence the replay already holds, and the reader
    would watch an answer be replaced by a second one."""
    model = _Model()

    _run(case, model, turns=_replay(case), intent="follow_up")

    assert model.choices == [None], (
        f"{case.id} went back for evidence on a question generated from evidence")


# ------------------------------------------------ 3. nothing to re-run on the click

CROSSING_CASES = [case for case in CASES if case.crossing]
TYPED_REACHING = [case for case in CROSSING_CASES if case.typed_reaches]


@pytest.mark.parametrize("case", CROSSING_CASES, ids=[case.id for case in CROSSING_CASES])
def test_a_declared_chip_is_not_offered_the_tool_it_was_generated_from(case: _Case):
    """`predict_readmission` and `search_literature` cross the free-text boundary on a
    cue test, and both cues are in the chip's own wording. A declaration is a fact about
    where the question came from, so the tool that would return what the replay already
    holds is withheld — absent, not refused."""
    model = _Model()

    _run(case, model, turns=_replay(case), intent="follow_up")

    assert case.crossing not in model.bound[0], (
        f"{case.id} was offered {case.crossing}, which can only re-fetch its own evidence")


@pytest.mark.parametrize("case", TYPED_REACHING, ids=[case.id for case in TYPED_REACHING])
def test_the_same_sentence_typed_still_reaches_the_tool(case: _Case):
    """The control, and the reason the withholding is keyed on the declaration rather
    than on the wording: a person who TYPES this question has no transcript behind it, so
    the tool is the only way the question can be answered at all. The literature chip's
    wording is unchanged, so this pairs with the test above."""
    model = _Model()

    _run(case, model, turns=[])

    assert case.crossing in model.bound[0]


def test_the_risk_chip_sentence_typed_reaches_the_record_not_the_model():
    """The cost of the grammar fix, asserted rather than left to be discovered.

    `_RISK_DRIVER` used to end "…readmission risk driver?", and "driver" was doing two
    jobs: it was part of the sentence and it was the token `routing.asks_why_risk_driver`
    matched. Dropping it for grammar unbound `predict_readmission` from the wording, which
    is right for the chip — its evidence is the replayed prediction — and a loss for
    anyone who types the same sentence, because nothing in it asks for the model's own
    attribution any more.

    Pinned as expected behaviour so it can be a decision rather than a surprise. If typed
    access to the attribution matters more than this wording, the sentence to use is
    "Why does {driver} drive this patient's readmission risk?" — grammatical, and it
    keeps the cue.
    """
    risk_case = next(case for case in CASES if case.id == "P6-risk-driver")
    model = _Model()

    _run(risk_case, model, turns=[])

    assert "predict_readmission" not in model.bound[0]
    # What it does reach: the standing record tools, which is what an answer without the
    # model's attribution is composed from.
    assert "rag_search" in model.bound[0]
