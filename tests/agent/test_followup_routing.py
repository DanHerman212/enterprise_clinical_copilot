# --- RC-3: a declared chip is answered in one pass ------------------------------

import asyncio
from unittest.mock import patch

from langchain_core.messages import AIMessage

from services.agent.postprocess import followups
from services.agent import graph
from services.agent.routing import routing

#: A series in the shape `get_lab_trends` returns, stored on the replayed turn.
LABS = {
    "trends": [
        {"label": "Red blood cells", "unit": "x10^6/uL",
         "points": [{"day": 1, "value": 4.2}, {"day": 3, "value": 3.47}]},
    ],
    "synthetic": True,
    "note": "Values are synthetic.",
    "hadm_id": 90000015,
}

#: The chip the production defect came from, verbatim: the `{label}` slot is the series
#: label, so the question carries the record's own wording rather than a clinician's.
THE_CHIP = ("How does the abnormal discharge Red blood cells (3.47 x10^6/uL) compare to "
            "its reference range and trend?")


class _AnswersInProse:
    """A model that answers and calls nothing, recording how each turn was bound.

    `tool_choice` is the whole signal: `None` is the AUTO binding a first pass gets, and
    `"any"` is the forced round `demand_retrieval` produces. Counting turns is how the
    test sees whether the loop went back.
    """

    def __init__(self):
        self.turns: list[str | None] = []

    def bind_tools(self, tools, tool_choice=None):
        return _BoundTurn(self, tool_choice)

    async def ainvoke(self, messages, **kwargs):  # pragma: no cover - always bound
        self.turns.append(None)
        return AIMessage(content=ANSWER)


class _BoundTurn:
    def __init__(self, owner, choice):
        self._owner = owner
        self._choice = choice

    async def ainvoke(self, messages, **kwargs):
        self._owner.turns.append(self._choice)
        return AIMessage(content=ANSWER)


ANSWER = "The red cell count was 3.47 x10^6/uL at discharge, below its reference range."


class _Toolbox:
    """Just enough toolbox for `graph._tools`, with a call that records itself."""

    description = "Retrieve passages from the discharge note."
    input_schema = {"type": "object", "properties": {}}


def _box():
    from services.agent.mcp_client import MCPToolbox

    box = MCPToolbox(session=None)
    box._tools = {"rag_search": _Toolbox()}

    async def call(name, arguments):
        return {"hadm_id": arguments.get("hadm_id"), "passages": []}

    box.call = call
    return box


def _run(model, **kwargs):
    with patch.object(graph, "_build_llm", lambda name: model):
        return asyncio.run(
            graph.ask(_box(), THE_CHIP, hadm_id=90000015, model="test", **kwargs)
        )


LAB_TURN = [{
    "question": "/labs",
    "answer": "Creatinine rose and the red cell count fell across the admission.",
    "tool_calls": [{
        "name": "get_lab_trends",
        "args": {"hadm_id": 90000015},
        "payload": LABS,
    }],
}]


class TestADeclaredFollowUpIsAnsweredInOnePass:
    def test_a_declared_chip_answers_from_the_transcript_and_stops(self):
        """The chip's evidence is the tool result the chip was generated from, and that
        result is in the transcript. One model turn: no forced retrieval, no second
        pass, and therefore no prose for the client to discard."""
        model = _AnswersInProse()

        _run(model, turns=LAB_TURN, intent="follow_up")

        assert model.turns == [None], (
            "the run went back for evidence on a question generated from evidence")

    def test_an_undeclared_question_still_gets_the_forced_retrieval(self):
        """The 2026-10-15 protection, unchanged for questions a person typed: a chart
        answer with no retrieval is sent back once, and the model must call something."""
        model = _AnswersInProse()

        _run(model, turns=LAB_TURN)

        assert model.turns == [None, "any"]

    def test_a_declaration_with_no_transcript_does_not_exempt_anything(self):
        """A first turn cannot be a follow-up — there is nothing for it to have been
        generated from — so the declaration alone is not a licence to skip retrieval."""
        model = _AnswersInProse()

        _run(model, intent="follow_up")

        assert model.turns == [None, "any"]


class TestTheLabVocabularyCoversTheSeries:
    def test_every_analyte_label_satisfies_the_predicate(self):
        """The chip's `{label}` slot is `Analyte.label`. If the predicate cannot read
        the label, the question the product generated about its own data is the one
        question the product cannot route."""
        from services.mcp.dependencies.labs.synthetic import PANEL

        unmatched = [analyte.label for analyte in PANEL
                     if not routing.asks_about_a_lab_analyte(analyte.label)]
        assert unmatched == [], f"the series labels the predicate cannot read: {unmatched}"

    def test_the_reported_chip_satisfies_the_predicate(self):
        assert routing.asks_about_a_lab_analyte(THE_CHIP) is True

    def test_a_rendered_abnormal_value_chip_satisfies_the_predicate(self):
        """P7, filled from every analyte the series can report."""
        from services.mcp.dependencies.labs.synthetic import PANEL

        for analyte in PANEL:
            question = followups._LAB_ABNORMAL.format(
                label=analyte.label, value="3.47", unit=analyte.unit)

            assert routing.asks_about_a_lab_analyte(question) is True, question

    def test_a_rendered_normal_panel_chip_satisfies_the_predicate(self):
        """P8 names two analytes and no value, and is offered when nothing is abnormal."""
        from services.mcp.dependencies.labs.synthetic import PANEL

        for first, second in zip(PANEL, PANEL[1:]):
            question = followups._LAB_NORMAL.format(first=first.label, second=second.label)

            assert routing.asks_about_a_lab_analyte(question) is True, question

    def test_a_question_that_names_no_analyte_is_still_refused(self):
        """The list grew; it did not become permissive. A disposition question is not a
        lab question, and the tool stays out of it."""
        for question in ("What was the discharge disposition?",
                         "Why do prior inpatient days drive this patient's risk?",
                         "What outpatient follow-up is scheduled?"):
            assert routing.asks_about_a_lab_analyte(question) is False, question
