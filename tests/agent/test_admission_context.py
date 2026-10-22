"""The pre-flight manifest: decoding, rendering, and failing silently.

The blind spot these cover is an asymmetry rather than a bug. The risk model is
scored on a tabular row, the retrieval tools read note text, and a discharge note
never declares an administrative field — so the agent could name `admission_type`
as a driver and then correctly report that the note does not document it. Both
statements true, the pair useless.

Two properties matter and they pull in opposite directions:

* The values must be *readable* — a clinician asking "what is the admission type?"
  needs "Urgent (URGENT)", not `admission_type_urgent = 1`.
* A value that is not there must be reported as absent, never inferred. The
  decoder resolves a one-hot group to a label or to "Not recorded", and there is
  no third outcome.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

from services.agent import admission_context, graph
from services.mcp.dependencies.features.profile import decode_profile


def _row(**overrides):
    """A minimal encoded row: an urgent admission, home discharge, and a count."""
    row = {
        "age": 61.0,
        "gender": 0.0,
        "admission_type_urgent": 1.0,
        "admission_type_ew_emer": 0.0,
        "discharge_location_home": 1.0,
        "insurance_medicare": 1.0,
        "race_white": 1.0,
        "prior_admission_count": 1.0,
        "prior_inpatient_days": 4.75,
        "recent_ed_visits": 0.0,
        "index_los_days": 3.0,
    }
    row.update(overrides)
    return row


def test_a_one_hot_group_decodes_to_the_level_it_names():
    profile = decode_profile(_row())

    assert profile["admission_type"] == "Urgent (URGENT)"
    assert profile["discharge_location"] == "Home"
    assert profile["insurance"] == "Medicare"
    assert profile["race"] == "White"


def test_a_group_with_no_hit_is_absent_rather_than_guessed():
    """Every dummy zero is a level the row does not carry."""
    row = _row()
    row["admission_type_urgent"] = 0.0

    assert decode_profile(row)["admission_type"] == "Not recorded"


def test_two_hits_in_one_group_are_absent_rather_than_taken_in_column_order():
    """The encoding forbids this, and resolving it would be an invention."""
    row = _row(admission_type_ew_emer=1.0)

    assert decode_profile(row)["admission_type"] == "Not recorded"


def test_a_null_group_is_absent():
    row = _row()
    row["admission_type_urgent"] = None

    assert decode_profile(row)["admission_type"] == "Not recorded"


def test_sex_is_read_from_its_dummy():
    """`gender` is `CAST(gender = 'M' AS INT64)` in the mart: 1 male, 0 female."""
    assert decode_profile(_row(gender=1.0))["gender"] == "Male"
    assert decode_profile(_row(gender=0.0))["gender"] == "Female"
    assert decode_profile(_row(gender=None))["gender"] == "Not recorded"


def test_counts_keep_whole_numbers_whole_and_fractions_fractional():
    profile = decode_profile(_row())

    assert profile["age"] == "61"
    assert profile["prior_inpatient_days"] == "4.75"
    assert profile["prior_admission_count"] == "1"


def test_every_declared_field_is_present_in_the_profile():
    """Callers index the dict; a missing key would be a KeyError at prompt time."""
    declared = {name for name, _label in admission_context._FIELDS}

    assert set(decode_profile(_row())) == declared


def test_the_block_names_the_admission_and_states_the_values():
    block = admission_context.render(90000015, decode_profile(_row()))

    assert block.startswith('<admission_metadata hadm_id="90000015"')
    assert block.rstrip().endswith("</admission_metadata>")
    assert "- admission_type: Urgent (URGENT)" in block
    assert "- age: 61" in block


def test_an_empty_profile_renders_nothing_at_all():
    """A block with no fields in it would imply a fact was found."""
    assert admission_context.render(90000015, {}) == ""
    assert admission_context.render(90000015, None) == ""


def test_a_value_cannot_change_the_shape_of_the_instruction():
    """Values reach a prompt; markup in one would escape its own block."""
    block = admission_context.render(90000015, {"admission_type": "EMER > <b>"})

    assert "<b>" not in block
    assert "EMER > " not in block


class _Box:
    """The sliver of the toolbox `hydrate` uses."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    async def call(self, name, arguments):
        self.calls.append((name, arguments))
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def test_hydration_reads_the_profile_once_for_the_admission():
    box = _Box({"hadm_id": 90000015, "profile": decode_profile(_row())})

    block = asyncio.run(admission_context.hydrate(box, 90000015))

    assert box.calls == [("get_admission_profile", {"hadm_id": 90000015})]
    assert "admission_type: Urgent (URGENT)" in block


def test_a_turn_with_no_admission_does_not_call_anything():
    box = _Box({})

    assert asyncio.run(admission_context.hydrate(box, None)) == ""
    assert box.calls == []


def test_a_refused_profile_leaves_the_prompt_untouched():
    box = _Box({"error": "unknown_patient", "message": "no row"})

    assert asyncio.run(admission_context.hydrate(box, 90000015)) == ""


def test_a_failing_profile_call_never_fails_the_turn():
    box = _Box(RuntimeError("transport down"))

    assert asyncio.run(admission_context.hydrate(box, 90000015)) == ""


# --- the manifest replaces the forced retrieval -------------------------------


class _AnsweringModel:
    """Answers with fixed text and counts how many turns it was asked for."""

    def __init__(self, text="The length of stay was 17.29 days."):
        self.text = text
        self.calls = 0

    def bind_tools(self, tools, **kwargs):
        return self

    async def ainvoke(self, messages, **kwargs):
        self.calls += 1
        return AIMessage(content=self.text)


class _FakeTool:
    """The two attributes `graph._tools` reads off an advertised tool."""

    description = "Retrieve passages from the discharge note."
    input_schema = {"type": "object", "properties": {}}


def _ask(model, question, manifest, **kwargs):
    from services.agent.mcp_client import MCPToolbox

    box = MCPToolbox(session=None)
    box._tools = {"rag_search": _FakeTool()}

    async def call(name, arguments):
        return {"hadm_id": arguments["hadm_id"], "passages": []}

    box.call = call
    with patch.object(graph, "_build_llm", lambda name: model):
        return asyncio.run(graph.ask(
            box, question, hadm_id=90000058, model="test",
            admission_context=manifest, **kwargs,
        ))


MANIFEST = (
    '<admission_metadata hadm_id="90000058">\n'
    "- length_of_stay_days: 17.2917\n"
    "</admission_metadata>"
)


def test_an_administrative_answer_is_not_sent_looking_for_note_evidence():
    """The manifest already IS the evidence; a forced search can only replace it.

    This is the defect that shipped and was caught on the demo screen: the graph
    demanded a retrieval after a correct tool-less answer, the notes could not
    contain a length of stay, and the second reply ("the discharge note does not
    document this") became the settled answer because only the last message counts.
    """
    model = _AnsweringModel()

    state = _ask(model, "what is the patient length of stay?", MANIFEST)

    assert model.calls == 1, "the turn was forced into a retrieval it did not need"
    assert state["tool_calls"] == []
    assert graph.final_text(state).startswith("The length of stay was 17.29 days.")


def test_a_clinical_question_still_has_its_retrieval_demanded():
    """The guard the change must not weaken: chart answers still have to read the chart."""
    model = _AnsweringModel("She was anxious throughout.")

    _ask(model, "What did the discharge note say about her anxiety?", MANIFEST)

    assert model.calls == 2, "a tool-less chart answer must still be sent to the notes"


def test_without_a_manifest_nothing_is_skipped():
    model = _AnsweringModel()

    _ask(model, "what is the patient length of stay?", "")

    assert model.calls == 2


# --- the same rule for a driver drill-down -----------------------------------

DRIVER_QUESTION = (
    "What in this patient's chart explains prior inpatient visits as a "
    "readmission risk driver?"
)

DRIVER_TURNS = [{
    "question": "What is this patient's readmission risk?",
    "tool_calls": [{
        "name": "predict_readmission",
        "args": {"hadm_id": 90000058},
        "payload": {
            "probability": 0.62,
            "threshold": 0.5,
            "decision": 1,
            "factors": [
                {"label": "prior inpatient visits", "contribution": 0.11},
            ],
        },
    }],
    "answer": "The probability is 0.62, above the 0.5 threshold.",
}]


def test_a_declared_driver_drilldown_answers_from_the_replayed_score():
    """The score and its factors are the evidence; a forced search replaces them.

    A clicked driver bar declares `intent: risk_driver`, and the turn it opens is a
    follow-up whose replay already holds the scoring call. Demanding a retrieval
    cannot improve that answer and costs it: the forced round's reply becomes the
    settled answer, so `demand_retrieval` must not fire here.
    """
    model = _AnsweringModel("Prior admissions contributed most of the estimate.")

    state = _ask(
        model, DRIVER_QUESTION, MANIFEST,
        turns=DRIVER_TURNS, intent="risk_driver",
    )

    assert model.calls == 1, "the drill-down was forced into a retrieval it did not need"
    assert state["tool_calls"] == []


def test_the_same_question_undeclared_still_has_its_retrieval_demanded():
    """The declaration is the signal, not the wording: only the control knows."""
    model = _AnsweringModel("Prior admissions contributed most of the estimate.")

    _ask(model, DRIVER_QUESTION, MANIFEST, turns=DRIVER_TURNS)

    assert model.calls == 2


def test_a_declaration_without_a_replayed_score_still_retrieves():
    """A first-turn declaration has nothing behind it yet, so it reads the chart."""
    model = _AnsweringModel("Prior admissions contributed most of the estimate.")

    _ask(model, DRIVER_QUESTION, MANIFEST, intent="risk_driver")

    assert model.calls == 2
