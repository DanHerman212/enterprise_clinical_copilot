"""The command boundary: one tool per command, and nothing else runs.

UX Phase 1. A slash command is deterministic, and "deterministic" is a property
of the path rather than a promise in a prompt:

  - the tool and its arguments come from `routing.py`, so the model is not asked
    which tool to call;
  - the formatting turn is made with no tools bound, so a secondary search is
    impossible rather than forbidden.

Both are asserted here against the code, because the failure they prevent is
invisible in the answer. A secondary search that succeeds, or a risk score the
model produced itself instead of reading, both look like a good answer.

Offline: a recording stub toolbox and a stub model, so no Vertex credentials and
no MCP server are involved.
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import SystemMessage  # noqa: E402

from services.agent import graph  # noqa: E402
from services.agent import routing  # noqa: E402
from services.agent import payload as payload_module  # noqa: E402
from services.agent.deterministic import (  # noqa: E402
    _NO_READMISSION,
    _presented_prediction,
    medication_narrative,
    parse_medications,
    run_command,
)
from services.agent.contracts import parse_agent_request  # noqa: E402
from services.agent.contracts import AgentRequestError  # noqa: E402
from services.agent.payload import build_ui_components  # noqa: E402
from services.mcp.tools import __all__ as REGISTERED_TOOLS  # noqa: E402


class _RecordingToolbox:
    """Records every call, and answers with a payload the caller sets."""

    def __init__(self, payload=None):
        self.calls: list[tuple[str, dict]] = []
        self.payload = payload if payload is not None else {"returned": 0,
                                                            "passages": []}

    async def call(self, name, args):
        self.calls.append((name, dict(args)))
        return self.payload


class _NoToolsModel:
    """A model that fails loudly if anything binds tools to it.

    `_build_llm` binds nothing, so the real object has no tools either. This
    stands in for it and turns the property into an assertion: if the formatting
    turn ever grows a `bind_tools` call, the test says so rather than passing
    while the model quietly gains the ability to search again.
    """

    def __init__(self, reply='{"medications": []}'):
        self.reply = reply
        self.seen: list[list] = []

    def bind_tools(self, *args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError(
            "the formatting turn was given tools; it must be unable to call any"
        )

    async def ainvoke(self, messages, config=None):
        self.seen.append(messages)
        from langchain_core.messages import AIMessage

        return AIMessage(content=self.reply)


# The component each command is drawn as, written out longhand from the design
# document and keyed by the WIRE name. Both halves matter: the left column is the
# vocabulary the request carries, and the right column is what the document asks
# for.
EXPECTED_COMPONENTS = {
    "risk": "risk_visualizer",
    "meds": "medication_table",
    "summarize": "narrative_with_sources",
    "literature": "literature_cards",
    # Phase 4B: the lab trajectories, which had no command at all until then.
    "labs": "lab_trends",
}

# The caller's own words, for the one command that needs them.
TOPIC = "a clinical topic"


def _topic_for(command):
    return TOPIC if command.topic_arg else None


def _run_command(command, hadm_id=90000023, *, payload=None, reply=None,
                 model=None):
    """Run one command against stubs, and return (state, toolbox, model)."""
    box = _RecordingToolbox(payload)
    if model is None:
        model = _NoToolsModel(reply) if reply is not None else _NoToolsModel()
    with patch.object(graph, "_build_llm", lambda name: model):
        state = asyncio.run(run_command(
            box, command, hadm_id, topic=_topic_for(command),
            question="a command", model="test-model",
        ))
    return state, box, model


# --- the table ---------------------------------------------------------------

def test_every_command_names_tools_the_server_registers():
    """A command that names a tool the server does not serve is a 500 at the
    moment a clinician presses the button, not a wrong answer.

    Every declared call is checked, not just the primary: a supplementary tool is
    as reachable as the one beside it, so it is as able to be unregistered."""
    declared = {name for c in routing.COMMANDS for name in c.tools}
    unknown = declared - set(REGISTERED_TOOLS)
    assert not unknown, f"commands name unregistered tools: {sorted(unknown)}"


def test_every_command_forbids_every_other_tool():
    """The declaration and the server's tool set are checked against each other.

    Written as the complement rather than as a literal list on purpose: a tool
    added to the server is then forbidden to every command until someone says
    otherwise here, which is the safe direction. The other direction fails
    silently — a new tool that no command forbids is a command that may call it.
    """
    for command in routing.COMMANDS:
        assert command.forbidden == set(REGISTERED_TOOLS) - set(command.tools), (
            f"{command.slash} does not forbid every tool but its own"
        )


def test_the_admission_is_an_argument_only_where_a_tool_takes_one():
    for command in routing.COMMANDS:
        if command.wire == "literature":
            assert routing.tool_arguments(command, 90000023, TOPIC) == {
                "query": TOPIC}
            continue
        assert routing.tool_arguments(command, 90000023) == {"hadm_id": 90000023}


def test_the_literature_command_will_not_run_without_the_callers_words():
    """There is no defensible default, and inventing one would mean reading the
    patient's chart to fill it in."""
    command = routing.command_for("literature")

    with pytest.raises(routing.MissingTopic) as caught:
        routing.tool_arguments(command, None)

    # The refusal a clinician reads names the command and gives the shape of the
    # answer, because a refusal without an example is a dead end.
    assert command.slash in caught.value.message
    assert "clinical topic" in caught.value.message


def test_a_command_that_takes_no_topic_refuses_the_callers_words():
    """Text beside a command that does not take any is a second intention. The
    contract refuses it before this, and this refuses it again for a caller that
    reached the tool layer directly."""
    with pytest.raises(ValueError):
        routing.tool_arguments(routing.command_for("risk"), 90000023, "a question")


def test_a_command_that_needs_an_admission_refuses_to_run_without_one():
    """Calling the tool with hadm_id=None would report the tool's complaint as
    the answer, which reads like a fact about the patient."""
    for command in routing.COMMANDS:
        if command.wire == "literature":
            continue
        with pytest.raises(ValueError):
            routing.tool_arguments(command, None)


def test_the_summary_command_carries_the_storage_spellings_of_its_sections():
    """Four sections, spelled as the index spells them.

    The document names `hospital_course` and `discharge_diagnoses`; storage calls
    them `brief_hospital_course` and `discharge_diagnosis`. A section name is part
    of every chunk id, so the document's spelling would retrieve nothing without
    a re-index — hence the storage spelling, pinned here so a future edit that
    "corrects" it fails loudly instead.
    """
    summary = routing.command_for("summarize")
    assert summary.sections == (
        "history_of_present_illness", "brief_hospital_course",
        "discharge_diagnosis", "discharge_instructions")
    assert "discharge_medications" not in summary.sections
    assert "discharge_summary" not in summary.sections


def test_an_unknown_wire_value_is_not_a_command():
    """The caller's allowlist and this table must agree about the vocabulary. A
    value that reaches here and is not a command is a disagreement, and the
    honest answer is a refusal rather than a fall back to the reasoning loop."""
    assert routing.is_command(None) is False
    assert routing.is_command("summarize") is True
    assert routing.is_command("not_a_command") is False
    with pytest.raises(routing.UnknownCommand):
        routing.command_for("not_a_command")


# --- one call, and only one --------------------------------------------------

@pytest.mark.parametrize("command", routing.COMMANDS, ids=lambda c: c.wire)
def test_a_command_runs_exactly_its_own_tools_once(command):
    state, box, _ = _run_command(command)

    # Every declared call, in declaration order, once each — and nothing else.
    # `/ehr` declares two, so this is the assertion that fails if a supplementary
    # call is dropped, duplicated, or run ahead of the tool it accompanies.
    assert [name for name, _ in box.calls] == list(command.tools), box.calls

    name, args = box.calls[0]
    assert name not in command.forbidden
    assert args == routing.tool_arguments(command, 90000023, _topic_for(command))

    # A supplementary call is built from the same template rules as the primary,
    # so a second call cannot acquire its own argument conventions.
    for index, (tool, extra) in enumerate(
            routing.supplementary_arguments(command, 90000023), start=1):
        assert box.calls[index] == (tool, extra), box.calls[index]

    # The record the answer is stored with names the same calls, so a replayed
    # turn shows what actually ran.
    assert [call["name"] for call in state["tool_calls"]] == list(command.tools)


@pytest.mark.parametrize(
    "command", [c for c in routing.COMMANDS if c.prose], ids=lambda c: c.wire)
def test_the_formatting_turn_cannot_call_anything(command):
    state, _, model = _run_command(command)

    assert len(model.seen) == 1, "the formatting turn ran more than once"
    system = model.seen[0][0]
    assert isinstance(system, SystemMessage)
    # The command's own formatting instruction is what the model is given, not
    # the tool-choosing system prompt.
    assert command.formatting in system.content


def test_the_supplementary_result_reaches_the_formatting_turn():
    """`/ehr` hands the risk result to the summarizer, so the narrative can explain it.

    This reverses a withholding that briefly stood here. Withholding the risk payload did
    not stop the model writing about risk; it stopped the model explaining risk, and the
    reader got a chart summary beside a score with nothing joining them. The division of
    labour that replaces it — the widget owns the score and the bars, the narrative's
    fourth section owns the clinical meaning — only works if the model can see the drivers.

    The drivers it sees are named and directed rather than quantified; that narrowing is
    the subject of the projection tests below.
    """
    command = routing.command_for("summarize")
    state, box, model = _run_command(command)

    # The turn really did make both calls...
    assert [name for name, _ in box.calls] == ["rag_search_sections",
                                              "predict_readmission"]
    # ...and the composer still sees the estimate, because it is what draws the chip.
    assert "predict_readmission" in [call["name"] for call in state["tool_calls"]]

    # ...and so does the prose: the risk result is in the formatting prompt now.
    human = model.seen[0][1]
    assert '<tool_result name="predict_readmission">' in human.content
    assert '<tool_result name="rag_search_sections">' in human.content


# A prediction payload at the precision the service serves it at: six decimals on the
# score and the base value, four on a driver contribution. The projection is what narrows
# it, and what it narrows to is the subject of the tests below.
SERVED_RISK = {
    "hadm_id": 90000023,
    "probability": 0.332758,
    "threshold": 0.11,
    "decision": 1,
    "base_value": 0.333758,
    "top_factors": [
        {"feature": "rdw_max", "contribution": 0.316788, "direction": "increases"},
        {"feature": "rbc_min", "contribution": 0.201144, "direction": "increases"},
    ],
    "model_version": "risk-1",
    "feature_source": "bigquery",
}


def test_the_risk_numbers_reach_the_model_as_stated_strings_only():
    """The prose is given the strings, not the floats (Phase 1.1, 2026-10-16).

    The leak this closes was guardrail-PERMITTED, which is precisely why it survived: a
    served probability quoted accurately is a number `verify_risk_numbers` has nothing to
    strip, so the turn was served with no flags at all and a fourth section reading like a
    debug dump beside a widget printing 9.0%. Three rounds of prompt wording did not stop
    it, because a float the model is holding is a float it can paste — so the copy it reads
    no longer holds one.
    """
    projected = _presented_prediction("predict_readmission", dict(SERVED_RISK))

    assert projected["probability_display"] == "33.3%"
    assert projected["threshold_display"] == "11.0%"

    for withheld in ("probability", "threshold", "base_value"):
        assert withheld not in projected, f"{withheld} is quotable while it is present"

    # The drivers survive as names and directions: the narrative still has to explain
    # them, and the widget still has to draw them.
    assert projected["top_factors"] == [
        {"feature": "red cell distribution width (RDW)", "direction": "increases"},
        {"feature": "lowest red blood cell count", "direction": "increases"},
    ]

    # And the payload handed in is untouched — the caller's copy is not this one.
    assert SERVED_RISK["probability"] == 0.332758


def test_the_execution_record_keeps_the_numbers_the_model_never_saw():
    """Withholding is a projection, not an edit.

    Load-bearing twice over: the guardrail reads the recorded call and not the model's
    copy, and a reader asking next week why a turn said 33.3% has to be able to get back
    to the number that produced it.
    """
    command = routing.command_for("summarize")
    state, _, _ = _run_command(command, payload=dict(SERVED_RISK))

    served = next(call["response"] for call in state["tool_calls"]
                  if call["name"] == "predict_readmission")

    assert served["probability"] == 0.332758
    assert served["base_value"] == 0.333758
    assert served["top_factors"][0]["contribution"] == 0.316788


def test_the_summary_prompt_names_the_strings_the_payload_carries():
    """The instruction and the projection move together.

    A prompt that names a field the payload no longer has is an instruction to invent it,
    which is the failure this pair exists to prevent — so the two are asserted against each
    other rather than separately.
    """
    formatting = routing.command_for("summarize").formatting
    projected = _presented_prediction("predict_readmission", dict(SERVED_RISK))

    for field in ("probability_display", "threshold_display"):
        assert field in formatting, f"the summary prompt does not name {field!r}"
        assert field in projected, f"the payload does not carry {field!r}"

    # Stated as a prohibition as well as a substitution: the strings are what to write,
    # and an unrounded decimal is what not to.
    assert "unrounded" in formatting


def test_the_summary_prompt_makes_sections_conditional_on_the_evidence():
    """A heading the passages do not support is a heading the model must omit.

    The sections are still named, because the interface renders them, but three of them
    are now conditional. Ordering all four unconditionally is what produced a fabricated
    discharge plan for 90000028 and an "admission" for 90000073 on 2026-10-20: to satisfy
    the format the model wrote sections the note did not contain.

    Asserted as the section names plus the omission rule, so the wording can be improved
    without failing a test that only ever checked the requirement is still stated.
    """
    formatting = routing.command_for("summarize").formatting

    # The inpatient heading, and the outpatient counterpart that replaces it.
    assert "Hospital course" in formatting
    assert "Clinical course" in formatting
    # Both inventable sections are named as conditional, and omission is sanctioned.
    assert "Discharge diagnosis" in formatting
    assert "Discharge instructions" in formatting
    assert "omit" in formatting.lower()
    # The risk section still has to be asked for.
    assert "Readmission risk & trajectory" in formatting

    # And it must not invite the model to re-derive the score the widget prints.
    assert "round" in formatting


def test_the_outpatient_caveat_forbids_calling_the_record_an_admission():
    """90000073 answered "this admission" about a speech-therapy visit.

    The caveat is the only place the formatting turn learns the record is an outpatient
    one, so the lexical rule has to live there: the answer may not call a visit a stay.
    """
    caveat = _NO_READMISSION

    assert "outpatient" in caveat.lower()
    assert "this admission" in caveat
    assert "this encounter" in caveat or "this visit" in caveat
    assert "Hospital course" in caveat  # redirected to 'Clinical course'


def test_a_prose_less_command_never_builds_a_model():
    """`/risk` has no formatting turn, so it must not construct an LLM either.

    The point of deleting the turn was to take the model out of the path, not to
    call it and discard the result.
    """
    _, _, model = _run_command(routing.command_for("risk"))

    assert model.seen == []


@pytest.mark.parametrize("command", routing.COMMANDS, ids=lambda c: c.wire)
def test_a_command_reports_each_tool_before_the_review_stage(command):
    """The run reads as: plan, run the declared tools, review.

    One `tool`/`result` pair per declared call, in the order the table declares
    them. A supplementary call is work that happened, so it announces like any
    other: a command that listed stages in another order would be narrating work it
    did not do, and silencing the second call would hide half of what `/ehr` ran.
    """
    seen = []
    box = _RecordingToolbox()
    model = _NoToolsModel()
    with patch.object(graph, "_build_llm", lambda name: model):
        asyncio.run(run_command(
            box, command, 90000023, topic=_topic_for(command),
            question="a command", model="test-model",
            on_event=seen.append,
        ))

    assert [event["stage"] for event in seen] == (
        ["planning"] + ["tool", "result"] * len(command.tools) + ["reviewing"]
    )
    assert [event["tool"] for event in seen if event["stage"] == "tool"] == (
        list(command.tools)
    )


def test_the_topic_reaches_the_tool_exactly_as_the_caller_wrote_it():
    """A search string the agent rewrote is a search nobody asked for."""
    state, box, _ = _run_command(
        routing.command_for("literature"), hadm_id=None)

    assert box.calls == [("search_literature", {"query": TOPIC})]
    assert state["command"] == "literature"


def test_the_literature_command_drops_only_its_score_when_there_is_no_patient():
    """`/pubmed` has two shapes, and the second one is the older.

    Its primary tool searches the literature and takes no admission, so a clinician
    reading about a condition with no chart open still gets an answer — the same single
    call the command has always made. What is missing is the comparison, because a
    comparison needs a patient to compare, and a dropped call is a call that did not
    happen rather than one that happened with `None` in it.
    """
    command = routing.command_for("literature")

    assert routing.supplementary_arguments(command, None) == []
    assert routing.supplementary_arguments(command, 90000023) == [
        ("predict_readmission", {"hadm_id": 90000023})]


def test_only_the_literature_command_may_run_short():
    """The flag is a shape, not a licence.

    Stated as the whole set rather than as one assertion about `/pubmed`, because the
    failure this prevents is the flag spreading: a command that silently drops a call it
    declared is a command whose answer depends on what the turn was handed, and that is
    the implicit variation the deterministic path exists to remove.
    """
    flagged = [
        (command.wire, spec.tool)
        for command in routing.COMMANDS
        for spec in command.supplementary
        if spec.skipped_without_admission
    ]

    assert flagged == [("literature", "predict_readmission")]


# --- the medication extraction ----------------------------------------------

_NOTE = ("MEDICATIONS: Tylenol 650 mg q.6h., Lasix 80 mg daily, "
         "Metformin 500 mg twice daily.")


def test_a_medication_the_note_does_not_mention_is_dropped():
    """The one failure a clinician cannot see: a medication the note does not
    contain, rendered in a table beside the ones it does."""
    reply = json.dumps({"medications": [
        {"medication": "Tylenol", "dosage": "650 mg", "frequency": "q.6h."},
        {"medication": "Warfarin", "dosage": "5 mg", "frequency": "daily"},
    ]})

    verified, dropped = parse_medications(reply, _NOTE)

    assert [row["medication"] for row in verified] == ["Tylenol"]
    assert len(dropped) == 1
    assert dropped[0]["medication"] == "Warfarin"
    assert "not present" in dropped[0]["reason"]


def test_a_dosage_the_note_does_not_state_is_dropped():
    """The name is in the note and the dose is not, which is how a plausible
    wrong number gets in: the row looks right and the number was invented."""
    reply = json.dumps({"medications": [
        {"medication": "Lasix", "dosage": "160 mg", "frequency": "daily"},
    ]})

    verified, dropped = parse_medications(reply, _NOTE)

    assert verified == []
    assert "dosage" in dropped[0]["reason"]


def test_an_extraction_with_no_dosage_is_kept():
    """A note line without a dose is a real line; the check is that a stated
    dose is supported, not that every row has one."""
    reply = json.dumps({"medications": [
        {"medication": "Metformin", "dosage": "", "frequency": "twice daily"},
    ]})

    verified, dropped = parse_medications(reply, _NOTE)

    assert dropped == []
    assert verified[0]["medication"] == "Metformin"


def test_a_fenced_json_reply_is_read():
    """The model sometimes wraps its JSON in a fence. Tolerating it here is
    cheaper than letting a formatting habit read as "no medications"."""
    reply = '```json\n{"medications": [{"medication": "Lasix", "dosage": "80 mg"}]}\n```'

    verified, _ = parse_medications(reply, _NOTE)

    assert [row["medication"] for row in verified] == ["Lasix"]


def test_a_reply_that_is_not_json_drops_every_row():
    verified, dropped = parse_medications("I could not read the list.", _NOTE)

    assert verified == []
    assert dropped == []


def test_the_fallback_sentence_needs_a_genuinely_empty_result():
    """The document reserves the disclaimer for an empty tool result. A tool
    that returned text this module could not verify is a different failure, and
    reporting it as "no information available" would turn a formatting bug into a
    clinical statement."""
    assert medication_narrative([], 0) == (
        "No specific discharge medication information is available for this admission."
    )

    unverified = medication_narrative([], 3)
    assert unverified != (
        "No specific discharge medication information is available for this admission."
    )
    assert "formatting failure" in unverified


def test_the_narrative_counts_the_verified_rows_rather_than_all_of_them():
    verified = [{"medication": "Lasix", "dosage": "80 mg", "frequency": "daily"}]
    assert "1 discharge medication" in medication_narrative(verified, 1)


def test_the_medication_command_hands_the_payload_the_verified_rows_only():
    """Through the command, not just the parser: the widget is built from the
    verified rows, so a dropped row cannot reach the grid."""
    reply = json.dumps({"medications": [
        {"medication": "Tylenol", "dosage": "650 mg"},
        {"medication": "Warfarin", "dosage": "5 mg"},
    ]})
    payload = {"returned": 1, "passages": [
        {"id": "MT-1-DS_discharge_medications_1", "section": "discharge_medications",
         "text": _NOTE},
    ]}
    model = _NoToolsModel(reply)

    state, _, _ = _run_command(routing.command_for("meds"), payload=payload,
                               model=model)

    medications = state["command_payload"]["medications"]
    assert [row["medication"] for row in medications] == ["Tylenol"]
    assert state["command_payload"]["dropped"][0]["medication"] == "Warfarin"

    components = build_ui_components("meds", state["tool_calls"],
                                     state["command_payload"])
    assert [c["component"] for c in components] == ["medication_table"]
    assert [row["medication"] for row in components[0]["props"]["medications"]] == [
        "Tylenol"]
    assert components[0]["props"]["dropped"][0]["medication"] == "Warfarin"


# --- the payload the answer is delivered as ---------------------------------

@pytest.mark.parametrize("command", routing.COMMANDS, ids=lambda c: c.wire)
def test_each_command_is_drawn_as_exactly_its_own_component(command):
    """One component per command, named by the table. A second component would
    mean the command reached for evidence it does not own."""
    components = build_ui_components(command.wire, [], {})

    assert [c["component"] for c in components] == [
        EXPECTED_COMPONENTS[command.wire]]


def test_the_component_table_is_keyed_by_the_wire_vocabulary():
    """Checked in both directions, because the mistake this guards against is
    invisible: a table keyed by the clinician-facing names covers half the
    commands and hands the other half to the derived path, which draws something
    plausible. `risk` and `meds` are the same word in both vocabularies;
    `summarize` and `literature` are not.
    """
    routed = {command.wire for command in routing.COMMANDS}

    assert set(payload_module.COMMAND_COMPONENTS) == routed
    assert payload_module.COMMAND_COMPONENTS == EXPECTED_COMPONENTS


def test_a_component_whose_data_is_missing_says_why_rather_than_nothing():
    """Omitting it renders as "nothing to show", which a clinician reads as "the
    notes say nothing" — a different and worse statement than "the tool failed"."""
    components = build_ui_components("literature", [], None)

    assert [c["component"] for c in components] == ["literature_cards"]
    assert "unavailable" in components[0]["props"]


def test_the_risk_component_carries_the_score_and_its_attributions():
    calls = [{
        "name": "predict_readmission",
        "args": {"hadm_id": 90000023},
        "response": {
            "hadm_id": 90000023, "probability": 0.3443, "threshold": 0.11,
            "decision": 1, "model_version": "risk-1",
            "top_factors": [{"feature": "length of stay", "contribution": 0.4,
                             "direction": "increases"}],
        },
    }]

    props = build_ui_components("risk", calls, None)[0]["props"]

    # The served probability is 0.3443; the component carries it at the precision
    # the narrative describes it at, so the prose and the widget agree.
    assert props["probability"] == 0.34
    assert props["threshold"] == 0.11
    assert props["top_factors"][0]["feature"] == "length of stay"


def test_a_free_text_answer_is_drawn_as_what_its_own_evidence_calls_for():
    """The field is not command-only: free text gets the components its calls
    produced, in call order, deduplicated."""
    calls = [
        {"name": "rag_search", "args": {}, "response": {"passages": [
            {"section": "brief_hospital_course", "text": "..."}]}},
        {"name": "rag_search", "args": {}, "response": {"passages": [
            {"section": "discharge_diagnosis", "text": "..."}]}},
        {"name": "search_literature", "args": {}, "response": {
            "returned": 1, "articles": [{"pmid": "1", "title": "T", "abstract": "A"}]}},
    ]

    components = build_ui_components(None, calls, None)

    assert [c["component"] for c in components] == [
        "source_cards", "literature_cards"]


# --- the request that carries both -------------------------------------------

def test_a_command_and_a_question_together_are_refused_not_resolved():
    """Two intentions in one request, and the request can only be answered as
    one. The old behaviour — the command won, the question was dropped — served a
    repeat of the previous answer to a clinician who had asked something new."""
    with pytest.raises(AgentRequestError) as caught:
        parse_agent_request(
            {"chip": "risk", "hadm_id": 90000023, "question": "and his meds?"},
            max_question_chars=2000,
        )

    assert caught.value.code == "ambiguous_intent"
    assert caught.value.status_code == 400
    assert "on its own" in caught.value.message


def test_a_command_alone_still_carries_its_admission_through_the_contract():
    parsed = parse_agent_request(
        {"chip": "meds", "hadm_id": 90000023}, max_question_chars=2000
    )

    assert parsed["kind"] == "meds"
    assert parsed["hadm_id"] == 90000023
    # A command that takes no topic carries none, and the contract says so
    # rather than leaving the field to be discovered downstream.
    assert parsed["topic"] is None


def test_a_command_that_needs_a_topic_is_refused_without_one():
    """The clinician is told what to type, not that something went wrong."""
    with pytest.raises(AgentRequestError) as caught:
        parse_agent_request(
            {"chip": "literature", "hadm_id": 90000023}, max_question_chars=2000
        )

    assert caught.value.code == "topic_required"
    assert caught.value.status_code == 400
    assert "/pubmed" in caught.value.message
    assert "clinical topic" in caught.value.message


def test_the_topic_travels_as_a_value_and_names_itself_in_the_question():
    """Two things at once, because they are one decision: the search string the
    tool is called with is a validated value, and the question the model is asked
    says what was searched rather than "this condition"."""
    parsed = parse_agent_request(
        {"chip": "literature", "hadm_id": 90000023,
         "question": "dual antiplatelet therapy"},
        max_question_chars=2000,
    )

    assert parsed["topic"] == "dual antiplatelet therapy"
    assert parsed["question"] == (
        "Search PubMed for recent literature on dual antiplatelet therapy. "
        "For admission 90000023."
    )


# --- the command typed rather than chosen --------------------------------------
#
# The composer sends a chip when the command is picked from the menu. A command
# typed by hand sent no chip, so the request was routed by the tool-choosing model
# like any other sentence — and "risk for readmission from knee replacement" reads
# as a risk question, so `/pubmed <that topic>` searched for nothing and answered
# with an estimate (observed live 2026-10-13). The lookup below is the fix: the
# command vocabulary is closed and lives in one table, so a slash token is resolved
# by that table rather than by inference.

def test_a_typed_command_is_the_same_intent_as_a_chip():
    parsed = parse_agent_request(
        {"question": "/pubmed risk for readmission from knee replacement",
         "hadm_id": 90000023},
        max_question_chars=2000,
    )

    assert parsed["kind"] == "literature"
    assert parsed["topic"] == "risk for readmission from knee replacement"
    assert parsed["question"] == (
        "Search PubMed for recent literature on risk for readmission from knee "
        "replacement. For admission 90000023."
    )


def test_a_typed_command_that_takes_no_topic_runs_alone():
    parsed = parse_agent_request(
        {"question": "/risk", "hadm_id": 90000023}, max_question_chars=2000
    )

    assert parsed["kind"] == "risk"
    assert parsed["topic"] is None


def test_a_typed_command_is_matched_regardless_of_case_and_surrounding_space():
    parsed = parse_agent_request(
        {"question": "  /PubMed dual antiplatelet therapy  ", "hadm_id": 90000023},
        max_question_chars=2000,
    )

    assert parsed["kind"] == "literature"
    assert parsed["topic"] == "dual antiplatelet therapy"


def test_a_typed_command_with_text_beside_it_is_still_two_intentions():
    """The slash token is resolved, and what remains beside it is judged by the
    same rule as a chip with a question. Without the resolution the whole line
    would have been read as free text, which is the quiet way round the refusal."""
    with pytest.raises(AgentRequestError) as caught:
        parse_agent_request(
            {"question": "/risk and his meds", "hadm_id": 90000023},
            max_question_chars=2000,
        )

    assert caught.value.code == "ambiguous_intent"


def test_a_sent_chip_is_not_overridden_by_typed_text():
    """The browser resolves the typed token too and sends the chip with it, so the
    server resolves only when no chip arrives. Two resolvers that could disagree is
    the bug being closed; the server's derivation is the fallback, not a second
    opinion."""
    with pytest.raises(AgentRequestError) as caught:
        parse_agent_request(
            {"chip": "risk", "hadm_id": 90000023, "question": "/risk"},
            max_question_chars=2000,
        )

    assert caught.value.code == "ambiguous_intent"


def test_an_unknown_slash_token_is_left_as_a_question():
    """A slash is also a ratio, a path and a date separator. Claiming those as
    commands would be a worse failure than the one the lookup prevents."""
    parsed = parse_agent_request(
        {"question": "/usr/bin/env is not a command", "hadm_id": 90000023},
        max_question_chars=2000,
    )

    assert parsed["kind"] is None
    assert "/usr/bin/env is not a command" in parsed["question"]
