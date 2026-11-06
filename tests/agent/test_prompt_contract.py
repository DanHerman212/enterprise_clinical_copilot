"""The prompt and the code it describes must agree.

The prompt is the least-covered artifact in this layer: every other piece has
tests, and the instructions the model is actually given had none. These tests
deliberately do *not* copy the prompt's sentences — that would fail on every
rewording and catch nothing. What they pin is the couplings, the places where the
prompt and the code describe the same thing and could drift apart, because that
is where a silent failure lives.

Two of them carry the weight.

The ``<tool_result>`` delimiter is a literal in two files: the graph wraps every
tool result in it, and the prompt tells the model that everything inside it is
patient data and never an instruction. Rename it on one side and the rule
describes a boundary that no longer exists — every retrieved note passage becomes
text the model was told it may obey. Nothing else in the suite would fail.

The tool names are written into the prompt ("you MUST call
``predict_readmission``") and defined again by the MCP server's registrations. If
they drift, the model is instructed to call a tool that is not there.
"""
import asyncio
import re

import pytest

from services.agent.engine import graph
from services.agent.engine.graph import _execute_tool_calls
from services.agent.postprocess import answer as answer_text
from services.agent.engine.mcp_client import MCPToolbox
from services.agent.engine import prompts
from services.agent.engine.prompts import SYSTEM_PROMPT


def _toolbox() -> MCPToolbox:
    """A toolbox whose calls return a payload and touch nothing."""
    box = MCPToolbox(session=object())

    async def call(name, args):  # noqa: ANN001 - mirrors MCPToolbox.call
        return {"name": name, "args": args}

    box.call = call
    return box


def _wrapped_tool_result() -> str:
    """The exact text the graph hands the model for one tool result.

    Driven through the real `_execute_tool_calls` rather than by formatting the
    tag here, so the test reads what the code emits instead of what this file
    believes it emits.
    """
    messages, _ = asyncio.run(
        _execute_tool_calls(
            _toolbox(),
            [{"name": "rag_search", "args": {"hadm_id": 1}, "id": "call-1"}],
        )
    )
    return messages[0].content


def test_the_prompt_names_a_literal_delimiter():
    """Without a literal tag the data-versus-instructions rule points at nothing."""
    assert re.findall(r"<([a-z_]+)>\.\.\.</\1>", SYSTEM_PROMPT), (
        "The prompt no longer names a literal delimiter, so its rule that note "
        "text is data and not instructions has nothing to refer to."
    )


def test_the_delimiter_the_wrapper_emits_is_the_one_the_prompt_names():
    tags = re.findall(r"<([a-z_]+)>\.\.\.</\1>", SYSTEM_PROMPT)
    wrapped = _wrapped_tool_result()
    for tag in tags:
        assert f"<{tag} name=" in wrapped, (
            f"The prompt calls the data boundary <{tag}> but the wrapper emits "
            f"{wrapped[:80]!r}. Renaming one side leaves the model trusting a "
            "boundary that does not exist."
        )


def test_the_prompt_names_the_tools_the_reasoning_loop_can_call():
    """The free-text prompt and the free-text toolkit must describe one system.

    Checked in BOTH directions now, because the failure on each side is real and
    they are not symmetric. A tool the model can call but is never told about is a
    tool it will not use. A tool it is told to call but does not have is worse:
    the prompt describes a call that cannot happen, and a model asked for a risk
    score it has no means to produce has every reason to write a plausible one —
    which is the exact failure the risk guardrail exists to catch after the fact.

    The set of tools the loop may use lives in `routing.py` beside the commands
    that own the others, so this compares the prompt against that rather than
    against everything the server registers. The reverse check is exact for the
    same reason: which names are tools is no longer a guess, it is the complement
    of the allow-list.
    """
    from services.agent.routing import routing

    for name in sorted(routing.FREETEXT_TOOLS):
        assert f"`{name}`" in SYSTEM_PROMPT, (
            f"the prompt never names {name}, which the loop can call"
        )

    # `predict_readmission` is the one command tool the loop may reach, and only on a
    # turn whose question asks for the model's own reasoning about a driver
    # (`routing.asks_why_risk_driver`), or for the estimate itself
    # (`routing.asks_for_risk_score`). Naming it without those conditions would be the
    # invitation the rule below forbids, so they are asserted with the name.
    assert "`predict_readmission`" in SYSTEM_PROMPT
    assert "two kinds of question" in SYSTEM_PROMPT

    # `search_literature` is the second conditional one: a command's tool that is bound
    # only when the question asks for published evidence (`routing.asks_for_literature`).
    # Naming it is right — the model must know it may search the literature — and the
    # condition has to be named with it, or the prompt reads as a standing invitation.
    assert "`search_literature`" in SYSTEM_PROMPT
    assert "Reach for `search_literature` ONLY when the question asks for" in SYSTEM_PROMPT

    hidden = ({command.tool for command in routing.COMMANDS}
              - routing.FREETEXT_TOOLS
              - {"predict_readmission"} - routing.LITERATURE_TOOLS)
    for name in sorted(hidden):
        assert f"`{name}`" not in SYSTEM_PROMPT, (
            f"the prompt names {name}, which the loop cannot call: either admit it "
            f"to FREETEXT_TOOLS or stop instructing the model to use it"
        )


# Each rule is checked by its operative phrase rather than by the sentence
# around it. A deletion fails; a rewrite has to keep the phrase, which is the
# same bargain the guardrail tests make — these are the rules the post-hoc
# guardrails exist to enforce, so losing one is a behaviour change.
RULE_MARKERS = (
    ("the data-versus-instructions rule", "never instructions"),
    ("the must-call-the-tool rule", "MUST call"),
    ("never invent a citation", "Never invent a citation"),
    # The `only from a tool call you made in THIS turn` marker was removed in Phase 4.
    # It had been load-bearing since 2026-09-18, when the citation guard did resolve
    # markers against this turn alone — and it stopped being true in Phase 1, when the
    # ledger started counting the replayed turns first. The test that replaced it asserts
    # the sentence is GONE; see `test_obsolete_single_turn_citation_rule_is_absent`.
    # The decline is a literal in two files — here and `graph.EVIDENCE_FLAG_KEY` —
    # and the sentence the clinician is owed is chosen from it (2026-10-12).
    ("the evidence-decline contract", '{"evidence_found": false}'),
    # Our record numbers are not clinical facts and are not something a clinician
    # can look up. Live on 2026-10-12 the model wrote one into a prose answer.
    ("the record-number ban", "no admission id"),
    # The free-text path had no equivalent of the command path's outpatient caveat,
    # so 90000073 "Summarize the note" described a speech-therapy visit as an
    # admission with a hospital course (2026-10-21). The prompt now names the block
    # `graph._context_block` injects, and the test below asserts the two agree.
    ("the outpatient-wording rule", "<encounter_notice>"),
    # 90000023 answered a creatinine question with "the discharge note does not
    # document it" on a turn whose only call was `get_lab_trends`: an absence claim
    # about a document nobody opened (2026-10-21).
    ("the absence-needs-a-retrieval rule",
     "An absence you state must be an absence you LOOKED FOR"),
    # The series carries a deterministic flag per timepoint; the prompt binds the
    # prose to it instead of asking the model to compare floats, and forbids the
    # summary sentence that hides a low reading (2026-10-21).
    ("the lab-flag rule", "remained within reference range"),
)


def test_no_prompt_still_prescribes_a_risk_band_word():
    """The rule that a risk is "above or below the operating threshold" and never
    "high", "low" or "borderline" has no prose to live in at all now: `/risk` was
    retired on 2026-10-26, the widget derives the band from the same number, and the
    estimate reaches prose only through `/ehr`, which is governed by
    `prompts.EHR_FORMATTING`.

    Asserted rather than dropped, because the constraint is still load-bearing: the band
    is a visual the interface derives from the served value, and any prose that invents
    its own band contradicts it. The day a risk prose turn comes back, the rule has to
    come back with it, and this test is what notices that it did not.
    """
    from services.agent.routing import routing

    assert [command.wire for command in routing.COMMANDS if command.wire == "risk"] == []
    # `RISK_FORMATTING` was deleted on 2026-10-29 (`00d7` Row 5): it had been marked
    # retired since 2026-10-13 and nothing read it. Asserted as an absence so the constant
    # cannot come back carrying the band wording.
    assert not hasattr(prompts, "RISK_FORMATTING")
    for band in ("high", "low", "borderline"):
        assert f'label the risk "{band}"' not in prompts.EHR_FORMATTING
    # The formatting that DOES carry the score states the same rule, so a `/ehr` answer
    # cannot invent a band either.
    assert "probability_display" in prompts.EHR_FORMATTING


@pytest.mark.parametrize("rule,marker", RULE_MARKERS)
def test_a_load_bearing_rule_is_still_stated(rule, marker):
    assert marker in SYSTEM_PROMPT, (
        f"The prompt no longer states {rule}. If the rule moved, point this at "
        "its new wording; if it was deleted, that is a guardrail change and not "
        "a test fix."
    )


def test_the_graph_uses_the_prompt_this_suite_tests():
    """The artifacts under test must be the shipped ones.

    Two seams since Phase 4, because there are now two artifacts. The suite reads
    `SYSTEM_PROMPT` — every spoke at once — so a rule is asserted once and found
    wherever the composition puts it. A TURN is given `compose_system_prompt`, so the
    composition function itself has to be the one the graph calls, and `SYSTEM_PROMPT`
    has to be the composition rather than a second literal that happens to look like
    one. `test_the_hub_and_spokes_are_the_alias` proves the second half.
    """
    assert graph.SYSTEM_PROMPT is SYSTEM_PROMPT
    assert prompts.SYSTEM_PROMPT is SYSTEM_PROMPT
    assert SYSTEM_PROMPT.startswith(prompts.CORE_HUB_PROMPT.rstrip())
    for spoke in (prompts.CHART_SPOKE, prompts.MEDS_SPOKE, prompts.LABS_SPOKE,
                  prompts.RISK_SPOKE, prompts.LITERATURE_SPOKE):
        assert spoke.strip() in SYSTEM_PROMPT, (
            "the composed alias is missing a spoke, so a rule the suite asserts "
            "against SYSTEM_PROMPT may be one no turn ever receives")


def test_an_obsolete_single_turn_citation_rule_is_absent():
    """The rule retired in Phase 4, asserted as an ABSENCE.

    `prompts.py` told the model that a citation to an earlier turn resolves to nothing
    and is removed before the answer reaches the clinician. That was true before Phase 1
    and has been false since: `citations.citation_bound` builds one ledger over the
    replayed calls first and this turn second, so a marker on replayed evidence resolves
    — and `CITATIONS` says so two bullets later ("must cite it like any other evidence").
    Two rules, one section, opposite instructions, and the obsolete one was the one the
    RULE_MARKERS table protected.

    Asserted as an absence so the sentence cannot come back. If a future prompt needs to
    say something about a marker addressing an unsupported passage, it must not borrow
    these words: they describe a mechanism the service does not have.
    """
    obsolete = "only from a tool call you made in THIS turn"
    assert obsolete not in prompts.CORE_HUB_PROMPT
    assert obsolete not in SYSTEM_PROMPT


def test_the_hub_and_every_spoke_stay_inside_their_word_budget():
    """The decomposition's only hard number, and the reason it exists.

    The 9,130-word monolith was the problem. A hub that grows back past its budget is
    the same problem with a new name, so the limit is enforced rather than described —
    and a spoke past 650 is a spoke that has absorbed rules from another turn type.
    """
    assert len(prompts.CORE_HUB_PROMPT.split()) <= 1500, (
        f"the hub is {len(prompts.CORE_HUB_PROMPT.split())} words; the budget is 1500. "
        "Move a rule to the spoke whose turns use it rather than raising the limit."
    )
    for name in ("CHART_SPOKE", "MEDS_SPOKE", "LABS_SPOKE", "RISK_SPOKE",
                 "LITERATURE_SPOKE"):
        spoke = getattr(prompts, name)
        assert spoke.split(), f"{name} is empty"
        assert len(spoke.split()) <= 650, (
            f"{name} is {len(spoke.split())} words; the budget is 650")


def test_compose_system_prompt_attaches_only_the_spokes_a_turn_can_use():
    """A rule that cannot apply is not free: it costs the window every round.

    A chart turn gets the hub and the two spokes whose tools it holds, and neither the
    risk rules (no `predict_readmission`, no risk question) nor the literature rules (no
    `search_literature`). The two conditional spokes are asserted by their own trigger as
    well, because a turn CAN reach those tools without holding them at the start: a
    driver question adds `predict_readmission`, and there the risk rules must arrive.
    """
    chart = prompts.compose_system_prompt(
        allowed_tools={"rag_search", "rag_search_sections", "discharge_medications"})

    assert chart.startswith(prompts.CORE_HUB_PROMPT.rstrip())
    assert prompts.CHART_SPOKE.strip() in chart
    assert prompts.MEDS_SPOKE.strip() in chart
    assert prompts.LABS_SPOKE.strip() not in chart
    assert prompts.RISK_SPOKE.strip() not in chart
    assert prompts.LITERATURE_SPOKE.strip() not in chart

    assert prompts.RISK_SPOKE.strip() in prompts.compose_system_prompt(
        allowed_tools={"rag_search"}, risk_driver_question=True)
    assert prompts.LITERATURE_SPOKE.strip() in prompts.compose_system_prompt(
        allowed_tools={"rag_search"}, wants_literature=True)
    # And the session's own evidence is a trigger: a follow-up that replayed a panel
    # gets the laboratory rules even though this turn's tool set never lists the tool.
    assert prompts.LABS_SPOKE.strip() in prompts.compose_system_prompt(
        allowed_tools={"rag_search"}, has_labs=True)


def test_the_decline_object_the_prompt_names_is_the_one_the_graph_reads():
    """A literal in two files that only work if they agree.

    The prompt tells the model which object to reply with; `read_evidence_reply`
    decides which replies count as a decline. Let them drift and a decline is served
    as an answer — a JSON object, in front of a clinician, which is the failure this
    contract exists to prevent.
    """
    assert f'{{"{graph.EVIDENCE_FLAG_KEY}": false}}' in SYSTEM_PROMPT


def test_the_record_number_rule_has_both_halves():
    """The prompt forbids the model from writing one of our record numbers, and
    `_asked_about` strips the admission the contract appends to the question, so the
    fixed sentence cannot carry one either. One half without the other is a rule with
    a hole in it, and the hole was open: live on 2026-10-12 a free-text answer read
    "the medical records for admission 90000019 do not indicate that the patient was
    intubated during this admission".
    """
    assert "no admission id" in SYSTEM_PROMPT
    assert "90000023" not in answer_text._asked_about(
        "Was she febrile? For admission 90000023.")


def test_the_free_text_prompt_is_told_when_the_record_is_an_outpatient_visit():
    """The prompt is admission-independent; the notice is not.

    `SYSTEM_PROMPT` is one constant for every request, so the only place the free-text
    loop can learn that a record is a visit rather than a stay is the block the graph
    appends to it. Live on 2026-10-21, "Summarize the note" on 90000073 — a
    speech-therapy progress note — produced "this admission" and a hospital course.
    """
    from services.agent.routing import encounter
    from services.agent.engine.prompts import OUTPATIENT_ENCOUNTER_NOTICE

    outpatient = 90000073  # a speech-therapy progress note, verified as such below
    encounter.set_current(encounter.OUTPATIENT)
    block = graph._context_block("", outpatient)
    assert OUTPATIENT_ENCOUNTER_NOTICE in block
    assert "<encounter_notice>" in block and "</encounter_notice>" in block

    # An inpatient admission is not told anything about outpatient records: a rule
    # stated on every turn is a rule the model learns to skim.
    encounter.set_current(encounter.INPATIENT)
    assert graph._context_block("", 90000023) == ""

    # And it composes with the manifest rather than replacing it: the notice qualifies
    # the record, the manifest states the features, and a turn needs both.
    encounter.set_current(encounter.OUTPATIENT)
    composed = graph._context_block("<admission_metadata>x</admission_metadata>", outpatient)
    assert composed.index("<admission_metadata>") < composed.index("<encounter_notice>")
    assert composed.startswith("\n\n")


def test_the_summary_headings_are_conditional_on_an_inpatient_stay():
    """`**Hospital course & procedures.**` is only right for an admission.

    The heading example is what the model copies, so an unconditional
    "**Hospital course.** The patient was admitted …" is an instruction to invent a stay
    for a record that had none (90000028 through the command path, 90000073 through this
    one). The outpatient form has to be shown, not merely permitted.

    The label set is the one `EHR_FORMATTING` uses (2026-11-03): the free-text summary
    and the `/ehr` briefing name the same four sections, so a reader who moves between
    the two surfaces is not reading two different documents. The legacy discharge-only
    labels are gone, and the rule that made them conditional is what remains.
    """
    assert "**Hospital course.** The patient was admitted" not in SYSTEM_PROMPT
    assert "**Clinical course.**" in SYSTEM_PROMPT
    assert "**Presentation & diagnosis.**" in SYSTEM_PROMPT
    assert "**Hospital course & procedures.**" in SYSTEM_PROMPT
    assert "**Discharge plan & medications.**" in SYSTEM_PROMPT
    # And the discharge-only label the summary used to require is no longer taught.
    assert "**Discharge diagnosis.**" not in SYSTEM_PROMPT


def test_the_medication_authority_separates_reading_from_listing():
    """/meds owns the reconciliation; it does not own reading a regimen the note records.

    The rule this replaces forbade both, and the cost was live on 2026-10-21 (`d7e4a1cb`):
    "is the literature behind this specific combination effective?" came back refused as a
    medications question, on the false premise that the agent was not permitted to read a
    regimen its retrieved passage named. Both halves are load-bearing — a rule that permits
    reading but forgets the reconciliation invites an invented list, and a rule that forbids
    reading refuses a question the record can answer.
    """
    assert "READING IS NOT LISTING" in SYSTEM_PROMPT

    # The reconciliation is still the command's, and the prohibition names the shape it
    # takes: an enumerated list with doses and frequencies.
    assert "enumerated list" in SYSTEM_PROMPT
    assert "/meds' answer to give" in SYSTEM_PROMPT

    # The false premise is refused in the prompt itself, not merely left unstated.
    assert "not permitted" in SYSTEM_PROMPT

    # The sentence this replaced forbade reading at all ("mining names out of another
    # section"); it must not return under the same words.
    assert "mining names" not in SYSTEM_PROMPT


def test_an_empty_sectioned_search_is_not_an_empty_record():
    """90000051 answered "no information in this record" after searching four headings.

    `rag_search_sections` reads named discharge-summary sections; a note with another
    layout returns none of them. The prompt has to say that the record-wide search is
    what decides, or the model reports an absence it never checked.
    """
    assert "ZERO PASSAGES FROM IT IS NOT AN EMPTY RECORD" in SYSTEM_PROMPT
    assert "call `rag_search` for this admission" in SYSTEM_PROMPT


def test_a_risk_driver_question_is_answered_with_the_models_attribution():
    """a6e4ca01: "what explains creatinine as a readmission risk driver?" got the labs only.

    The chart says what the value did; only the model says why it counts. The estimate
    is a second, separate reason the same tool is bound — a question asking for the score
    is answered with the score, not redirected to a command — and the prompt states both.
    """
    assert "`predict_readmission` is in front of you on two kinds of question" \
        in SYSTEM_PROMPT
    # Wrapped across lines in the prompt, so the halves are asserted rather than the
    # whitespace between them — the rule is that the drivers and their direction are
    # both named, not that they are named in one line.
    assert "drivers contributed most" in SYSTEM_PROMPT
    assert "which drivers led" in SYSTEM_PROMPT
    assert "direction" in SYSTEM_PROMPT
    # The second condition: a question about the estimate is answered from the tool.
    assert "probability_display" in SYSTEM_PROMPT
    assert "Never redirect the reader to a command" in SYSTEM_PROMPT
    # Never a float: the projection gives the model the display strings and no number.
    assert "threshold_display" in SYSTEM_PROMPT


def test_the_prompt_rules_that_gate_two_showed_were_missing():
    """Three rules the full dev suite showed were absent from the prompt.

    * `dev_t5_01` named one anticoagulant where the record carried two: a class question
      must enumerate every matching drug and say which was continued.
    * `dev_t1_05` opened a `/labs` answer with "all laboratory values remained within
      their reference ranges" and then listed a creatinine that had nearly doubled. A
      blanket reassurance followed by changes is a contradiction.
    * `dev_t3_03` answered "which was worst" from a tie the `flag` cannot break. The
      answer is a comparison of the FLAGGED readings against their own reference bounds —
      the flag says which values are candidates, the bounds say how far out each one sits
      — and the ordinal the panel used to carry never enters the answer (`00c8`: the
      float form was quoted as "an excursion of 1.258" and scored as fabrication).
    """
    assert "enumerate EVERY matching" in SYSTEM_PROMPT
    assert "given inpatient and stopped or held" in SYSTEM_PROMPT
    assert "Never open a lab answer with a blanket" not in SYSTEM_PROMPT
    assert "blanket reassurance that contradicts" in SYSTEM_PROMPT
    assert "`_abnormality_rank`" not in SYSTEM_PROMPT
    assert "NEVER written into an answer" not in SYSTEM_PROMPT
    assert "cites each analyte's series" in SYSTEM_PROMPT


def test_the_prompt_pins_medication_names_sections_and_both_timepoints():
    """Three rules the full Gate 2 run showed were absent.

    * `dev_t3_01`: the answer glossed brand names with generics the record does not
      contain ("Coumadin (warfarin)"), which is a clinical claim the evidence lacks.
    * `dev_t4_02`: a fact from `brief_hospital_course` was attributed in prose to
      `discharge_instructions`, and a turn that had said the medication section was
      unreadable later produced a drug without reconciling the two.
    * `dev_t4_01`: the lab summary named an abnormal extreme but not the discharge value,
      so the reader could not see where the series ended.
    """
    assert "Never add a parenthetical generic or brand-name" in SYSTEM_PROMPT
    # `dev_t4_02` in the four-scenario retest: the synthesis wrote "proximal and distal
    # LAD" where the passage said "mid LAD and the distal LAD". The rule covers what a
    # finding is and where it is, not only what it is called.
    assert "Copy anatomical locations" in SYSTEM_PROMPT
    assert "is not" in SYSTEM_PROMPT  # the substitution examples
    assert "Paraphrasing spatial or anatomical detail" in SYSTEM_PROMPT
    # Wrapped across lines in the prompt, so the halves are asserted.
    assert "name the section the cited" in SYSTEM_PROMPT
    assert "passage actually is" in SYSTEM_PROMPT
    # `dev_t4_02` in the Gate 2 full suite: Turn 2 reported the medications section was
    # retrieved and held none, and Turn 3 called it "unavailable". The instruction that
    # asked for that word is gone; the distinction it collapsed is the rule now.
    assert "section itself was unavailable" not in SYSTEM_PROMPT
    assert 'Reserve "unavailable"' in SYSTEM_PROMPT
    assert "RETRIEVED and contained no listed items is empty" in SYSTEM_PROMPT
    assert "state both the extreme value of the admission" in SYSTEM_PROMPT

    from services.agent.routing import routing

    assert "BOTH the most extreme value" in prompts.LABS_FORMATTING
    assert "value the series ends on at discharge" in prompts.LABS_FORMATTING


def test_the_lab_command_prompt_carries_the_same_three_rules():
    """`/labs` is a formatting turn, so the rules live in its own prompt as well.

    The citation rule inverted on 2026-11-03: the prompt used to ask for a marker per
    analyte series, and now asks for none, because the trajectory table under the sentence
    already attributes every reading it states.
    """
    from services.agent.routing import routing

    assert "Never open with \"all laboratory values remained within their reference ranges\"" \
        in prompts.LABS_FORMATTING
    assert "`_abnormality_rank`" not in prompts.LABS_FORMATTING
    assert "Write no citation markers on this wire" in prompts.LABS_FORMATTING
    assert "cites each one's series" not in prompts.LABS_FORMATTING


def test_a_concern_question_is_answered_with_every_abnormal_analyte_not_a_winner():
    """Different analytes are in different units, so a ratio of excursions is not a
    comparison. The rule names all of them with their own bounds instead — the ranking
    used to be delegated to a cross-unit division the model was asked to perform."""
    from services.agent.routing import routing

    prompt = " ".join(prompts.LABS_FORMATTING.split())

    assert 'which value was "worst"' in prompt
    assert "ratio of one value's excursion to its interval width is not a clinical" in prompt
    assert "Do not pick a single winner across analytes of different kinds" in prompt


def test_a_number_is_not_distributed_across_the_items_beside_it():
    """a4b7930e: "100% accuracy" reported for three goals when the passage said two.

    A percentage stated for a list is not a percentage for every member of it, and the
    judge scores the difference as an altered value.
    """
    assert "A number belongs to the item it is stated for." in SYSTEM_PROMPT
    assert "Do not group or distribute" in SYSTEM_PROMPT

# --- Phase 01: the prompt has to permit the multi-turn behaviours ------------

def test_an_analytical_medication_question_is_answered_not_redirected():
    """The old rule refused every medication question. That is what failed Turn 2.

    "Are any high-risk or anticoagulant medications listed on discharge?" was answered
    with the count from Turn 1, because the prompt told the model a medications
    question was not its to answer. The boundary is now between the LIST (a command's
    answer) and a question ABOUT the list (the loop's).
    """
    assert "Repeating a count" in SYSTEM_PROMPT
    assert "is an anticoagulant" in SYSTEM_PROMPT


def test_a_comparative_lab_question_is_answered_in_the_first_sentence():
    assert "was most abnormal" in SYSTEM_PROMPT
    assert "not an answer to" in SYSTEM_PROMPT


def test_feature_values_are_stated_exactly_as_recorded():
    assert "18.5833" in SYSTEM_PROMPT
    assert "not 18.58" in SYSTEM_PROMPT


def test_the_manifest_outranks_a_historical_facility():
    assert "outranks the notes" in SYSTEM_PROMPT
    assert "discharge_location" in SYSTEM_PROMPT


def test_an_answer_from_the_transcript_cites_like_any_other():
    """Superseded 2026-10-23 (TF-10).

    This test used to pin the opposite rule — that transcript evidence needs no marker —
    because the citation layer resolved markers against this turn's passages only, so a
    marker on replayed evidence would have been stripped. Numbering now runs through the
    whole conversation and the source table is the same concatenation, so the exemption
    is gone and an uncited follow-up is a defect rather than a contract.
    """
    assert "without a citation marker" not in SYSTEM_PROMPT
    assert "must cite it like any other evidence" in SYSTEM_PROMPT
    assert "the replayed turns' passages come first" in SYSTEM_PROMPT


# --- Phase 04 Sprint 4.1: the prompt describes the system that now exists ----
#
# Two drifts, and each one described a mechanism the code had already removed.
#
# The `hadm_id`: Pillar 2 strips it from every model-facing schema and injects the
# session's admission after the guard has seen what the model asked for. The prompt went
# on telling the model to pass one — and, worse, to ask the user for one — so a compliant
# model wrote an argument its own tool no longer has, or turned a question it could
# answer into a request for an identifier the caller cannot supply.
#
# The `<session_evidence_state>` block: `graph._working_memory` injects it before every
# model round and the prompt never named it. The data-versus-instructions rule therefore
# covered the one channel that carries live tool results and not the one that carries the
# evidence a replayed turn retrieved, which is where a hostile note passage actually
# arrives on a follow-up.

#: What a `hadm_id` instruction looked like before Pillar 2. None of these may come back:
#: each tells the model to perform a call shape the schema does not have.
OBSOLETE_ADMISSION_ARGUMENT = (
    "that admission's hadm_id",
    "the admission's hadm_id",
    "with the hadm_id",
    "given you a hadm_id",
)

SESSION_EVIDENCE_TAGS = (
    "<session_evidence_state>",
    "<evidence_items>",
    "<section_inventory>",
    "<readmission_risk>",
    "<literature>",
)


def test_the_prompt_names_the_session_evidence_block_where_the_data_rule_lives():
    """Named, and named in DATA VS INSTRUCTIONS rather than anywhere in the file.

    The placement is the assertion. A tag mentioned in a formatting section is a
    formatting instruction; the tag mentioned beside `<tool_result>` is the one the
    data-versus-instructions rule extends to, which is the rule that decides whether an
    imperative sentence inside a note passage is reported or obeyed.

    Sliced inside the HUB since Phase 4: the data rule is hub content, and a copy of it
    in a spoke would be the second statement of one boundary.
    """
    start = prompts.CORE_HUB_PROMPT.index("DATA VS INSTRUCTIONS")
    end = prompts.CORE_HUB_PROMPT.index("EVIDENCE & ABSTENTION")
    section = prompts.CORE_HUB_PROMPT[start:end]

    for tag in SESSION_EVIDENCE_TAGS:
        assert tag in section, (
            f"the prompt never names {tag} in DATA VS INSTRUCTIONS, so the block "
            "`graph._working_memory` injects is a second data channel the rule about "
            "note text does not cover"
        )


def test_the_tool_result_delimiter_is_still_the_first_pair_the_prompt_names():
    """The delimiter test above reads the FIRST `<tag>...</tag>` in the prompt.

    Adding a second elided pair above it would silently re-point that test at a tag the
    wrapper does not emit — and it would fail confusingly, or worse, pass against the
    wrong boundary. Asserted as an ordering so the specification can grow below this line
    and not above it.
    """
    assert SYSTEM_PROMPT.index("<tool_result>...</tool_result>") < SYSTEM_PROMPT.index(
        "<session_evidence_state>"
    )


@pytest.mark.parametrize("phrase", OBSOLETE_ADMISSION_ARGUMENT)
def test_no_prompt_tells_the_model_to_pass_an_admission_id(phrase):
    """The argument is injected by the session; there is no schema field to put it in."""
    assert phrase not in SYSTEM_PROMPT, (
        f"The prompt still instructs `{phrase}`, which Pillar 2 removed from every "
        "model-facing schema: the call cannot be made that way"
    )
    assert phrase not in prompts.RETRIEVAL_DEMAND, (
        f"`RETRIEVAL_DEMAND` still instructs `{phrase}` — the forced round asks for a "
        "call the tool no longer accepts"
    )


def test_every_tool_the_prompt_tells_the_model_to_call_takes_no_admission():
    """The other half: the prompt must say WHO supplies the admission.

    Deleting the instruction without saying where the binding comes from leaves a model
    with a call it cannot complete and no account of why — and the failure that follows is
    the one the prompt used to invite, asking the user for an identifier.
    """
    assert "bound to this admission by the session" in SYSTEM_PROMPT
    assert "Never ask the user for an" in SYSTEM_PROMPT


def test_the_transcript_notice_points_at_the_block_that_holds_the_evidence():
    """`replay_messages` emits the dialogue; the passages are in the overlay.

    "The transcript above" was true when a replayed turn's tool bodies sat in the
    messages. Since Sprint 1.4 the replay carries Human/AI pairs and the record is
    injected per round, so a model sent to look "above" is being sent where the evidence
    is not — and re-asks for the tool it was told it already had.
    """
    assert "<session_evidence_state>" in prompts.TRANSCRIPT_NOTICE
    assert "The transcript above" not in prompts.TRANSCRIPT_NOTICE
    assert "{contents}" in prompts.TRANSCRIPT_NOTICE, (
        "the notice is formatted with the withdrawn tools' nouns; losing the field "
        "turns it into a sentence that names none of them"
    )

