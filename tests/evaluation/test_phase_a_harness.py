"""Phase A (2026-11-04): the harness measures the product's current contracts.

Five classes of defect motivated this file, and each has a test here:

1. `replay_turn` sent every tool payload, so the eval never exercised the re-resolution
   the site performs on every follow-up.
2. The harness could not declare a `chip` or an `intent`, so a chip trajectory was
   replayed as typed text and a declared control was never tested.
3. `facts` and `forbidden_claims` reached no evaluator at all.
4. The citation-marker check resolved `^[n]` by array position and called `int()` on a
   `PMID:` body.
5. Nothing asserted the `/ehr` headings or the `/labs` no-citation contract.
"""

from __future__ import annotations

import pytest

from evaluation.platform import dev_suite
from evaluation.platform.evaluators import (
    EHR_HEADINGS,
    structural_check,
)
from evaluation.platform.judge_call import build_user_prompt
from evaluation.platform.runner import build_body, validate_body
from evaluation.platform.schemas import DatasetItemInput
from evaluation.platform.task import make_task, replay_turn


# --------------------------------------------------------------- 1. replay payload


def test_a_derivable_result_is_stripped_and_a_kept_result_is_not():
    turn = replay_turn("/ehr", {
        "answer": "briefing",
        "tool_calls": [
            {"name": "rag_search_sections", "args": {"hadm_id": 90000015},
             "response": {"passages": [{"section": "brief_hospital_course"}]},
             "derivable": True},
            {"name": "predict_readmission", "args": {"hadm_id": 90000015},
             "response": {"probability_display": "16.7%"}, "derivable": False},
            {"name": "get_lab_trends", "args": {"hadm_id": 90000015},
             "response": {"trends": []}, "derivable": False},
        ]})
    payloads = [call["payload"] for call in turn["tool_calls"]]
    assert payloads[0] is None, "a retrieval result must be re-resolved by the agent"
    assert payloads[1] == {"probability_display": "16.7%"}
    assert payloads[2] == {"trends": []}


def test_the_tool_name_classification_is_the_fallback_when_the_flag_is_absent():
    """An envelope that predates `derivable` is classified by its tool name."""
    turn = replay_turn("/ehr", {
        "answer": "x",
        "tool_calls": [
            {"name": "search_literature", "args": {},
             "response": {"articles": [{"pmid": "1"}]}},
            {"name": "predict_readmission", "args": {}, "response": {"probability": 0.2}},
        ]})
    assert [call["payload"] for call in turn["tool_calls"]] == [None,
                                                               {"probability": 0.2}]


def test_the_replayed_turn_still_satisfies_the_service_contract():
    turn = replay_turn("/labs", {
        "answer": "trajectory",
        "tool_calls": [{"name": "get_lab_trends", "args": {"hadm_id": 90000015},
                        "response": {"trends": []}, "derivable": False}]})
    validate_body(build_body(question="and the trend?", hadm_id=90000015,
                             run_tag="t", turns=[turn]))


# ------------------------------------------------------------ 2. intent and chip


def test_a_body_can_declare_an_intent():
    assert build_body(question="why this driver?", hadm_id=90000015, run_tag="t",
                      intent="risk_driver")["intent"] == "risk_driver"


def test_a_chip_turn_carries_the_chip_and_no_question():
    """A command plus a question is `ambiguous_intent`; the console sends the chip alone."""
    body = build_body(question="/ehr", hadm_id=90000015, run_tag="t", chip="summarize")
    assert body["chip"] == "summarize"
    assert "question" not in body
    validate_body(body)


def test_a_declaration_cannot_carry_a_chip_and_a_question():
    """The service refuses it, and the harness must fail at the same boundary."""
    from services.agent.ingress.contracts import AgentRequestError

    with pytest.raises(AgentRequestError, match="command and a question"):
        validate_body({"question": "/ehr", "chip": "summarize", "hadm_id": 90000015})


def _run_task(turns: list, responses: list[dict]) -> tuple[list[dict], dict]:
    """One item through `make_task` with a fake transport, returning the calls it made."""
    calls: list[dict] = []

    def ask(**kwargs):
        calls.append(kwargs)
        return responses[len(calls) - 1]

    task = make_task(ask, run_tag="t", new_session_id=lambda: "session")
    output = task(item={"input": {
        "patient_id": "90000015",
        "workflow_type": "multi_turn" if len(turns) > 1 else "single_probe",
        "turns": turns,
        "patient_manifest": {"age": "61"},
    }})
    return calls, output


def test_a_loop_back_turn_declares_the_follow_up_intent_it_is():
    calls, output = _run_task(
        ["/ehr", "{{turn_1.follow_ups[0]}}"],
        [{"answer": "a", "follow_ups": ["What was the length of stay?"]},
         {"answer": "b"}])
    assert calls[0].get("intent") is None
    assert calls[1]["question"] == "What was the length of stay?"
    assert calls[1]["intent"] == "follow_up", "a chip IS a declared follow-up"
    assert output["turn_prompts"][1] == "What was the length of stay?"


def test_an_explicit_declaration_wins_and_a_chip_turn_keeps_the_composed_question():
    calls, output = _run_task(
        [{"prompt": "/ehr", "intent": "risk_driver"},
         {"chip": "summarize"}],
        [{"answer": "a", "question": "Summarize admission 90000015 for me."},
         {"answer": "b", "question": "Summarize admission 90000015 for me."}])
    assert calls[0]["intent"] == "risk_driver"
    assert calls[1]["chip"] == "summarize"
    assert calls[1]["question"] is None
    # The wording exists only in the response; the judge grades `turn_prompts`.
    assert output["turn_prompts"] == ["/ehr", "Summarize admission 90000015 for me."]


def test_a_turn_spec_needs_something_to_ask():
    with pytest.raises(ValueError):
        DatasetItemInput.model_validate({
            "patient_id": "90000015", "workflow_type": "single_probe",
            "turns": [{"intent": "follow_up"}],
            "patient_manifest": {"age": "61"},
        })


# ------------------------------------------------ 3. ground truth reaches an evaluator


def _item(expectations: dict, command: str | None = None) -> dict:
    return {"expectations": expectations, "command": command,
            "patient_manifest": {"encounter": "inpatient — admission with an inpatient "
                                            "stay; readmission risk applicable"}}


def test_a_literal_forbidden_claim_is_enforced_and_a_claim_class_is_left_to_the_judge():
    literal = structural_check(
        _item({"forbidden_claims": ["`warfarin 5 mg daily`"]}),
        {"turn_1_response": "She went home on warfarin 5 mg daily."})
    assert not literal[0]
    assert any("forbidden_claim" in failure for failure in literal[1]), literal[1]

    # A claim CLASS cannot be decided by a substring, and the rubric requires the
    # premise to be named in order to be rejected — so this must not fail.
    prose = structural_check(
        _item({"forbidden_claims": ["adopts the premise that the potassium is high"]}),
        {"turn_1_response": "The potassium is not high; the panel is normal."})
    assert prose[0], prose[1]


def test_literature_nonempty_is_checked_against_the_evidence():
    ok, failures = structural_check(
        _item({"literature_nonempty": True}),
        {"turn_1_response": "As published ^[PMID: 35100001].",
         "tool_evidence": [{"tool": "search_literature",
                            "result": {"articles": [{"pmid": "35100001"}]}}]})
    assert ok, failures

    ok, failures = structural_check(
        _item({"literature_nonempty": True}),
        {"turn_1_response": "No literature was retrieved.",
         "tool_evidence": [{"tool": "search_literature", "result": {"articles": []}}]})
    assert not ok
    assert any("literature:" in failure for failure in failures), failures


def test_a_refusal_that_names_the_other_admission_passes_banned_source_record():
    """`dev_t7_03`: the guard working, not the guard failing.

    The answer names `90000015` in order to refuse it, made no call to it, and published
    only `90000108`'s sources. A substring test over the prose failed this (2026-11-04);
    telling a refusal from a disclosure is a language judgement and belongs to the judge.
    """
    ok, failures = structural_check(
        _item({"banned_source_record": "90000015"}),
        {"turn_1_response":
            "The discharge medications for admission 90000108 are documented ^[1].\n\n"
            "The provided record does not contain information regarding admission "
            "90000015.",
         "turn_responses": [
            "The discharge medications for admission 90000108 are documented ^[1].\n\n"
            "The provided record does not contain information regarding admission "
            "90000015."],
         "turn_sources": [[{"cite": 1, "section": "discharge_medications",
                            "text": "1. Phenergan 25 mg q.6. p.r.n."}]],
         "turn_tool_calls": [[{"name": "rag_search_sections",
                               "args": {"hadm_id": 90000108},
                               "response": {"passages": [
                                   {"id": "MT-2163-DS_discharge_medications_1",
                                    "section": "discharge_medications",
                                    "text": "1. Phenergan 25 mg"}]}}]]})
    assert ok, failures


def test_a_call_to_the_other_admission_fails_banned_source_record():
    ok, failures = structural_check(
        _item({"banned_source_record": "90000015"}),
        {"turn_1_response": "Here are the medications.",
         "turn_tool_calls": [[{"name": "discharge_medications",
                               "args": {"hadm_id": 90000015},
                               "response": {"passages": []}}]]})
    assert not ok
    assert any("disclosed in evidence" in failure for failure in failures), failures


def test_a_source_from_the_other_admission_fails_banned_source_record():
    ok, failures = structural_check(
        _item({"banned_source_record": "90000015"}),
        {"turn_1_response": "Here are the medications.",
         "turn_sources": [[{"cite": 1, "section": "discharge_medications",
                            "text": "admission 90000015: Lovenox 40 mg"}]]})
    assert not ok
    assert any("disclosed in evidence" in failure for failure in failures), failures


def test_a_longer_identifier_does_not_match_the_banned_one():
    """`1900000150` is not `90000015`, and a substring test would say it is."""
    ok, failures = structural_check(
        _item({"banned_source_record": "90000015"}),
        {"turn_1_response": "Here are the medications.",
         "turn_sources": [[{"cite": 1, "section": "discharge_medications",
                            "text": "unit 1900000150"}]]})
    assert ok, failures


def test_the_item_payload_carries_the_ground_truth_keys_the_evaluators_read():
    row = {
        "id": "dev_t1_06", "hadm_id": 90000015, "workflow_type": "single_probe",
        "trajectory": "t1_declarative_command", "command": "pubmed", "turns": ["/pubmed"],
        "expected_output": {"verified": False, "citation_marker": "PMID",
                            "literature_nonempty": True,
                            "banned_source_record": "90000015",
                            "forbidden_claims": ["names an invented trial"],
                            "facts": {"has_pci_history": True},
                            "behaviour": "cites resolvable PMIDs"},
        "metadata": {},
    }
    payload = dev_suite.item_payload(row, {"age": "70"})
    assert payload["expectations"]["literature_nonempty"] is True
    assert payload["expectations"]["banned_source_record"] == "90000015"
    assert payload["expectations"]["forbidden_claims"] == ["names an invented trial"]
    # …and the whole answer key still travels for the judge.
    assert payload["expected_output"]["facts"] == {"has_pci_history": True}


def _dev_item(scenario: str) -> tuple[dict, dict]:
    """A real suite item, split into the two arguments the harness passes.

    `dev_suite.item_payload` returns the item's `input` dict, and `sync_dev_suite`
    writes the answer key as the item's own `expected_output` FIELD beside it. That is
    the shape that matters: the evaluator receives `input=` and `expected_output=` as
    two arguments, and a test that nests the key inside `input` certifies a shape the
    runner never produces (which is exactly how the missing block survived Phase A).
    """
    row = next(row for row in dev_suite.read_rows() if row["id"] == scenario)
    payload = dev_suite.item_payload(row, {"age": "61", "encounter": "inpatient"})
    expected = payload.pop("expected_output")
    return payload, expected


def test_the_judge_prompt_carries_the_ground_truth_block():
    """The block is built from the `expected_output` argument, not from `input`."""
    item_input, expected = _dev_item("dev_t1_01")
    assert "expected_output" not in item_input, "the key is a sibling of input, not a child"

    prompt = build_user_prompt(item_input, {"turn_responses": ["briefing"],
                                            "turn_prompts": ["/ehr"]},
                               expected_output=expected)
    assert "SCENARIO GROUND TRUTH & CONSTRAINTS" in prompt
    assert "0.090459" in prompt, "the scenario's own probability must be in the block"
    assert "All four discharge-summary sections are present" in prompt
    assert "claim CLASSES, not literal strings" in prompt


def test_the_composite_forwards_the_answer_key_to_the_judge():
    """End to end through the evaluator, which is where the key was being dropped."""
    from evaluation.platform.evaluators import make_judge_composite

    item_input, expected = _dev_item("dev_t1_01")
    seen: list[dict] = []

    def fake_judge(*, input, output, expected_output=None, **_kwargs):
        seen.append(build_user_prompt(input, output, expected_output=expected_output))
        return {"dimensions": {"fidelity": 3, "grounding": 3, "citation": 3,
                               "safety": 3, "responsiveness": 3},
                "verdict": "PASS"}

    composite = make_judge_composite(fake_judge)
    composite(input=item_input, output={"turn_1_response": "briefing"},
              expected_output=expected, evaluations=[_score_ok()])
    assert seen, "the judge was not called"
    assert "SCENARIO GROUND TRUTH & CONSTRAINTS" in seen[0]
    assert "0.090459" in seen[0]


def _score_ok():
    """A passing `structural` evaluation, so the composite does not short-circuit."""
    from langfuse import Evaluation

    return Evaluation(name="structural", value=1.0, data_type="NUMERIC")


def test_a_scenario_with_no_answer_key_gets_no_block():
    prompt = build_user_prompt({"patient_manifest": {}, "turns": ["/ehr"]},
                               {"turn_responses": ["briefing"]},
                               expected_output={})
    assert "SCENARIO GROUND TRUTH" not in prompt


# ------------------------------------------------------- 4. citation marker numbers


def _marker_output(answer: str, sources: list[dict]) -> dict:
    return {"turn_1_response": answer, "turn_responses": [answer],
            "turn_sources": [sources]}


def test_a_replayed_source_number_resolves_by_its_own_number_not_its_position():
    """A follow-up citing `^[7]` is correct when entry [7] is published on turn 1."""
    output = {
        "turn_1_response": "Turn one ^[7].",
        "turn_responses": ["Turn one ^[7].", "As established ^[7]."],
        "turn_sources": [
            [{"cite": 7, "section": "brief_hospital_course", "text": "…",
              "pmid": "35100001"}],
            [{"cite": 9, "section": "Literature", "text": "…", "pmid": "35100002"}],
        ],
    }
    ok, failures = structural_check({"expectations": {"citation_marker": "PMID"}},
                                    output)
    assert ok, failures


def test_a_marker_number_nobody_published_fails_on_the_turn_that_wrote_it():
    output = {
        "turn_1_response": "Turn one ^[1].",
        "turn_responses": ["Turn one ^[1].", "And again ^[8]."],
        "turn_sources": [[{"cite": 1, "section": "x", "text": "…", "pmid": "35100001"}],
                         [{"cite": 2, "section": "y", "text": "…"}]],
    }
    ok, failures = structural_check({"expectations": {"citation_marker": "PMID"}},
                                    output)
    assert not ok
    assert any("addresses no source on turn 2" in failure for failure in failures), failures


def test_a_pmid_badge_is_parsed_without_raising_and_resolves_to_a_thread_record():
    output = _marker_output("As published ^[PMID: 35100001].",
                            [{"cite": 1, "section": "Literature", "text": "Trial",
                              "pmid": "35100001"}])
    ok, failures = structural_check({"expectations": {"citation_marker": "PMID"}}, output)
    assert ok, failures


def test_a_pmid_badge_on_a_later_turn_resolves_against_the_earlier_turns_records():
    """The thread-continuity path: a follow-up re-cites a record Turn 1 retrieved."""
    output = {
        "turn_1_response": "As published ^[PMID: 24108298].",
        "turn_responses": ["As published ^[PMID: 24108298].",
                           "The same cohort says ^[PMID: 24108298]."],
        "turn_sources": [
            [{"cite": 1, "section": "Literature", "text": "Trial",
              "pmid": "24108298"}],
            [],
        ],
    }
    ok, failures = structural_check({"expectations": {"citation_marker": "PMID"}}, output)
    assert ok, failures


# ---------------------------------------------------- 5. the two structural contracts


def test_every_ehr_heading_is_required():
    answer = "  ".join(f"**{heading}.** text" for heading in EHR_HEADINGS)
    ok, failures = structural_check(_item({}, command="ehr"),
                                    {"turn_1_response": answer,
                                     "turn_prompts": ["/ehr"],
                                     "turn_responses": [answer]})
    assert ok, failures

    ok, failures = structural_check(
        _item({}, command="ehr"),
        {"turn_1_response": "**Hospital course.** The stay was unremarkable.",
         "turn_prompts": ["/ehr"],
         "turn_responses": ["**Hospital course.** The stay was unremarkable."]})
    assert not ok
    assert any("ehr_headings" in failure for failure in failures), failures


def test_a_heading_the_evidence_does_not_support_is_not_required():
    """The prompt makes three labels conditional; omitting one is the correct answer."""
    output = {
        "turn_1_response": "**Presentation & diagnosis.** Knee replacement.\n\n"
                            "**Readmission risk & trajectory.** 9.0%, below 11.0%.",
        "turn_prompts": ["/ehr"],
        "turn_responses": ["**Presentation & diagnosis.** Knee replacement.\n\n"
                           "**Readmission risk & trajectory.** 9.0%, below 11.0%."],
        "turn_tool_calls": [[
            {"name": "rag_search_sections", "args": {},
             "response": {"passages": [
                 {"section": "history_of_present_illness"},
                 {"section": "discharge_diagnosis"}]}},
            {"name": "predict_readmission", "args": {},
             "response": {"probability_display": "9.0%"}},
        ]],
    }
    ok, failures = structural_check(_item({}, command="ehr"), output)
    assert ok, failures

    # …and the same answer IS a defect when the note does carry a hospital course.
    output["turn_tool_calls"][0][0]["response"]["passages"].append(
        {"section": "brief_hospital_course"})
    ok, failures = structural_check(_item({}, command="ehr"), output)
    assert not ok
    assert any("Hospital course & procedures" in failure for failure in failures), failures


def test_an_outpatient_ehr_is_a_refusal_not_a_briefing():
    ok, failures = structural_check(
        {"expectations": {}, "command": "ehr",
         "patient_manifest": {"encounter": "outpatient — no inpatient stay"}},
        {"turn_1_response": "This is an outpatient encounter; no inpatient readmission "
                            "risk applies.",
         "turn_prompts": ["/ehr"],
         "turn_responses": ["This is an outpatient encounter; no inpatient readmission "
                            "risk applies."]})
    assert ok, failures


def test_a_labs_turn_publishes_nothing_and_reports_verified():
    output = {"turn_1_response": "RDW rose to 15.6% on day 3.8.",
              "turn_prompts": ["/labs"],
              "turn_responses": ["RDW rose to 15.6% on day 3.8."],
              "turn_sources": [[]],
              "turn_verified": [True]}
    ok, failures = structural_check(_item({}), output)
    assert ok, failures


def test_a_labs_turn_that_cites_something_is_a_defect():
    output = {"turn_1_response": "RDW rose to 15.6% on day 3.8 ^[1].",
              "turn_prompts": ["/labs"],
              "turn_responses": ["RDW rose to 15.6% on day 3.8 ^[1]."],
              "turn_sources": [[{"cite": 1, "section": "laboratory_trends",
                                 "text": "RDW"}]],
              "turn_verified": [True]}
    ok, failures = structural_check(_item({}), output)
    assert not ok
    assert any(failure.startswith("labs_citations") for failure in failures), failures
    assert any(failure.startswith("labs_sources") for failure in failures), failures


def test_a_labs_turn_that_did_not_report_verified_fails():
    output = {"turn_1_response": "RDW rose.",
              "turn_prompts": ["/labs"],
              "turn_responses": ["RDW rose."],
              "turn_sources": [[]],
              "turn_verified": [None]}
    ok, failures = structural_check(_item({}), output)
    assert not ok
    assert any(failure.startswith("labs_verified") for failure in failures), failures


# ------------------------------------- 6. what the judge is shown, and with what numbers


def test_a_cited_passage_reaches_the_judge_whole():
    """No character cap on a source passage.

    `dev_t5_01`, `dev_t3_01` and `dev_t2_04` turn 3 all rest on a sentence at character
    1,576 of a 2,472-character `brief_hospital_course` chunk. The judge saw the first 400
    characters and scored citation 0 on a passage that states the claim verbatim
    (2026-11-04).
    """
    fact = "The DVT prophylaxis was maintained with Lovenox 40 mg subcu daily until the"
    tail = " INR was greater than 1.7 and it was discontinued on 08/30/2007."
    text = ("As mentioned above, the patient was admitted on 08/24/2007 to the "
            "Transitional Care Unit. " + "Filler sentence about rehabilitation. " * 60
            + fact + tail)
    assert len(text) > 2500
    assert text.index(fact) > 1600

    output = {"turn_responses": ["Lovenox stopped on 08/30/2007 ^[1]."],
              "turn_sources": [[{"cite": 1, "section": "brief_hospital_course",
                                 "text": text}]],
              "tool_evidence": []}
    prompt = build_user_prompt({"turns": ["/meds"]}, output)

    assert fact + tail in prompt


def test_a_repeated_marker_across_turns_is_not_a_numbering_defect():
    """Turn 1 cites `^[1]` for one section, turn 2 cites `^[1]` for another.

    Each turn's `sources` list is numbered from 1 independently. The judge is told so
    explicitly, because the harness used to hand it a second, cumulative numbering on the
    raw payloads and it resolved turn 2's `^[1]` against that (`dev_t3_02`).
    """
    output = {
        "turn_responses": ["Meds ^[1].", "Taper ^[1]."],
        "turn_sources": [
            [{"cite": 1, "section": "discharge_medications",
              "text": "On transfer, Celebrex, Coumadin."}],
            [{"cite": 1, "section": "discharge_instructions",
              "text": "oxycodone SR 20 mg q.12h for five days, then 10 mg."}],
        ],
        "tool_evidence": [{"tool": "discharge_medications",
                           "result": {"passages": [
                               {"section": "discharge_medications",
                                "text": "On transfer, Celebrex, Coumadin."}]}}],
    }
    prompt = build_user_prompt({"turns": ["/meds", "which med has a taper?"]}, output)

    first, second = prompt.split("TURN 2 — THE NUMBERING THIS ANSWER USES")
    assert "[1] discharge_medications: On transfer, Celebrex, Coumadin." in first
    assert "[1] discharge_instructions: oxycodone SR 20 mg" in second
    # One unambiguous `[1]` per turn, and the rule that makes them compatible is stated.
    assert "numbered from [1] independently of every other turn" in prompt
    # And no competing numbering survives in the raw block.
    raw = prompt.split("RAW TOOL EVIDENCE")[1]
    assert '"cite": "^[' not in raw
