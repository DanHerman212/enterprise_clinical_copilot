"""The four root-seam fixes of 2026-11-06, one test each and a few for the edges.

Each test names the defect it was written for, because these four were found by a live
probe and every one of them would pass a test that only checked the shape of the call:
a marker that resolves to the wrong passage, an extraction failure reported as a
literature that said nothing, an identifier the model was never supposed to see, and an
absence asserted about a value nobody read. Offline — no MCP, no model, no network.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from services.agent.engine import deterministic
from services.agent.engine import mcp_client
from services.agent.engine import prompts
from services.agent.postprocess import answer
from services.agent.postprocess.evidence_state import SessionEvidenceState
from services.agent.presentation import canvas
from services.agent.routing import admission_context
from services.agent.routing import routing

HADM = 90000015


# ---------------------------------------------------------------- Fix 1: the marker
# was the literal `^[1]`, so a `/meds` turn after an `/ehr` cited position 1 no matter
# which passage the sentence reported on.


def test_the_narrative_cites_the_position_the_record_gave_the_passage():
    rows = [{"medication": "Celebrex", "dosage": "200 mg", "frequency": "daily"}]

    cited = answer.medication_narrative(
        rows, 1, sections=("discharge_medications",), cite=5)

    assert cited.endswith("^[5]."), cited
    assert "^[1]" not in cited


def test_the_narrative_writes_no_marker_without_a_position():
    rows = [{"medication": "Celebrex", "dosage": "200 mg", "frequency": "daily"}]

    uncited = answer.medication_narrative(
        rows, 1, sections=("discharge_medications",), cite=0)

    assert "^[" not in uncited


def _passage(passage_id: str, section: str, text: str) -> dict:
    return {"id": passage_id, "section": section, "text": text}


def test_a_record_resolves_a_passage_id_to_the_position_it_already_holds():
    state = SessionEvidenceState(hadm_id=HADM)
    state.ingest_tool_output("rag_search_sections", {"hadm_id": HADM}, {
        "query": "sections", "returned": 2,
        "passages": [_passage("p1", "discharge_diagnosis", "Cervical spondylosis."),
                     _passage("p2", "brief_hospital_course", "Discharged home.")],
    })

    assert state.position_of_note_passage("p2") == 2
    assert state.position_of_note_passage("never-seen") is None


def test_the_medication_position_is_read_from_the_record_not_from_the_turn():
    """The claim the defect broke: after `/ehr` numbered four sections, `/meds` must
    report the position the record gave its own passage, which is 5, not 1."""
    state = SessionEvidenceState(hadm_id=HADM)
    state.ingest_tool_output("rag_search_sections", {"hadm_id": HADM}, {
        "query": "sections", "returned": 4,
        "passages": [_passage(f"p{index}", section, "text") for index, section in
                     enumerate(("chief_complaint", "allergies", "past_medical_history",
                                "physical_exam"), start=1)],
    })
    recorded = [{
        "name": "discharge_medications", "args": {"hadm_id": HADM},
        "response": {"hadm_id": HADM, "returned": 2, "passages": [
            _passage("m1", "discharge_medications", "Celebrex 200 mg daily."),
            _passage("m2", "discharge_instructions", "Continue Celebrex."),
        ]},
    }]
    state.ingest_tool_output("discharge_medications", {"hadm_id": HADM},
                             recorded[0]["response"])
    deterministic._consolidate_medication_passages(recorded)

    assert deterministic._medication_passage_position(recorded, state) == 5


def test_no_record_means_no_marker_rather_than_a_wrong_one():
    recorded = [{
        "name": "discharge_medications", "args": {}, "response": {
            "returned": 1,
            "passages": [_passage("m1", "discharge_medications", "Celebrex 200 mg.")],
        },
    }]

    assert deterministic._medication_passage_position(recorded, None) is None


# ------------------------------------------------- Fix 2: one unparseable reply made
# the table say the literature stated no rate for the condition.

FEMORAL = deterministic.ClinicalTarget(
    condition="Hip fracture", procedure="Hemiarthroplasty",
    anchors=("hip fracture",),
)


def _abstract(pmid: str) -> dict:
    return {"pmid": pmid, "title": f"Study {pmid}", "journal": "J Test", "year": "2024",
            "abstract": "Patients were followed for 30 days after surgery."}


class _Reply:
    """A model that answers each call from a queue, so a retry can be seen."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0
        self.bound: list[dict] = []

    def bind(self, **kwargs):
        self.bound.append(kwargs)
        return self

    async def ainvoke(self, messages, config=None, **kwargs):
        self.calls += 1
        text = self._replies.pop(0) if self._replies else "no json here"
        return _Message(text)


class _Message:
    def __init__(self, content):
        self.content = content
        self.response_metadata = {"finish_reason": "STOP"}
        self.tool_calls = []


ROW = {
    "pmid": "111",
    "clinical_match": True,
    "horizon_days": 30,
    "reported_rate": "8.4%",
    "rate_sentence": "The 30-day readmission rate was 8.4%.",
    "cohort_summary": "420 patients with a hip fracture",
    "evidence_tier": "retrospective cohort",
}


def _reply_json(row: dict | None) -> str:
    return json.dumps({"comparators": [row] if row else [], "rejected": []})


def test_a_truncated_reply_is_a_degraded_pass_not_an_empty_literature():
    llm = _Reply(["no json object in this reply"])

    rows, rejected, outcome = asyncio.run(
        deterministic._extract_pass(llm, FEMORAL, [_abstract("111")], None))

    assert rows == []
    assert rejected and rejected[0]["pmid"] == ""
    assert outcome == routing.OUTCOME_EXTRACTION_DEGRADED


def test_a_failed_reply_is_retried_over_halves_and_a_half_that_reads_wins():
    abstract = dict(_abstract("111"))
    abstract["abstract"] = "The 30-day readmission rate was 8.4%."
    pool = [abstract, _abstract("222")]
    llm = _Reply(["not json", _reply_json(ROW), _reply_json(None)])

    rows, _rejected, outcome = asyncio.run(
        deterministic._extract_pass(llm, FEMORAL, pool, None))

    assert [row["pmid"] for row in rows] == ["111"]
    assert outcome == routing.OUTCOME_VERIFIED
    assert llm.calls == 3, "one whole-pool call, then one per half"


def test_the_extraction_call_carries_an_explicit_output_ceiling():
    abstract = dict(_abstract("111"))
    abstract["abstract"] = "The 30-day readmission rate was 8.4%."
    llm = _Reply([_reply_json(ROW)])

    asyncio.run(deterministic._extract_pass(llm, FEMORAL, [abstract], None))

    assert llm.bound == [{"max_output_tokens": deterministic.COMPARATOR_REPLY_TOKENS}]


def test_the_outcomes_reduce_to_one_word_for_the_turn():
    verified = deterministic.ComparatorPass(
        query="q", wide=True, candidates=1, aligned=1, verified=1,
        outcome=routing.OUTCOME_VERIFIED)
    empty = deterministic.ComparatorPass(
        query="q", wide=True, candidates=1, aligned=1, verified=0,
        outcome=routing.OUTCOME_VERIFIED_EMPTY)
    failed = deterministic.ComparatorPass(
        query="q", wide=True, candidates=1, aligned=1, verified=0,
        outcome=routing.OUTCOME_EXTRACTION_DEGRADED)

    assert deterministic.benchmark_outcome([verified, failed]) == \
        routing.OUTCOME_VERIFIED
    assert deterministic.benchmark_outcome([empty]) == routing.OUTCOME_VERIFIED_EMPTY
    assert deterministic.benchmark_outcome([empty, failed]) == \
        routing.OUTCOME_EXTRACTION_DEGRADED
    assert deterministic.benchmark_outcome([]) == routing.OUTCOME_VERIFIED_EMPTY


def test_the_table_says_which_empty_it_is_drawing():
    estimate = {"probability": 0.075, "threshold": 0.11, "decision": 0,
                "model_version": "v1"}

    degraded = canvas._comparative_matrix(estimate, [], extraction_degraded=True)
    empty = canvas._comparative_matrix(estimate, [], extraction_degraded=False)

    assert "did not complete" in degraded["baseline_note"]
    assert "no cohort has been substituted" in degraded["baseline_note"].lower()
    assert "did not complete" not in empty["baseline_note"]
    assert "No published 30-day readmission rate" in empty["baseline_note"]


# ------------------------------------------- Fix 3: the model was shown the admission
# id and wrote it back into tool arguments.

#: The tool mcp_client builds its model-facing copy from.
class _AdmittedTool:
    description = "Read the discharge medication list."
    input_schema = {
        "type": "object",
        "properties": {"hadm_id": {"type": "integer"}, "analyte": {"type": "string"}},
        "required": ["hadm_id"],
    }


def _toolbox(*names: str):
    box = mcp_client.MCPToolbox(session=None)
    box._tools = {name: _AdmittedTool() for name in names}
    return box


def test_the_model_facing_schema_never_advertises_the_admission():
    assert "hadm_id" not in json.dumps(
        mcp_client._model_schema("discharge_medications", _AdmittedTool.input_schema))


def test_an_admission_the_model_wrote_is_not_carried_into_the_call():
    box = _toolbox("discharge_medications")

    assert box.model_arguments(
        "discharge_medications", {"hadm_id": "90000069"}) == {}


def test_an_argument_the_model_was_shown_survives_the_projection():
    box = _toolbox("get_lab_trends")

    assert box.model_arguments(
        "get_lab_trends", {"hadm_id": 90000015, "analyte": "creatinine"}) == {
        "analyte": "creatinine"}


def test_a_tool_that_declares_no_admission_is_left_untouched():
    box = mcp_client.MCPToolbox(session=None)
    box._tools = {"search_literature": type("_T", (), {
        "description": "", "input_schema": {"type": "object", "properties": {
            "query": {"type": "string"}}},
    })()}

    assert box.model_arguments("search_literature", {"query": "hip fracture"}) == {
        "query": "hip fracture"}


def test_the_session_still_completes_the_call_it_completed_before():
    assert mcp_client._with_session_admission(
        "discharge_medications", {}, HADM) == {"hadm_id": HADM}
    assert mcp_client._with_session_admission(
        "search_literature", {"query": "q"}, HADM) == {"query": "q"}
    assert mcp_client._with_session_admission(
        "discharge_medications", {}, None) == {}


def test_the_manifest_block_states_clinical_facts_and_no_identifier():
    block = admission_context.render(HADM, {
        "age": 61, "gender": "Female", "admission_type": "Urgent (URGENT)",
        "discharge_location": "Home", "encounter_type": "inpatient",
    })

    assert "<admission_metadata source=" in block
    assert "hadm_id" not in block
    assert str(HADM) not in block
    assert "- age: 61" in block


# -------------------------------------- Fix 4: an absence was asserted about an analyte
# no call had asked for.

def _lab_payload(*keys: str) -> dict:
    return {
        "hadm_id": HADM, "synthetic": True,
        "trends": [
            {"key": key, "unit": "u", "reference_low": 1.0, "reference_high": 2.0,
             "series": [{"day": 0.0, "value": 1.5, "flag": "normal"}]}
            for key in keys
        ],
    }


def test_a_read_analyte_is_inventory_present():
    state = SessionEvidenceState(hadm_id=HADM)

    result = state.ingest_tool_output("get_lab_trends", {"hadm_id": HADM},
                                      _lab_payload("hemoglobin"))

    assert state.checked_labs == {"hemoglobin": "present"}
    assert result.lab_checked == ("hemoglobin: present",)


def test_an_analyte_asked_for_and_not_returned_is_inventory_absent():
    state = SessionEvidenceState(hadm_id=HADM)

    state.ingest_tool_output("get_lab_trends", {"hadm_id": HADM, "analyte": "creatinine"},
                             {"hadm_id": HADM, "trends": []})

    assert state.checked_labs == {"creatinine": "absent"}


def test_the_panel_does_not_license_an_absence_about_a_value_nobody_named():
    """A whole-panel read records what the panel held. It says nothing about a value the
    caller never asked for, and it must not be recorded as one."""
    state = SessionEvidenceState(hadm_id=HADM)

    state.ingest_tool_output("get_lab_trends", {"hadm_id": HADM},
                             _lab_payload("hemoglobin", "sodium"))

    assert set(state.checked_labs) == {"hemoglobin", "sodium"}


def test_the_inventory_is_rendered_for_the_model_and_survives_a_reload():
    state = SessionEvidenceState(hadm_id=HADM)
    state.ingest_tool_output("get_lab_trends", {"hadm_id": HADM, "analyte": "creatinine"},
                             {"hadm_id": HADM, "trends": []})

    rendered = state.render_for_prompt()
    reloaded = SessionEvidenceState.from_dict(state.to_dict())

    assert "<lab_inventory>" in rendered
    assert "creatinine: absent" in rendered
    assert reloaded.checked_labs == {"creatinine": "absent"}


def test_a_failed_read_never_downgrades_an_analyte_that_has_values():
    state = SessionEvidenceState(hadm_id=HADM)
    state.ingest_tool_output("get_lab_trends", {"hadm_id": HADM},
                             _lab_payload("hemoglobin"))

    state.ingest_tool_output("get_lab_trends", {"hadm_id": HADM, "analyte": "hemoglobin"},
                             {"hadm_id": HADM, "trends": []})

    assert state.checked_labs == {"hemoglobin": "present"}


def test_reading_one_analyte_does_not_spend_the_tool_for_the_turn():
    """The second lock the once-per-turn rule applies, refined: a filtered call leaves the
    rest of the panel unread, so the panel call is still the one worth making."""
    assert not routing.spends_turn_allowance(
        "get_lab_trends", {"hadm_id": HADM, "analyte": "hemoglobin"})
    assert routing.spends_turn_allowance("get_lab_trends", {"hadm_id": HADM})
    assert routing.spends_turn_allowance("rag_search", {"query": "q"})


def test_the_question_names_the_analytes_the_overlay_must_account_for():
    assert routing.lab_analytes_named(
        "What were her hemoglobin and creatinine trends?") == ("hemoglobin", "creatinine")
    assert routing.lab_analytes_named("Summarise the discharge course") == ()


def test_the_overlay_names_only_what_is_still_unread():
    from services.agent.engine import graph

    record = SessionEvidenceState(hadm_id=HADM)
    record.ingest_tool_output("get_lab_trends", {"hadm_id": HADM},
                              _lab_payload("hemoglobin"))
    state = {"question": "What were her hemoglobin and creatinine trends?",
             "allowed_tools": frozenset({"get_lab_trends"}),
             "evidence_state": record}

    assert graph._unqueried_analytes(state) == ["creatinine"]


def test_the_overlay_is_silent_when_the_tool_cannot_be_called():
    from services.agent.engine import graph

    state = {"question": "What were her hemoglobin and creatinine trends?",
             "allowed_tools": frozenset({"rag_search"}),
             "evidence_state": SessionEvidenceState(hadm_id=HADM)}

    assert graph._unqueried_analytes(state) == []


def test_the_laboratory_spoke_states_both_new_rules():
    lowered = prompts.LABS_SPOKE.lower()

    assert "<lab_inventory>" in prompts.LABS_SPOKE
    assert "no `analyte`" in lowered
    assert "never \"the record does not document\" a series no call requested" in lowered
    assert "<lab_inventory>" in prompts.CORE_HUB_PROMPT


def test_the_medication_spoke_states_the_fallback_provenance():
    """The rule `/meds` makes in code, stated for the free-text path: drugs read out of
    the hospital course are not a discharge list, and the inventory is what says so."""
    assert "MARKS `discharge_medications` ABSENT" in prompts.MEDS_SPOKE
    assert "hospital course documents these drugs" in prompts.MEDS_SPOKE


@pytest.mark.parametrize("budget,text", [
    (1500, prompts.CORE_HUB_PROMPT),
    (650, prompts.LABS_SPOKE),
    (650, prompts.MEDS_SPOKE),
])
def test_the_new_rules_did_not_grow_a_prompt_past_its_budget(budget, text):
    assert len(text.split()) <= budget
