"""The three repairs `00d2` asked for, at the layer each one lives in.

  * the `/meds` extraction ran out of output allowance mid-JSON on a sixteen-drug list
    (`t3_01`, `t4_01`, `t6_02`), and the truncated reply was refused — the allowance is
    now sized for a list of that length;
  * a synthesis turn wrote `^[0]` against a section it had been told was not retrieved
    (`t4_02`) — out-of-range markers are now removed where the answer is settled;
  * the harness reported `gate_pass: True` beside `structural_failures: 1` (`t6_02`) —
    the boolean now carries the criterion the report quotes.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from evaluation.platform.run_evals import gate_verdict
from services.agent.engine.deterministic import (
    parse_medications,
)
from services.agent.postprocess.answer import medication_narrative
from services.agent.postprocess.answer import sanitize_answer_markers
from services.agent.postprocess.citations import enforce_citations


def sanitize_citation_range(answer: str, *, source_total: int) -> str:
    """The settlement half of `enforce_citations`, as these tests used to call it.

    The two functions were merged into one on 2026-10-29; this shim keeps the assertions
    below written against the bound alone.
    """
    return enforce_citations(answer, source_total)[0]
from services.mcp.config import GEMINI_MAX_OUTPUT_TOKENS


# --- 1. the extraction allowance fits a sixteen-drug discharge list ------------------

#: The discharge list from `90000067` as the note writes it — sixteen drugs, three
#: passages, two of them carrying a multi-step taper. This is the record whose
#: extraction truncated at 1,729 characters on `00d2`.
_DISCHARGE_LIST = (
    "1. Home oxygen 1 to 2 liters to maintain O2 saturations at 89 to 91% at all times.\n"
    "2. Ativan 1 mg p.o. t.i.d.\n"
    "3. Metformin 1000 mg p.o. b.i.d.\n"
    "4. Glucotrol 5 mg p.o. daily.\n"
    "5. Spiriva 1 puff b.i.d.\n"
    "6. Lantus 25 units subcu q.a.m.\n"
    "7. Cardizem CD 180 mg p.o. q.a.m.\n"
    "8. Advair 250/50 mcg, 1 puff b.i.d.\n"
    "9. Iron 325 mg p.o. b.i.d.\n"
    "10. Aspirin 325 mg p.o. daily.\n"
    "11. Lipitor 10 mg p.o. bedtime.\n"
    "12. Digoxin 0.25 mg p.o. daily.\n"
    "13. Lisinopril 20 mg p.o. q.a.m.\n"
    "14. DuoNeb every 4 hours for the next several weeks, then q.6 h. thereafter.\n"
    "15. Prednisone 40 mg p.o. q.a.m. x3 days followed by 30 mg p.o. q.a.m. x3 days, "
    "then followed by 20 mg p.o. q.a.m. x5 days, then 10 mg p.o. q.a.m. x14 days, then "
    "discontinue, #30 days supply given. No refills.\n"
    "16. Clindamycin 300 mg p.o. q.i.d. x2 weeks, dispensed #64 with one refill."
)

#: The reply the model writes for that list, in the shape it writes it: one row per drug,
#: with the note's own long strings copied into `dosage` and `frequency`. Verbatim from
#: the `00d1` transaction, which fit inside the old allowance by about a hundred
#: characters — the point of the test is that the margin no longer decides the outcome.
_EXTRACTION_REPLY = """{
  "medications": [
    {"medication": "Home oxygen", "dosage": "1 to 2 liters",
     "frequency": "to maintain O2 saturations at 89 to 91% at all times"},
    {"medication": "Ativan", "dosage": "1 mg", "frequency": "t.i.d."},
    {"medication": "Metformin", "dosage": "1000 mg", "frequency": "b.i.d."},
    {"medication": "Glucotrol", "dosage": "5 mg", "frequency": "daily"},
    {"medication": "Spiriva", "dosage": "", "frequency": "1 puff b.i.d."},
    {"medication": "Lantus", "dosage": "25 units", "frequency": "subcu q.a.m."},
    {"medication": "Cardizem CD", "dosage": "180 mg", "frequency": "q.a.m."},
    {"medication": "Advair", "dosage": "250/50 mcg", "frequency": "1 puff b.i.d."},
    {"medication": "Iron", "dosage": "325 mg", "frequency": "b.i.d."},
    {"medication": "Aspirin", "dosage": "325 mg", "frequency": "daily"},
    {"medication": "Lipitor", "dosage": "10 mg", "frequency": "bedtime"},
    {"medication": "Digoxin", "dosage": "0.25 mg", "frequency": "daily"},
    {"medication": "Lisinopril", "dosage": "20 mg", "frequency": "q.a.m."},
    {"medication": "DuoNeb", "dosage": "",
     "frequency": "every 4 hours for the next several weeks, then q.6 h. thereafter"},
    {"medication": "Prednisone", "dosage": "40 mg p.o. q.a.m. x3 days followed by 30 mg",
     "frequency": "q.a.m. x14 days, then discontinue"},
    {"medication": "Clindamycin", "dosage": "300 mg", "frequency": "q.i.d. x2 weeks"}
  ]
}"""


def test_the_output_allowance_leaves_room_for_a_sixteen_drug_extraction():
    """The allowance covers thinking AND the answer, and truncation is silent.

    Measured on `00d2`: the reply for this record stopped inside its fifteenth `dosage`
    at 1,729 characters and the truncated object was refused. A ceiling below the length
    of one discharge list is a defect in the configuration, not in the model.
    """
    assert GEMINI_MAX_OUTPUT_TOKENS >= 4096


def test_a_sixteen_drug_reply_parses_to_sixteen_verified_rows():
    """The reply the model writes for this record survives verification intact."""
    verified, dropped = parse_medications(_EXTRACTION_REPLY, _DISCHARGE_LIST)

    assert dropped == []
    assert len(verified) == 16
    assert [row["medication"] for row in verified][-2:] == ["Prednisone", "Clindamycin"]

    narrative = medication_narrative(verified, returned_passages=3)
    assert narrative.startswith("16 discharge medications, as recorded in the discharge")
    assert "Aspirin" in narrative


def test_a_reply_truncated_mid_string_is_still_refused():
    """The guard stays: the fix is more room, not tolerance for a partial object.

    Accepting the prefix would serve fifteen drugs to a reader who asked what she went
    home on, and the sixteenth is not less important than the first fifteen.
    """
    truncated = _EXTRACTION_REPLY[: _EXTRACTION_REPLY.rindex('"Clindamycin"')]
    assert not truncated.rstrip().endswith("}")
    verified, dropped = parse_medications(truncated, _DISCHARGE_LIST)

    assert verified == []
    assert dropped == []
    assert medication_narrative(verified, returned_passages=3) == (
        "The discharge medications section was retrieved, but no medication could be "
        "read from it ^[1]."
    )


# --- 2. markers that address no citable unit are removed ----------------------------

def test_zero_and_out_of_range_markers_are_removed_without_a_dangling_space():
    """`00d2`, `t4_02` turn 3, verbatim: "the section is empty ^[0]." """
    answer = (
        "The record does not document the discharge medication regimen, and the "
        "discharge medications section is empty ^[0]."
    )

    assert sanitize_citation_range(answer, source_total=5) == (
        "The record does not document the discharge medication regimen, and the "
        "discharge medications section is empty."
    )


def test_a_marker_past_the_last_unit_is_removed():
    answer = "The taper started at 40 mg ^[99], then stepped down ^[2]."

    assert sanitize_citation_range(answer, source_total=3) == (
        "The taper started at 40 mg, then stepped down ^[2]."
    )


def test_a_valid_marker_is_left_exactly_as_it_was():
    answer = "The nadir was 7.2 g/dL ^[3]."

    assert sanitize_citation_range(answer, source_total=3) == answer


def test_a_partly_valid_marker_keeps_its_valid_numbers():
    """The supported half of the claim still points at its evidence.

    Kept numbers are emitted one marker each: the browser's marker regex reads a
    single number, so `^[1, 2, 3]` would draw nothing at all.
    """
    assert sanitize_citation_range("Values rose ^[1, 9].", source_total=3) == (
        "Values rose ^[1]."
    )
    assert sanitize_citation_range("Values rose ^[1-9].", source_total=3) == (
        "Values rose ^[1] ^[2] ^[3]."
    )
    assert sanitize_citation_range("Values rose ^[1, 2].", source_total=3) == (
        "Values rose ^[1] ^[2]."
    )


def test_nothing_is_citable_so_every_number_is_removed():
    assert sanitize_citation_range("No list was returned ^[1].", source_total=0) == (
        "No list was returned."
    )


def test_a_pmid_badge_must_name_a_record_the_turn_retrieved():
    """The literature badge addresses a record, and only a record this turn saw.

    Changed 2026-10-29. This test used to pin the opposite rule — that a PMID badge was
    left alone because the settlement pass knew nothing about literature records. It
    left a fabricated identifier standing until a later pass removed it, and only if that
    pass ran. The bound now carries the retrieved PMIDs, so the badge is checked where
    every other marker is checked.
    """
    answer = "Trial evidence supports this ^[PMID: 12345678]."

    kept, flags = enforce_citations(answer, 2, {"12345678"})
    assert kept == answer
    assert flags == []

    stripped, flags = enforce_citations(answer, 2)
    assert "PMID" not in stripped
    assert stripped == "Trial evidence supports this."
    # Flagged in the same `^<digits>` form as a passage number, so one filter counts every
    # unresolvable marker: a labelled badge and a bare identifier name the same record.
    assert flags == ["citation_out_of_range:^12345678"]


def test_a_section_name_marker_is_left_to_the_unknown_marker_pass():
    """Not this function's case: `renumber_citations` already drops it."""
    answer = "The course is described ^[brief_hospital_course]."

    assert sanitize_citation_range(answer, source_total=2) == answer


# --- 3. the wired settlement path, both channels ------------------------------------

def _state(answer: str, units: int, **extra) -> dict:
    passage = {"text": "text", "section": "discharge_medications"}
    return {
        "messages": [AIMessage(content=answer)],
        "tool_calls": [{"name": "rag_search_sections", "args": {},
                        "response": {"passages": [dict(passage) for _ in range(units)]}}],
        "replayed_tool_calls": [],
        **extra,
    }


def test_the_settled_prose_loses_its_out_of_range_marker():
    state = _state("The section is empty ^[0].", units=2)

    sanitize_answer_markers(state)

    assert state["messages"][-1].content == "The section is empty."


def test_a_command_answer_written_in_code_is_sanitized_too():
    """`/meds` writes its own `^[1]`; a thread with no units must not keep it."""
    state = _state("", units=0, deterministic_answer="No list was returned ^[1].")

    sanitize_answer_markers(state)

    assert state["deterministic_answer"] == "No list was returned."


def test_replayed_units_count_towards_the_range():
    """A follow-up cites the conversation's units, not only this turn's."""
    replayed = [{"name": "get_lab_trends", "args": {},
                 "response": {"trends": [{"label": "Creatinine"}]}}]
    state = _state("The value fell ^[1].", units=0, replayed_tool_calls=replayed)

    sanitize_answer_markers(state)

    assert state["messages"][-1].content == "The value fell ^[1]."


# --- 4. the harness verdict carries every criterion it reports ----------------------

def _stats(**overrides) -> dict:
    stats = {"pass_rate": 0.9333, "safety_failures": 0, "structural_failures": 0,
             "unscored": 0, "unattributed": 0}
    stats.update(overrides)
    return stats


def test_a_run_with_a_structural_failure_does_not_pass_the_gate():
    """`00d2` in one line: 14/15, no safety failure, and `gate_pass: True`."""
    assert gate_verdict(_stats()) is True
    assert gate_verdict(_stats(structural_failures=1)) is False


def test_every_other_criterion_still_blocks():
    assert gate_verdict(_stats(pass_rate=0.80)) is False
    assert gate_verdict(_stats(safety_failures=1)) is False
    assert gate_verdict(_stats(unscored=1)) is False
    assert gate_verdict(_stats(unattributed=1)) is False
