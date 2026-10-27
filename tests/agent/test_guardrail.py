"""Offline tests for the deterministic guardrails — no cloud credentials.

Covers the P3.2 per-med frequency verification:
  - the canonicalizer now sees "once daily"/"twice daily" (the judge's swap class)
  - a freq that contradicts the SAME med's discharge entry is dropped
  - conservative guards: single-dose chunks only, no held/stopped meds,
    no multi-med chunks, no global whitespace collapse
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.agent.postprocess import guardrail as g  # noqa: E402
from services.agent.feature_labels import display_probability  # noqa: E402


# --- risk-number guard (ECC-04) ----------------------------------------------

_PREDICT_CALL = [{
    "name": "predict_readmission",
    "args": {"hadm_id": 90000009},
    "response": {"probability": 0.131398, "threshold": 0.12, "decision": 1},
}]


def test_supported_risk_numbers_pass_untouched():
    answer = "The probability is 0.131398 (13.1%), above the 0.12 threshold."
    cleaned, flags = g.verify_risk_numbers(answer, _PREDICT_CALL)
    assert cleaned == answer
    assert flags == []


def test_fabricated_probability_is_stripped_and_flagged():
    answer = "The readmission risk is 0.14, above the 0.12 threshold."
    cleaned, flags = g.verify_risk_numbers(answer, _PREDICT_CALL)
    assert "0.14" not in cleaned
    assert "0.12" in cleaned  # the real threshold survives
    assert "risk_number_unsupported:0.14" in flags


def test_risk_number_with_no_predict_call_is_stripped():
    """The exact failure the module docstring names: a confident fabricated
    number with zero tool calls, served with 200 (ECC-04)."""
    answer = "The 30-day readmission risk is 0.14 (14%)."
    cleaned, flags = g.verify_risk_numbers(answer, [])
    assert "0.14" not in cleaned
    assert "14%" not in cleaned
    assert any(f.startswith("risk_number_unsupported:") for f in flags)


def test_errored_predict_response_supports_nothing():
    calls = [{"name": "predict_readmission",
              "response": {"error": "prediction_failed"}}]
    cleaned, flags = g.verify_risk_numbers("Risk is 0.131398.", calls)
    assert "0.131398" not in cleaned
    assert flags


# --- the comparative matrix (Phase 2, 2026-10-16) -----------------------------

# One captured record, abridged to the sentence a comparator turn quotes. The rate is
# inside the abstract, which is where `_literature_text` looks for it.
_LITERATURE_CALL = [{
    "name": "search_literature",
    "args": {"query": "congestive heart failure 30-day readmission risk"},
    "response": {
        "returned": 1,
        "articles": [{
            "pmid": "37495280",
            "title": "Global Comparison of Readmission Rates for Patients With "
                     "Heart Failure.",
            "abstract": ("Of the 1.5 million individuals at risk, 13.2% (95% CI: "
                         "10.5%-16.1%) were readmitted within 30 days."),
        }],
    },
}]


def test_a_cohort_rate_quoted_from_the_retrieved_literature_survives():
    """The sentence the matrix exists to make, kept intact.

    Without the exemption this became "sits below the pooled rate reported for …" with
    the figure cut out and a safety swap raised over a correct sentence — on every
    comparative turn, which is what made it worth a guard change rather than a note.
    """
    answer = ("Her 13.1% sits below the 13.2% pooled across 24 reports of "
              "hospitalised heart failure ^[1].")

    cleaned, flags = g.verify_risk_numbers(
        answer, _PREDICT_CALL + _LITERATURE_CALL, evidence_text="",
        literature_text=g._literature_text(_LITERATURE_CALL))

    assert "13.2%" in cleaned
    assert "13.1%" in cleaned          # the patient's own figure, from the prediction
    assert flags == []


def test_a_rate_the_literature_does_not_state_is_still_stripped():
    """The exemption is verbatim or it is nothing.

    A comparator turn is where an invented percentage is most tempting and least
    visible: the sentence around it is about published cohorts, so a number nobody
    published reads as one with a source.
    """
    answer = "Her 13.1% sits below the 9.4% pooled across 24 reports ^[1]."

    cleaned, flags = g.verify_risk_numbers(
        answer, _PREDICT_CALL + _LITERATURE_CALL, evidence_text="",
        literature_text=g._literature_text(_LITERATURE_CALL))

    assert "9.4%" not in cleaned
    assert "13.1%" in cleaned
    assert "risk_number_unsupported:9.4%" in flags


def test_a_dose_in_a_retrieved_abstract_is_not_a_dose_the_note_states():
    """The exemption is one guard wide, and this is the line it must not cross.

    Asserted through `guard_answer` rather than by calling the med guard directly,
    because the property being protected is which text reaches which guard. The note
    passage is real and simply does not mention the drug, so the med guard has evidence
    to act on — and the only way this answer survives intact is if the abstract's dose
    was allowed to verify a claim about the patient.
    """
    calls = [
        {"name": "rag_search_sections", "args": {}, "response": {
            "returned": 1, "passages": [{
                "id": "note-1", "section": "discharge_instructions",
                "text": "Continue home medications as previously prescribed.",
            }],
        }},
        {"name": "search_literature", "args": {"query": "q"}, "response": {
            "returned": 1, "articles": [{
                "pmid": "1", "title": "T",
                "abstract": "Patients received warfarin 5 mg daily after discharge.",
            }],
        }},
    ]

    result = g.guard_answer("She was discharged on warfarin 5 mg daily.", calls)

    assert "5 mg" not in result["answer"], result
    assert "med_dose_mismatch:5 mg" in result["flags"], result["flags"]


def test_truncated_but_honest_decimal_passes():
    cleaned, flags = g.verify_risk_numbers("About 0.13.", _PREDICT_CALL)
    assert cleaned == "About 0.13."
    assert flags == []


# --- the third channel: laboratory trajectories (Phase 4B, 2026-10-16) ---------

# The real shape `get_lab_trends` returns, abridged to two analytes. RDW's unit is `%`,
# which is the whole problem: a lab turn writes percentages, and before this channel
# existed every one of them was read as a claim about readmission risk.
_LAB_CALL = [{
    "name": "get_lab_trends",
    "args": {"hadm_id": 90000015},
    "response": {
        "hadm_id": 90000015, "returned": 2, "synthetic": True,
        "trends": [
            {"key": "rdw", "label": "RDW", "unit": "%",
             "reference_low": 11.5, "reference_high": 14.5, "decimals": 1,
             "anchored": ["rdw_max"],
             "series": [{"day": 0.0, "value": 12.0, "flag": "normal"},
                        {"day": 1.3, "value": 13.1, "flag": "normal"},
                        {"day": 2.7, "value": 14.1, "flag": "normal"},
                        {"day": 4.0, "value": 15.6, "flag": "high"}]},
            {"key": "creatinine", "label": "Creatinine", "unit": "mg/dL",
             "reference_low": 0.5, "reference_high": 1.1, "decimals": 2,
             "anchored": [],
             "series": [{"day": 0.0, "value": 0.87, "flag": "normal"},
                        {"day": 4.0, "value": 1.04, "flag": "normal"}]},
        ],
    },
}]


def test_an_rdw_percentage_from_a_trajectory_is_not_a_risk_number():
    """The exact sentence the directive names, and the exact corruption it found.

    RDW is reported in `%` and comes from `get_lab_trends` — neither the notes nor a
    record — so both figures were stripped as unsupported risk claims and the sentence was
    served as "The RDW increased from to, moving into the high range" under a visible
    safety swap.

    No prediction in the calls, deliberately. With one present this test passed while the
    defect was still there: `_PREDICT_CALL`'s threshold is 0.12, and `12.0%` is 0.12, so
    the number was grounded by the threshold rather than by the trajectory it came from.
    """
    answer = "The RDW increased from 12.0% to 15.6%, moving into the high range."

    cleaned, flags = g.verify_risk_numbers(
        answer, _LAB_CALL, lab_text=g._lab_text(_LAB_CALL))

    assert cleaned == answer, cleaned
    assert flags == [], flags


def test_the_reference_interval_of_a_trajectory_is_grounded_too():
    """A trajectory's own interval is drawn from the panel, so quoting it is reporting
    what was retrieved — not a claim about the patient's risk."""
    answer = "Her RDW of 15.6% is above the reference range of 11.5%-14.5%."

    cleaned, flags = g.verify_risk_numbers(
        answer, _LAB_CALL, lab_text=g._lab_text(_LAB_CALL))

    assert cleaned == answer, cleaned
    assert flags == []


def test_a_percentage_the_trajectory_does_not_carry_is_still_stripped():
    """The channel is verbatim or it is nothing, exactly like the literature one.

    Otherwise this fix would be indistinguishable from switching the guard off: a
    percentage that appears in no trajectory is still a percentage nobody produced.
    """
    answer = "The RDW increased to 17.4%, which is high."

    cleaned, flags = g.verify_risk_numbers(
        answer, _LAB_CALL, lab_text=g._lab_text(_LAB_CALL))

    assert "17.4%" not in cleaned
    assert "risk_number_unsupported:17.4%" in flags


def test_a_lab_decimal_written_bare_of_units_is_grounded():
    """The decimal branch's version of the same defect, and why the trajectory channel is
    checked there too rather than only against `%`.

    `_UNIT_TAIL` saves a lab value written with its unit; one written without it
    ("creatinine peaked at 0.87") reaches the decimal branch, which is looking for
    probabilities.
    """
    grounded, flags = g.verify_risk_numbers(
        "Creatinine peaked at 0.87 during the admission.", _LAB_CALL,
        lab_text=g._lab_text(_LAB_CALL))

    assert grounded == "Creatinine peaked at 0.87 during the admission."
    assert flags == []

    # ...and the widening did not blind the branch: a figure in no channel is still caught.
    stripped, flags = g.verify_risk_numbers(
        "Creatinine peaked at 0.83 during the admission.", _LAB_CALL,
        lab_text=g._lab_text(_LAB_CALL))

    assert "0.83" not in stripped
    assert "risk_number_unsupported:0.83" in flags


def test_guard_answer_wires_the_trajectory_channel_through():
    """End to end through the entry point the chain actually calls.

    A flag-free result is what keeps the Evidence-Check badge off, because the badge is
    raised from `guardrail_flags` — so "the prose settled without a swap" and "no flags"
    are the same statement here. A test on `verify_risk_numbers` alone would pass while
    `guard_answer` still handed the guard nothing for labs.
    """
    result = g.guard_answer(
        "The RDW increased from 12.0% to 15.6%, moving into the high range.",
        _LAB_CALL,
    )

    assert result["answer"] == (
        "The RDW increased from 12.0% to 15.6%, moving into the high range.")
    assert result["flags"] == [], result["flags"]


def test_the_med_guard_still_cannot_see_the_trajectories():
    """The channels are separate arguments, and this is the property that keeps them so.

    The same line the literature channel had to hold: `evidence_text` is the notes, handed
    to the med guard alone. Merging the third channel into it — the tempting shortcut, one
    string instead of three — would let a dose-shaped value from a generated panel verify a
    medication claim the patient's own note never states.
    """
    calls = [
        {"name": "rag_search_sections", "args": {}, "response": {
            "returned": 1, "passages": [{
                "id": "note-1", "section": "discharge_instructions",
                "text": "Continue home medications as previously prescribed.",
            }],
        }},
        {"name": "get_lab_trends", "args": {}, "response": {
            "trends": [{"key": "creatinine", "label": "Creatinine", "unit": "g",
                        "reference_low": 0.5, "reference_high": 1.1,
                        "series": [{"day": 0.0, "value": 5.0, "flag": "normal"}]}],
        }},
    ]

    result = g.guard_answer("She was discharged on warfarin 5 g daily.", calls)

    assert "5 g" not in result["answer"], result
    assert "med_dose_mismatch:5 g" in result["flags"], result["flags"]


def test_the_number_the_interface_shows_is_always_supported():
    """The display precision and the guardrail's tolerance are checked against
    each other, because they are two independent constants that have to agree.

    The prose is written from the SHOWN value — the interface rounds the score to
    two decimals, and the narrative describes the same number. If the tolerance
    were ever tighter than half a unit in that last displayed place, the guardrail
    would strip the number the widget was drawing, and the sentence would arrive
    with a hole in it. That failure is invisible in a diff and silent at runtime.
    """
    # Served values across the range the demo cohort produces, including the
    # half-way cases where rounding moves the digit.
    for served in (0.090459, 0.131398, 0.135729, 0.316788, 0.3443, 0.125, 0.115,
                   0.0049, 0.985):
        shown = display_probability(served)
        answer = f"The readmission risk score is {shown:.2f}."

        cleaned, flags = g.verify_risk_numbers(
            answer,
            [{"name": "predict_readmission",
              "response": {"probability": served, "threshold": 0.11}}],
        )

        assert cleaned == answer, f"{served} -> {shown} was rewritten"
        assert flags == [], f"{served} -> {shown} was flagged"


def test_a_display_value_that_is_not_a_number_is_not_rounded_to_zero():
    """Absence is not a score: a rounded 0.0 would be the most confident wrong
    number in the payload."""
    assert display_probability(None) is None
    assert display_probability("0.13") is None
    assert display_probability(True) is None


def test_quoted_lab_value_from_evidence_is_not_a_risk_number():
    evidence = "Labs: creatinine 0.9, INR 0.95 on discharge."
    answer = "Discharge creatinine was 0.9 ^[1]."
    cleaned, flags = g.verify_risk_numbers(answer, _PREDICT_CALL, evidence)
    assert cleaned == answer
    assert flags == []


def test_dose_decimals_and_concentrations_are_skipped():
    evidence = "1. Fluticasone 0.05% cream\n2. Digoxin 0.25 mg PO DAILY"
    answer = "Continue digoxin 0.25 mg daily and fluticasone 0.05% cream."
    cleaned, flags = g.verify_risk_numbers(answer, _PREDICT_CALL, evidence)
    assert cleaned == answer
    assert flags == []


def test_guard_answer_strips_fabricated_risk_with_no_tools():
    out = g.guard_answer("The readmission risk is 0.14.", [])
    assert "0.14" not in out["answer"]
    assert any(f.startswith("risk_number_unsupported:") for f in out["flags"])


def test_a_lab_concentration_is_not_a_dose_and_survives_the_med_guard():
    """A number with a lab denominator is a concentration, not a prescription.

    Observed live on the day the lab card shipped: "11.2 g/dL" matched the dose pattern as
    "11.2 g", was found nowhere in the retrieved notes, and was stripped — so the prose read
    "dropped to a low of /dL" and the Evidence-Check badge announced a correction to a
    sentence that was already correct.

    The other half of the same bug is the more dangerous one: that match also ran over the
    NOTE text, so a lab value entered the note's dose vocabulary and an invented dose could
    have been verified by a creatinine.

    The expectation below changed on 2026-10-17, in one direction only. This fixture's note
    says nothing about haemoglobin, so the value is genuinely ungrounded and the lab guard
    now removes the sentence — the gap that let a follow-up invent an eGFR trajectory in the
    pilot. What this test still protects is the half it was written for: a concentration must
    never be read as a dose, so no `med_*` flag may fire.
    """
    note = [{"name": "rag_search_sections", "args": {}, "response": {
        "returned": 1, "passages": [{"id": "n", "section": "brief_hospital_course",
                                     "text": "Continue home medications."}]}}]
    answer = "Her hemoglobin dropped to a low of 11.2 g/dL, from 13.0 g/dL."

    out = g.guard_answer(answer, note)

    assert not any(flag.startswith("med_") for flag in out["flags"]), out["flags"]
    assert any(flag.startswith("lab_value_unsupported") for flag in out["flags"]), (
        out["flags"]
    )


def test_a_lab_value_the_note_documents_is_served_unchanged():
    """The other side of the same guard: grounded measurements must survive it."""
    note = [{"name": "rag_search_sections", "args": {}, "response": {
        "returned": 1, "passages": [{"id": "n", "section": "brief_hospital_course",
                                     "text": "Hemoglobin fell to 11.2 g/dL from "
                                             "13.0 g/dL."}]}}]
    answer = "Her hemoglobin dropped to a low of 11.2 g/dL, from 13.0 g/dL."

    out = g.guard_answer(answer, note)

    assert out["answer"] == answer, out
    assert not any(flag.startswith("lab_") for flag in out["flags"]), out["flags"]


def test_a_weight_based_dose_is_still_a_dose():
    """The exclusion names lab denominators, so "40 mg/kg" is still checked.

    Excluding every slash would have been the one-line fix and would have opened a hole
    exactly where dosing is least forgiving.
    """
    note = [{"name": "rag_search_sections", "args": {}, "response": {
        "returned": 1, "passages": [{"id": "n", "section": "discharge_instructions",
                                     "text": "Start enoxaparin 40 mg/kg once daily."}]}}]

    out = g.guard_answer("Start enoxaparin 40 mg/kg once daily.", note)

    assert "40 mg" in out["answer"]
    assert not any(flag.startswith("med_dose_mismatch") for flag in out["flags"]), out["flags"]


# --- A follow-up is answered against the whole conversation -------------------

_REPLAYED_PREDICTION = [{
    "name": "predict_readmission", "args": {"hadm_id": 90000003},
    "response": {"probability": 0.206685, "threshold": 0.11, "prediction": 1},
}]


def test_a_replayed_prediction_supports_the_numbers_it_published():
    """The live defect (2026-09-18), reduced.

    A follow-up that repeats the stored score made no tool call of its own, so
    the risk guard saw no prediction and struck both numbers out of a correct
    answer: "the risk for this patient is, which is above the operating
    threshold." The replayed turn's prediction is the evidence for it.
    """
    answer = ("The 30-day unplanned readmission risk for this patient is 0.206685, "
              "which is above the 0.11 operating threshold.")

    alone = g.guard_answer(answer, [])
    # The doubled space is the removal's own artifact, kept as the guard really
    # produces it: the browser collapses it, so the page rendered the sentence as
    # "the risk for this patient is, which is above the operating threshold."
    assert alone["answer"] == ("The 30-day unplanned readmission risk for this "
                               "patient is, which is above the  operating threshold.")
    assert alone["flags"] == ["risk_number_unsupported:0.11",
                              "risk_number_unsupported:0.206685"]

    replayed = g.guard_answer(answer, [], _REPLAYED_PREDICTION)
    assert replayed["answer"] == answer
    assert replayed["flags"] == []


def test_a_replayed_turn_does_not_license_a_different_number():
    """The guard still has to catch what the conversation never produced."""
    out = g.guard_answer("The risk is 0.14.", [], _REPLAYED_PREDICTION)

    assert "0.14" not in out["answer"]
    assert out["flags"] == ["risk_number_unsupported:0.14"]


def test_a_note_value_quoted_from_a_replayed_passage_is_supported():
    """The med guard reads the notes the conversation retrieved, not only this
    turn's: a dose quoted from an earlier turn's discharge section is still
    quoted from the note."""
    replayed = [{
        "name": "rag_search", "args": {"hadm_id": 90000003, "query": "discharge meds"},
        "response": {"passages": [{
            "section": "discharge_medications",
            "text": "Discharge Medications:\n1. Warfarin 2.5 mg PO DAILY",
        }]},
    }]

    out = g.guard_answer("Continue warfarin 2.5 mg daily.", [], replayed)

    assert out["answer"] == "Continue warfarin 2.5 mg daily."
    assert out["flags"] == []


def test_a_marker_into_a_replayed_passage_resolves_against_the_conversation():
    """Superseded 2026-10-24 (TF-10's other half).

    This test used to pin the opposite rule — that `^[1]` could only address a slot in
    this turn's own presentation — because the renderer restarted numbering at 1 per
    tool result. Numbering now runs continuously across the replayed turns and this one
    (`graph._render_tool_result`), and `a2ui.compose_presentation` resolves against the
    same concatenation, so the guard has to count the same array or a correct marker is
    deleted. Measured live on `agent-00175-ttr`: a chip answer citing `^[6]` against six
    passages was served as `chart_synthesis_without_citations`.
    """
    replayed = [{
        "name": "rag_search", "args": {},
        "response": {"passages": [{"section": "brief_hospital_course",
                                   "text": "Admitted with hip pain."}]},
    }]

    out = g.guard_answer("He was admitted with hip pain ^[1].", [], replayed)

    assert "^[1]" in out["answer"]
    assert out["flags"] == []


def test_a_marker_past_the_conversation_is_still_stripped():
    """Widening the count must not accept a citation to nothing."""
    replayed = [{
        "name": "rag_search", "args": {},
        "response": {"passages": [{"section": "brief_hospital_course",
                                   "text": "Admitted with hip pain."}]},
    }]

    out = g.guard_answer("He was admitted with hip pain ^[2].", [], replayed)

    assert "^[2]" not in out["answer"]
    assert out["flags"] == ["citation_out_of_range:^2"]


# --- dose removal by span (ECC-11) --------------------------------------------

def test_dropping_a_dose_does_not_corrupt_a_similar_dose():
    """str.replace on the surface "5 mg" also destroyed "2.5 mg" (ECC-11);
    span removal must leave the supported dose intact."""
    evidence = "Discharge Medications:\n1. Warfarin 2.5 mg PO DAILY"
    answer = "Take warfarin 2.5 mg daily and unknowndrug 5 mg nightly."
    cleaned, flags = g.verify_med_tokens(answer, evidence)
    assert "2.5 mg" in cleaned
    assert "unknowndrug 5 mg" not in cleaned
    assert "med_dose_mismatch:5 mg" in flags


# --- citation stripping (ECC-14) ----------------------------------------------

def test_out_of_range_citation_is_stripped_and_flagged():
    cleaned, flags = g.check_citations("Recovered well ^[1]. See also ^[4].", slots=1)
    assert "^[1]" in cleaned
    assert "^[4]" not in cleaned
    assert flags == ["citation_out_of_range:^4"]


def test_citation_with_zero_slots_is_stripped():
    cleaned, flags = g.check_citations("The notes say so ^[1].", slots=0)
    assert "^[1]" not in cleaned
    assert flags == ["citation_out_of_range:^1"]


# --- invented dates (ECC-35) ----------------------------------------------------

def test_concrete_date_with_redacted_source_is_flagged():
    passages = [{"section": "discharge_instructions",
                 "text": "Follow up on ___ with Dr. ___."}]
    flags = g.flag_invented_dates("Follow up on March 5, 2024.", passages)
    assert flags == ["redacted_date_filled:March 5, 2024"]


def test_date_present_in_source_is_not_flagged():
    passages = [{"section": "discharge_instructions",
                 "text": "Surgery was 3/14/23; other dates are ___."}]
    flags = g.flag_invented_dates("Surgery was on 3/14/23.", passages)
    assert flags == []


# --- citation slots on a literature turn (the /pubmed marker defect) ------------

def _search(*articles):
    return [{"name": "search_literature",
             "response": {"returned": len(articles), "articles": list(articles)}}]


def test_a_literature_turn_offers_one_slot_per_record_that_can_be_cited():
    """Counted from the records holding a PMID, which is the set the canvas draws,
    so a slot always has something behind it to open."""
    calls = _search({"pmid": "1"}, {"pmid": "2"}, {"title": "no pmid"})

    assert g._literature_slots(calls) == 2


def test_a_turn_with_no_search_offers_no_slots():
    assert g._literature_slots([{"name": "rag_search",
                                 "response": {"passages": [{"text": "x"}]}}]) == 0
    assert g._literature_slots([]) == 0
    assert g._literature_slots(None) == 0


def test_markers_survive_on_a_literature_turn():
    """Live defect 2026-10-13: the slot count came from note passages alone, a
    literature turn reads none, so every marker the model wrote was stripped as out
    of range — a grounded, cited summary served as an uncited one, which reads as a
    model that made its claims up."""
    calls = _search({"pmid": "1"}, {"pmid": "2"}, {"pmid": "3"})

    cleaned, flags = g.check_citations(
        "ERAS did not change readmission ^[1]; day-case surgery lowered it ^[2].",
        slots=g._literature_slots(calls),
    )

    assert "^[1]" in cleaned
    assert "^[2]" in cleaned
    assert flags == []


def test_a_marker_past_the_last_record_is_still_stripped():
    """The fix widens the range; it does not remove the guard."""
    calls = _search({"pmid": "1"}, {"pmid": "2"}, {"pmid": "3"})

    cleaned, flags = g.check_citations(
        "A claim ^[4].", slots=g._literature_slots(calls)
    )

    assert "^[4]" not in cleaned
    assert flags == ["citation_out_of_range:^4"]


# --- markers written as identifiers -------------------------------------------
#
# The instruction asks for `^[1]` and forbids PMIDs in the prose. The model writes
# `^[PMID: 35186616]` anyway, because on a literature turn the marker it displays IS the
# identifier — and those markers reached a production reader as literal text, because
# neither the renderer nor this guard recognised the form (2026-10-13).

def test_a_marker_written_as_a_pmid_resolves_against_the_retrieved_records():
    calls = _search({"pmid": "35186616"}, {"pmid": "42337790"})

    cleaned, flags = g.check_citations(
        "Bilateral disease is common ^[PMID: 35186616].",
        slots=g._literature_slots(calls),
        pmids=g._literature_pmids(calls),
    )

    assert "^[PMID: 35186616]" in cleaned
    assert flags == []


def test_a_marker_naming_a_pmid_that_was_never_retrieved_is_stripped():
    """The form is accepted; the claim behind it is still checked against the records.
    An invented identifier left in the prose reads as evidence."""
    cleaned, flags = g.check_citations(
        "Supported ^[PMID: 35186616]. Also invented ^[PMID: 99999999].",
        slots=1,
        pmids={"35186616"},
    )

    assert "^[PMID: 35186616]" in cleaned
    assert "^[PMID: 99999999]" not in cleaned
    assert flags == ["citation_out_of_range:^99999999"]


def test_a_bare_identifier_resolves_the_same_as_a_labelled_one():
    cleaned, flags = g.check_citations(
        "A claim ^[35186616].", slots=1, pmids={"35186616"}
    )

    assert "^[35186616]" in cleaned
    assert flags == []


def test_a_note_turn_has_no_identifiers_to_name():
    """A note turn carries no PMIDs, so nothing but a position resolves on it — the
    identifier form is not a way past the position rule."""
    cleaned, flags = g.check_citations("The notes say so ^[PMID: 1].", slots=0)

    assert "^[PMID: 1]" not in cleaned
    assert flags == ["citation_out_of_range:^1"]


def test_the_identifiers_a_literature_turn_may_name_come_from_the_search():
    calls = _search({"pmid": "1"}, {"pmid": "2"}, {"title": "no pmid"})

    assert g._literature_pmids(calls) == frozenset({"1", "2"})
    assert g._literature_pmids([{"name": "rag_search", "response": {}}]) == frozenset()
    assert g._literature_pmids([]) == frozenset()
    assert g._literature_pmids(None) == frozenset()


def test_guard_answer_wires_the_identifiers_through():
    """The guardrail runs on the answer the model wrote, so the wiring matters as much
    as the rule: slots and identifiers have to arrive from the same search call. They
    are read from THIS turn's calls — a marker addresses a slot in this turn's
    presentation — so passing the calls as replayed history instead resolves nothing."""
    calls = _search({"pmid": "35186616"})

    out = g.guard_answer("Bilateral disease is common ^[PMID: 35186616].", calls)

    assert "^[PMID: 35186616]" in out["answer"]
    assert out["flags"] == []


def test_a_marker_resolved_against_replayed_history_is_not_a_slot_this_turn_has():
    """The same answer, with the search arriving as an earlier turn's call: the
    presentation this turn composes has no records in it, so the marker cannot
    resolve and is stripped rather than left pointing at nothing."""
    calls = _search({"pmid": "35186616"})

    out = g.guard_answer("Bilateral disease is common ^[PMID: 35186616].", [], calls)

    assert "^[PMID: 35186616]" not in out["answer"]
    assert out["flags"] == ["citation_out_of_range:^35186616"]


def test_dates_without_source_redactions_are_left_alone():
    passages = [{"section": "brief_hospital_course",
                 "text": "Admitted January 2 and recovered."}]
    assert g.flag_invented_dates("Seen on March 5.", passages) == []


# --- freq canonicalization (P3.2) -------------------------------------------

def test_canon_freqs_sees_once_daily():
    assert "DAILY" in g._canon_freqs("metoprolol tartrate 25 mg tablet once daily")


def test_canon_freqs_sees_twice_daily():
    assert "BID" in g._canon_freqs("Bupropion 150 mg sustained release twice daily")


def test_canon_freqs_sees_twice_a_day():
    assert "BID" in g._canon_freqs("docusate sodium 100 mg PO twice a day")


def test_canon_freqs_sees_qd():
    assert "DAILY" in g._canon_freqs("aspirin 81 mg PO QD")


# --- regression: compound freqs must not double-count (2026-08-19) -----------

def test_canon_freqs_twice_daily_is_bid_only():
    # "twice daily" -> BID; the bare-DAILY branch used to re-match the "daily"
    # inside it and add a spurious DAILY token.
    assert g._canon_freqs("furosemide 20 mg tablet twice daily") == {"BID"}


def test_canon_freqs_once_daily_is_daily_only():
    assert g._canon_freqs("metoprolol tartrate 25 mg tablet once daily") == {"DAILY"}


def test_canon_freqs_keeps_standalone_daily_beside_twice_daily():
    # Masking must not eat a legit standalone "daily" elsewhere in the text.
    freqs = g._canon_freqs("furosemide 20 mg tablet twice daily, "
                           "simvastatin 10 mg tablet daily")
    assert freqs == {"BID", "DAILY"}


# --- per-med swap detection (the judge's metoprolol/Bupropion class) ---------

# 27382649 discharge_medications entry.
METOPROLOL_SECTION = (
    "Discharge Medications:\n"
    "1. metoprolol tartrate 25 mg Tablet Sig: One (1) Tablet PO BID (2 times a day).  \n"
    "2. sertraline 50 mg Tablet Sig: One (1) Tablet PO DAILY (Daily). \n"
)

# 29916192 discharge_medications entry.
BUPROPION_SECTION = (
    "Discharge Medications:\n"
    "1. Aspirin 325 mg Tablet Sig: One (1) Tablet PO DAILY (Daily).\n"
    "2. Bupropion 150 mg Tablet Sustained Release Sig: Two (2) Tablet "
    "Sustained Release PO QAM (once a day (in the morning)).\n"
    "3. Citalopram 20 mg Tablet Sig: Three (3) Tablet PO at bedtime.\n"
)


def test_permed_drops_once_daily_contradicting_bid():
    answer = ("Discharge Medications.** At discharge, the patient was prescribed "
              "metoprolol tartrate 25 mg tablet once daily, sertraline 50 mg "
              "tablet daily ^[3].")
    cleaned, flags = g.verify_med_freqs_per_med(answer, METOPROLOL_SECTION)
    assert "once daily" not in cleaned
    assert "metoprolol tartrate" in cleaned
    assert any("metoprolol tartrate:DAILY" in f for f in flags)
    assert "sertraline" in cleaned  # other med untouched


def test_permed_drops_twice_daily_contradicting_qam():
    answer = ("Discharge Medications.** At discharge, the patient was prescribed "
              "Aspirin 325 mg daily, Bupropion 150 mg sustained release twice "
              "daily, and Citalopram 20 mg three tablets at bedtime ^[3].")
    cleaned, flags = g.verify_med_freqs_per_med(answer, BUPROPION_SECTION)
    assert "twice daily" not in cleaned
    assert "Bupropion 150 mg sustained release" in cleaned
    assert any("bupropion:BID" in f for f in flags)
    assert "Aspirin 325 mg daily" in cleaned  # correct med untouched


def test_permed_leaves_correct_freq_alone():
    answer = ("Discharge Medications.** At discharge, the patient was prescribed "
              "metoprolol tartrate 25 mg tablet twice a day, sertraline 50 mg "
              "tablet daily ^[3].")
    cleaned, flags = g.verify_med_freqs_per_med(answer, METOPROLOL_SECTION)
    assert cleaned == answer  # "twice a day" == BID -> no change
    assert flags == []


def test_permed_twice_daily_matching_bid_is_untouched():
    # Regression (2026-08-18 dry-run, 22247761): "twice daily" was read as
    # {BID, DAILY}; the spurious DAILY dropped a correct "daily" from a
    # PASSING answer. Now "twice daily" canonicalizes to BID only -> no change.
    section = ("Discharge Medications:\n"
               "1. furosemide 20 mg Tablet Sig: One (1) Tablet PO BID (2 times a day).\n")
    answer = ("Discharge Medications.** At discharge, the patient was prescribed "
              "furosemide 20 mg tablet twice daily, simvastatin 10 mg tablet "
              "daily ^[1].")
    cleaned, flags = g.verify_med_freqs_per_med(answer, section)
    assert cleaned == answer
    assert flags == []


# --- conservative guards -----------------------------------------------------

def test_multi_med_chunk_is_skipped():
    # "Diltiazem … and amLODIPine …" glued with 'and' — two doses -> skip.
    section = ("Discharge Medications:\n"
               "1. Diltiazem 60 mg Tablet Sig: One (1) Tablet PO TID.\n"
               "2. amLODIPine 5 mg Tablet Sig: One (1) Tablet PO DAILY.\n")
    answer = ("Discharge Medications.** the patient was prescribed Diltiazem 60 mg "
              "PO TID and amLODIPine 5 mg PO DAILY ^[2].")
    cleaned, flags = g.verify_med_freqs_per_med(answer, section)
    assert cleaned == answer
    assert flags == []


def test_held_med_chunk_is_skipped():
    # HYDROcodone is HELD, not a discharge med — its freq must not be dropped.
    section = ("Discharge Medications:\n"
               "1. Acetaminophen 650 mg Tablet Sig: One (1) Tablet PO Q6H.\n")
    answer = ("Discharge Medications**\n* Acetaminophen 650 mg PO Q6H\n"
              "* HYDROcodone-Acetaminophen (5mg-325mg) 2 TAB PO Q6H:PRN "
              "Pain - Moderate (HELD - Do not restart until talking with PCP)\n")
    cleaned, flags = g.verify_med_freqs_per_med(answer, section)
    assert "Q6H:PRN" in cleaned or "Q6H" in cleaned  # held med's PRN not stripped
    assert flags == []


def test_multiple_doses_in_chunk_skipped():
    # valacyclovir + held GlipiZIDE/olmesartan in one chunk -> multiple doses.
    section = ("Discharge Medications:\n"
               "1. ValACYclovir 1000 mg Tablet Sig: One (1) Tablet PO Q24H.\n")
    answer = ("Discharge Medications.** … and ValACYclovir 1000 mg PO Q24H ^[3]. "
              "GlipiZIDE XL 10 mg PO DAILY and olmesartan 20 mg oral DAILY were held.")
    cleaned, flags = g.verify_med_freqs_per_med(answer, section)
    assert "DAILY" in cleaned
    assert flags == []


def test_no_evidence_does_nothing():
    answer = "Discharge Medications.** Some text ^[3]."
    cleaned, flags = g.verify_med_freqs_per_med(answer, "")
    assert cleaned == answer
    assert flags == []


def test_newlines_preserved():
    # The cleanup must NOT collapse markdown structure.
    section = ("Discharge Medications:\n"
               "1. metoprolol tartrate 25 mg Tablet Sig: One (1) Tablet PO BID.\n")
    answer = ("Discharge Medications.** … once daily ^[1].\n\n**Hospital Course.** "
              "The patient was treated ^[2].")
    cleaned, _ = g.verify_med_freqs_per_med(answer, section)
    assert "\n\n**Hospital Course.**" in cleaned


def test_verify_med_tokens_keeps_paragraph_break_after_dose_drop():
    # Regression (2026-08-18 dry-run, 24592634): dropping an unverifiable dose
    # collapsed the \n\n paragraph break via re.sub(r"\s{2,}"). The cleanup
    # must collapse only intra-line space runs, never newlines.
    evidence = "cyanocobalamin 1000 mcg oral, vitamin D 800 IU oral"
    answer = ("The patient was given cyanocobalamin 2000 UNIT oral.\n\n"
              "**Discharge Diagnosis.** Pneumonia ^[1].")
    cleaned, flags = g.verify_med_tokens(answer, evidence)
    assert "\n\n**Discharge Diagnosis.**" in cleaned
    assert any("2000 UNIT" in f for f in flags)


def test_dose_with_thousands_separator_is_not_dropped():
    # Regression (2026-08-18 dry-run, 24592634): source "2,000 mcg" / "2,000
    # unit" (RX line) never matched the answer's "2000 mcg" / "2000 UNIT"
    # because the comma broke _DOSE_RE, so a CORRECT dose was dropped from a
    # PASSING answer. Normalization must equate "2,000" with "2000".
    evidence = ("5.  Cyanocobalamin ___ mcg PO DAILY\n"
                "RX *cyanocobalamin (vitamin B-12) 2,000 mcg 1 tablet(s) by mouth\n"
                "11.  Vitamin D ___ UNIT PO DAILY\n"
                "RX *ergocalciferol (vitamin D2) 2,000 unit 1 tablet(s) by mouth\n")
    answer = ("Discharge Medications.** Cyanocobalamin 2000 mcg PO DAILY, "
              "Vitamin D 2000 UNIT PO DAILY ^[1].")
    cleaned, flags = g.verify_med_tokens(answer, evidence)
    assert cleaned == answer
    assert flags == []


# --- section entry parsing ---------------------------------------------------

def test_section_entries_parse_numbered_meds():
    entries = g._section_med_entries(METOPROLOL_SECTION)
    names = [e["name"] for e in entries]
    assert "metoprolol tartrate" in names
    assert "sertraline" in names
    meta = next(e for e in entries if e["name"] == "metoprolol tartrate")
    assert meta["freqs"] == {"BID"}
    assert "25mg" in meta["doses"]


def test_match_entry_uses_dose_to_disambiguate():
    # Levetiracetam 500 mg QAM + 1000 mg HS: a 500 mg chunk must pick the QAM entry.
    section = ("Discharge Medications:\n"
               "1. LeVETiracetam 500 mg Tablet Sig: One (1) Tablet PO QAM.\n"
               "2. LeVETiracetam 1000 mg Tablet Sig: One (1) Tablet PO HS.\n")
    entries = g._section_med_entries(section)
    chunk = "LeVETiracetam 500 mg tablet in the morning"
    e = g._match_entry(chunk, g._norm_doses(chunk), entries)
    assert e is not None
    assert e["doses"] == {"500mg"}

# --- dispositions and measurements must be in the record ---------------------

def _note(text, section="brief_hospital_course"):
    return [{"name": "rag_search_sections", "args": {}, "response": {
        "returned": 1, "passages": [{"id": "n", "section": section, "text": text}]}}]


REHAB_NOTE = _note(
    "The patient remained medically stable during her hospital stay and arrangements "
    "were made for discharge to a rehabilitation program given her recent falls."
)


def test_a_disposition_the_note_does_not_say_is_removed():
    """The pilot's safety failure, reproduced as a test.

    Turn 2 of a risk conversation answered "discharged to a skilled nursing facility"
    citing the hospital-course passage, which says "discharge to a rehabilitation
    program". The citation resolved, so the citation guard passed it; the claim was
    invented, and SNF and inpatient rehab are different readmission risks.
    """
    answer = ("She was discharged to a skilled nursing facility due to her recent "
              "falls ^[1].")

    out = g.guard_answer(answer, REHAB_NOTE)

    assert "skilled nursing" not in out["answer"].lower(), out
    assert "disposition_unsupported:skilled_nursing" in out["flags"], out["flags"]


def test_a_disposition_the_note_does_say_is_served():
    answer = "Arrangements were made for discharge to a rehabilitation program ^[1]."

    out = g.guard_answer(answer, REHAB_NOTE)

    assert "rehabilitation program" in out["answer"], out
    assert not any(f.startswith("disposition_") for f in out["flags"]), out["flags"]


def test_a_denied_disposition_is_not_a_claim():
    """'She was not discharged home' asserts nothing about being discharged home.

    Stripping it would delete a correct answer for saying the opposite of the thing it
    is accused of claiming.
    """
    answer = "She was not discharged home from this admission."

    out = g.guard_answer(answer, REHAB_NOTE)

    assert "not discharged home" in out["answer"], out
    assert not any(f.startswith("disposition_") for f in out["flags"]), out["flags"]


def test_an_answer_that_is_only_an_invented_disposition_abstains():
    """Removing every claim must leave an honest sentence, not an empty answer."""
    answer = "She was discharged to a skilled nursing facility ^[1]."

    out = g.guard_answer(answer, REHAB_NOTE)

    assert out["answer"] == g.GUARDED_ABSTENTION, out
    assert "abstained_after_guard" in out["flags"], out["flags"]


def test_an_invented_lab_trajectory_is_removed():
    """fab-01: eGFR and creatinine endpoints the tool output never contained."""
    note = _note("Renal function was monitored; the note does not report the "
                 "laboratory results.")
    answer = ("Her eGFR declined from 117 to 94 mL/min/1.73m2 and creatinine rose "
              "from 0.55 to 1.06 mg/dL.")

    out = g.guard_answer(answer, note)

    assert "94" not in out["answer"], out
    assert any(f.startswith("lab_value_unsupported") for f in out["flags"]), out["flags"]


def test_a_lab_value_carried_by_a_tool_payload_is_grounded():
    """A measurement a tool returned is evidence, wherever in the payload it sits."""
    note = [{"name": "predict_readmission", "args": {"hadm_id": 1}, "response": {
        "probability": 0.24, "threshold": 0.11,
        "top_factors": [{"feature": "creatinine_max", "value": 1.06, "unit": "mg/dL"}],
    }}]
    answer = "Creatinine peaked at 1.06 mg/dL during the stay."

    out = g.guard_answer(answer, note)

    assert out["answer"] == answer, out
