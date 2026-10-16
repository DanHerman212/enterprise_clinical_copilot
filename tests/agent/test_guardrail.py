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

from services.agent import guardrail as g  # noqa: E402
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
    """
    note = [{"name": "rag_search_sections", "args": {}, "response": {
        "returned": 1, "passages": [{"id": "n", "section": "brief_hospital_course",
                                     "text": "Continue home medications."}]}}]
    answer = "Her hemoglobin dropped to a low of 11.2 g/dL, from 13.0 g/dL."

    out = g.guard_answer(answer, note)

    assert out["answer"] == answer, out
    assert not any(flag.startswith("med_") for flag in out["flags"]), out["flags"]


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


def test_citations_stay_scoped_to_the_turn_that_composed_them():
    """`^[1]` addresses a slot in THIS turn's presentation.

    The presentation is composed from this turn's calls, so a marker counted
    against a replayed turn's passages would validate a citation the caller
    cannot resolve. Replayed passages are evidence for what the answer may
    claim, never for how it numbers what it cites.
    """
    replayed = [{
        "name": "rag_search", "args": {},
        "response": {"passages": [{"section": "brief_hospital_course",
                                   "text": "Admitted with hip pain."}]},
    }]

    out = g.guard_answer("He was admitted with hip pain ^[1].", [], replayed)

    assert "^[1]" not in out["answer"]
    assert out["flags"] == ["citation_out_of_range:^1"]


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
