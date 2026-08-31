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

from agent import guardrail as g  # noqa: E402


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


def test_truncated_but_honest_decimal_passes():
    cleaned, flags = g.verify_risk_numbers("About 0.13.", _PREDICT_CALL)
    assert cleaned == "About 0.13."
    assert flags == []


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
    passages = [{"section": "brief_hospital_course", "text": "recovered"}]
    cleaned, flags = g.check_citations("Recovered well ^[1]. See also ^[4].", passages)
    assert "^[1]" in cleaned
    assert "^[4]" not in cleaned
    assert flags == ["citation_out_of_range:^4"]


def test_citation_with_zero_passages_is_stripped():
    cleaned, flags = g.check_citations("The notes say so ^[1].", [])
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
