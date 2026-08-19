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
