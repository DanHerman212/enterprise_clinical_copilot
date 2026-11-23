"""Gate C2, fixes 1 and 2: the dose grammar the record writes, and the unit of removal.

Two defects, one shape. A guard that reads less of the note than the note writes deletes
correct prose, and a guard that edits inside a sentence leaves a fragment that reads as an
instruction nothing removed. The first is `_source_doses` — a discharge sig writes ranges
and combination strengths that `_DOSE_RE` cannot see, so "Benadryl 25-50 mg" indexed as no
dose at all and "50 mg" was judged invented. The second is the unit of redaction: the token
used to be cut out by character offset, which published `Benadryl 25 to orally`.

The third section is the schedule that belongs to one drug and is written across two. A
summary that writes "as-needed Ativan and trazodone" over a note that schedules Ativan twice
daily has given a benzodiazepine a schedule nothing states, and the per-chunk frequency pass
cannot see it: its "exactly one dose in the chunk" gate skips a pair with no dose at all.
"""

from __future__ import annotations

from services.agent.postprocess import guardrail as g

#: The canary item's own medication line, which is a RANGE, and its own answer, which is
#: the range written out in words. This pair is the defect that produced Gate C2.
_BENADRYL_NOTE = "Discharge Medications:\n3. Benadryl 25-50 mg p.o. q.i.d. for pruritus."

_DISCHARGE_LABEL = "**Discharge plan & medications.**"


# ---------------------------------------------------------------------------------------
# (a) the source index reads the grammar a discharge sig actually uses
# ---------------------------------------------------------------------------------------


def test_a_dose_range_is_indexed_at_both_endpoints():
    """"25-50 mg" states two doses, and `_DOSE_RE` alone sees neither.

    The lookbehind excludes a hyphen before the number and the lookahead excludes one after
    the unit, so a range indexed as the empty set — which made an answer that wrote "50 mg"
    look like an invention about a note that had just stated it.
    """
    assert g._source_doses("Benadryl 25-50 mg p.o. q.i.d.") == {"25mg", "50mg"}


def test_every_dash_a_note_might_use_indexes_the_same_way():
    assert g._source_doses("Benadryl 25-50 mg") == {"25mg", "50mg"}
    assert g._source_doses("Benadryl 25\u201350 mg") == {"25mg", "50mg"}
    assert g._source_doses("Benadryl 25 to 50 mg") == {"25mg", "50mg"}


def test_a_combination_tablet_is_indexed_at_both_strengths():
    """"5/325 mg" is two active ingredients, and an answer may name either one."""
    assert g._source_doses("Percocet 5/325 mg q.6h p.r.n.") == {"5mg", "325mg"}
    assert g._source_doses("amoxicillin/clavulanate 325/10 mg") == {"325mg", "10mg"}


def test_a_weight_based_dose_is_still_one_dose_and_not_a_combination():
    """"40 mg/kg" is dosing over a weight; it is not 40 mg plus kg.

    The exclusion the plain dose pattern already carried is applied to the combination
    rule here, so widening the grammar did not turn a weight denominator into a strength.
    """
    assert g._source_doses("Start enoxaparin 40 mg/kg once daily") == {"40mg"}


def test_cc_and_ml_are_the_same_volume():
    """The note writes `cc` and the answer writes `mL`, and they must verify each other."""
    assert g._source_doses("flush with 15 cc of saline") == {"15ml"}
    assert g._source_doses("flush with 15 mL of saline") == {"15ml"}
    assert g._norm_doses("flush with 15 cc of saline") == {"15ml"}


def test_the_canary_answer_survives_the_med_guard_untouched():
    """The whole point: a dose the note states as a range is a dose the note states.

    Before this, `verify_med_tokens` dropped the token and published
    "Benadryl 25 to orally four times daily" — a fragment whose surviving words are part of
    the dose that was removed (`dev_d1_slash_01`, 2026-11-22).
    """
    answer = (f"{_DISCHARGE_LABEL} Benadryl 25 to 50 mg orally four times daily "
              "for pruritus.")

    cleaned, flags = g.verify_med_tokens(answer, _BENADRYL_NOTE)

    assert cleaned == answer
    assert flags == []


# ---------------------------------------------------------------------------------------
# (b) the unit of redaction is the sentence, and a heading survives it
# ---------------------------------------------------------------------------------------


def test_an_unsupported_dose_removes_its_sentence_and_leaves_no_fragment():
    """No "25 to", no doubled space, no half a dose left behind.

    A reader cannot tell a redaction from a typo, so the sentence goes whole — the rule the
    laboratory guard already applies to a measurement taken out of "rose from 0.55 to 1.06".
    """
    evidence = "Discharge Medications:\n1. warfarin 2.5 mg PO DAILY"
    answer = (f"{_DISCHARGE_LABEL} The patient takes warfarin 9 mg nightly. "
              "He was discharged to home.")

    cleaned, flags = g.verify_med_tokens(answer, evidence)

    assert cleaned == f"{_DISCHARGE_LABEL} He was discharged to home."
    assert flags == ["med_dose_mismatch:9 mg"]


def test_the_section_label_re_attaches_to_the_sentence_that_survives():
    """The label opened the paragraph; the sentence it opened was the one removed.

    The heading belongs to the section, not to the sentence, so it moves to the first
    sentence under it that survives — the rule `verify_dispositions` states, and the reason
    `/ehr` is never served three headings out of four.
    """
    evidence = "Discharge Medications:\n1. warfarin 2.5 mg PO DAILY"
    answer = (f"{_DISCHARGE_LABEL} He takes warfarin 9 mg nightly. "
              "He was discharged to home.\n\n"
              "**Readmission risk & trajectory.** His estimate is above the threshold.")

    cleaned, _ = g.verify_med_tokens(answer, evidence)

    assert cleaned.startswith(f"{_DISCHARGE_LABEL} He was discharged to home.")
    assert "**Readmission risk & trajectory.**" in cleaned
    assert "9 mg" not in cleaned


def test_a_section_that_loses_every_sentence_loses_its_heading():
    evidence = "Discharge Medications:\n1. warfarin 2.5 mg PO DAILY"
    answer = f"{_DISCHARGE_LABEL} He takes warfarin 9 mg nightly."

    cleaned, flags = g.verify_med_tokens(answer, evidence)

    # Nothing left under the label, so the label goes rather than promising a gap — and the
    # empty answer is what `guard_answer` turns into the turn's abstention.
    assert cleaned == ""
    assert "med_dose_mismatch:9 mg" in flags


def test_a_supported_dose_elsewhere_in_the_answer_is_never_touched():
    """The pass only ever rebuilds text a sentence was actually removed from."""
    evidence = "Discharge Medications:\n1. warfarin 2.5 mg PO DAILY"
    answer = ("She takes warfarin 2.5 mg daily. **Discharge instructions.** "
              "Follow up in two weeks.")

    cleaned, flags = g.verify_med_tokens(answer, evidence)

    assert cleaned == answer
    assert flags == []


def test_a_frequency_defect_is_flagged_without_editing_the_prose():
    """A wrong frequency is lower-risk than a wrong dose, and stays a flag."""
    cleaned, flags = g.verify_med_tokens(
        "She leaves on Ativan 0.5 mg three times daily.",
        "Ativan 0.5 mg p.o. b.i.d.")

    assert cleaned == "She leaves on Ativan 0.5 mg three times daily."
    assert flags == ["med_freq_mismatch:TID"]


# ---------------------------------------------------------------------------------------
# (c) a schedule stated across two drugs is checked against each drug's own entry
# ---------------------------------------------------------------------------------------

#: Ativan scheduled twice daily and trazodone at bedtime — the note `dev_d3_fup_01` has, in
#: the shape `_section_med_entries` parses.
_SCHEDULED_PAIR = (
    "Discharge Medications:\n"
    "1. Ativan 0.25 mg Tablet Sig: One (1) Tablet PO BID.\n"
    "2. trazodone 50 mg Tablet Sig: One (1) Tablet PO QHS.\n"
)


def test_as_needed_stated_across_two_drugs_is_checked_against_both():
    """"as-needed Ativan and trazodone" over a note that schedules Ativan BID.

    Neither drug's entry says as-needed, and the reader titrates a benzodiazepine on the
    strength of the sentence. Both are named in the flag, because the sentence made the
    claim about both.
    """
    answer = (f"{_DISCHARGE_LABEL} She was prescribed short-term as-needed Ativan and "
              "trazodone. She was discharged to home.")

    cleaned, flags = g.verify_med_freqs_per_med(answer, _SCHEDULED_PAIR)

    assert "as-needed" not in cleaned
    assert cleaned == f"{_DISCHARGE_LABEL} She was discharged to home."
    assert "med_freq_permed_mismatch:ativan:PRN" in flags
    assert "med_freq_permed_mismatch:trazodone:PRN" in flags


def test_a_pair_whose_entries_both_state_the_schedule_is_left_alone():
    """The check asks whether the claim is true, not whether a pair was written."""
    both_prn = (
        "Discharge Medications:\n"
        "1. Ativan 0.5 mg Tablet Sig: One (1) Tablet PO Q6H PRN.\n"
        "2. trazodone 50 mg Tablet Sig: One (1) Tablet PO QHS PRN.\n"
    )
    answer = f"{_DISCHARGE_LABEL} She was prescribed as-needed Ativan and trazodone."

    cleaned, flags = g.verify_med_freqs_per_med(answer, both_prn)

    assert cleaned == answer
    assert flags == []


def test_two_drugs_in_separate_clauses_are_not_read_as_one_schedule():
    """Only a conjoined noun phrase carries a schedule across both names.

    "aspirin was continued and metoprolol was increased to twice daily" states one
    schedule, for metoprolol. Dropping it would delete a correct sentence, so the pass
    requires nothing but a joiner between the two names before it reads the schedule as
    theirs.
    """
    section = (
        "Discharge Medications:\n"
        "1. aspirin 81 mg Tablet Sig: One (1) Tablet PO DAILY.\n"
        "2. metoprolol tartrate 25 mg Tablet Sig: One (1) Tablet PO BID.\n"
    )
    answer = (f"{_DISCHARGE_LABEL} Aspirin 81 mg was continued and metoprolol tartrate "
              "was increased to twice daily.")

    cleaned, flags = g.verify_med_freqs_per_med(answer, section)

    assert cleaned == answer
    assert flags == []


def test_the_discharge_window_ends_at_the_next_section_heading():
    """A schedule stated in the hospital course is not a discharge prescription.

    `/ehr` labels its paragraph `**Discharge plan & medications.**` and the window used to
    be the whole briefing, so the two sections' frequencies were checked against each
    other (`dev_d3_fup_01`, 2026-11-22).
    """
    session = (
        "Discharge Medications:\n"
        "1. valsartan 80 mg Tablet Sig: One (1) Tablet PO DAILY.\n"
    )
    answer = (
        "**Hospital course & procedures.** She was given hydralazine 10 mg three times "
        "daily on the ward.\n\n"
        f"{_DISCHARGE_LABEL} She was prescribed valsartan 80 mg daily. "
        "She was discharged to home.\n\n"
        "**Readmission risk & trajectory.** Her estimate is above the threshold."
    )

    base, region = g._med_region(answer)

    assert answer[:base].endswith(_DISCHARGE_LABEL)
    assert "hydralazine" not in region
    assert "Readmission risk" not in region
    assert "valsartan" in region
