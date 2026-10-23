"""The follow-up chips: a deterministic selector over verified chart signals.

Replaces the model-based suite that stood here until Phase 05 §2. The questions are no
longer proposed by a model, so what is worth testing changed: not "did we parse the
model's JSON" but "does each template fire only when the record carries the fact it
asks about, and is every chip answerable by the Turn-2 loop".

Three things are asserted over and over, because they are the contract:

* a slot is filled from the payload or the template does not fire;
* no patient-chart chip trips `routing.asks_for_literature`, which is the predicate that
  would bind PubMed on the click; and
* the risk chip satisfies `routing.asks_why_risk_driver`, which is what unlocks
  `predict_readmission`.

All fixtures are synthetic. No Dev or Holdout record is referenced by id.
"""

from __future__ import annotations

import pytest

from services.agent import followups, routing

ANSWER = "Discharged home in stable condition."


# --------------------------------------------------------------------------- fixtures


def _med_state(text: str, *, rows: list[str] | None = None,
               returned: int | None = None) -> dict:
    payload = {
        "tool_calls": [{
            "name": "discharge_medications",
            "response": {"returned": 1 if text else 0,
                         "passages": [{"section": "discharge_medications",
                                       "text": text}] if text else []},
        }],
        "command": "meds",
    }
    if returned is not None:
        payload["command_payload"] = {
            "medications": [{"medication": name, "dose": "1 mg", "frequency": "daily"}
                            for name in (rows or [])],
            "dropped": [],
            "returned_passages": returned,
        }
    return payload


def _note_state(sections: dict[str, str]) -> dict:
    return {
        "tool_calls": [{
            "name": "rag_search_sections",
            "response": {"returned": len(sections),
                         "passages": [{"section": name, "text": text}
                                      for name, text in sections.items()]},
        }],
        "command": "summarize",
    }


def _lab_state(*trends: dict) -> dict:
    return {
        "tool_calls": [{"name": "get_lab_trends",
                        "response": {"returned": len(trends),
                                     "trends": list(trends)}}],
        "command": "labs",
    }


def _trend(label: str, value: float, low: float, high: float, *,
           unit: str = "mg/dL", decimals: int = 1) -> dict:
    flag = "normal"
    if value > high:
        flag = "high"
    elif value < low:
        flag = "low"
    return {"label": label, "unit": unit, "reference_low": low,
            "reference_high": high, "decimals": decimals,
            "series": [{"day": 0, "value": low, "flag": "normal"},
                       {"day": 3, "value": value, "flag": flag}]}


def _risk_state(*, factors: list[dict] | None = None,
                decision: str = "above") -> dict:
    return {
        "tool_calls": [{
            "name": "predict_readmission",
            "response": {"probability": 0.2, "threshold": 0.11,
                         "decision": decision,
                         "probability_display": "20.0%",
                         "threshold_display": "11.0%",
                         "top_factors": factors if factors is not None else [
                             {"feature": "oncology_flag", "contribution": 0.31,
                              "direction": "increases",
                              "value": 1}]},
        }],
        "command": "risk",
    }


def _chips(state: dict, *, scope: str = "patient_chart", outpatient: bool = False,
           question: str = "/ehr") -> list[str]:
    return followups.suggest(question, ANSWER, scope=scope, state=state,
                             outpatient=outpatient)


def _assert_routable(chips: list[str]) -> None:
    for chip in chips:
        assert len(chip) <= followups._MAX_QUESTION_CHARS, chip
        assert not routing.asks_for_literature(chip), chip


# ------------------------------------------------------------------- contract basics


def test_an_empty_answer_offers_nothing():
    """`/risk` answers in a widget and carries no prose, so it has no chip row."""
    assert followups.suggest("/risk", "", scope="patient_chart",
                             state=_risk_state()) == []
    assert followups.suggest("/risk", "   ", state=_risk_state()) == []


def test_no_state_offers_nothing():
    """Without a chart there is nothing to ask about; a generic chip would be the
    primitive row this replaced."""
    assert followups.suggest("/ehr", ANSWER, state=None) == []


def test_a_turn_offers_at_most_three():
    state = _med_state(
        "Oxycodone SR 20 mg for five days, then decrease to oxycodone SR 10 mg. "
        "Continue Coumadin 7 mg daily, INR every Monday and Thursday. "
        "No driving for two days. Follow up with Dr. A on 09/21/2007.",
        rows=["Oxycodone SR", "Coumadin"], returned=1)
    assert len(_chips(state)) == followups.FOLLOW_UP_COUNT


def test_every_chip_is_within_the_cap_and_routable():
    state = _med_state(
        "Continue Coumadin 7 mg daily. Follow up with Dr. A. "
        "No driving for two days.", rows=["Coumadin"], returned=1)
    _assert_routable(_chips(state))


# -------------------------------------------------------------------------- P1 taper


def test_p1_a_taper_names_the_drug_the_cue_belongs_to():
    state = _med_state(
        "Oxycodone SR 20 mg p.o. q.12h. for five days, then decrease to "
        "oxycodone SR 10 mg p.o. q.12h. for five days.", returned=1)
    chips = _chips(state)

    assert chips[0] == ("What is the dosing and taper schedule for oxycodone SR "
                        "at discharge?")


def test_p1_does_not_fire_without_a_taper_cue():
    state = _med_state("Celebrex 200 mg daily. Colace 100 mg b.i.d.", returned=1)
    assert not any("taper" in chip.lower() for chip in _chips(state))


def test_p1_is_dropped_when_the_drug_cannot_be_resolved():
    """The cue alone is not enough: a chip with an empty slot asks about nothing."""
    state = _med_state("The dose will be tapered over the coming weeks.", returned=1)
    assert not any("taper" in chip.lower() for chip in _chips(state))


# --------------------------------------------------------------------- P2 anticoag


def test_p2_names_the_anticoagulant_as_the_note_writes_it():
    state = _med_state("Continue Coumadin 7 mg daily for 30 days.", returned=1)
    chips = _chips(state)

    assert "Coumadin" in chips[0]
    assert "monitoring" not in chips[0], "no monitoring is documented in this record"


def test_p2_adds_the_monitoring_half_only_when_monitoring_is_documented():
    state = _med_state(
        "Continue Coumadin 7 mg daily. INR every Monday and Thursday.", returned=1)
    assert "monitoring" in _chips(state)[0]


def test_p2_recognises_brand_names_not_only_generics():
    """The false negative the class table exists to prevent: this note says Lovenox."""
    state = _med_state("Lovenox 40 mg subcutaneous daily.", returned=1)
    assert any("Lovenox" in chip for chip in _chips(state))


# ------------------------------------------------------------------- P3 precautions


def test_p3_fires_on_a_documented_restriction():
    state = _note_state({"discharge_instructions":
                         "No driving for two days; do not lift more than 10 pounds."})
    assert followups._CARE_RED_FLAGS in _chips(state)


def test_p3_does_not_fire_on_a_note_without_one():
    state = _note_state({"discharge_diagnosis": "Status post knee replacement."})
    assert followups._CARE_RED_FLAGS not in _chips(state)


def test_p3_fires_on_a_return_precaution_cue():
    state = _note_state({"discharge_instructions":
                         "Return for repeat ERCP if there are signs of stent occlusion."})
    assert followups._CARE_RED_FLAGS in _chips(state)


# ---------------------------------------------------------------------- P4 follow-up


def test_p4_fires_on_a_documented_appointment():
    state = _note_state({"discharge_instructions":
                         "Follow-up appointment with Dr. A on 09/21/2007 at noon."})
    assert followups._CARE_FOLLOW_UP in _chips(state)


def test_p4_fires_on_a_clinic_without_a_named_provider():
    state = _note_state({"discharge_instructions": "Follow up with Urology in clinic."})
    assert followups._CARE_FOLLOW_UP in _chips(state)


# ----------------------------------------------------------------------- P5 regimen


def test_p5_names_two_verified_drugs_when_no_taper_or_anticoagulant_chip_ran():
    state = _med_state("Celebrex 200 mg daily. Colace 100 mg b.i.d. Synthroid 137 mcg.",
                       rows=["Celebrex", "Colace", "Synthroid"], returned=1)
    assert any(chip.startswith("What are the discharge dosing instructions for")
               for chip in _chips(state))


def test_p5_is_suppressed_when_the_taper_and_anticoagulant_chips_both_ran():
    """Two questions on one row must not be repeated by a third."""
    state = _med_state(
        "Coumadin 7 mg daily. Oxycodone SR 20 mg for five days, then decrease to "
        "oxycodone SR 10 mg.",
        rows=["Coumadin", "Oxycodone SR"], returned=1)
    chips = _chips(state)

    assert len(chips) == 2, chips
    assert not any(chip.startswith("What are the discharge dosing instructions")
                   for chip in chips)


# ------------------------------------------------------------------------- P6 driver


def test_p6_asks_why_the_top_increasing_driver_raises_the_risk():
    state = _risk_state()
    chips = _chips(state)

    # The label is the curated feature label, not the raw column name: a chip that says
    # "oncology_flag" is a developer string in a clinician's row.
    assert chips[0] == ("Why does oncology history increase this patient's readmission "
                        "risk driver?")
    assert routing.asks_why_risk_driver(chips[0]), (
        "the chip must unlock predict_readmission on the click")


def test_p6_never_asks_for_the_number():
    for chip in _chips(_risk_state()):
        assert "%" not in chip
        assert not any(char.isdigit() for char in chip)


def test_p6_does_not_fire_without_an_increasing_driver():
    state = _risk_state(factors=[{"feature": "monocytes_min", "contribution": -0.5,
                                  "direction": "decreases", "value": 0.2}])
    assert not any("readmission risk driver" in chip for chip in _chips(state))


# --------------------------------------------------------------------------- P7/P8


def test_p7_ranks_by_distance_outside_the_reference_range_not_by_raw_value():
    """A wide reference interval must not outrank a narrower one because its number
    is larger: the ranking is in range-widths."""
    wide = _trend("Sodium", 150.0, 130.0, 148.0, unit="mmol/L", decimals=0)
    narrow = _trend("Potassium", 6.0, 3.5, 5.0, unit="mmol/L", decimals=1)
    chips = _chips(_lab_state(wide, narrow))

    assert "Potassium" in chips[0]
    assert "(6.0 mmol/L)" in chips[0]


def test_p7_states_the_value_at_the_precision_the_tool_reports():
    state = _lab_state(_trend("Red blood cells", 2.69, 4.2, 5.4,
                              unit="x10^6/uL", decimals=2))
    assert "(2.69 x10^6/uL)" in _chips(state)[0]


def test_p8_lists_two_values_when_every_analyte_is_normal():
    state = _lab_state(_trend("Sodium", 140.0, 135.0, 145.0),
                       _trend("Potassium", 4.1, 3.5, 5.0))
    chips = _chips(state)

    assert chips[0].startswith("What were the discharge values for Sodium and")
    assert "Potassium" in chips[0]


def test_p8_needs_two_analytes():
    state = _lab_state(_trend("Sodium", 140.0, 135.0, 145.0))
    assert not any(chip.startswith("What were the discharge values") for chip in _chips(state))


# ---------------------------------------------------------------------------- P9 note


def test_p9_offers_the_note_overview_when_turn_1_was_not_the_summary():
    state = _note_state({"discharge_diagnosis": "Status post knee replacement."})
    state["command"] = "labs"
    assert followups._NOTE_OVERVIEW in _chips(state)


def test_p9_is_suppressed_when_turn_1_was_the_summary_itself():
    state = _note_state({"discharge_diagnosis": "Status post knee replacement."})
    assert state["command"] == "summarize"
    assert followups._NOTE_OVERVIEW not in _chips(state)


# ------------------------------------------------------------------ P10 empty meds


def test_p10_no_medication_chip_on_a_record_with_no_discharge_medications():
    """`90000106`'s shape: `/meds` returns nothing, so a drug chip would ask about a
    list the record does not hold."""
    state = _med_state("", rows=[], returned=0)
    state["tool_calls"].append({
        "name": "rag_search", "response": {"returned": 1, "passages": [
            {"section": "brief_hospital_course", "text": "Admitted to the emergency room."}]}})
    chips = _chips(state)

    assert not any("dosing" in chip.lower() or "taper" in chip.lower() for chip in chips)
    assert followups._NOTE_OVERVIEW in chips


# -------------------------------------------------------------------- outpatient


def test_an_outpatient_record_with_no_note_offers_no_chart_chip():
    """`90000051`'s shape after the encounter notice: there is nothing to drill into."""
    state = {"tool_calls": [], "command": "summarize"}
    assert _chips(state, outpatient=True) == []


def test_an_outpatient_record_keeps_its_lab_chip_because_the_labs_exist():
    state = _lab_state(_trend("RDW", 23.1, 11.5, 14.5, unit="%", decimals=1))
    state["command"] = "labs"
    chips = _chips(state, outpatient=True)

    assert len(chips) == 1
    assert "RDW" in chips[0]


# --------------------------------------------------------------------- literature


def test_a_literature_turn_asks_about_the_evidence_and_nothing_else():
    chips = followups.suggest(
        "/pubmed dual antiplatelet therapy after coronary stenting", ANSWER,
        scope="literature",
        state={"tool_calls": [{"name": "search_literature",
                               "response": {"articles": [{"title": "A trial"}]}}]})

    assert len(chips) == 1
    assert "published literature" in chips[0]
    assert "dual antiplatelet therapy" in chips[0]


def test_a_literature_turn_ignores_chart_signals():
    chips = followups.suggest(
        "/pubmed anticoagulation", ANSWER, scope="literature",
        state=_med_state("Continue Coumadin 7 mg daily.", returned=1))

    assert all("published literature" in chip for chip in chips)


# ------------------------------------------------------------------- the veto itself


def test_a_chip_that_would_bind_pubmed_is_dropped():
    """The veto is the loop's own predicate, not a blacklist maintained here."""
    assert not followups._routable("What does the evidence say about her Coumadin?", None)
    assert followups._routable("What is the discharge dosing plan for Coumadin?", None)


def test_no_template_carries_a_literature_cue():
    """Every template in the module, filled with a plausible slot, must survive."""
    templates = {
        followups._MED_TAPER: {"drug": "oxycodone SR"},
        followups._MED_ANTICOAG_MONITORED: {"drug": "Coumadin"},
        followups._MED_ANTICOAG: {"drug": "Coumadin"},
        followups._MED_REGIMEN: {"first": "Celebrex", "second": "Colace"},
        followups._RISK_DRIVER: {"driver": "oncology history"},
        followups._LAB_ABNORMAL: {"label": "Potassium", "value": "6.0",
                                 "unit": "mmol/L"},
        followups._LAB_NORMAL: {"first": "Sodium", "second": "Potassium"},
    }
    for template, slots in templates.items():
        chip = followups._filled(template, **slots)
        assert chip, template
        assert not routing.asks_for_literature(chip), chip


def test_the_fixed_templates_are_routable():
    for chip in (followups._CARE_RED_FLAGS, followups._CARE_FOLLOW_UP,
                 followups._NOTE_OVERVIEW):
        assert not routing.asks_for_literature(chip), chip
        assert len(chip) <= followups._MAX_QUESTION_CHARS


# ------------------------------------------------------------------ dedup and replay


def test_an_identical_chip_is_offered_once():
    assert followups._distinct(["A?", "a?", "B?"]) == ["A?", "B?"]


def test_replayed_calls_count_as_chart_state():
    """A follow-up turn's chart facts may belong to the turn before."""
    state = {
        "tool_calls": [],
        "replayed_tool_calls": [{
            "name": "discharge_medications",
            "response": {"returned": 1,
                         "passages": [{"section": "discharge_medications",
                                       "text": "Continue Coumadin 7 mg daily."}]},
        }],
        "command": None,
    }
    assert any("Coumadin" in chip for chip in _chips(state))


@pytest.mark.parametrize("chip", followups.suggest(
    "/ehr", ANSWER, scope="patient_chart",
    state=_med_state("Continue Coumadin 7 mg daily. No driving for two days. "
                     "Follow up with Dr. A.", rows=["Coumadin"], returned=1)))
def test_a_real_turn_s_chips_are_short_enough_for_a_row(chip):
    assert len(chip) <= followups._MAX_QUESTION_CHARS


# ------------------------------------------------------- Sprint 1: asked questions


TAPER_NOTE = ("Oxycodone SR 20 mg q.12h for five days, then decrease to "
              "oxycodone SR 10 mg q.12h for five days. Continue Coumadin 7 mg daily. "
              "No driving for two days. Follow up with Dr. A.")
TAPER_CHIP = ("What is the dosing and taper schedule for oxycodone SR at discharge?")
COUMADIN_CHIP = "What is the discharge dosing plan for Coumadin?"


def test_clicking_a_chip_never_offers_that_chip_back():
    """The production defect: clicking the taper chip re-rendered the taper chip."""
    state = _med_state(TAPER_NOTE, rows=["Oxycodone SR", "Coumadin"], returned=1)
    first = followups.suggest("/meds", ANSWER, scope="patient_chart", state=state)
    assert first[0] == TAPER_CHIP

    second = followups.suggest("/meds", ANSWER, scope="patient_chart", state=state,
                               asked=["/meds", TAPER_CHIP])
    assert TAPER_CHIP not in second, second
    assert second[0] == COUMADIN_CHIP


def test_an_asked_question_matches_through_punctuation_and_case():
    """A click and its replayed text differ in whitespace and punctuation only."""
    state = _med_state(TAPER_NOTE, rows=["Oxycodone SR", "Coumadin"], returned=1)
    chips = followups.suggest(
        "/meds", ANSWER, scope="patient_chart", state=state,
        asked=["what is the DOSING and taper schedule for oxycodone sr at discharge"])
    assert TAPER_CHIP not in chips


def test_the_current_turn_advances_down_the_taxonomy_across_three_turns():
    """Turn 1 → Turn 2 → Turn 3 consumes what was asked and keeps advancing."""
    state = _med_state(TAPER_NOTE, rows=["Oxycodone SR", "Coumadin"], returned=1)

    turn_1 = followups.suggest("/meds", ANSWER, scope="patient_chart", state=state)
    turn_2 = followups.suggest("/meds", ANSWER, scope="patient_chart", state=state,
                               asked=["/meds", *turn_1])
    turn_3 = followups.suggest("/meds", ANSWER, scope="patient_chart", state=state,
                               asked=["/meds", *turn_1, *turn_2])

    for earlier, later in ((turn_1, turn_2), (turn_1 + turn_2, turn_3)):
        assert not set(map(str.lower, later)) & set(map(str.lower, earlier)), (
            f"repeat across turns: {later} vs {earlier}")
    # Each turn advances: the second offers what the first did not, and once the
    # taxonomy for this record is spent the row shrinks rather than repeating.
    assert turn_2, "turn 2 offered nothing after three questions were asked"
    assert len(set(turn_1) & set(turn_2)) == 0
    assert len(turn_3) < len(turn_2)


def test_a_thread_that_has_asked_everything_offers_nothing():
    state = _med_state(TAPER_NOTE, rows=["Oxycodone SR", "Coumadin"], returned=1)
    everything = followups._candidates(state=state, scope="patient_chart",
                                       question="/meds", outpatient=False)
    chips = followups.suggest("/meds", ANSWER, scope="patient_chart", state=state,
                              asked=["/meds", *everything])
    assert chips == []


# ------------------------------------------------- Sprint 1: current turn first


def test_the_current_turn_s_evidence_is_offered_before_replayed_evidence():
    """A lab turn's own chip outranks the medication chips of the turn before it."""
    state = {
        "tool_calls": [{"name": "get_lab_trends", "response": {
            "returned": 1, "trends": [_trend("RDW", 23.7, 11.5, 14.5,
                                             unit="%", decimals=1)]}}],
        "replayed_tool_calls": [{"name": "discharge_medications", "response": {
            "returned": 1, "passages": [{"section": "discharge_medications",
                                         "text": "Continue Coumadin 7 mg daily."}]}}],
        "command": None,
    }
    chips = followups.suggest("what was abnormal?", ANSWER, scope="patient_chart",
                              state=state)

    assert chips[0].startswith("How does the abnormal discharge RDW")
    # And the replayed chart fills the remaining slots rather than being ignored.
    assert any("Coumadin" in chip for chip in chips[1:])
    assert len(chips) >= 2


def test_the_replayed_fallback_is_not_used_when_the_current_turn_fills_the_row():
    state = _med_state(TAPER_NOTE, rows=["Oxycodone SR", "Coumadin"], returned=1)
    state["replayed_tool_calls"] = [{"name": "get_lab_trends", "response": {
        "returned": 1, "trends": [_trend("RDW", 23.7, 11.5, 14.5)]}}]

    chips = followups.suggest("/meds", ANSWER, scope="patient_chart", state=state)
    assert not any(chip.startswith("How does the abnormal") for chip in chips)


# -------------------------------------------------- Sprint 1: literature isolation


def _literature_state(articles: int, *, replayed: bool = True) -> dict:
    state = {
        "tool_calls": [{"name": "search_literature",
                        "response": {"query": "pain control", "returned": articles,
                                     "articles": [{"title": f"Trial {n}"}
                                                  for n in range(articles)]}}],
        "replayed_tool_calls": [],
        "command": None,
    }
    if replayed:
        state["replayed_tool_calls"] = [{
            "name": "discharge_medications",
            "response": {"returned": 1, "passages": [
                {"section": "discharge_medications",
                 "text": TAPER_NOTE}]},
        }]
    return state


def test_a_literature_turn_with_no_articles_offers_nothing():
    """Zero PubMed records is a turn with no follow-up — never the chart chips."""
    chips = followups.suggest("how long do patients need pain meds after surgery?",
                              ANSWER, scope="literature",
                              state=_literature_state(0))
    assert chips == []


def test_a_literature_turn_never_falls_back_to_replayed_chart_calls():
    chips = followups.suggest("how long do patients need pain meds after surgery?",
                              ANSWER, scope="literature",
                              state=_literature_state(2))

    assert len(chips) == 1
    assert "published literature" in chips[0]
    assert not any("oxycodone" in chip or "Coumadin" in chip for chip in chips)


def test_a_literature_turn_ignores_the_chart_even_when_it_is_in_the_same_turn():
    state = _literature_state(1)
    state["tool_calls"].append({"name": "discharge_medications", "response": {
        "returned": 1, "passages": [{"section": "discharge_medications",
                                     "text": TAPER_NOTE}]}})
    chips = followups.suggest("/pubmed pain control", ANSWER, scope="literature",
                              state=state)

    assert all("published literature" in chip for chip in chips)
