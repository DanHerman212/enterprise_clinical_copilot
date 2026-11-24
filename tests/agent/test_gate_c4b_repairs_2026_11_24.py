"""Gate C4b-preparation: the offline half, one test per repaired contract.

Four behaviours, and each of them was measured wrong in the C4a slice against `511fbbe`:
a schedule cell that lost its durations, a `/ehr` instruction that kept a heading the record
could not support, a driver caveat that disclaimed a value the model had scored, and a plural
blood-count request that could be answered from one analyte.
"""

from __future__ import annotations

import asyncio
import json

from services.agent.engine import graph
from services.agent.engine import prompts
from services.agent.engine.deterministic import (
    _has_discharge_section,
    _panel_has_abnormal,
    _panel_has_critical,
    _split_regimen_detail,
    lab_flag_annotation,
    parse_medications,
)
from services.agent.engine.mcp_client import _presented_factor
from services.agent.postprocess.guardrail import normalise_factor_values
from services.agent.routing import admission_context
from services.agent.routing import routing

# ------------------------------------------------------------------ 1. the regimen cell

_TAPER = "20 mg p.o. q.12h. for five days, then 10 mg p.o. q.12h. for five days"


def test_a_step_down_taper_keeps_its_durations_in_the_schedule():
    """A taper is expressed as time, so the time has to survive the split.

    `for five days, then 10 mg … for five days` is the whole of a step-down. Moving it out
    of `frequency` turned one dated regimen into two open-ended rows (`dev_d1_slash_02`,
    `dev_d5_slash_01`, measured 2026-11-24).
    """
    frequency, moved = _split_regimen_detail(_TAPER)

    assert frequency == "20 mg q.12h. for five days, then 10 mg q.12h. for five days"
    # Both steps state the route; both are moved, and `_regimen_route` keeps one.
    assert moved == ["p.o.", "p.o."]


def test_a_two_week_course_stays_beside_its_bedtime_dose():
    frequency, moved = _split_regimen_detail("at bedtime p.r.n. for two weeks")

    assert frequency == "at bedtime p.r.n. for two weeks"
    assert moved == []


def test_an_indication_still_moves_while_the_schedule_stays_whole():
    frequency, moved = _split_regimen_detail("q.6h. to be used p.r.n. for breakthrough pain")

    assert frequency == "q.6h. p.r.n."
    assert moved == ["for breakthrough pain"]


def test_parse_medications_keeps_the_taper_and_moves_the_route():
    """The whole cell survives the round trip, and `p.o.` lands in `route` beside it."""
    note = (
        "DISCHARGE INSTRUCTIONS: oxycodone SR 20 mg p.o. q.12h. for five days, then 10 mg "
        "p.o. q.12h. for five days. trazodone 50 mg at bedtime p.r.n. for two weeks. "
        "Percocet 5/325 mg one to two tablets q.6h. p.r.n. for breakthrough pain."
    )
    output = json.dumps({"medications": [
        {"medication": "oxycodone SR", "dosage": "20 mg", "frequency": _TAPER,
         "route": "", "setting": "discharge"},
        {"medication": "trazodone", "dosage": "50 mg",
         "frequency": "at bedtime p.r.n. for two weeks", "route": "",
         "setting": "discharge"},
        {"medication": "Percocet", "dosage": "5/325 mg",
         "frequency": "q.6h. p.r.n. for breakthrough pain", "route": "",
         "setting": "discharge"},
    ]})

    rows, dropped = parse_medications(output, note)

    assert dropped == []
    by_name = {row["medication"].lower(): row for row in rows}
    taper = by_name["oxycodone sr"]
    assert taper["frequency"] == (
        "20 mg q.12h. for five days, then 10 mg q.12h. for five days")
    assert taper["route"] == "p.o."
    assert by_name["trazodone"]["frequency"] == "at bedtime p.r.n. for two weeks"
    assert by_name["percocet"]["frequency"] == "q.6h. p.r.n."
    assert by_name["percocet"]["route"] == "for breakthrough pain"


def test_a_route_the_model_stated_is_kept_beside_the_one_it_misplaced():
    """Two supported fragments combine; neither overwrites the other."""
    note = "DISCHARGE INSTRUCTIONS: oxycodone SR 20 mg p.o. q.12h. for five days."
    output = json.dumps({"medications": [
        {"medication": "oxycodone SR", "dosage": "20 mg",
         "frequency": "p.o. q.12h. for five days", "route": "p.o.",
         "setting": "discharge"},
    ]})

    rows, _ = parse_medications(output, note)

    assert rows[0]["frequency"] == "q.12h. for five days"
    # The same route twice is one entry, not "p.o., p.o.".
    assert rows[0]["route"] == "p.o."


# ------------------------------------------------------- 2. the /ehr discharge heading


def test_a_record_with_no_discharge_section_gets_no_discharge_heading():
    """The heading is a claim about the document, so the retrieval decides whether it exists.

    90000035's `/ehr` turn retrieved only the diagnosis, procedure, indications and course,
    and wrote the heading anyway over facts the course had already carried
    (`dev_d5_fup_02` turn 1, measured 2026-11-24).
    """
    without = " ".join(prompts.ehr_formatting_for(has_discharge_section=False).split())

    assert "**Discharge plan & medications.**" not in without
    # And the fact the heading was carrying is still a fact: where the patient went is an
    # uncited sentence at the close of the course, read from the manifest.
    assert "no discharge-plan heading" in without
    assert "`<admission_metadata>`" in without


def test_a_record_with_a_discharge_section_keeps_the_heading():
    assert "**Discharge plan & medications.**" in prompts.ehr_formatting_for(
        has_discharge_section=True)
    # The default composition is the one every non-live caller reads, so it keeps it too.
    assert "**Discharge plan & medications.**" in prompts.EHR_FORMATTING


# ------------------------------------------------------------ 3. the driver caveat


def test_the_series_caveat_belongs_only_to_a_row_with_no_value():
    """A valued driver's series IS that value's course, and is not disclaimed.

    `dev_d4_risk_01` turn 3 stated `lowest red blood cell count (3.82 ×10⁶/µL) decreases the
    risk` and then said those readings were not what the model scored. Both halves cannot be
    true (`2026-11-24`).
    """
    spoke = " ".join(prompts.RISK_SPOKE.split())

    assert ("Qualify the series ONLY where part one said the row carries no recorded value"
            in spoke)
    assert ("Where part one stated a value, the series is that value's own course and is "
            "never disclaimed" in spoke)

    valued = _presented_factor(
        {"feature": "rbc_min", "direction": "decreases", "value": 3.82})
    nulled = _presented_factor(
        {"feature": "monocytes_min", "direction": "decreases", "value": None})
    assert valued["value"]
    assert "value" not in nulled


# ------------------------------------------------------ 4. the plural panel request

_PLURAL = ("Compare those two opposing factors with her discharge blood counts and 16.6% "
           "risk score.")


def test_a_plural_blood_count_request_keeps_the_panel_reachable():
    """One analyte's series cannot answer "her blood counts", and a replay must not close it.

    90000046's fourth turn held `get_lab_trends(analyte='rbc')` from a driver chip, asked
    for the discharge blood counts, and made no tool call at all (`dev_d4_risk_01`,
    measured 2026-11-24).
    """
    assert routing.asks_for_the_whole_panel(_PLURAL)
    assert not routing.asks_for_the_whole_panel(
        "How does lowest red blood cell count affect this patient's readmission risk?")

    allowed = graph._freetext_allowed(
        True, False, True, False, False, question=_PLURAL)
    assert "get_lab_trends" in allowed

    # The single-analyte call does not spend the turn's allowance, so the panel call that
    # answers the plural question is still available after it.
    assert routing.spends_turn_allowance("get_lab_trends", {"analyte": "rbc"}) is False
    assert routing.spends_turn_allowance("get_lab_trends", {}) is True


# ------------------------------------------------- 5. the panel is read, not just reachable


def test_a_plural_request_with_only_an_analyte_read_prefetches_the_whole_panel(monkeypatch):
    """Binding the tool is not enough; the panel has to be in the record before the write.

    90000046's fourth turn held `rbc` from a driver chip and the note's own `pertinent_results`
    snapshot, and answered with both and no tool call at all (`dev_d4_risk_01`, 2026-11-24).
    """
    seen: list[dict] = []

    async def fake_execute(toolbox, calls, **kwargs):
        seen.extend(calls)
        return [], [{"name": "get_lab_trends", "args": {}, "response": {"trends": []}}]

    monkeypatch.setattr(graph, "_execute_tool_calls", fake_execute)

    recorded = asyncio.run(graph._prefetch_full_panel(
        object(), object(), question=_PLURAL, hadm_id=90000046,
        on_event=None, cite_offset=0,
        replayed=[{"name": "get_lab_trends", "args": {"analyte": "rbc"},
                   "response": {"trends": []}}],
    ))

    assert seen == [{"name": "get_lab_trends", "args": {}, "id": ""}], seen
    assert recorded


def test_a_thread_that_already_holds_the_panel_reads_it_again(monkeypatch):
    called = False

    async def fake_execute(toolbox, calls, **kwargs):
        nonlocal called
        called = True
        return [], []

    monkeypatch.setattr(graph, "_execute_tool_calls", fake_execute)

    recorded = asyncio.run(graph._prefetch_full_panel(
        object(), object(), question=_PLURAL, hadm_id=90000046,
        on_event=None, cite_offset=0,
        replayed=[{"name": "get_lab_trends", "args": {}, "response": {"trends": []}}],
    ))

    assert recorded == []
    assert called is False


def test_a_single_analyte_question_prefetches_nothing(monkeypatch):
    called = False

    async def fake_execute(toolbox, calls, **kwargs):
        nonlocal called
        called = True
        return [], []

    monkeypatch.setattr(graph, "_execute_tool_calls", fake_execute)

    recorded = asyncio.run(graph._prefetch_full_panel(
        object(), object(),
        question="How does lowest red blood cell count affect her readmission risk?",
        hadm_id=90000046, on_event=None, cite_offset=0, replayed=[],
    ))

    assert recorded == []
    assert called is False


# ---------------------------------------------- 6. a fractional duration survives every layer


def test_a_fractional_day_duration_is_never_rounded_to_an_integer():
    """The payload says 4.75 and every layer repeats it, or the prose contradicts the bar."""
    assert admission_context.clinical_value("prior_inpatient_days", 4.75) == 4.75
    assert admission_context.clinical_measure("prior_inpatient_days", 4.75) == "4.75"
    # A served fraction in a column whose NAME promises a tally is still the served
    # fraction: this assertion required 5 until 2026-11-24, and the same rounding of a
    # served `medication_order_count` from 13.64 to 14 is what the judge docked
    # `dev_d2_risk_01` for. Whole in, whole out — 14.0 is 14 — is the whole rule.
    assert admission_context.clinical_value("medication_order_count", 4.85) == 4.85
    assert admission_context.clinical_value("medication_order_count", 14.0) == 14


def test_the_guardrail_leaves_the_served_fraction_alone():
    call = {"name": "predict_readmission", "response": {"top_factors": [
        {"feature": "prior_inpatient_days", "value": 4.75, "direction": "increases"}]}}
    answer = "Prior inpatient days (4.75) increased the estimate."

    out, flags = normalise_factor_values(answer, [call])

    assert out == answer
    assert flags == []


# ------------------------------------------------- 7. the panel's critical status is stated


_PANEL = [{"name": "get_lab_trends", "response": {"trends": [
    {"label": "Hemoglobin", "unit": "g/dL", "series": [
        {"day": 0.0, "value": 8.5, "flag": "low"},
        {"day": 3.0, "value": 10.7, "flag": "low"}]},
    {"label": "Sodium", "unit": "mmol/L", "series": [
        {"day": 0.0, "value": 136.0, "flag": "normal"}]},
]}}]


def test_an_abnormal_panel_with_no_critical_reading_says_so():
    assert _panel_has_abnormal(_PANEL) is True
    assert _panel_has_critical(_PANEL) is False

    annotation = lab_flag_annotation(_PANEL)

    assert "No reading in this panel carries a critical flag." in annotation


def test_an_all_normal_panel_is_told_nothing_about_criticality():
    normal = [{"name": "get_lab_trends", "response": {"trends": [
        {"label": "Sodium", "series": [{"day": 0.0, "value": 136.0, "flag": "normal"}]}]}}]

    assert _panel_has_abnormal(normal) is False
    assert "critical flag" not in lab_flag_annotation(normal)

