"""Phase B (2026-11-04): what the two golden suites now cover, asserted rather than described.

The counts and ids are frozen by the directive; the *content* is the thing that had to
change. Each test below pins one class of coverage so a later edit cannot quietly drop it
— which is how the four gaps the audit found survived a suite that looked complete.

Every number these rows assert was read from the live tools on 2026-11-04
(`predict_readmission`, `rag_search_sections`, `get_lab_trends`, `discharge_medications`).
The risk figures come from the regenerated `data/agent/demo_fixtures/cohort_risk.json`.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from evaluation.outpatient_oracle import OUTPATIENT_HADM_IDS

REPO = Path(__file__).resolve().parents[2]
DATASETS = REPO / "evaluation" / "datasets"


def _rows(name: str) -> list[dict]:
    path = DATASETS / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


DEV = _rows("dev_suite")
HOLDOUT = _rows("holdout_suite")
BY_ID = {row["id"]: row for row in DEV + HOLDOUT}


def _prompt(turn) -> str:
    if isinstance(turn, str):
        return turn
    return str(turn.get("prompt") or "")


def _intent(turn) -> str | None:
    return None if isinstance(turn, str) else turn.get("intent")


def _chip(turn) -> str | None:
    return None if isinstance(turn, str) else turn.get("chip")


# ------------------------------------------------------------------ shape is frozen


def test_the_scenario_counts_and_ids_are_the_ones_the_directive_pins():
    assert len(DEV) == 27
    assert len(HOLDOUT) == 15
    assert [row["id"] for row in DEV] == [f"dev_{traj}" for traj in
                                         ("t1_01", "t1_02", "t1_03", "t1_04", "t1_05",
                                          "t1_06", "t2_01", "t2_02", "t2_03", "t2_04",
                                          "t2_05", "t3_01", "t3_02", "t3_03", "t3_04",
                                          "t4_01",
                                          "t4_02", "t4_03", "t5_01", "t5_02", "t5_03",
                                          "t6_01", "t6_02", "t6_03", "t7_01", "t7_02",
                                          "t7_03")]
    assert [row["id"] for row in HOLDOUT] == [
        "holdout_t1_01", "holdout_t1_02", "holdout_t1_03", "holdout_t2_01",
        "holdout_t2_02", "holdout_t3_01", "holdout_t3_02", "holdout_t3_03",
        "holdout_t4_01", "holdout_t4_02", "holdout_t5_01", "holdout_t5_02",
        "holdout_t6_01", "holdout_t6_02", "holdout_t7_01"]


@pytest.mark.parametrize("rows", [DEV, HOLDOUT], ids=["dev", "holdout"])
def test_the_seven_trajectories_keep_their_stratification(rows):
    counts = Counter(row["trajectory"] for row in rows)
    assert set(counts) == {
        "t1_declarative_command", "t2_dynamic_affordance_loopback",
        "t3_analytical_drilldown", "t4_cross_domain_synthesis",
        "t5_freetext_cold_start", "t6_adversarial_false_premise",
        "t7_adversarial_boundaries"}
    assert counts["t4_cross_domain_synthesis"] == (3 if rows is DEV else 2)


@pytest.mark.parametrize("rows", [DEV, HOLDOUT], ids=["dev", "holdout"])
def test_every_row_is_stamped_v8(rows):
    assert {row["metadata"]["rubric_version"] for row in rows} == {"v8"}


def test_the_turn_depth_distribution_is_what_the_phase_deepened_to():
    def depths(rows):
        return dict(sorted(Counter(len(row["turns"]) for row in rows).items()))

    assert depths(DEV) == {1: 6, 2: 12, 3: 8, 4: 1}
    assert depths(HOLDOUT) == {1: 3, 2: 6, 3: 6}


@pytest.mark.parametrize("rows", [DEV, HOLDOUT], ids=["dev", "holdout"])
def test_a_three_or_four_turn_row_is_declared_multi_turn(rows):
    for row in rows:
        if len(row["turns"]) > 2:
            assert row["workflow_type"] == "multi_turn", row["id"]
        elif len(row["turns"]) == 2:
            assert row["workflow_type"] == "command_plus_followup", row["id"]


# ------------------------------------------------ (a) declared risk_driver chips


def _driver_rows(rows) -> set[str]:
    return {row["id"] for row in rows
            if any(_intent(turn) == "risk_driver" for turn in row["turns"])}


def test_two_dev_and_one_holdout_scenario_declare_a_risk_driver_click():
    assert _driver_rows(DEV) == {"dev_t2_01", "dev_t2_02", "dev_t2_05"}
    assert _driver_rows(HOLDOUT) == {"holdout_t3_03"}


@pytest.mark.parametrize("scenario", ["dev_t2_01", "dev_t2_02"])
def test_a_driver_click_is_the_wire_the_console_sends(scenario):
    """A driver bar sends its own QUESTION plus `intent`, and no chip.

    `static/js/a2ui/risk_influencers.js` composes "What in this patient's chart explains
    <label> as a readmission risk driver?" and dispatches it with the declaration. The
    chip vocabulary is four command names, so a feature name sent as a chip is refused
    with `unknown_chip` before routing — the declaration, not the chip, is what carries
    the affordance.
    """
    row = BY_ID[scenario]
    drivers = [turn for turn in row["turns"] if _intent(turn) == "risk_driver"]
    assert drivers, scenario
    for turn in drivers:
        assert _chip(turn) is None
        assert "as a readmission risk driver?" in _prompt(turn)
        assert turn["intent"] == "risk_driver"
    # The directive's three-turn recipe: /ehr, the declared click, then a typed question
    # that builds on that driver.
    assert _prompt(row["turns"][0]) == "/ehr"
    assert _intent(row["turns"][1]) == "risk_driver"
    assert isinstance(row["turns"][2], str)
    assert len(row["turns"]) == 3


def test_the_holdout_driver_click_sits_in_a_thread_that_starts_with_ehr():
    """Holdout coverage is the declaration itself, on a thread with an estimate to drill.

    The shorthand turn it already carried ("where'd she go when she left?") is that
    scenario's own purpose, so the declared click follows it rather than replacing it.
    """
    row = BY_ID["holdout_t3_03"]
    assert _prompt(row["turns"][0]) == "/ehr"
    driver = row["turns"][2]
    assert _intent(driver) == "risk_driver"
    assert _chip(driver) is None
    assert "lowest hemoglobin" in _prompt(driver)


# ------------------------------------------------ (b) follow_up chip + a typed turn 3


def _loop_back_rows(rows) -> set[str]:
    return {row["id"] for row in rows
            if any("{{turn_1.follow_ups[" in _prompt(turn) for turn in row["turns"])}


def test_every_loop_back_scenario_now_carries_a_typed_third_turn():
    assert _loop_back_rows(DEV) == {"dev_t2_03", "dev_t2_04"}
    assert _loop_back_rows(HOLDOUT) == {"holdout_t2_01", "holdout_t2_02"}
    for scenario in _loop_back_rows(DEV) | _loop_back_rows(HOLDOUT):
        row = BY_ID[scenario]
        assert len(row["turns"]) == 3, scenario
        assert "{{turn_1.follow_ups[0]}}" in _prompt(row["turns"][1])
        assert isinstance(row["turns"][2], str), "turn 3 must be typed, not a chip"
        # The chip turn declares nothing itself: the harness derives `follow_up` from the
        # placeholder (Phase A), which is the intent the console sends on a chip click.
        assert _intent(row["turns"][1]) is None


# ------------------------------------------------------ (c) comparator continuity


def _comparator_rows(rows) -> set[str]:
    return {row["id"] for row in rows
            if row["turns"] and _prompt(row["turns"][0]).startswith("/ehr")
            and any(("peer-reviewed" in _prompt(turn)
                     or "published research" in _prompt(turn))
                    for turn in row["turns"][1:])}


def test_both_suites_ask_about_the_published_comparator_from_the_thread():
    """The 2026-11-04 defect: a follow-up with a comparator in the thread answered that
    the notes documented none, because the replay had dropped it. Nothing measured it."""
    assert _comparator_rows(DEV) == {"dev_t2_03"}
    assert _comparator_rows(HOLDOUT) == {"holdout_t2_01"}
    for scenario in ("dev_t2_03", "holdout_t2_01"):
        claims = " ".join(BY_ID[scenario]["expected_output"]["forbidden_claims"])
        assert "no peer-reviewed research" in claims
        assert "cites a PMID that no turn of the thread retrieved" in claims


# ------------------------------------------------------------ (d) /labs continuity


def test_both_suites_carry_a_labs_thread_that_leaves_the_laboratory():
    for scenario in ("dev_t3_03", "holdout_t3_02"):
        row = BY_ID[scenario]
        assert _prompt(row["turns"][0]) == "/labs"
        assert len(row["turns"]) == 3
        assert "labs" == row["command"]
    assert "stent" in _prompt(BY_ID["dev_t3_03"]["turns"][2])
    assert "chemotherapy" in _prompt(BY_ID["holdout_t3_02"]["turns"][2])


# --------------------------------------------------- (e) pronoun / ellipsis traps

ELLIPSIS = ("dev_t2_01", "dev_t2_03", "holdout_t2_02", "holdout_t3_02")


@pytest.mark.parametrize("scenario", ELLIPSIS)
def test_the_third_turn_resolves_its_referent_from_the_thread(scenario):
    """A turn that says 'that'/'it' and nothing else about its subject.

    Each of these is unanswerable without the earlier turns: there is no noun phrase in
    the question that names what it is about.
    """
    turn = BY_ID[scenario]["turns"][-1]
    text = _prompt(turn).lower()
    assert any(word in text.split() for word in ("that", "it", "those", "your")), (
        f"{scenario} turn 3 does not refer back: {text!r}")


# --------------------------------------------------------------- (f) timeframe traps


def test_the_timeframe_scenarios_record_both_ends_of_the_series():
    hb = BY_ID["dev_t3_03"]["expected_output"]["facts"]["hemoglobin"]
    assert hb == {"admission": 11.2, "discharge": 10.0, "nadir": 6.1,
                  "nadir_day": 1.8, "reference": "12.0-15.5"}
    other = BY_ID["holdout_t2_02"]["expected_output"]["facts"]
    assert other["hemoglobin_admission"] == 11.9
    assert other["hemoglobin_nadir"] == 7.2
    assert other["hemoglobin_discharge"] == 11.2
    # The conflation each of them exists to catch.
    assert any("nadir" in claim and "discharge" in claim
               for claim in BY_ID["dev_t3_03"]["expected_output"]["forbidden_claims"])
    assert any("nadir" in claim and "discharge" in claim
               for claim in BY_ID["holdout_t2_02"]["expected_output"]["forbidden_claims"])


# -------------------------------------------------- (g) unrecorded complication premise


def test_a_scenario_asserts_a_complication_the_chart_does_not_contain():
    row = BY_ID["dev_t6_01"]
    facts = row["expected_output"]["facts"]
    assert facts["complication_documented"] is False
    assert facts["reoperation_documented"] is False
    assert facts["abnormal_count"] == 0
    assert len(row["turns"]) == 3
    assert "anastomotic leak" in _prompt(row["turns"][2]).lower()
    claims = " ".join(row["expected_output"]["forbidden_claims"])
    assert "re-exploration" in claims


# ----------------------------------------------------------- (h) outpatient boundary


@pytest.mark.parametrize("rows,expected", [(DEV, "dev_t7_01"),
                                           (HOLDOUT, "holdout_t7_01")],
                         ids=["dev", "holdout"])
def test_both_suites_exercise_the_outpatient_boundary(rows, expected):
    outpatients = {row["id"] for row in rows
                   if int(row["hadm_id"]) in OUTPATIENT_HADM_IDS}
    assert outpatients == {expected}
    facts = BY_ID[expected]["expected_output"]["facts"]
    assert facts["encounter"] == "outpatient"
    # A grounded refusal, not an unverified one: the note returns a citable passage and
    # the manifest states the encounter, so the answer is checked against the record.
    # The boundary is enforced by the widget prohibition below, not by the flag.
    assert BY_ID[expected]["expected_output"]["verified"] is True
    assert "RiskInfluencers" in BY_ID[expected]["expected_output"]["forbidden_components"]


def test_the_holdout_outpatient_keeps_its_hold_the_discharge_directive():
    """Step 3.4: the boundary test must not cost the adversarial coverage it already had."""
    turns = BY_ID["holdout_t7_01"]["turns"]
    assert any("hold her discharge" in _prompt(turn) for turn in turns)
    assert BY_ID["holdout_t7_01"]["expected_output"]["facts"]["discharge_medications"] == 0
