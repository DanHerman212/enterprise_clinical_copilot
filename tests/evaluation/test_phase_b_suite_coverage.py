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
    assert len(DEV) == 40
    assert len(HOLDOUT) == 15
    assert len({row["id"] for row in DEV}) == 40
    # The Dev file is the matrix, so its ids name the depth and the surface rather than a
    # trajectory, and every id agrees with the depth its turns actually have.
    for row in DEV:
        assert row["id"].startswith("dev_d"), row["id"]
        assert row["metadata"]["depth_tier"] == f"{len(row['turns'])}_turn", row["id"]
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
    assert counts["t4_cross_domain_synthesis"] == (11 if rows is DEV else 2)


@pytest.mark.parametrize(
    ("rows", "expected_version"),
    [(DEV, "v9"), (HOLDOUT, "v8")],
    ids=["dev", "holdout"],
)
def test_every_row_is_stamped_with_its_expected_rubric_version(rows, expected_version):
    assert {row["metadata"]["rubric_version"] for row in rows} == {expected_version}


def test_the_turn_depth_distribution_is_what_the_phase_deepened_to():
    def depths(rows):
        return dict(sorted(Counter(len(row["turns"]) for row in rows).items()))

    assert depths(DEV) == {1: 8, 2: 8, 3: 8, 4: 8, 5: 8}
    assert depths(HOLDOUT) == {1: 3, 2: 6, 3: 6}


@pytest.mark.parametrize("rows", [DEV, HOLDOUT], ids=["dev", "holdout"])
def test_a_three_or_four_turn_row_is_declared_multi_turn(rows):
    for row in rows:
        if len(row["turns"]) > 2:
            assert row["workflow_type"] == "multi_turn", row["id"]
        elif len(row["turns"]) == 2:
            assert row["workflow_type"] in {"command_plus_followup", "multi_turn"}, \
                row["id"]


# ------------------------------------------------ (a) declared risk_driver chips


def _driver_rows(rows) -> set[str]:
    return {row["id"] for row in rows
            if any(_intent(turn) == "risk_driver" for turn in row["turns"])}


def test_seven_dev_and_one_holdout_scenario_declare_a_risk_driver_click():
    assert _driver_rows(DEV) == {
        "dev_d2_risk_01", "dev_d2_risk_02", "dev_d3_risk_01", "dev_d4_risk_01",
        "dev_d4_risk_02", "dev_d5_risk_01", "dev_d5_risk_02"}
    assert _driver_rows(HOLDOUT) == {"holdout_t3_03"}


@pytest.mark.parametrize("scenario", ["dev_d2_risk_01", "dev_d2_risk_02"])
def test_a_driver_click_is_the_wire_the_console_sends(scenario):
    """A driver bar sends the question `shared.labels` composes, plus its intent.

    The chip under an answer and the bar inside the risk card ask the same thing of the
    same evidence, and that sentence is composed once, in `shared.labels.driver_question`.
    The chip vocabulary is four command names, so a feature name sent as a chip is refused
    with `unknown_chip` before routing — the declaration, not the chip, carries the
    affordance.
    """
    row = BY_ID[scenario]
    drivers = [turn for turn in row["turns"] if _intent(turn) == "risk_driver"]
    assert drivers, scenario
    for turn in drivers:
        assert _chip(turn) is None
        prompt = _prompt(turn)
        assert prompt.startswith("How does "), prompt
        assert prompt.endswith("affect this patient's readmission risk?"), prompt
    # The command first, then the declared click it offered.
    assert _prompt(row["turns"][0]) == "/ehr"
    assert _intent(row["turns"][1]) == "risk_driver"


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


def test_every_loop_back_scenario_carries_a_chip_turn_and_a_typed_turn():
    assert _loop_back_rows(DEV) == {
        "dev_d2_fup_01", "dev_d2_fup_02", "dev_d3_fup_01", "dev_d3_fup_02",
        "dev_d4_fup_01", "dev_d5_fup_01", "dev_d5_fup_02", "dev_d5_risk_01"}
    assert _loop_back_rows(HOLDOUT) == {"holdout_t2_01", "holdout_t2_02"}
    for scenario in _loop_back_rows(DEV) | _loop_back_rows(HOLDOUT):
        row = BY_ID[scenario]
        chip_turns = [index for index, turn in enumerate(row["turns"])
                      if "{{turn_1.follow_ups[" in _prompt(turn)]
        assert chip_turns, scenario
        for index in chip_turns:
            # The chip turn declares nothing itself: the harness derives `follow_up` from
            # the placeholder, which is the intent the console sends on a chip click.
            assert _intent(row["turns"][index]) is None, scenario
        assert any(isinstance(turn, str) and not turn.startswith("/")
                   for turn in row["turns"]), "the thread carries a typed turn"
        assert row["metadata"]["turn1_follow_ups"] is True, scenario
        # Only a command with a deterministic payload may be indexed: `/meds` slots move
        # with the live extracted rows.
        assert row["turns"][0] in {"/ehr", "/labs"}, scenario


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
    assert _comparator_rows(DEV) == {"dev_d3_fup_01"}
    assert _comparator_rows(HOLDOUT) == {"holdout_t2_01"}
    for scenario in ("dev_d3_fup_01", "holdout_t2_01"):
        claims = " ".join(BY_ID[scenario]["expected_output"]["forbidden_claims"])
        assert ("no published evidence" in claims
                or "no peer-reviewed research" in claims), scenario
        assert ("cites a PMID that no turn of the thread retrieved" in claims
                or "cites an identifier no turn of the thread retrieved" in claims)


# ------------------------------------------------------------ (d) /labs continuity


def test_both_suites_carry_a_labs_thread_that_leaves_the_laboratory():
    for scenario in ("dev_d3_adv_01", "holdout_t3_02"):
        row = BY_ID[scenario]
        assert _prompt(row["turns"][0]) == "/labs"
        assert len(row["turns"]) == 3
        assert "labs" == row["command"]
    assert "leak" in _prompt(BY_ID["dev_d3_adv_01"]["turns"][2]).lower()
    assert "chemotherapy" in _prompt(BY_ID["holdout_t3_02"]["turns"][2])


# --------------------------------------------------- (e) pronoun / ellipsis traps

ELLIPSIS = ("dev_d3_fup_01", "dev_d4_risk_01", "dev_d5_risk_02",
            "holdout_t2_02", "holdout_t3_02")


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
    facts = BY_ID["dev_d3_free_02"]["expected_output"]["turn_2"]["facts"]
    assert "11.2 g/dL" in facts["arrival"]
    assert "6.1 g/dL" in facts["nadir"]
    assert "10.0 g/dL" in facts["discharge"]
    assert "10.8" in facts["discharge"], "the note's own prose value is named too"
    other = BY_ID["holdout_t2_02"]["expected_output"]["facts"]
    assert other["hemoglobin_admission"] == 11.9
    assert other["hemoglobin_nadir"] == 7.2
    assert other["hemoglobin_discharge"] == 11.2
    # The conflation the holdout row exists to catch.
    assert any("nadir" in claim and "discharge" in claim
               for claim in BY_ID["holdout_t2_02"]["expected_output"][
                   "forbidden_claims"])


# -------------------------------------------------- (g) unrecorded complication premise


def test_a_scenario_asserts_a_complication_the_chart_does_not_contain():
    row = BY_ID["dev_d3_adv_01"]
    facts = row["expected_output"]["turn_3"]["facts"]
    assert "no leak is documented" in facts["leak"]
    assert "no re-exploration occurred" in facts["leak"]
    assert len(row["turns"]) == 3
    assert "anastomotic leak" in _prompt(row["turns"][2]).lower()
    claims = " ".join(row["expected_output"]["forbidden_claims"])
    assert "reoperation" in claims


# ----------------------------------------------------------- (h) outpatient boundary


@pytest.mark.parametrize("rows,expected",
                         [(DEV, {"dev_d1_adv_01", "dev_d1_free_02", "dev_d2_adv_01",
                                  "dev_d3_free_01"}),
                          (HOLDOUT, {"holdout_t7_01"})],
                         ids=["dev", "holdout"])
def test_both_suites_exercise_the_outpatient_boundary(rows, expected):
    outpatients = {row["id"] for row in rows
                   if int(row["hadm_id"]) in OUTPATIENT_HADM_IDS}
    assert outpatients == expected
    for scenario in sorted(outpatients):
        facts = BY_ID[scenario]["expected_output"]["facts"]
        assert facts["encounter"] == "outpatient"
    # A grounded refusal, not an unverified one: the note returns a citable passage and
    # the manifest states the encounter, so the answer is checked against the record.
    # The boundary is enforced by the widget prohibition, not by the flag.
    command = BY_ID["dev_d1_adv_01"]["expected_output"]
    assert command["verified"] is True
    assert "RiskInfluencers" in command["forbidden_components"]


def test_the_holdout_outpatient_keeps_its_hold_the_discharge_directive():
    """Step 3.4: the boundary test must not cost the adversarial coverage it already had."""
    turns = BY_ID["holdout_t7_01"]["turns"]
    assert any("hold her discharge" in _prompt(turn) for turn in turns)
    assert BY_ID["holdout_t7_01"]["expected_output"]["facts"]["discharge_medications"] == 0
