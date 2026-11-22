"""A thread may run to five turns, and the judge must be shown turn 5's answer key.

The Tier 2 dev and holdout suites are about to carry threads longer than four turns. Three
ceilings had to move together, and each of them failed differently:

* `schemas.MAX_TURNS` refused a five-turn `multi_turn` item at validation time;
* `dev_suite.validate_rows` and `sync_holdout_suite.validate_rows` indexed a dict by turn
  count, so a five-turn row raised `KeyError` — the loader crashed instead of naming the
  scenario and the rule it broke (`sync_holdout_suite` capped at three turns as well);
* `judge_call._scenario_constraints` named `turn_2`, `turn_3` and `turn_4` key by key, so a
  `turn_5` answer key would have been graded by a judge that never saw it.

The two-turn rule is pinned here too. A two-turn item is either a command and its follow-up
or a short multi-turn thread, and the old table allowed only the first.
"""

from __future__ import annotations

import pytest

from evaluation.platform import dev_suite, judge_call, schemas, sync_holdout_suite


def _row(scenario: str, hadm: str, turns: list[str], workflow: str,
         trajectory: str = "t4_cross_domain_synthesis") -> dict:
    return {
        "id": scenario,
        "hadm_id": hadm,
        "workflow_type": workflow,
        "trajectory": trajectory,
        "turns": turns,
        "expected_output": {"facts": {"encounter": "inpatient"}},
    }


# --- the ceiling ---------------------------------------------------------------------


def test_the_ceiling_is_five_turns():
    assert schemas.MAX_TURNS == 5


def test_a_five_turn_item_validates():
    item = schemas.DatasetItemInput.model_validate({
        "patient_id": "90000015",
        "workflow_type": "multi_turn",
        "turns": ["/ehr", "/labs", "and the meds?", "which was worst?", "and the risk?"],
        "patient_manifest": {"encounter": "inpatient"},
    })

    assert len(item.turns) == 5


def test_a_six_turn_item_is_refused():
    with pytest.raises(ValueError):
        schemas.DatasetItemInput.model_validate({
            "patient_id": "90000015",
            "workflow_type": "multi_turn",
            "turns": ["one", "two", "three", "four", "five", "six"],
            "patient_manifest": {"encounter": "inpatient"},
        })


# --- the suite validators ------------------------------------------------------------


def test_the_dev_validator_accepts_a_five_turn_row():
    dev_suite.validate_rows([_row("dev_t4_05", "90000015",
                                  ["/ehr", "/labs", "and?", "which was worst?", "risk?"],
                                  "multi_turn")])


def test_the_dev_validator_names_a_six_turn_row_rather_than_crashing():
    """Before the fix this raised `KeyError: 6` from inside the loader."""
    with pytest.raises(dev_suite.DevSuiteError) as raised:
        dev_suite.validate_rows([_row("dev_t4_06", "90000015",
                                      ["a", "b", "c", "d", "e", "f"], "multi_turn")])

    assert "6 turns exceeds 5" in str(raised.value)


def test_a_two_turn_row_may_be_a_short_multi_turn_thread():
    """The old table allowed only `command_plus_followup` at two turns."""
    dev_suite.validate_rows([_row("dev_t4_02", "90000015", ["/ehr", "and the labs?"],
                                  "multi_turn")])
    dev_suite.validate_rows([_row("dev_t1_02", "90000015", ["/ehr", "and the labs?"],
                                  "command_plus_followup")])


def test_a_three_turn_row_may_not_be_a_command_plus_followup():
    with pytest.raises(dev_suite.DevSuiteError) as raised:
        dev_suite.validate_rows([_row("dev_t3_09", "90000015", ["/ehr", "a?", "b?"],
                                      "command_plus_followup")])

    assert "need workflow_type" in str(raised.value)


def test_the_holdout_validator_accepts_five_turns_and_still_caps_at_five():
    rows = [_row(f"holdout_t4_{index:02}", "90000087",
                 ["/ehr", "/labs", "and?", "which was worst?", "risk?"], "multi_turn")
            for index in range(sync_holdout_suite.EXPECTED_ITEMS)]
    sync_holdout_suite.validate_rows(rows)

    rows[0] = _row("holdout_t4_00", "90000087",
                   ["a", "b", "c", "d", "e", "f"], "multi_turn")
    with pytest.raises(sync_holdout_suite.HoldoutSuiteError) as raised:
        sync_holdout_suite.validate_rows(rows)

    assert "6 turns exceeds 5" in str(raised.value)


# --- the judge -----------------------------------------------------------------------


def test_the_judge_is_shown_the_fifth_turn_answer_key():
    prompt = judge_call.build_user_prompt(
        {"patient_manifest": {"encounter": "inpatient"}, "turns": ["/ehr"]},
        {"turn_responses": ["Answer one."]},
        {"facts": {"encounter": "inpatient"},
         "turn_5": {"facts": {"prior_inpatient_days": 0},
                    "behaviour": "names the zero prior days"}},
    )

    assert "turn_5" in prompt
    assert "names the zero prior days" in prompt


def test_an_absent_fifth_turn_leaves_the_block_unchanged():
    prompt = judge_call.build_user_prompt(
        {"patient_manifest": {}, "turns": ["/ehr"]},
        {"turn_responses": ["Answer one."]},
        {"facts": {"encounter": "inpatient"}, "turn_4": {"facts": {"a": 1}}},
    )

    assert "turn_4" in prompt
    assert "turn_5" not in prompt
