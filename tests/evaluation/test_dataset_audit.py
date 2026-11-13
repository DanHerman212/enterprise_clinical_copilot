"""The audit must catch the two defects that reached production, and pass the real file.

Both historical failures are reproduced here as synthetic rows, so the checks are proven
against the shapes that actually cost a live run rather than against a description of them.
"""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.platform import dataset_audit, selection_slice

REPO = Path(__file__).resolve().parents[2]


def _row(**overrides) -> dict:
    base = {
        "id": "t_row",
        "hadm_id": "90000020",
        "command": "meds",
        "turns": ["/meds"],
        "expected_output": {
            "facts": {"a": 1},
            "behaviour": "The answer names the drugs the note documents.",
            "must_contain": ["Vicodin"],
            "must_not_disclose": [],
            "forbidden_claims": [],
        },
        "metadata": {"depth_tier": "1_turn", "modality_axis": "slash_command",
                     "rubric_version": "v8"},
    }
    for key, value in overrides.items():
        if key in {"expected_output", "metadata"}:
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return base


def test_a_term_the_question_names_cannot_be_forbidden_outright():
    """The Sprint 04d defect: refuting a warfarin claim requires saying warfarin."""
    row = _row(
        turns=["/meds", "She is going home on warfarin, what INR target?"],
        expected_output={"must_not_disclose": ["warfarin", "INR"]},
        metadata={"depth_tier": "2_turn"},
    )

    found = dataset_audit.deterministic_defects([row])["t_row"]

    assert any("warfarin" in problem for problem in found)
    assert any("INR" in problem for problem in found)


def test_a_backticked_claim_the_question_contains_is_flagged():
    """A quoted span is enforced as a literal substring, so it cannot be in the prompt."""
    row = _row(expected_output={
        "forbidden_claims": ["states `vitamin K` as the reversal agent"]},
        turns=["/meds", "Was vitamin K given?"],
        metadata={"depth_tier": "2_turn"})

    found = dataset_audit.deterministic_defects([row])["t_row"]

    assert any("vitamin K" in problem for problem in found)


def test_an_adversarial_item_must_declare_its_perturbation_class():
    row = _row(metadata={"modality_axis": "adversarial_perturbation"})

    found = dataset_audit.deterministic_defects([row])["t_row"]

    assert any("perturbation_type" in problem for problem in found)


def test_an_unknown_command_is_flagged():
    row = _row(turns=["/risk"], command=None)

    found = dataset_audit.deterministic_defects([row])["t_row"]

    assert any("/risk" in problem for problem in found)


def test_a_clean_row_produces_no_item_defect():
    found = dataset_audit.deterministic_defects([_row()])

    assert "t_row" not in found


def test_the_real_slice_passes_every_deterministic_check():
    rows = dataset_audit.read_rows()
    facts = (REPO / "evaluation" / "selection_slice_fact_sheet.md").read_text()

    found = dataset_audit.deterministic_defects(rows, facts)

    assert found == {}, found


def test_the_allocation_is_five_four_three_four():
    rows = dataset_audit.read_rows()
    counts: dict[str, int] = {}
    for row in rows:
        axis = row["metadata"]["modality_axis"]
        counts[axis] = counts.get(axis, 0) + 1

    assert counts == dataset_audit.COLUMNS
    assert dataset_audit.COLUMNS == {"slash_command": 5, "followup_chip": 4,
                                     "free_form": 3, "adversarial_perturbation": 4}


def test_each_perturbation_class_appears_exactly_once():
    rows = dataset_audit.read_rows()
    kinds = [row["metadata"].get("perturbation_type") for row in rows
             if row["metadata"]["modality_axis"] == "adversarial_perturbation"]

    assert sorted(kinds) == sorted(dataset_audit.PERTURBATIONS)


def test_the_audit_records_a_hash_of_the_file_it_checked():
    assert dataset_audit.sha256(dataset_audit.SLICE) == dataset_audit.sha256(
        selection_slice.SELECTION_SLICE_PATH)
    assert len(dataset_audit.sha256(dataset_audit.SLICE)) == 64


def test_the_critic_may_only_answer_pass_or_defect():
    """The reviewer's vocabulary is two words; anything else is a harness fault."""
    assert "PASS" in dataset_audit.CRITIC_SYSTEM
    assert "DEFECT" in dataset_audit.CRITIC_SYSTEM
    assert "never rewrite" in dataset_audit.CRITIC_SYSTEM.lower()
