"""The four harness contracts the selection-slice calibration added.

Each one is a case where the harness and the product disagreed about what the reader was
shown, and every disagreement scored a correct answer as a defect:

* a `/meds` turn's answer is its table. The harness kept the sentence and dropped the rows,
  so the judge graded a claim about where the list came from and never saw the list.
* a three-turn scenario carries its answer key in `turn_3`, and the judge's constraints
  block read only `turn_2` — so the third turn was graded against the first turn's facts.
* a `/pubmed` marker resolves to a PMID, and the sources the judge was handed carried the
  number and the abstract but not the identifier or the title.
* the v2 slice was seeded as a dataset no runner would accept.

The display-formatting changes are in `evaluation/rubric.md` and are not testable here:
they are instructions to a model, not code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evaluation.platform import judge_call, run_evals, runner, task  # noqa: E402
from evaluation.platform.evaluators import structural_check  # noqa: E402


def _medication_surface(rows: list[dict]) -> dict:
    """A `/meds` canvas, shaped as `canvas.compose` emits it."""
    return {"messages": [{"updateComponents": {"components": [
        {"id": "root", "component": "Column", "children": ["medications"]},
        {"id": "medications", "component": "MedicationGrid",
         "medications": rows, "dropped_count": 0, "returned_passages": 1},
    ]}}]}


# --- the rendered table ---------------------------------------------------------------

ROWS = [
    {"medication": "Vicodin", "dosage": "1 tablet", "frequency": "q6h prn",
     "context": "oral"},
    {"medication": "Macrodantin", "dosage": "100 mg", "frequency": "at bedtime",
     "context": "oral"},
]


def test_the_grid_rows_are_captured_per_turn():
    rendered = task._rendered({"a2ui": _medication_surface(ROWS)})
    assert [row["medication"] for row in rendered] == ["Vicodin", "Macrodantin"]
    assert rendered[0]["dosage"] == "1 tablet"
    assert rendered[1]["frequency"] == "at bedtime"


def test_a_turn_with_no_grid_captures_nothing():
    canvas = {"messages": [{"updateComponents": {"components": [
        {"id": "root", "component": "Column", "children": []},
    ]}}]}
    assert task._rendered({"a2ui": canvas}) == []
    assert task._rendered({}) == []


def test_a_rendered_row_can_satisfy_a_token_check():
    """`must_contain` reads the table as well as the sentence.

    The prose of a `/meds` answer is one sentence about provenance, so an anchor naming a
    drug could previously only be satisfied by the sentence mentioning it.
    """
    output = {"turn_1_response": "The discharge note lists two drugs.",
              "turn_responses": ["The discharge note lists two drugs."],
              "turn_rendered": [ROWS]}
    ok, failures = structural_check(
        {"expectations": {"must_contain": ["Macrodantin"]}}, output)
    assert ok, failures


def test_a_rendered_row_can_fail_a_disclosure_check():
    """And a drug that appears only in the grid is still a disclosure."""
    output = {"turn_1_response": "Two drugs are documented.",
              "turn_responses": ["Two drugs are documented."],
              "turn_rendered": [[*ROWS,
                                {"medication": "Warfarin", "dosage": "5 mg",
                                 "frequency": "daily", "context": "oral"}]]}
    ok, failures = structural_check(
        {"expectations": {"must_not_disclose": ["warfarin"]}}, output)
    assert not ok
    assert any("must_not_disclose" in failure for failure in failures), failures


def test_the_judge_is_shown_the_medication_table_beneath_the_answer():
    output = {"turn_responses": ["Two drugs are documented."],
              "turn_rendered": [ROWS]}
    prompt = judge_call.build_user_prompt({"turns": ["/meds"]}, output)
    assert "RENDERED MEDICATION TABLE" in prompt
    assert "| Vicodin | 1 tablet | q6h prn | oral |" in prompt
    # And it sits with the turn it belongs to, after that turn's answer.
    assert prompt.index("ANSWER:") < prompt.index("RENDERED MEDICATION TABLE")


def test_a_thread_without_a_table_gains_no_heading():
    prompt = judge_call.build_user_prompt(
        {"turns": ["/ehr"]}, {"turn_responses": ["A briefing."]})
    assert "RENDERED MEDICATION TABLE" not in prompt


# --- the three-turn answer key --------------------------------------------------------

THREE_TURN_KEY = {
    "facts": {"encounter": "inpatient"},
    "behaviour": "answers the first question",
    "turn_2": {"facts": {"discharge_location": "Home"}},
    "turn_3": {"facts": {"anticoagulant_documented": False},
               "behaviour": "states that no anticoagulant is documented"},
    "turn_4": {"facts": {"prior_inpatient_days": 0}},
}


def test_a_later_turns_answer_key_reaches_the_judge():
    block = judge_call._scenario_constraints({}, THREE_TURN_KEY)
    assert "anticoagulant_documented" in block
    assert "no anticoagulant is documented" in block
    assert "prior_inpatient_days" in block


def test_a_three_turn_block_is_still_emitted_when_turn_2_is_absent():
    """A 3-turn scenario may key only its later turns."""
    block = judge_call._scenario_constraints(
        {}, {"facts": {"encounter": "inpatient"}, "turn_3": {"facts": {"x": 1}}})
    assert block
    assert '"x": 1' in block


# --- the published record's identity --------------------------------------------------

PUBLISHED = {"cite": 2, "kind": "Literature", "pmid": "39165878",
             "title": "Total knee arthroplasty in the elderly",
             "section": "J Arthroplasty · 2024",
             "text": "Ninety-day readmission was 4.1%."}


def test_a_published_record_is_named_by_pmid_and_title():
    rendered = judge_call._cited_sources({"turn_sources": [[PUBLISHED]]}, 0)
    assert "[2] PubMed PMID 39165878 (Total knee arthroplasty in the elderly):" in rendered


def test_a_published_record_without_a_title_still_resolves():
    record = {key: value for key, value in PUBLISHED.items() if key != "title"}
    rendered = judge_call._cited_sources({"turn_sources": [[record]]}, 0)
    assert "[2] PubMed PMID 39165878:" in rendered


def test_a_note_passage_is_named_by_its_section():
    passage = {"cite": 1, "section": "brief_hospital_course", "text": "Admitted."}
    rendered = judge_call._cited_sources({"turn_sources": [[passage]]}, 0)
    assert rendered == "  [1] brief_hospital_course: Admitted."


def test_the_task_keeps_the_identity_fields_the_judge_needs():
    kept = task._sources({"sources": [PUBLISHED]})
    assert kept[0]["pmid"] == "39165878"
    assert kept[0]["title"] == "Total knee arthroplasty in the elderly"
    assert kept[0]["kind"] == "Literature"


def test_a_note_passage_gains_no_identifier_it_does_not_have():
    kept = task._sources({"sources": [{"cite": 1, "section": "x", "text": "y"}]})
    for absent in ("pmid", "title", "kind"):
        assert absent not in kept[0]


@pytest.mark.parametrize("value", ["", "   ", None])
def test_a_blank_identifier_is_not_kept(value):
    kept = task._sources({"sources": [
        {"cite": 1, "section": "x", "text": "y", "pmid": value, "title": value}]})
    assert "pmid" not in kept[0]
    assert "title" not in kept[0]


# --- the dataset is reachable ---------------------------------------------------------

def test_the_selection_slice_is_a_named_target():
    assert run_evals.SELECTION_SLICE_DATASET == "ecc-model-selection-slice-v2"
    # Named, not default: a bare `run_evals` keeps measuring the two Tier 2 suites.
    assert run_evals.SELECTION_SLICE_DATASET not in run_evals.DATASETS


def test_every_runner_accepts_the_same_datasets():
    """Two scopes for one set of suites is how a purge half-deleted a run."""
    assert run_evals.SELECTION_SLICE_DATASET in runner.ALLOWED_DATASETS

    from evaluation.platform import purge_run

    assert run_evals.SELECTION_SLICE_DATASET in purge_run._datasets()


def test_the_slice_is_accepted_by_the_runner_and_reaches_the_run(monkeypatch):
    """`--datasets ecc-model-selection-slice-v2` is allowed, and the name travels.

    The allowlist line and the run name are both inside `main`, so both are exercised:
    a name that is accepted but not used as the run label would attribute the run to a
    timestamp nobody can find later.
    """
    captured: dict = {}

    monkeypatch.setattr(run_evals, "verify_environment",
                        lambda: {"API_URL": "https://agent.example.invalid"})
    monkeypatch.setattr(run_evals, "served_revision", lambda url: None)
    monkeypatch.setattr(run_evals, "run",
                        lambda targets, **kwargs: captured.update(
                            targets=targets, **kwargs) or {})

    code = run_evals.main(["--datasets", run_evals.SELECTION_SLICE_DATASET,
                           "--run-name", "baseline arm"])
    assert code == 0
    assert captured["targets"] == (run_evals.SELECTION_SLICE_DATASET,)
    # Whitespace would make the name unusable as the `run:<name>` trace tag.
    assert captured["run_name"] == "baseline-arm"


def test_an_unknown_dataset_is_still_refused(monkeypatch):
    monkeypatch.setattr(run_evals, "verify_environment",
                        lambda: {"API_URL": "https://agent.example.invalid"})
    code = run_evals.main(["--datasets", "clinical-census-full-v1",
                           "--dry-payloads"])
    assert code == 2
