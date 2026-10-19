"""Phase 02: report aggregation keeps float scores, and keeps the suites apart.

Three regressions are pinned here, each one a bug that was invisible in a report:

1. **Floats dropped.** Consensus scores a dimension with the median of its samples, so
   a split sample set produces 1.5. The aggregator tested `isinstance(v, int)`, which
   is `False` for 1.5 — the score vanished from the denominator instead of counting as
   the partial result it is, and a dimension judged hundreds of times reported
   `total: 0`.

2. **Populations pooled.** The census is a fixed denominator the release gate reads;
   Suite 2 is thirty probes aimed at coverage and boundaries. Judged into one pool, a
   clean modality run can lift a weak census, and the number stops meaning anything.

3. **Corrupt merge.** Two JSONL files joined as bytes lose a record when the first has
   no trailing newline, and the judge parses its input with a bare `json.loads`.

Run: `./.venv/bin/python -m pytest tests/agent/test_judge_float_aggregation.py -q`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "evaluation" / "agent"))

import entrypoint  # noqa: E402
import judge  # noqa: E402


def _judged(verdict: str = "PASS", suite: str = "census", **dims) -> dict:
    """One judged row with every dimension at 2 unless overridden."""
    dimensions = {name: 2 for name in judge.DIMS}
    dimensions.update(dims)
    rec = {
        "hadm_id": 90000015,
        "prompt": "suite2" if suite == "suite2" else "summary",
        "turn": 1,
        "answer": "the answer under review",
        "judge": {"dimensions": dimensions, "verdict": verdict},
    }
    if suite == "suite2":
        rec["id"] = "S2-01"
        rec["suite"] = "suite2"
    return rec


# --- 1. floats ---------------------------------------------------------------


def test_a_split_sample_counts_as_a_fail_not_as_nothing():
    """1.5 is a real score. The old `isinstance(v, int)` dropped it entirely."""
    out = judge._aggregate([_judged(grounding=1.5)])
    grounding = out["census"]["dimensions"]["grounding"]
    assert grounding["total"] == 1
    assert grounding["fail"] == 1
    assert grounding["pass"] == 0


def test_float_two_is_a_pass():
    """`2.0` is a median that rounded to whole numbers; it must not be dropped."""
    out = judge._aggregate([_judged(grounding=2.0)])
    assert out["census"]["dimensions"]["grounding"] == {
        "pass": 1, "fail": 0, "total": 1, "pass_rate": 1.0,
    }


def test_mixed_int_and_float_aggregate_together():
    """The case the bug hid: half the samples split, and the rate is 1 of 2."""
    out = judge._aggregate([_judged(grounding=2), _judged(grounding=1.5)])
    grounding = out["census"]["dimensions"]["grounding"]
    assert (grounding["pass"], grounding["fail"], grounding["total"]) == (1, 1, 2)
    assert grounding["pass_rate"] == 0.5


def test_a_bool_is_not_a_score():
    """`isinstance(True, int)` is True, so the check has to exclude bool explicitly."""
    out = judge._aggregate([_judged(safety=True)])
    assert out["census"]["dimensions"]["safety"]["total"] == 0


# --- 2. populations ----------------------------------------------------------


def test_suite2_does_not_enter_the_census_block():
    """The gate reads the census; a suite-2 row must not move its denominator."""
    out = judge._aggregate([_judged(), _judged(suite="suite2")])
    assert out["census"]["scored"] == 1
    assert out["suite2"]["scored"] == 1
    assert out["census"]["dimensions"]["fidelity"]["total"] == 1
    assert out["suite2"]["dimensions"]["fidelity"]["total"] == 1


def test_both_populations_are_always_present():
    """A report reader indexes `by_suite["suite2"]` without asking if it exists."""
    out = judge._aggregate([_judged()])
    assert set(out) >= {"census", "suite2"}
    assert out["suite2"]["verdict"]["pass_rate"] is None


def test_suite2_safety_failures_stay_visible():
    """Safety is absolute, so a suite-2 boundary crossing is not reclassified away."""
    out = judge._aggregate([_judged(), _judged(suite="suite2", safety=0)])
    assert out["census"]["safety_failures"] == 0
    assert out["suite2"]["safety_failures"] == 1


def test_suite2_safety_does_not_reach_the_gated_number():
    """Regression: the 2026-10-19 smoke run failed here.

    Census safety was 0 and suite 2 safety was 4; summing them made the release gate
    read 4 and fail a clinical run whose own population had no safety failure at all.
    """
    by_suite = judge._aggregate([_judged(), _judged(suite="suite2", safety=0)])
    report = judge._build_report([{"hadm_id": 1}], by_suite)
    assert report["safety_failures"] == 0, "the gate must read census safety only"
    assert report["by_suite"]["suite2"]["safety_failures"] == 1, "still published"
    assert report["scored"] == 1, "the suite-2 row must not enter the census count"


def test_an_error_is_counted_against_its_own_population():
    failed = {"hadm_id": 1, "prompt": "suite2", "turn": 1, "id": "S2-02",
              "suite": "suite2", "error": "boom", "judge": {"error": "agent run failed"}}
    out = judge._aggregate([_judged(), failed])
    assert out["census"]["agent_errors"] == 0
    assert out["suite2"]["agent_errors"] == 1
    assert out["suite2"]["scored"] == 0


def test_the_aggregate_still_reports_a_plain_int_report_correctly():
    """The ordinary case, so the fix cannot be a special case for floats."""
    out = judge._aggregate([_judged(verdict="FAIL", fidelity=0)])
    assert out["census"]["verdict"] == {"pass": 0, "fail": 1, "pass_rate": 0.0}
    assert out["census"]["dimensions"]["fidelity"]["fail"] == 1


# --- resume key --------------------------------------------------------------


def test_suite2_probes_are_not_collapsed_into_one_on_resume():
    """Every suite-2 case shares prompt "suite2" and several share an admission."""
    a = {"hadm_id": 90000015, "prompt": "suite2", "turn": 1, "id": "S2-01"}
    b = {"hadm_id": 90000015, "prompt": "suite2", "turn": 1, "id": "S2-02"}
    assert judge._resume_key(a) != judge._resume_key(b)
    census_a = {"hadm_id": 90000015, "prompt": "summary", "turn": 1}
    census_b = {"hadm_id": 90000015, "prompt": "summary", "turn": 2}
    assert judge._resume_key(census_a) != judge._resume_key(census_b)


# --- 3. merge + expected refusals --------------------------------------------


def test_expected_refusal_is_recognised():
    rec = {"contract_refusal": True, "status": 400, "error": "unknown_command",
           "expect": {"status": 400, "error": "unknown_command"}}
    assert entrypoint._expected_refusal(rec) is True


def test_a_refusal_the_case_did_not_predict_is_still_judged():
    """An unexpected 400 is a real failure; excluding it would hide the failure."""
    rec = {"contract_refusal": True, "status": 400, "error": "unknown_command",
           "expect": {"status": 200}}
    assert entrypoint._expected_refusal(rec) is False


def test_merge_survives_a_file_with_no_trailing_newline(tmp_path, monkeypatch):
    """Byte concatenation corrupts the boundary line; parse-then-emit cannot."""
    traces = tmp_path / "traces.jsonl"
    traces.write_text(json.dumps({"hadm_id": 1, "prompt": "summary", "turn": 1}))
    suite2 = tmp_path / "suite2.jsonl"
    suite2.write_text(json.dumps({"hadm_id": 2, "prompt": "suite2", "id": "S2-01",
                                  "turn": 1}) + "\n")
    monkeypatch.setattr(entrypoint, "RESULTS", tmp_path)

    out = entrypoint._merge_for_judging({"traces": traces, "suite2": suite2}, "unit")
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert [r["hadm_id"] for r in rows] == [1, 2]


def test_merge_drops_expected_refusals_only_from_suite2(tmp_path, monkeypatch):
    refusal = {"hadm_id": 3, "prompt": "suite2", "id": "S2-16", "turn": 1,
               "contract_refusal": True, "status": 400, "error": "unknown_command",
               "expect": {"status": 400, "error": "unknown_command"}}
    traces = tmp_path / "traces.jsonl"
    traces.write_text(json.dumps({"hadm_id": 1, "prompt": "summary", "turn": 1}) + "\n")
    suite2 = tmp_path / "suite2.jsonl"
    suite2.write_text(json.dumps(refusal) + "\n")
    monkeypatch.setattr(entrypoint, "RESULTS", tmp_path)

    out = entrypoint._merge_for_judging({"traces": traces, "suite2": suite2}, "unit")
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert [r["hadm_id"] for r in rows] == [1]


def test_adversarial_expectations_are_never_filtered(tmp_path, monkeypatch):
    """The filter is scoped to suite 2; applying it to probes would shrink a 20-run."""
    probe = {"hadm_id": 5, "prompt": "adversarial", "family": "fab", "turn": 1,
             "contract_refusal": True, "status": 400, "error": "bad_request",
             "expect": {"status": 400, "error": "bad_request"}}
    traces = tmp_path / "adv.jsonl"
    traces.write_text(json.dumps(probe) + "\n")
    monkeypatch.setattr(entrypoint, "RESULTS", tmp_path)

    out = entrypoint._merge_for_judging({"traces": traces, "suite2": None}, "unit")
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(rows) == 1


# --- fixtures ----------------------------------------------------------------


def test_cases_dir_override_is_honoured(tmp_path, monkeypatch):
    (tmp_path / "suite2_cases.json").write_text('{"cases": []}')
    monkeypatch.setenv("EVAL_CASES_DIR", str(tmp_path))
    assert entrypoint._fixture("suite2_cases.json") == tmp_path / "suite2_cases.json"


def test_missing_fixture_falls_back_to_the_image_copy(tmp_path, monkeypatch):
    """A mount that is empty must not leave a run with nothing to evaluate."""
    monkeypatch.setenv("EVAL_CASES_DIR", str(tmp_path))
    assert entrypoint._fixture("suite2_cases.json") == entrypoint.HERE / "suite2_cases.json"


@pytest.mark.parametrize("name", ["suite2_cases.json", "adversarial_cases.json"])
def test_shipped_fixtures_still_parse(name):
    payload = json.loads((entrypoint.HERE / name).read_text())
    assert payload["cases"], f"{name} carries no cases"
