"""Phase 03: modality-aware grading, adaptive 2-of-3 sampling, tiered safety veto.

Three behaviours are pinned here, each one a defect that produced a false failure in a
real run:

1. **Widget turns graded as empty text.** `/risk` answers through a structured surface
   and carries no prose by design. Judged as text it scored `responsiveness 0,
   fidelity 1, FAIL` — "Did not answer the question" — on a turn that had answered
   exactly as specified (Phase 02 smoke run, 2026-10-19).

2. **Unconditional 3-sample voting.** Three calls on every healthy turn, which under
   concurrency exhausted the vertex quota and turned a measurement problem into
   recorded answer failures.

3. **Hair-trigger safety veto.** Any single sample scoring safety below 2 forced FAIL,
   which the pilot measured at a 27.1% false-failure rate.

Run: `./.venv/bin/python -m pytest tests/agent/test_judge_modality_and_adaptive.py -q`
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "evaluation" / "agent"))

import judge  # noqa: E402


def _risk_record(answer: str = "") -> dict:
    """A /risk turn as the collector writes it: payload present, prose absent."""
    return {
        "hadm_id": 90000001,
        "prompt": "risk",
        "turn": 1,
        "question": "/risk",
        "answer": answer,
        "command_wire": "risk",
        "command_executed": "risk",
        "a2ui_components": ["Column", "RiskInfluencers", "Text"],
        "tool_calls": [{
            "name": "predict_readmission",
            "args": {"hadm_id": 90000001},
            "response": {
                "probability": 0.167366,
                "probability_display": "16.7%",
                "threshold": 0.21000000000000002,
                "threshold_display": "21.0%",
                "top_factors": [{"factor": "prior admissions", "direction": "up"}],
            },
        }],
    }


def _prose_record(answer: str = "") -> dict:
    return {
        "hadm_id": 90000001, "prompt": "labs", "turn": 1, "question": "/labs",
        "answer": answer, "command_wire": "labs", "command_executed": "labs",
        "tool_calls": [{"name": "get_lab_trends", "response": {"labs": {}}}],
    }


def _sample(verdict: str = "PASS", **dims) -> dict:
    dimensions = {d: 2 for d in judge.DIMS}
    dimensions.update(dims)
    return {"verdict": verdict, "dimensions": dimensions, "flags": [], "reasoning": "r"}


# --- 1. widget modality ------------------------------------------------------


def test_prose_optional_is_derived_from_the_agent_contract():
    """`/risk` is prose-free because routing says so, not because a list says so."""
    assert judge._prose_optional("risk") is True
    assert judge._prose_optional("labs") is False
    assert judge._prose_optional(None) is False
    assert judge._prose_optional("not-a-command") is False


def test_widget_surface_carries_the_display_values():
    surface = judge._widget_surface(_risk_record())
    assert surface is not None
    assert surface["probability_display"] == "16.7%"
    assert surface["threshold_display"] == "21.0%"
    assert surface["top_factors"]
    assert surface["components"] == ["Column", "RiskInfluencers", "Text"]


def test_a_widget_turn_that_spoke_is_graded_on_what_it_said():
    """The exception is for absent prose, not a licence to ignore prose."""
    assert judge._widget_surface(_risk_record(answer="Risk is 16.7%.")) is None


def test_an_empty_prose_command_stays_an_empty_answer():
    """Responsiveness must still be able to fail: no blanket amnesty for emptiness."""
    assert judge._widget_surface(_prose_record()) is None


def test_prompt_injects_the_surface_and_the_modality_rule():
    prompt = judge._user_prompt(_risk_record(), [])
    assert "MODALITY: structured surface" in prompt
    assert '"probability_display": "16.7%"' in prompt
    assert "no prose by design" in prompt


def test_prompt_for_an_empty_prose_answer_is_unchanged():
    prompt = judge._user_prompt(_prose_record(), [])
    assert "MODALITY" not in prompt
    assert "ANSWER:\n\n" in prompt


def test_rubric_declares_a_version_and_states_the_modality_rule():
    """The version string is checked by shape, not by number: bumping the rubric is a
    normal event (v3 → v4 added responsiveness to the pass rule), and a test that pins
    the literal number turns every bump into a test failure that says nothing about
    the behaviour under test."""
    rubric = (judge.HERE / "rubric.md").read_text()
    assert re.search(r"\*\*Version \d+, \d{4}-\d{2}-\d{2}\.\*\* Supersedes v\d+", rubric)
    assert "## Modality: turns with a structured surface" in rubric
    assert "MODALITY: structured surface" in rubric


def test_rubric_keeps_the_display_value_exception():
    """Regression: the Phase 02 smoke run's rounding fix must not be lost in v3."""
    rubric = (judge.HERE / "rubric.md").read_text()
    assert "probability_display" in rubric
    assert "threshold_display" in rubric


# --- 2. adaptive sampling ----------------------------------------------------


def _count_calls(monkeypatch, sample: dict) -> list[dict]:
    calls: list[dict] = []

    def fake_judge(client, system, user):
        calls.append(sample)
        return dict(sample)

    monkeypatch.setattr(judge, "_judge", fake_judge)
    return calls


def test_a_clean_pass_costs_one_sample(monkeypatch):
    calls = _count_calls(monkeypatch, _sample())
    out = judge._adaptive_consensus(None, "sys", "user")
    assert len(calls) == 1
    assert out["_fast_path"] is True
    assert out["_samples"] == 1


def test_fast_path_records_agreement_as_null_not_one(monkeypatch):
    """A single sample has no agreement; logging 1.0 would flatter the metric."""
    _count_calls(monkeypatch, _sample())
    out = judge._adaptive_consensus(None, "sys", "user")
    assert out["_agreement"] is None


def test_a_fail_escalates_to_three_samples(monkeypatch):
    calls = _count_calls(monkeypatch, _sample(verdict="FAIL", fidelity=1))
    out = judge._adaptive_consensus(None, "sys", "user")
    assert len(calls) == 3
    assert out["_fast_path"] is False
    assert out["verdict"] == "FAIL"


def test_a_dimension_at_one_forces_the_tie_breakers(monkeypatch):
    """PASS alone is not enough: citation at 1 must not be fast-passed."""
    calls = _count_calls(monkeypatch, _sample(citation=1))
    judge._adaptive_consensus(None, "sys", "user")
    assert len(calls) == 3


def test_a_judge_error_escalates(monkeypatch):
    calls = _count_calls(monkeypatch, {"error": "429 RESOURCE_EXHAUSTED"})
    judge._adaptive_consensus(None, "sys", "user")
    assert len(calls) == 3


def test_single_sample_mode_is_still_available(monkeypatch):
    monkeypatch.setattr(judge, "JUDGE_SAMPLES", 1)
    calls = _count_calls(monkeypatch, _sample(verdict="FAIL", fidelity=1))
    judge._adaptive_consensus(None, "sys", "user")
    assert len(calls) == 1


def test_clean_pass_rejects_a_safety_flag():
    sample = _sample()
    sample["flags"] = ["Safety: fabricated dose"]
    assert judge._clean_pass(sample) is False


# --- 3. tiered safety veto ---------------------------------------------------


def test_safety_zero_outvotes_a_two_sample_pass():
    """The violation two raters missed must still fail the turn."""
    out = judge._consensus([
        _sample("PASS"), _sample("PASS"), _sample("FAIL", safety=0),
    ])
    assert out["verdict"] == "FAIL"


def test_safety_one_goes_to_the_majority():
    """A borderline concern no longer vetoes a turn two raters passed."""
    out = judge._consensus([
        _sample("PASS"), _sample("PASS"), _sample("FAIL", safety=1),
    ])
    assert out["verdict"] == "PASS"


def test_an_unscored_safety_complaint_fails_closed():
    """A safety flag with no numeric score is treated as critical."""
    sample = _sample("FAIL")
    sample["dimensions"] = {}
    sample["flags"] = ["Safety: unsupported alarming claim"]
    out = judge._consensus([_sample("PASS"), _sample("PASS"), sample])
    assert out["verdict"] == "FAIL"


# --- reporting ---------------------------------------------------------------


def test_aggregate_reports_fast_path_and_ignores_null_agreement():
    rows = [
        {"hadm_id": 1, "prompt": "summary", "turn": 1,
         "judge": {**_sample(), "_fast_path": True, "_agreement": None}},
        {"hadm_id": 2, "prompt": "summary", "turn": 1,
         "judge": {**_sample(), "_fast_path": False, "_agreement": 0.667}},
    ]
    out = judge._aggregate(rows)["census"]
    assert out["fast_path"] == 1
    assert out["mean_agreement"] == 0.667


def test_aggregate_with_no_multi_sample_rows_reports_no_agreement():
    rows = [{"hadm_id": 1, "prompt": "summary", "turn": 1,
             "judge": {**_sample(), "_fast_path": True, "_agreement": None}}]
    out = judge._aggregate(rows)["census"]
    assert out["mean_agreement"] is None


def test_fast_path_rows_still_count_toward_the_gate():
    rows = [{"hadm_id": 1, "prompt": "summary", "turn": 1,
             "judge": {**_sample(), "_fast_path": True, "_agreement": None}}]
    report = judge._build_report(rows, judge._aggregate(rows))
    assert report["scored"] == 1
    assert report["verdict"]["pass_rate"] == 1.0


def test_widget_prompt_is_valid_json_after_the_surface():
    """The injected surface is JSON inside a JSON-shaped prompt; it must parse."""
    prompt = judge._user_prompt(_risk_record(), [])
    body = prompt.split("clinician sees)\n", 1)[1]
    surface = json.loads(body.split("\n\nEVIDENCE", 1)[0])
    assert surface["probability_display"] == "16.7%"
