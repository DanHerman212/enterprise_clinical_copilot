"""Langfuse scores carry judgements, not telemetry.

Sprint 03 wrote token counts and USD prices onto traces as scores so the Datasets view
could be sorted by cost. The cost of that was a Scores table where a reader comparing
`grounding` had to skip six `agent_*` rows and six `judge_*` rows per item, and a price in
the same column as a clinical judgement. Sprint 04d removes the writers and keeps the
numbers locally: the run summary, the diagnostics JSONL and the report.

These tests exist so the writers cannot come back quietly, and so the local accounting
they replaced is proven to still work.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from evaluation.platform import evaluators as ev
from evaluation.platform import run_evals

REPO = Path(__file__).resolve().parents[2]
PLATFORM = REPO / "evaluation" / "platform"

TELEMETRY_PREFIXES = ("agent_", "judge_")


def _telemetry_names(names) -> list[str]:
    return [name for name in names if str(name).startswith(TELEMETRY_PREFIXES)]


class _Recorder:
    """A Langfuse client stand-in that remembers every score it was asked to write."""

    def __init__(self) -> None:
        self.scores: list[dict] = []

    def create_score(self, **kwargs) -> None:  # noqa: D401 - a stub, not a writer
        self.scores.append(kwargs)

    def flush(self) -> None:
        return None

    @property
    def api(self):  # pragma: no cover - only reached by a priced item
        raise AssertionError("this test never reads a trace")


class _Outcome:
    """One item, priced or not."""

    def __init__(self, item_id: str, *, trace_id: str | None, output: dict,
                 evaluations: list | None = None) -> None:
        self.item_id = item_id
        self.trace_id = trace_id
        self.output = output
        self.evaluations = evaluations or []


def test_the_judge_composite_scores_judgements_and_nothing_else():
    def judge_fn(**_kwargs):
        return {"dimensions": {"grounding": 3, "citation": 3, "fidelity": 3,
                               "responsiveness": 3, "safety": 3, "context": 3},
                "verdict": "PASS", "reason": "ok", "flags": [],
                "_usage": {"input": 100, "output": 10, "thinking": 5, "total": 110},
                "_model": "xai/grok-4.7"}

    composite = ev.make_judge_composite(judge_fn)
    produced = composite(input={"scenario_id": "x"}, output={"langfuse_trace_id": "t1"})

    names = [evaluation.name for evaluation in produced]
    assert not _telemetry_names(names), names
    assert "verdict" in names or "grounding" in names


def test_the_grading_cost_is_recorded_in_process_not_on_the_trace():
    """Written by the same function that scores the item, keyed by the trace it graded."""
    ev.JUDGE_CHANNEL.clear()
    ev._record_judge_channel(
        {"langfuse_trace_id": "trace-abc"},
        {"_usage": {"input": 1_000, "output": 100}, "_model": "xai/grok-4.7"})

    assert "trace-abc" in ev.JUDGE_CHANNEL
    assert ev.JUDGE_CHANNEL["trace-abc"]["model"] == "xai/grok-4.7"


def test_a_run_writes_no_cost_scores_and_still_reports_its_totals():
    client = _Recorder()
    outcomes = [_Outcome("v2_d1_slash_01", trace_id=None, output={})]

    totals = run_evals._record_costs(client, outcomes, dataset_name="slice")

    assert client.scores == []
    assert totals["agent_cost_usd"] == 0.0
    assert totals["judge_cost_usd"] == 0.0
    assert totals["unattributed"] == 1
    assert "agent_input_tokens" in totals


def test_the_judge_channel_prices_from_the_record_it_is_handed():
    ev.JUDGE_CHANNEL.clear()
    ev.JUDGE_CHANNEL["trace-1"] = {
        "usage": {"input": 1_000_000, "output": 0, "thinking": 0, "total": 1_000_000},
        "model": "xai/grok-4.7",
    }
    outcome = _Outcome("v2_d1_free_01", trace_id="trace-1", output={})

    channel = run_evals._judge_channel(outcome)

    assert channel["cost_usd"] == 2.00  # 1,000,000 input tokens at 2.00 per million
    assert channel["usage"]["input"] == 1_000_000
    assert "trace-1" not in ev.JUDGE_CHANNEL, "the record is consumed, not accumulated"


def test_an_unrecorded_grading_reports_unattributed_rather_than_zero():
    ev.JUDGE_CHANNEL.clear()
    outcome = _Outcome("v2_d1_free_01", trace_id="trace-2", output={})

    channel = run_evals._judge_channel(outcome)

    assert channel["cost_usd"] is None


def test_the_run_level_scores_are_the_gate_terms_and_nothing_else():
    """No `judge_errors`, no prices: the gate terms a reader can act on."""
    builder = ev.make_run_evaluators(min_pass_rate=0.85)[0]
    produced = builder(stats={"pass_rate": 1.0, "safety_failures": 0,
                              "structural_failures": 0, "unscored": 2,
                              "unattributed": 0, "passed": 16, "scored": 16})

    names = [evaluation.name for evaluation in produced]
    assert "judge_errors" not in names
    assert not _telemetry_names(names), names
    assert set(names) <= {"pass_rate", "safety_failures", "structural_failures",
                          "unattributed", "gate_pass"}


def test_the_unscored_count_still_reaches_the_gate_without_a_score_row():
    """The count is what blocks a gate; the row was only ever a display of it."""
    stats = {"pass_rate": 1.0, "safety_failures": 0, "structural_failures": 0,
             "unscored": 3, "unattributed": 0, "passed": 13, "scored": 13}

    assert ev.gate_verdict(stats, 0.85) is False


def test_no_module_in_the_platform_writes_a_telemetry_score():
    """A source scan, because the failure mode is a writer added back quietly."""
    offenders: list[str] = []
    for path in sorted(PLATFORM.glob("*.py")):
        source = path.read_text()
        if "create_score(" not in source:
            continue
        for line in source.splitlines():
            if "create_score(" not in line:
                continue
            window = line + source.split(line, 1)[1][:400]
            if any(f'name="{prefix}' in window or f"name='{prefix}" in window
                   or f'"{{prefix}}' in window or "cost_evaluations(" in window
                   for prefix in TELEMETRY_PREFIXES):
                offenders.append(f"{path.name}: {line.strip()}")
    assert not offenders, offenders


def test_the_cost_functions_still_exist_for_local_reporting():
    """The numbers are kept; only the score writers are gone."""
    assert callable(ev.cost_evaluations)
    source = inspect.getsource(run_evals)
    assert "agent_cost_usd" in source
    assert "unattributed" in source
