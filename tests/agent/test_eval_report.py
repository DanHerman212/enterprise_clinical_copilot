"""Unit tests for rag/eval_report.py — the artifact generator is pure stdlib,
so the report shape is testable without any cloud dependency."""

import csv

import pytest

from services.mcp.retrieval.eval_report import EvalResult, render_html, write_artifacts


def _passing_result() -> EvalResult:
    return EvalResult(
        corpus="mimic",
        index_name="rag-tree-ah-20260902",
        data_fingerprint="abc123",
        config_hash="def456",
        num_queries=2,
        metrics={"recall_at_10": 0.97, "empty_result_rate": 0.01},
        thresholds={"recall_at_10": 0.90},
        max_thresholds={"empty_result_rate": 0.05},
        per_query=[
            {"query_id": 0, "text": "q0", "recall_at_10": 1.0},
            {"query_id": 1, "text": "q1", "recall_at_10": 0.94},
        ],
        examples=[{"text": "q0", "top_passage": "passage 0"}],
    )


def test_passing_verdict_and_html():
    result = _passing_result()
    passed, failing = result.verdict()
    assert passed is True
    assert failing == []
    html = render_html(result)
    assert "PASS" in html
    assert "rag-tree-ah-20260902" in html


def test_failing_metric_flags_and_csv():
    result = _passing_result()
    result.metrics["recall_at_10"] = 0.70  # below 0.90 threshold
    result.per_query[1]["recall_at_10"] = 0.70

    passed, failing = result.verdict()
    assert passed is False
    assert failing == ["recall_at_10"]

    failures = result.failures()
    assert [f["query_id"] for f in failures] == [1]


def test_an_unmeasured_threshold_refuses_without_being_called_a_failure():
    """The report a reviewer opens must not read as a corpus verdict.

    A threshold whose number is absent is neither satisfied nor breached. It
    used to be evaluated as infinity and reported as a failed metric, so the
    gate said "the corpus failed" about a corpus that was never measured — the
    measurement component was in a stale image (2026-09-18).
    """
    result = _passing_result()
    del result.metrics["empty_result_rate"]

    passed, failing = result.verdict()
    assert passed is False, "fail-closed: an unevaluated threshold cannot pass"
    assert failing == [], "nothing was measured, so nothing was breached"
    assert result.unmeasured() == ["empty_result_rate"]


def test_the_html_gives_an_unmeasured_threshold_a_row(tmp_path):
    """A missing metric used to vanish from the table, not appear in it.

    The rows were built from the measured metrics, so a threshold with no
    measurement was invisible: the reader saw a shorter table rather than a
    missing number, and nothing said a threshold had never been evaluated.
    """
    result = _passing_result()
    del result.metrics["empty_result_rate"]

    html = render_html(result)
    assert "not measured" in html
    assert "empty_result_rate" in html, "the row has to name the threshold"
    assert "FAIL" in html, "the run still refuses"


def test_write_artifacts_produces_all_three(tmp_path):
    result = _passing_result()
    paths = write_artifacts(result, tmp_path)

    assert (tmp_path / "eval_report.html").exists()
    assert (tmp_path / "eval_results.json").exists()
    assert (tmp_path / "failures.csv").exists()

    # results.json is machine-readable and carries the verdict.
    import json
    payload = json.loads((tmp_path / "eval_results.json").read_text())
    assert payload["passed"] is True
    assert payload["failing_metrics"] == []
    assert payload["unmeasured_metrics"] == []

    # failures.csv has a header even when there are no failures.
    with (tmp_path / "failures.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["query_id", "text", "recall_at_10", "reason"]
    assert len(rows) == 1  # header only


def test_html_escapes_user_text():
    result = _passing_result()
    result.examples[0]["text"] = "<script>alert(1)</script>"
    html = render_html(result)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_ratio_vs_count_formatting():
    result = EvalResult(
        corpus="demo",
        index_name="ingest",
        num_queries=454,
        metrics={
            "vector_count": 454.0,
            "embedding_dims": 768.0,
            "unique_id_ratio": 1.0,
            "empty_embedding_count": 0.0,
        },
        thresholds={"unique_id_ratio": 1.0},
        max_thresholds={"empty_embedding_count": 0.0},
        ratio_metrics=("unique_id_ratio",),
    )
    html = render_html(result)
    assert "vector_count</td><td>454</td>" in html
    assert "embedding_dims</td><td>768</td>" in html
    assert "unique_id_ratio</td><td>100.0%</td>" in html
    assert "empty_embedding_count</td><td>0</td>" in html
