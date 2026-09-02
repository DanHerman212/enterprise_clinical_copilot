"""Unit tests for rag/gate.py — the online recall gate decision is pure."""

from rag.gate import recall_result

_REPORT = {
    "num_queries": 2,
    "recall": {"@1": 0.80, "@5": 0.90, "@10": 0.95},
    "empty_result_rate": 0.01,
    "per_query": [
        {"q": 0, "@1": 1.0, "@5": 1.0, "@10": 1.0},
        {"q": 1, "@1": 0.6, "@5": 0.8, "@10": 0.9},
    ],
}


def test_passing_gate():
    result = recall_result(
        report=_REPORT, corpus="demo", index_name="rag-tree-ah-x",
        recall_min={"recall_at_10": 0.90},
        empty_max={"empty_result_rate": 0.05},
    )
    passed, failing = result.verdict()
    assert passed is True
    assert failing == []
    assert result.metrics["recall_at_10"] == 0.95


def test_failing_recall_gate():
    result = recall_result(
        report=_REPORT, corpus="demo", index_name="rag-tree-ah-x",
        recall_min={"recall_at_10": 0.99},  # above the measured 0.95
    )
    passed, failing = result.verdict()
    assert passed is False
    assert failing == ["recall_at_10"]
    # the failing per-query rows are surfaced for the review queue
    assert [f["query_id"] for f in result.failures()] == [1]


def test_empty_result_rate_is_lower_better():
    result = recall_result(
        report=_REPORT, corpus="demo", index_name="rag-tree-ah-x",
        recall_min={"recall_at_10": 0.90},
        empty_max={"empty_result_rate": 0.005},  # measured 0.01 > 0.005
    )
    passed, failing = result.verdict()
    assert passed is False
    assert "empty_result_rate" in failing
