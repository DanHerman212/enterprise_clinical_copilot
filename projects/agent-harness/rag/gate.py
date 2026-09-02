"""Pure gate helpers: turn a recall report into an EvalResult + pass/fail.

The online recall job (``pipelines/recall_k.py``) writes ``recall_report.json``
with ``{"recall": {"@1": …, "@5": …, "@10": …}, "per_query": […]}``. This
module folds that report plus the configured thresholds into an
:class:`rag.eval_report.EvalResult` — which renders the HTML report and decides
pass/fail — so the deploy step's gate decision is pure and unit-testable
without any cloud calls.
"""

from __future__ import annotations

from typing import Any

from rag.eval_report import EvalResult


def recall_result(
    *,
    report: dict[str, Any],
    corpus: str,
    index_name: str,
    data_fingerprint: str = "",
    config_hash: str = "",
    recall_min: dict[str, float] | None = None,
    empty_max: dict[str, float] | None = None,
) -> EvalResult:
    """Build an EvalResult from a ``recall_k.py`` report + thresholds.

    ``recall_min`` maps metric names (``recall_at_10``) to minimums (higher is
    better); ``empty_max`` maps metric names (``empty_result_rate``) to maximums
    (lower is better).
    """
    recall = report.get("recall", {})
    metrics: dict[str, float] = {
        f"recall_at_{str(k).lstrip('@')}": float(v) for k, v in recall.items()
    }
    thresholds: dict[str, float] = {
        name: float(minimum)
        for name, minimum in (recall_min or {}).items()
    }
    max_thresholds: dict[str, float] = {
        name: float(maximum)
        for name, maximum in (empty_max or {}).items()
    }
    if "empty_result_rate" in report:
        metrics["empty_result_rate"] = float(report["empty_result_rate"])

    per_query: list[dict[str, Any]] = []
    for q in report.get("per_query", []):
        row: dict[str, Any] = {"query_id": q.get("q")}
        for k, v in q.items():
            if str(k).startswith("@"):
                row[f"recall_at_{str(k).lstrip('@')}"] = float(v)
        per_query.append(row)

    return EvalResult(
        corpus=corpus,
        index_name=index_name,
        data_fingerprint=data_fingerprint,
        config_hash=config_hash,
        num_queries=int(report.get("num_queries", 0)),
        metrics=metrics,
        thresholds=thresholds,
        max_thresholds=max_thresholds,
        primary_metric="recall_at_10",
        per_query=per_query,
        # every metric in a recall_k report is a rate in [0, 1]
        ratio_metrics=tuple(metrics),
    )
