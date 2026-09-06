"""Eval report generator for the RAG pipeline.

Pure standard library, no cloud calls — same constraint as the rest of ``rag/``.
Takes a structured eval result and renders the three artifacts a stakeholder
review needs:

  * ``eval_report.html``  — self-contained HTML (inline CSS, no CDN) with an
    executive summary, per-metric pass/fail tables, the failure list, and a
    lineage block. Opens offline; safe under a strict CSP.
  * ``eval_results.json`` — machine-readable results for regression diffing.
  * ``failures.csv``      — the failed queries as a human-review queue.

The HTML is deliberately dependency-free: a stakeholder must be able to open
the file directly, with no external requests, and still read the full story.
"""

from __future__ import annotations

import csv
import html
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EvalResult:
    """One RAG eval run, with thresholds and the raw per-query detail."""

    corpus: str
    index_name: str
    data_fingerprint: str = ""
    config_hash: str = ""
    num_queries: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)      # higher is better (>=)
    max_thresholds: dict[str, float] = field(default_factory=dict)  # lower is better (<=)
    # Per-query detail. Each entry carries the metric values; a query whose
    # primary metric falls below threshold is a "failure".
    per_query: list[dict[str, Any]] = field(default_factory=list)
    primary_metric: str = "recall_at_10"
    examples: list[dict[str, Any]] = field(default_factory=list)
    generated_at_utc: str = field(default_factory=_now_utc)
    # Metrics that are rates in [0, 1] (render as percentages); everything
    # else is a raw count/dimension and renders as a plain number.
    ratio_metrics: tuple[str, ...] = ()

    def verdict(self) -> tuple[bool, list[str]]:
        """(passed, failing metrics)."""
        failing = [
            name for name, minimum in self.thresholds.items()
            if self.metrics.get(name, -1.0) < minimum
        ]
        failing += [
            name for name, maximum in self.max_thresholds.items()
            if self.metrics.get(name, float("inf")) > maximum
        ]
        return (not failing), failing

    def failures(self) -> list[dict[str, Any]]:
        """Per-query rows that missed the primary threshold."""
        minimum = self.thresholds.get(self.primary_metric)
        if minimum is None:
            return []
        out = []
        for row in self.per_query:
            if row.get(self.primary_metric, -1.0) < minimum:
                out.append(row)
        return out


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def _num(value: Any) -> str:
    """Format a raw count/dimension (not a rate) as a plain number."""
    try:
        f = float(value)
        return str(int(f)) if f.is_integer() else f"{f:.4g}"
    except (TypeError, ValueError):
        return "—"


def render_html(result: EvalResult) -> str:
    """Render the self-contained HTML report."""
    passed, failing = result.verdict()
    verdict_badge = (
        '<span class="badge pass">PASS</span>'
        if passed else '<span class="badge fail">FAIL</span>'
    )
    rows = ""
    for name in result.metrics:
        if name in result.thresholds:
            op, limit = "&ge;", result.thresholds[name]
        elif name in result.max_thresholds:
            op, limit = "&le;", result.max_thresholds[name]
        else:
            op, limit = None, None
        status = ("✅" if name not in failing else "❌") if op else "·"
        cell_cls = ("ok" if name not in failing else "bad") if op else ""
        is_rate = name in result.ratio_metrics
        fmt = _pct if is_rate else _num
        thr = f"{op} {fmt(limit)}" if op else "—"
        rows += (
            f"<tr><td>{_esc(name)}</td><td>{fmt(result.metrics.get(name))}</td>"
            f"<td>{thr}</td><td class=\"{cell_cls}\">{status}</td></tr>"
        )
    failures = result.failures()
    failure_rows = "".join(
        f"<tr><td>{_esc(row.get('query_id', '?'))}</td>"
        f"<td>{_esc(row.get('text', '')[:120])}</td>"
        f"<td>{_pct(row.get(result.primary_metric))}</td>"
        f"<td>{_esc(row.get('reason', ''))}</td></tr>"
        for row in failures
    ) or "<tr><td colspan=\"4\" class=\"ok\">No failures.</td></tr>"
    examples_html = "".join(
        f"<div class=\"example\"><div class=\"q\">{_esc(e.get('text', ''))}</div>"
        f"<div class=\"a\">{_esc(e.get('top_passage', '')[:400])}</div></div>"
        for e in result.examples
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>RAG eval — {_esc(result.index_name)}</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:2rem;color:#1a1a1a}}
h1{{font-size:1.4rem}} h2{{font-size:1.1rem;margin-top:1.6rem}}
table{{border-collapse:collapse;width:100%;margin-top:.5rem}}
td,th{{border:1px solid #ddd;padding:.4rem .6rem;text-align:left;font-size:.9rem}}
th{{background:#f5f5f5}}
.badge{{display:inline-block;padding:.2rem .6rem;border-radius:.25rem;font-weight:600}}
.pass{{background:#d9f2d9;color:#1e6b1e}} .fail{{background:#fadbd8;color:#a11}}
.ok{{color:#1e6b1e}} .bad{{color:#a11}}
.meta{{color:#555;font-size:.85rem}} .example{{border:1px solid #eee;padding:.6rem;margin:.5rem 0}}
.q{{font-weight:600}} .a{{color:#444;margin-top:.3rem}}
</style></head><body>
<h1>RAG evaluation report {verdict_badge}</h1>
<div class="meta">index <b>{_esc(result.index_name)}</b> · corpus {_esc(result.corpus)}
 · {result.num_queries} queries · generated {_esc(result.generated_at_utc)}</div>

<h2>Metrics</h2>
<table><tr><th>metric</th><th>value</th><th>threshold</th><th>status</th></tr>{rows}</table>

<h2>Failures ({len(failures)})</h2>
<table><tr><th>query</th><th>text</th><th>{_esc(result.primary_metric)}</th><th>reason</th></tr>{failure_rows}</table>

<h2>Annotated examples</h2>{examples_html}

<h2>Lineage</h2>
<div class="meta">
data_fingerprint <code>{_esc(result.data_fingerprint)}</code><br>
config_hash <code>{_esc(result.config_hash)}</code><br>
index <code>{_esc(result.index_name)}</code>
</div>
</body></html>"""


def write_report_files(
    result: EvalResult,
    *,
    html_path: Path | str,
    results_path: Path | str,
    failures_path: Path | str,
) -> None:
    """Write the three report files to explicit paths (one per artifact)."""
    Path(html_path).write_text(render_html(result), encoding="utf-8")

    payload = asdict(result)
    payload["passed"], payload["failing_metrics"] = result.verdict()
    Path(results_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    failures = result.failures()
    with Path(failures_path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["query_id", "text", result.primary_metric, "reason"]
        )
        writer.writeheader()
        for row in failures:
            writer.writerow({
                "query_id": row.get("query_id", ""),
                "text": row.get("text", ""),
                result.primary_metric: row.get(result.primary_metric, ""),
                "reason": row.get("reason", ""),
            })


def write_artifacts(result: EvalResult, out_dir: Path | str) -> dict[str, str]:
    """Write report.html, results.json, failures.csv into a directory.

    Convenience wrapper for local/deploy use; KFP components use
    :func:`write_report_files` so each file is its own artifact.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "html": out / "eval_report.html",
        "json": out / "eval_results.json",
        "csv": out / "failures.csv",
    }
    write_report_files(
        result,
        html_path=paths["html"],
        results_path=paths["json"],
        failures_path=paths["csv"],
    )
    return {k: str(v) for k, v in paths.items()}
