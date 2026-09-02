"""eval_ingest — offline integrity gate for the built ingest artifact.

Runs inside the ingest pipeline, after embed_chunks and before build_index, to
validate the artifact that is about to be indexed:

  * vector count vs expected
  * embedding dimensionality vs expected
  * no duplicate datapoint ids
  * no empty/zero embeddings

Emits the stakeholder artifacts via rag/eval_report.py (HTML + JSON + CSV) and
fails the pipeline loudly on any violation, so a silently short or corrupt
index can never ship. The report is written even on a threshold failure so the
reason is inspectable, not just an exit code.
"""

import json
from collections import Counter

from kfp import dsl

from ._image import RAG_IMAGE, component


def run_eval_ingest(
    *,
    ingest_path: str,
    expected: int,
    dimensions: int,
    report_path: str,
    results_path: str,
    failures_path: str,
    corpus: str,
    index_name: str,
) -> bool:
    """Validate the ingest and write the report. Returns True if it passes."""
    from rag.eval_report import EvalResult, write_report_files

    total = 0
    ids: Counter[str] = Counter()
    dims_seen: set[int] = set()
    empty = 0

    with open(ingest_path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            total += 1
            ids[record["id"]] += 1
            emb = record.get("embedding") or []
            dims_seen.add(len(emb))
            if not any(emb):
                empty += 1

    duplicates = sum(1 for n in ids.values() if n > 1)

    # Hard integrity violations are bugs — fail before writing a report.
    if total != expected:
        raise SystemExit(f"vector count {total} != expected {expected}")
    if dims_seen != {dimensions}:
        raise SystemExit(f"embedding dims {sorted(dims_seen)} != expected {dimensions}")
    if duplicates:
        raise SystemExit(f"{duplicates} duplicate datapoint ids")

    result = EvalResult(
        corpus=corpus,
        index_name=index_name,
        num_queries=total,
        metrics={
            "vector_count": float(total),
            "embedding_dims": float(dimensions),
            "unique_id_ratio": 1.0,
            "empty_embedding_count": float(empty),
        },
        thresholds={"unique_id_ratio": 1.0},
        max_thresholds={"empty_embedding_count": 0.0},
        ratio_metrics=("unique_id_ratio",),
    )
    write_report_files(
        result,
        html_path=report_path,
        results_path=results_path,
        failures_path=failures_path,
    )

    passed, failing = result.verdict()
    if not passed:
        raise SystemExit(f"eval gate failed: {failing}")
    print(
        f"eval_ingest: {total} vectors, {dimensions} dims, "
        f"{empty} empty, {duplicates} dup — PASS"
    )
    return passed


@component(base_image=RAG_IMAGE)
def eval_ingest(
    ingest: dsl.Input[dsl.Dataset],
    expected: int,
    dimensions: int,
    corpus: str,
    index_name: str,
    report: dsl.Output[dsl.HTML],
    results: dsl.Output[dsl.Artifact],
    failures: dsl.Output[dsl.Artifact],
) -> bool:
    """KFP component: validate the ingest and write the eval report artifacts.

    ``report`` is typed ``dsl.HTML`` (schema ``system.HTML``) so the Vertex
    Pipelines UI renders the self-contained report inline from the node,
    instead of leaving only a GCS URI to chase down.
    """
    from pipelines.components.eval_ingest import run_eval_ingest

    return run_eval_ingest(
        ingest_path=ingest.path,
        expected=expected,
        dimensions=dimensions,
        report_path=report.path,
        results_path=results.path,
        failures_path=failures.path,
        corpus=corpus,
        index_name=index_name,
    )
