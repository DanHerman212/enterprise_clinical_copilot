"""
rag_ingest_pipeline — Vertex AI (KFP v2) RAG ingest pipeline DAG.

Wiring::

    chunk_notes ─▶ embed_chunks ─▶ build_index

All components move data only through BigQuery and GCS; nothing is stored on a
developer machine. `embed_chunks` can reuse embeddings from a previous ingest
(via `previous_ingest_uri`) so a chunking fix re-embeds only the changed
chunks instead of paying for the whole corpus again.
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import aiplatform
from kfp import compiler, dsl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.components.build_index import build_index  # noqa: E402
from pipelines.components.chunk_notes import DEFAULT_SECTIONS, chunk_notes  # noqa: E402
from pipelines.components.embed_chunks import embed_chunks  # noqa: E402

PIPELINE_NAME = "rag-ingest"
PROJECT_ID = os.environ.get("PROJECT_ID", "trim-icon-498815-a0")
LOCATION = "us-east1"
NOTES_TABLE = f"{PROJECT_ID}.mimiciv_note.discharge"
SPLIT_TABLE = f"{PROJECT_ID}.readmission.analytics_dataset_encoded"
PIPELINE_ROOT = os.environ.get(
    "PIPELINE_ROOT", f"gs://{PROJECT_ID}-mlops/pipeline-root"
)


@dsl.pipeline(
    name=PIPELINE_NAME,
)
def rag_ingest_pipeline(
    project_id: str = PROJECT_ID,
    notes_table_ref: str = NOTES_TABLE,
    split_table_ref: str = SPLIT_TABLE,
    split_name: str = "test",
    pack_to: int = 700,
    sections_csv: str = ",".join(DEFAULT_SECTIONS),
    previous_ingest_uri: str = "",
    dimensions: int = 768,
    embed_workers: int = 1,
    brute_sample: int = 2000,
    approximate_neighbors: int = 40,
    expected_vectors: int = 555770,
    # Resource limits per step. Defaults are sized for the real ~560k-chunk
    # corpus; small synthetic runs pass tiny values (e.g. "1"/"2Gi") so Vertex
    # schedules cheap machines. KFP set_* takes string quantities.
    chunk_cpu: str = "2",
    chunk_mem: str = "8Gi",
    embed_cpu: str = "8",
    embed_mem: str = "24Gi",
    index_cpu: str = "2",
    index_mem: str = "8Gi",
) -> None:
    """Assemble the RAG ingest DAG: chunk → embed → index."""
    chunks_task = chunk_notes(
        project_id=project_id,
        notes_table_ref=notes_table_ref,
        split_table_ref=split_table_ref,
        split_name=split_name,
        pack_to=pack_to,
        sections_csv=sections_csv,
    )
    chunks_task.set_cpu_limit(chunk_cpu).set_memory_limit(chunk_mem)

    ingest_task = embed_chunks(
        project_id=project_id,
        location=LOCATION,
        chunks=chunks_task.outputs["chunks"],
        previous_ingest_uri=previous_ingest_uri,
        workers=embed_workers,
    )
    # Defaults sized for the real corpus: downloads the previous ingest
    # (~4-8 GB, streamed) and holds ~560k chunk records plus file-I/O page
    # cache. cpu 8 -> e2-standard-8 (32 GB) so the 24 Gi limit has real
    # headroom; a tight limit on a 16 GB machine OOMs (observed across runs
    # 2-4). A fresh synthetic ingest (previous_ingest_uri="") needs none of
    # that and passes e.g. "2"/"4Gi".
    ingest_task.set_cpu_limit(embed_cpu).set_memory_limit(embed_mem)

    index_task = build_index(
        project_id=project_id,
        location=LOCATION,
        ingest=ingest_task.outputs["ingest"],
        dimensions=dimensions,
        brute_sample=brute_sample,
        approximate_neighbors=approximate_neighbors,
        expected=expected_vectors,
    )
    index_task.set_cpu_limit(index_cpu).set_memory_limit(index_mem)


def compile_pipeline(package_path: str = "rag_ingest_pipeline.yaml") -> str:
    """Compile the pipeline to a KFP IR YAML and return the path."""
    compiler.Compiler().compile(
        pipeline_func=rag_ingest_pipeline, package_path=package_path
    )
    return package_path


def submit() -> None:
    """Compile and submit the pipeline to Vertex AI Pipelines.

    All parameter values can be overridden via env vars so one definition
    serves both the real ~560k-chunk corpus and small synthetic runs:

      NOTES_TABLE_REF / SPLIT_TABLE_REF / SPLIT_NAME     source tables
      PREVIOUS_INGEST_URI / EMBED_WORKERS / EXPECTED_VECTORS / BRUTE_SAMPLE
      CHUNK_CPU / CHUNK_MEM / EMBED_CPU / EMBED_MEM / INDEX_CPU / INDEX_MEM
    """
    package_path = compile_pipeline()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    job = aiplatform.PipelineJob(
        display_name=f"{PIPELINE_NAME}-{ts}",
        template_path=package_path,
        pipeline_root=PIPELINE_ROOT,
        parameter_values={
            "project_id": PROJECT_ID,
            "notes_table_ref": os.environ.get("NOTES_TABLE_REF", NOTES_TABLE),
            "split_table_ref": os.environ.get("SPLIT_TABLE_REF", SPLIT_TABLE),
            "split_name": os.environ.get("SPLIT_NAME", "test"),
            "previous_ingest_uri": os.environ.get(
                "PREVIOUS_INGEST_URI",
                f"gs://{PROJECT_ID}-mlops/rag/embeddings/ingest/embed_ingest.jsonl.gz",
            ),
            "embed_workers": int(os.environ.get("EMBED_WORKERS", "1")),
            "expected_vectors": int(os.environ.get("EXPECTED_VECTORS", "555770")),
            "brute_sample": int(os.environ.get("BRUTE_SAMPLE", "2000")),
            "chunk_cpu": os.environ.get("CHUNK_CPU", "2"),
            "chunk_mem": os.environ.get("CHUNK_MEM", "8Gi"),
            "embed_cpu": os.environ.get("EMBED_CPU", "8"),
            "embed_mem": os.environ.get("EMBED_MEM", "24Gi"),
            "index_cpu": os.environ.get("INDEX_CPU", "2"),
            "index_mem": os.environ.get("INDEX_MEM", "8Gi"),
        },
        # Enable KFP step caching: unchanged steps (chunk-notes, embed) are
        # skipped on retry, so a fix to build-index alone no longer re-pays the
        # ~11 min chunk run. A step whose code or inputs changed still re-runs.
        enable_caching=True,
    )
    job.submit(service_account=os.environ.get("PIPELINE_SA") or None)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "compile"
    if command == "submit":
        submit()
    else:
        path = compile_pipeline()
        print(f"compiled → {path}")
