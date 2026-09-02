"""
rag_ingest_pipeline — Vertex AI (KFP v2) RAG ingest pipeline DAG.

Wiring::

    chunk_notes ─▶ embed_chunks ─▶ build_index

All components move data only through BigQuery and GCS; nothing is stored on a
developer machine. `embed_chunks` can reuse embeddings from a previous ingest
(via `previous_ingest_uri`) so a chunking fix re-embeds only the changed
chunks instead of paying for the whole corpus again.
"""

import hashlib
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
from mcp_server.config import PROJECT as PROJECT_ID  # noqa: E402
from rag.chunking import DEFAULT_PACK_TO  # noqa: E402
from rag.config import load as load_rag_config  # noqa: E402

PIPELINE_NAME = "rag-ingest"
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
    data_fingerprint: str = "",
    pack_to: int = DEFAULT_PACK_TO,
    sections_csv: str = ",".join(DEFAULT_SECTIONS),
    previous_ingest_uri: str = "",
    dimensions: int = 768,
    embed_workers: int = 1,
    brute_sample: int = 2000,
    approximate_neighbors: int = 40,
    expected_vectors: int = 555770,
    shard_size: str = "SHARD_SIZE_MEDIUM",
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
        data_fingerprint=data_fingerprint,
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
        shard_size=shard_size,
    )
    index_task.set_cpu_limit(index_cpu).set_memory_limit(index_mem)


def compile_pipeline(package_path: str = "rag_ingest_pipeline.yaml") -> str:
    """Compile the pipeline to a KFP IR YAML and return the path."""
    compiler.Compiler().compile(
        pipeline_func=rag_ingest_pipeline, package_path=package_path
    )
    return package_path


def _source_fingerprint(project_id: str, refs: tuple[str, str]) -> str:
    """Fingerprint the source tables so KFP caching invalidates on data change.

    KFP step caching keys on code + inputs, NOT table contents (ECC-70): without
    this, a re-submission after the source tables change would silently reuse
    stale chunks/embeddings and build an index that does not match the data.
    Folding each table's modified time + row count into the chunk_notes input
    makes a data change re-run the DAG.
    """
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)
    parts = []
    for ref in refs:
        table = client.get_table(ref)
        modified = table.modified.timestamp() if table.modified else 0
        parts.append(f"{ref}:{int(modified)}:{table.num_rows}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def submit() -> None:
    """Compile and submit the pipeline to Vertex AI Pipelines.

    Reads rag_config.yaml (config-as-code) for the corpus and every knob; the
    corpus switch (MIMIC vs demo) is ``RAG_CORPUS``, not a scatter of env vars.
    Per-run knobs that legitimately vary stay env-driven:
    ``PREVIOUS_INGEST_URI``, ``EMBED_WORKERS``.
    """
    cfg = load_rag_config()

    package_path = compile_pipeline()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    aiplatform.init(project=cfg.project, location=cfg.location)
    job = aiplatform.PipelineJob(
        display_name=f"{PIPELINE_NAME}-{ts}",
        template_path=package_path,
        pipeline_root=os.environ.get(
            "PIPELINE_ROOT", f"gs://{cfg.project}-mlops/pipeline-root"
        ),
        parameter_values={
            "project_id": cfg.project,
            "notes_table_ref": cfg.corpus.notes_table_ref,
            "split_table_ref": cfg.corpus.split_table_ref,
            "split_name": cfg.corpus.split_name,
            "data_fingerprint": _source_fingerprint(
                cfg.project,
                (cfg.corpus.notes_table_ref, cfg.corpus.split_table_ref),
            ),
            "previous_ingest_uri": os.environ.get(
                "PREVIOUS_INGEST_URI",
                f"gs://{cfg.project}-mlops/rag/embeddings/ingest/embed_ingest.jsonl.gz",
            ),
            "embed_workers": int(os.environ.get("EMBED_WORKERS", "1")),
            "pack_to": cfg.pack_to,
            "sections_csv": ",".join(DEFAULT_SECTIONS),
            "dimensions": cfg.dimensions,
            "approximate_neighbors": cfg.approximate_neighbors,
            "expected_vectors": cfg.corpus.expected_vectors,
            "brute_sample": cfg.brute_sample,
            "shard_size": cfg.shard_size,
            "chunk_cpu": cfg.chunk_cpu,
            "chunk_mem": cfg.chunk_mem,
            "embed_cpu": cfg.embed_cpu,
            "embed_mem": cfg.embed_mem,
            "index_cpu": cfg.index_cpu,
            "index_mem": cfg.index_mem,
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
