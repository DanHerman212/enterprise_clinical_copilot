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
    embed_workers: int = 6,
    brute_sample: int = 2000,
    approximate_neighbors: int = 40,
    expected_vectors: int = 555770,
) -> None:
    """Assemble the RAG ingest DAG: chunk → embed → index."""
    chunks = chunk_notes(
        project_id=project_id,
        notes_table_ref=notes_table_ref,
        split_table_ref=split_table_ref,
        split_name=split_name,
        pack_to=pack_to,
        sections_csv=sections_csv,
    )

    ingest = embed_chunks(
        project_id=project_id,
        location=LOCATION,
        chunks=chunks.outputs["chunks"],
        previous_ingest_uri=previous_ingest_uri,
        workers=embed_workers,
    )

    build_index(
        project_id=project_id,
        location=LOCATION,
        ingest=ingest.outputs["ingest"],
        dimensions=dimensions,
        brute_sample=brute_sample,
        approximate_neighbors=approximate_neighbors,
        expected=expected_vectors,
    )


def compile_pipeline(package_path: str = "rag_ingest_pipeline.yaml") -> str:
    """Compile the pipeline to a KFP IR YAML and return the path."""
    compiler.Compiler().compile(
        pipeline_func=rag_ingest_pipeline, package_path=package_path
    )
    return package_path


def submit() -> None:
    """Compile and submit the pipeline to Vertex AI Pipelines."""
    package_path = compile_pipeline()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    job = aiplatform.PipelineJob(
        display_name=f"{PIPELINE_NAME}-{ts}",
        template_path=package_path,
        pipeline_root=PIPELINE_ROOT,
        parameter_values={
            "project_id": PROJECT_ID,
            "notes_table_ref": NOTES_TABLE,
            "split_table_ref": SPLIT_TABLE,
            "previous_ingest_uri": os.environ.get(
                "PREVIOUS_INGEST_URI",
                f"gs://{PROJECT_ID}-mlops/rag/embeddings/ingest/embed_ingest.jsonl.gz",
            ),
            "embed_workers": int(os.environ.get("EMBED_WORKERS", "6")),
        },
        enable_caching=False,
    )
    job.submit(service_account=os.environ.get("PIPELINE_SA") or None)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "compile"
    if command == "submit":
        submit()
    else:
        path = compile_pipeline()
        print(f"compiled → {path}")
