"""Helpers for embedding chunks with gemini-embedding-001 (batch mode).

Pure functions, no cloud calls (same constraint as the rest of rag/). The
batch-input row schema and the Vector Search ingest format live here so they
are testable and shared between the build script and any later docs.

Design (D1, decided 2026-08-06):
  * gemini-embedding-001, truncated to 768 dims via output_dimensionality
  * task_type RETRIEVAL_DOCUMENT at index time (queries use RETRIEVAL_QUERY)
  * datapoint ids are chunk_ids with ':' folded to '_': Vector Search ids only
    allow [A-Za-z0-9_-]. The chunk store keys by datapoint_id, so rag_search
    maps a returned id to its chunk row in BigQuery and never reverse-parses.
"""

from __future__ import annotations

EMBEDDING_MODEL = "gemini-embedding-001"
OUTPUT_DIMENSIONALITY = 768
DOCUMENT_TASK_TYPE = "RETRIEVAL_DOCUMENT"
QUERY_TASK_TYPE = "RETRIEVAL_QUERY"
RESTRICT_NAMESPACE = "hadm_id"


def datapoint_id(chunk_id: str) -> str:
    """Vector Search-safe id for a chunk_id ({note_id}:{section}:{ordinal})."""
    return chunk_id.replace(":", "_")


def batch_input_row(
    chunk_id: str,
    text: str,
    output_dimensionality: int = OUTPUT_DIMENSIONALITY,
    task_type: str = DOCUMENT_TASK_TYPE,
) -> dict:
    """One row of the gemini-embedding-001 batch-prediction input JSONL."""
    return {
        "key": chunk_id,
        "request": {"content": {"parts": [{"text": text}]}},
        "embed_content_config": {
            "output_dimensionality": output_dimensionality,
            "task_type": task_type,
        },
    }


def vector_search_record(chunk_id: str, hadm_id: int, embedding: list[float]) -> dict:
    """One Vector Search ingest record, with the hadm_id restrict attached."""
    return {
        "id": datapoint_id(chunk_id),
        "embedding": embedding,
        "restricts": [
            {"namespace": RESTRICT_NAMESPACE, "allow": [str(hadm_id)]},
        ],
    }
