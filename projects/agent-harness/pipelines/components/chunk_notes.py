"""chunk_notes — read test-split discharge notes from BigQuery, chunk them with
the section-aware chunker, and emit the chunk corpus artifact.

Cloud home of scripts/build_chunks.py: the same fixed logic (rag/chunking.py,
pack_to, narrative-section whitelist) runs as a pipeline component so note text
never lands on a developer machine. chunk_ids are asserted unique — the
2026-08-06 duplicate-section bug would fail this component instead of shipping
a silently-short index.
"""

import gzip
import json
import re
from collections import Counter

from google.cloud import bigquery
from kfp import dsl

from rag.chunking import INDEX_SECTIONS

from ._image import RAG_IMAGE, component

# Narrative/assessment sections worth indexing — single-sourced from
# rag.chunking, the same tuple the serving-side datapoint-id parser uses,
# so build and serving can never drift.
DEFAULT_SECTIONS = INDEX_SECTIONS

# ECC-37: table identifiers cannot be bound as query parameters, so the
# runtime pipeline params are validated against a strict shape instead —
# project.dataset.table, plain identifier characters only.
_TABLE_REF_RE = re.compile(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+")


def _validated_table_ref(ref: str, name: str) -> str:
    if not _TABLE_REF_RE.fullmatch(ref):
        raise ValueError(
            f"{name} is not a valid project.dataset.table reference: {ref!r}"
        )
    return ref


def run_chunk_notes(
    *,
    project_id: str,
    notes_table_ref: str,
    split_table_ref: str,
    split_name: str,
    pack_to: int,
    sections_csv: str,
    chunks_path: str,
    manifest_path: str,
) -> None:
    from rag.chunking import DEFAULT_MAX_CHARS, chunk_note

    whitelist = {s.strip() for s in sections_csv.split(",") if s.strip()}
    notes_table_ref = _validated_table_ref(notes_table_ref, "notes_table_ref")
    split_table_ref = _validated_table_ref(split_table_ref, "split_table_ref")
    client = bigquery.Client(project=project_id)

    # split_name is a value, so it is bound as a query parameter (ECC-37) —
    # never interpolated into the SQL text.
    sql = f"""
        SELECT d.hadm_id, d.note_id, d.text
        FROM `{notes_table_ref}` AS d
        JOIN `{split_table_ref}` AS a ON d.hadm_id = a.hadm_id
        WHERE a.split_name = @split_name
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("split_name", "STRING", split_name)
        ]
    )

    section_counts: Counter[str] = Counter()
    seen: set[str] = set()
    total = 0
    with gzip.open(chunks_path, "wt", encoding="utf-8") as out:
        for row in client.query(sql, job_config=job_config).result():
            note = {
                "hadm_id": row["hadm_id"],
                "note_id": row["note_id"],
                "text": row["text"],
            }
            for chunk in chunk_note(note, max_chars=DEFAULT_MAX_CHARS, pack_to=pack_to):
                if chunk.section not in whitelist:
                    continue
                if chunk.chunk_id in seen:
                    raise SystemExit(f"Duplicate chunk_id: {chunk.chunk_id}")
                seen.add(chunk.chunk_id)
                section_counts[chunk.section] += 1
                total += 1
                out.write(json.dumps(chunk.__dict__) + "\n")

    with open(manifest_path, "w") as handle:
        json.dump(
            {
                "chunks": total,
                "unique_chunk_ids": len(seen),
                "sections": dict(section_counts),
                "pack_to": pack_to,
            },
            handle,
            indent=2,
        )
    print(f"chunk_notes: {total} chunks, {len(seen)} unique ids")


@component(
    base_image=RAG_IMAGE,
    packages_to_install=["google-cloud-bigquery"],
)
def chunk_notes(
    project_id: str,
    notes_table_ref: str,
    split_table_ref: str,
    split_name: str,
    pack_to: int,
    sections_csv: str,
    chunks: dsl.Output[dsl.Artifact],
    manifest: dsl.Output[dsl.Artifact],
) -> None:
    """KFP component: chunk the test-split discharge notes in BigQuery."""
    from pipelines.components.chunk_notes import run_chunk_notes

    run_chunk_notes(
        project_id=project_id,
        notes_table_ref=notes_table_ref,
        split_table_ref=split_table_ref,
        split_name=split_name,
        pack_to=pack_to,
        sections_csv=sections_csv,
        chunks_path=chunks.path,
        manifest_path=manifest.path,
    )
