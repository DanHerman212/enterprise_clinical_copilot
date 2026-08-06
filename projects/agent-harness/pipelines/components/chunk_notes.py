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
from collections import Counter

from google.cloud import bigquery
from kfp import dsl

from ._image import RAG_IMAGE, component

# Narrative/assessment sections worth indexing (matches scripts/build_chunks.py).
DEFAULT_SECTIONS = (
    "history_of_present_illness",
    "past_medical_history",
    "family_history",
    "social_history",
    "physical_exam",
    "brief_hospital_course",
    "discharge_condition",
    "discharge_diagnosis",
    "discharge_medications",
    "medications_on_admission",
    "discharge_disposition",
    "discharge_instructions",
)


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
    client = bigquery.Client(project=project_id)

    sql = f"""
        SELECT d.hadm_id, d.note_id, d.text
        FROM `{notes_table_ref}` AS d
        JOIN `{split_table_ref}` AS a ON d.hadm_id = a.hadm_id
        WHERE a.split_name = '{split_name}'
    """

    section_counts: Counter[str] = Counter()
    seen: set[str] = set()
    total = 0
    with gzip.open(chunks_path, "wt", encoding="utf-8") as out:
        for row in client.query(sql).result():
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
