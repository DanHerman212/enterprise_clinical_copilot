"""embed_chunks — embed the chunk corpus with gemini-embedding-001 @ 768 dims
and emit the Vector Search ingest artifact (uncompressed .json).

Cloud home of scripts/embed_chunks.py. Reads chunks from the pipeline artifact
(no local data), optionally reuses embeddings from a previous ingest on GCS
(first-wins dedup by datapoint id — so the duplicate-id fix only re-embeds the
renumbered chunks), embeds the remainder via the Vertex batchEmbedContents
surface, and writes the clean ingest artifact. All embedding compute runs in
Vertex; this component is the driver.
"""

import gzip
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.cloud import storage
from google.genai import types
from kfp import dsl

from ._image import RAG_IMAGE, component

BATCH_SIZE = 100  # batchEmbedContents max requests per call


def run_embed_chunks(
    *,
    project_id: str,
    location: str,
    chunks_path: str,
    previous_ingest_uri: str,
    ingest_path: str,
    manifest_path: str,
    workers: int,
) -> None:
    from rag.embed import (
        DOCUMENT_TASK_TYPE,
        EMBEDDING_MODEL,
        OUTPUT_DIMENSIONALITY,
        datapoint_id,
        vector_search_record,
    )

    chunks: list[dict] = []
    with gzip.open(chunks_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            chunks.append(json.loads(line))
    n_total = len(chunks)
    print(f"chunks: {n_total}")

    base_path = ingest_path + ".base"  # reused embeddings (deduped)
    new_path = ingest_path + ".new"    # freshly embedded records
    done: set[str] = set()
    reused = 0

    if previous_ingest_uri:
        client_s = storage.Client(project=project_id)
        parts = previous_ingest_uri.replace("gs://", "").split("/", 1)
        raw = client_s.bucket(parts[0]).blob(parts[1]).download_as_bytes()
        text = gzip.decompress(raw).decode() if parts[1].endswith(".gz") else raw.decode()
        with open(base_path, "w", encoding="utf-8") as out:
            for line in text.splitlines():
                record = json.loads(line)
                if record["id"] in done:
                    continue
                done.add(record["id"])
                out.write(line + "\n")
                reused += 1
        print(f"reused {reused} embeddings from {previous_ingest_uri}")

    pending = [c for c in chunks if datapoint_id(c["chunk_id"]) not in done]
    print(f"pending embed: {len(pending)}")

    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
        # A stalled connection must raise, not freeze the run.
        http_options=types.HttpOptions(timeout=120_000),
    )
    lock = threading.Lock()
    embedded = 0
    retries = 0
    started = time.monotonic()

    def embed_batch(batch: list) -> None:
        nonlocal embedded, retries
        texts = [c["text"] for c in batch]
        for attempt in range(6):
            try:
                resp = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=texts,
                    config=types.EmbedContentConfig(
                        output_dimensionality=OUTPUT_DIMENSIONALITY,
                        task_type=DOCUMENT_TASK_TYPE,
                    ),
                )
                break
            except Exception as exc:
                with lock:
                    retries += 1
                if attempt == 5:
                    raise SystemExit(f"embed batch failed after retries: {exc}")
                time.sleep(2 ** attempt + random.uniform(0, 1))
        records = [
            vector_search_record(c["chunk_id"], c["hadm_id"], list(e.values))
            for c, e in zip(batch, resp.embeddings)
        ]
        for rec in records:
            if len(rec["embedding"]) != OUTPUT_DIMENSIONALITY:
                raise SystemExit(f"bad dims for {rec['id']}: {len(rec['embedding'])}")
        with lock:
            with open(new_path, "a", encoding="utf-8") as out:
                for rec in records:
                    out.write(json.dumps(rec) + "\n")
            embedded += len(records)

    batches = [pending[i:i + BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(embed_batch, b) for b in batches]):
            future.result()

    elapsed = time.monotonic() - started
    with open(ingest_path, "w", encoding="utf-8") as out:
        if os.path.exists(base_path):
            with open(base_path, encoding="utf-8") as handle:
                out.write(handle.read())
        with open(new_path, encoding="utf-8") as handle:
            out.write(handle.read())

    total = 0
    unique: set[str] = set()
    with open(ingest_path, encoding="utf-8") as handle:
        for line in handle:
            total += 1
            unique.add(json.loads(line)["id"])
    if total != n_total or len(unique) != n_total:
        raise SystemExit(
            f"ingest mismatch: {total} records / {len(unique)} unique vs {n_total} chunks"
        )

    manifest = {
        "chunks": n_total,
        "ingest": total,
        "unique_ids": len(unique),
        "reused": reused,
        "new_embedded": embedded,
        "retries": retries,
        "dimensions": OUTPUT_DIMENSIONALITY,
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"embed_chunks: {total} records ({embedded} new, {reused} reused) in {elapsed/60:.1f} min")


@component(
    base_image=RAG_IMAGE,
    packages_to_install=["google-genai", "google-cloud-storage"],
)
def embed_chunks(
    project_id: str,
    location: str,
    chunks: dsl.Input[dsl.Artifact],
    previous_ingest_uri: str,
    workers: int,
    ingest: dsl.Output[dsl.Artifact],
    manifest: dsl.Output[dsl.Artifact],
) -> None:
    """KFP component: embed chunks and emit the Vector Search ingest artifact."""
    from pipelines.components.embed_chunks import run_embed_chunks

    run_embed_chunks(
        project_id=project_id,
        location=location,
        chunks_path=chunks.path,
        previous_ingest_uri=previous_ingest_uri,
        ingest_path=ingest.path,
        manifest_path=manifest.path,
        workers=workers,
    )
