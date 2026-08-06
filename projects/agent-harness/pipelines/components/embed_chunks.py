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
import io
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


def _rss_mib() -> int:
    """Current process RSS in MiB (Linux /proc), for OOM debugging."""
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return -1


def _path_to_gcs_uri(path: str) -> str:
    """Map a KFP artifact path (mounted under /gcs/) to its gs:// URI."""
    if path.startswith("/gcs/"):
        return "gs://" + path[len("/gcs/"):]
    raise SystemExit(f"cannot derive gs:// URI from artifact path: {path}")


def _stream_count(client_s: storage.Client, uri: str) -> tuple[int, int]:
    """Stream a GCS ingest file once; return (record_count, unique_id_count)."""
    bucket_name, obj = uri.replace("gs://", "", 1).split("/", 1)
    blob = client_s.bucket(bucket_name).blob(obj)
    total = 0
    unique: set[str] = set()
    with blob.open("rb") as src:
        reader = gzip.open(src, "rt", encoding="utf-8") \
            if obj.endswith(".gz") else io.TextIOWrapper(src, encoding="utf-8")
        with reader as text:
            for line in text:
                total += 1
                unique.add(json.loads(line)["id"])
    return total, unique


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

    # Scratch in /tmp (local disk), NOT next to ingest_path: ingest_path lives
    # on the GCS FUSE mount and run 6 showed ~35 GB of slow network I/O from
    # writing .base/.new there. Copy the final file to ingest_path once.
    base_path = "/tmp/ingest.base"  # reused embeddings (deduped)
    new_path = "/tmp/ingest.new"    # freshly embedded records
    done: set[str] = set()
    reused = 0

    if previous_ingest_uri:
        # Stream-decompress from GCS: the previous ingest is ~7-8 GB
        # decompressed, so loading it whole OOMs the component (hit on run 2).
        client_s = storage.Client(project=project_id)
        parts = previous_ingest_uri.replace("gs://", "").split("/", 1)
        blob = client_s.bucket(parts[0]).blob(parts[1])
        with blob.open("rb") as src:
            reader = gzip.open(src, "rt", encoding="utf-8") \
                if parts[1].endswith(".gz") else io.TextIOWrapper(src, encoding="utf-8")
            with reader as text:
                with open(base_path, "w", encoding="utf-8") as out:
                    for line in text:
                        record = json.loads(line)
                        if record["id"] in done:
                            continue
                        done.add(record["id"])
                        out.write(line)
                        reused += 1
        print(f"reused {reused} embeddings from {previous_ingest_uri}")

    pending = [c for c in chunks if datapoint_id(c["chunk_id"]) not in done]
    print(f"pending embed: {len(pending)}")

    # FULL-REUSE FAST PATH: nothing left to embed, so skip the 9 GB
    # download-to-/tmp + GCS-FUSE re-upload that run 8 showed takes ~10 min of
    # pure I/O. Instead do a server-side GCS copy of the previous ingest
    # straight to the output object (same bytes, no data through this pod) and
    # verify the count/unique-ids in one streaming pass.
    if previous_ingest_uri and not pending and not previous_ingest_uri.endswith(".gz"):
        started_fast = time.monotonic()
        client_s = storage.Client(project=project_id)
        total, unique = _stream_count(client_s, previous_ingest_uri)
        if total != n_total or len(unique) != n_total:
            raise SystemExit(
                f"previous ingest mismatch: {total} records / {len(unique)} unique "
                f"vs {n_total} chunks"
            )
        dst_uri = _path_to_gcs_uri(ingest_path)
        dst_bucket, dst_obj = dst_uri.replace("gs://", "", 1).split("/", 1)
        src_bucket, src_obj = previous_ingest_uri.replace("gs://", "", 1).split("/", 1)
        src_blob = client_s.bucket(src_bucket).blob(src_obj)
        dst_blob = client_s.bucket(dst_bucket).blob(dst_obj)
        dst_blob.rewrite(src_blob)  # server-side; no bytes through this pod
        elapsed_fast = time.monotonic() - started_fast
        manifest = {
            "chunks": n_total,
            "ingest": total,
            "unique_ids": len(unique),
            "reused": total,
            "new_embedded": 0,
            "retries": 0,
            "dimensions": OUTPUT_DIMENSIONALITY,
            "elapsed_seconds": round(elapsed_fast, 1),
            "path": "server-side copy",
        }
        with open(manifest_path, "w") as handle:
            json.dump(manifest, handle, indent=2)
        print(f"embed_chunks: {total} records via server-side copy in {elapsed_fast:.1f}s")
        return

    lock = threading.Lock()
    embedded = 0
    retries = 0
    started = time.monotonic()
    last_progress = [started]
    thread_local = threading.local()

    def get_client():
        # One client per worker thread. A single shared genai/httpx client
        # under sustained concurrent load deadlocks (all threads blocked, no
        # HTTP requests for 12+ min, timeout not firing) - observed twice.
        if not hasattr(thread_local, "client"):
            thread_local.client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location,
                http_options=types.HttpOptions(timeout=120_000),
            )
        return thread_local.client

    def embed_batch(batch: list) -> None:
        nonlocal embedded, retries
        texts = [c["text"] for c in batch]
        for attempt in range(6):
            try:
                resp = get_client().models.embed_content(
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
            last_progress[0] = time.monotonic()
            with open(new_path, "a", encoding="utf-8") as out:
                for rec in records:
                    out.write(json.dumps(rec) + "\n")
            embedded += len(records)
            if embedded % (BATCH_SIZE * 50) < BATCH_SIZE:
                rss_mib = _rss_mib()
                print(f"  …embedded {embedded}/{len(pending)} rss={rss_mib}MiB",
                      flush=True)

    stop_wd = threading.Event()

    def watchdog() -> None:
        # A silent hang must fail the step loudly, not block forever.
        while not stop_wd.wait(60):
            stalled = time.monotonic() - last_progress[0]
            if stalled > 300:
                print(f"STALLED: no successful embed for {stalled/60:.1f} min; aborting",
                      flush=True)
                os._exit(1)

    threading.Thread(target=watchdog, daemon=True).start()

    batches = [pending[i:i + BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(embed_batch, b) for b in batches]):
            future.result()
    stop_wd.set()

    elapsed = time.monotonic() - started
    with open(ingest_path, "w", encoding="utf-8") as out:
        # Stream line-by-line: the base file is ~3 GB, and handle.read() would
        # load it all into memory (an OOM risk against the container limit).
        if os.path.exists(base_path):
            with open(base_path, encoding="utf-8") as handle:
                for line in handle:
                    out.write(line)
        # new_path only exists when there were pending chunks to embed; on a
        # full-reuse run (pending == 0) it is never created (hit on run 7).
        if os.path.exists(new_path):
            with open(new_path, encoding="utf-8") as handle:
                for line in handle:
                    out.write(line)

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
