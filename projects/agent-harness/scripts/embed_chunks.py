"""Embed the chunk corpus with gemini-embedding-001 @ 768 dims.

D1 config (decided 2026-08-06): gemini-embedding-001, output_dimensionality
768, task_type RETRIEVAL_DOCUMENT.

Mechanism: google-genai embed_content with a list of contents (the Vertex
batchEmbedContents surface). The async BatchPredictionJob path was tried first
and failed - the preview JSONL schema (nested embed_content_config) is rejected
by the batch engine, so this sync path is the reliable one. Online rate applies
(~$0.15/1M tokens; ~$10 for the corpus).

Resumable: embeddings are appended to the ingest file as they are produced, so
a crash or rate-limit stall resumes on re-run (already-done chunk ids skipped).

    .venv/bin/python scripts/embed_chunks.py [--prepare-only] [--limit N] [--workers W]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.embed import (  # noqa: E402
    DOCUMENT_TASK_TYPE,
    EMBEDDING_MODEL,
    OUTPUT_DIMENSIONALITY,
    batch_input_row,
    datapoint_id,
    vector_search_record,
)
from rag.notes import CACHE_DIR, iter_chunks  # noqa: E402

PROJECT = "trim-icon-498815-a0"
LOCATION = "us-east1"
BUCKET = "trim-icon-498815-a0-mlops"

INPUT_PATH = CACHE_DIR / "embed_input.jsonl.gz"
INGEST_PATH = CACHE_DIR / "embed_ingest.jsonl.gz"
INPUT_MANIFEST = CACHE_DIR / "embed_input.manifest.json"
INGEST_MANIFEST = CACHE_DIR / "embed_ingest.manifest.json"

# gemini-embedding-001 online is billed per token; 4 chars per token is the
# standard approximation for English prose.
CHARS_PER_TOKEN = 4
ONLINE_RATE_PER_M = 0.15
BATCH_SIZE = 100  # batchEmbedContents max requests per call


def prepare(limit: int | None) -> dict:
    """Build the batch input JSONL locally; returns its manifest."""
    total_chars = 0
    rows = 0
    with gzip.open(INPUT_PATH, "wt", encoding="utf-8") as out:
        for chunk in iter_chunks():
            if limit is not None and rows >= limit:
                break
            row = batch_input_row(chunk["chunk_id"], chunk["text"])
            out.write(json.dumps(row) + "\n")
            total_chars += len(chunk["text"])
            rows += 1
            if rows % 100_000 == 0:
                print(f"  …{rows} rows")

    est_tokens = total_chars / CHARS_PER_TOKEN
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model": EMBEDDING_MODEL,
        "output_dimensionality": OUTPUT_DIMENSIONALITY,
        "task_type": DOCUMENT_TASK_TYPE,
        "rows": rows,
        "total_chars": total_chars,
        "est_tokens": int(est_tokens),
        "est_embed_cost_usd": round(est_tokens * ONLINE_RATE_PER_M / 1e6, 2),
        "input_path": str(INPUT_PATH),
    }
    INPUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def run_embedding(limit: int | None, workers: int) -> int:
    """Embed all chunks via batchEmbedContents; append-only and resumable.

    Returns the process exit code: 0 on success, 1 when any batch failed
    (the ingest is NOT uploaded to GCS on failure — ECC-39).
    """
    import random
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from google import genai
    from google.genai import types

    client = genai.Client(
        vertexai=True,
        project=PROJECT,
        location=LOCATION,
        # A stalled connection must raise, not freeze the run (hit on 2026-08-06
        # at 344,200/555,770: 0% CPU, no progress, no error).
        http_options=types.HttpOptions(timeout=120_000),
    )

    done: set[str] = set()
    if INGEST_PATH.exists():
        with gzip.open(INGEST_PATH, "rt", encoding="utf-8") as handle:
            for line in handle:
                done.add(json.loads(line)["id"])
        print(f"Resuming: {len(done)} chunks already embedded")

    pending = [
        (chunk["chunk_id"], chunk["hadm_id"], chunk["text"])
        for chunk in iter_chunks()
        if datapoint_id(chunk["chunk_id"]) not in done
    ]
    if limit is not None:
        pending = pending[:limit]
    print(f"To embed: {len(pending)} chunks "
          f"(~{len(pending) * 118:,} est tokens, ~${len(pending)*118*ONLINE_RATE_PER_M/1e6:.2f})")

    batches = [pending[i:i + BATCH_SIZE]
               for i in range(0, len(pending), BATCH_SIZE)]
    lock = threading.Lock()
    written = 0
    failed = 0
    retries = 0
    started = time.monotonic()

    def embed_batch(batch: list) -> None:
        nonlocal written, failed, retries
        texts = [row[2] for row in batch]
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
                    print(f"  FAILED batch after retries: {exc}", flush=True)
                    with lock:
                        failed += 1
                    return
                time.sleep(2 ** attempt + random.uniform(0, 1))
        if len(resp.embeddings) != len(batch):
            raise SystemExit(
                f"embed API returned {len(resp.embeddings)} embeddings for "
                f"{len(batch)} inputs — refusing to silently drop chunks (ECC-38)"
            )
        records = [
            vector_search_record(cid, hadm, list(emb.values))
            for (cid, hadm, _), emb in zip(batch, resp.embeddings, strict=True)
        ]
        for rec in records:
            if len(rec["embedding"]) != OUTPUT_DIMENSIONALITY:
                raise SystemExit(f"Bad dims for {rec['id']}: {len(rec['embedding'])}")
        with lock:
            with gzip.open(INGEST_PATH, "at", encoding="utf-8") as out:
                for rec in records:
                    out.write(json.dumps(rec) + "\n")
            written += len(records)
            if written % 5000 < BATCH_SIZE:
                elapsed = time.monotonic() - started
                rate = written / max(elapsed, 1e-9)
                remaining = (len(pending) - written) / max(rate, 1e-9)
                print(f"  …{written}/{len(pending)} ({rate:.0f}/s, "
                      f"~{remaining/60:.0f} min left)")

    stop_hb = threading.Event()

    def heartbeat() -> None:
        """Print a status line every 60s so a stall is visible, not silent."""
        while True:
            if stop_hb.wait(60):
                break
            elapsed = time.monotonic() - started
            with lock:
                print(f"  [hb] {elapsed/60:.1f} min  written={written}/{len(pending)}"
                      f"  retries={retries}  failed={failed}", flush=True)

    threading.Thread(target=heartbeat, daemon=True).start()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(embed_batch, b) for b in batches]):
            future.result()
    stop_hb.set()

    elapsed = time.monotonic() - started
    with gzip.open(INGEST_PATH, "rt", encoding="utf-8") as handle:
        total = sum(1 for _ in handle)
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model": EMBEDDING_MODEL,
        "output_dimensionality": OUTPUT_DIMENSIONALITY,
        "task_type": DOCUMENT_TASK_TYPE,
        "this_run_written": written,
        "ingest_total": total,
        "failed_batches": failed,
        "retries": retries,
        "elapsed_seconds": round(elapsed, 1),
        "ingest_path": str(INGEST_PATH),
    }
    INGEST_MANIFEST.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nWrote {INGEST_PATH} ({total} records, {OUTPUT_DIMENSIONALITY} dims)")
    print(f"  this run: {written} in {elapsed/60:.1f} min; "
          f"failed batches: {failed}; retries: {retries}")
    print(f"  manifest: {INGEST_MANIFEST}")
    if failed:
        print(f"ERROR: {failed} batch(es) failed — NOT uploading to GCS. "
              "The run is resumable; fix and re-run (ECC-39).")
        return 1
    upload_ingest()
    return 0


def upload_ingest() -> None:
    """Push the finished ingest file + manifest to GCS (durable copy).

    The local file is the working scratch for resume; GCS is where the artifact
    lives and where the §6 index build reads it from. User preference (2026-08-06):
    cloud storage, not local, as the durable home for PHI-adjacent artifacts.
    """
    from google.cloud import storage

    client = storage.Client(project=PROJECT)
    bucket = client.bucket(BUCKET)
    prefix = f"rag/embeddings/ingest/"
    for path in (INGEST_PATH, INGEST_MANIFEST):
        if not path.exists():
            continue
        dest = f"{prefix}{path.name}"
        bucket.blob(dest).upload_from_filename(str(path))
        print(f"Uploaded {path.name} → gs://{BUCKET}/{dest}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true",
                        help="build the input sizing manifest locally; no spend")
    parser.add_argument("--workers", type=int, default=4,
                        help="concurrent embed calls (rate-limit aware)")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.prepare_only:
        manifest = prepare(args.limit)
        print(f"\nPrepared {manifest['rows']} rows → {INPUT_PATH}")
        print(f"  est tokens:  {manifest['est_tokens']:,}")
        print(f"  est cost:    ${manifest['est_embed_cost_usd']} (online rate)")
        return 0
    return run_embedding(args.limit, args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
