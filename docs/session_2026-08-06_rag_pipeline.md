# Session 2026-08-06 — RAG Ingest Pipeline to Vertex Indexes

## Goal

Finish the cloud RAG ingest pipeline (KFP on Vertex AI) so the demo's `rag_search`
tool can run: **chunk notes → embed chunks → build Vector Search indexes**, with
all data processed in the cloud (BigQuery / GCS / Vertex — nothing local).

Constraint in force: **cloud-only processing**. No PHI stored on this machine.

---

## What was completed today

### 1. Ran the cloud pipeline end-to-end through the embed step
- `chunk_notes` **SUCCEEDED** — read 555,770 discharge-note chunks from BigQuery,
  produced unique chunk IDs (the earlier duplicate-ID bug stayed fixed; 14 unit tests pass).
- `embed_chunks` **SUCCEEDED** (run 6) — all 555,770 chunks embedded with
  `gemini-embedding-001` @ 768 dims. 544,039 reused from the previous ingest +
  11,731 re-embedded. Verified clean: 555,770 records, 555,770 unique IDs.

### 2. Diagnosed and fixed the run-6 build-index failure
- Root cause: `create_tree_ah_index` was called **without the leaf-node params** the
  SDK needs. Vertex rejected the request:
  `FAILED_PRECONDITION: algorithmConfig is required but missing from the metadata`.
- Fix in `pipelines/components/build_index.py`:
  - `leaf_node_embedding_count=1000`
  - `leaf_nodes_to_search_percent=10`
- Confirmed the brute-force index (2,000 vectors) built fine in run 6 — only tree-AH
  was broken.

### 3. Fixed embed-chunks GCS-FUSE slowness
- `embed_chunks.py` wrote its `.base` / `.new` scratch files **next to the output
  artifact on the GCS FUSE mount** → ~35 GB of slow network I/O in run 6.
- Fix: scratch files now go to local `/tmp`; the final file is copied to the output
  once.

### 4. Ran run 7, found a second bug, fixed it
- Run 7 embed failed with `FileNotFoundError: /tmp/ingest.new`.
- Root cause: on a **full-reuse** run (`pending embed: 0`) the `.new` file is never
  created, but the merge step opened it unconditionally.
- Fix in `embed_chunks.py`: guard the merge with `if os.path.exists(new_path)`.

### 5. Submitted run 8 (in flight)
- Reuses run 6's clean ingest via `PREVIOUS_INGEST_URI`, so embed is a fast copy
  (0 pending), then builds both indexes.
- Rebuilt the pipeline image with both fixes baked in.

### 6. Enabled KFP step caching
- `enable_caching=True` in `rag_ingest_pipeline.py` so future retries skip unchanged
  steps (chunk-notes ≈ 11 min saved per retry; a run-9 would only re-run build-index).

---

## Run history (today)

| Run | Pipeline job | Result |
|-----|--------------|--------|
| 6 | `rag-ingest-20260806161511` | chunk ✅ / embed ✅ / **build-index FAILED** (tree-AH missing algorithmConfig) |
| 7 | `rag-ingest-20260806171017` | chunk ✅ / **embed FAILED** (`/tmp/ingest.new` missing on full-reuse) / index not triggered |
| 8 | `rag-ingest-20260806173635` | **RUNNING** (chunk → embed → build-index) |

---

## Challenges encountered (root-cause lessons)

1. **KFP lightweight components only serialize the decorated function body.**
   Module imports don't carry over — caused a `NameError` in run 1. Fix: wrappers
   import their `run_*` helpers from the image on PYTHONPATH.

2. **Memory limits must fit the machine.** `set_memory_limit` is applied to the pod,
   but cpu→machine mapping is fixed (cpu 4 → e2-standard-4 = 16 GB). A 16 Gi limit on
   a 16 GB machine won't schedule; an in-code OOM hit from loading the whole
   decompressed ingest. Fix: e2-standard-8 + 24 Gi + streamed decompress.

3. **Output artifacts are GCS FUSE mounts — slow network I/O.** Scratch files there
   (`.base`/`.new`) caused ~35 GB of I/O. Fix: local `/tmp` scratch, copy once.

4. **Shared genai client deadlocks under concurrency** (observed twice). Fix:
   per-thread clients + sequential workers + a 300 s watchdog.

5. **Vertex Vector Search / SDK gotchas:**
   - index files must be `.json` (not `.jsonl.gz`); gzip objects → 500.
   - datapoint IDs only `[A-Za-z0-9_-]`.
   - querying requires a deployed index endpoint (no offline query).
   - `create_tree_ah_index` defaults for leaf-node params are `None` → must pass
     them or the build is rejected.

6. **The biggest structural pain: cloud-only = slow iteration.** Every failure was a
   small, deterministic bug a fast local test would catch, but the cloud-only rule
   means embed/build-index only run inside Vertex — each catch costs a ~20-25 min
   pipeline run + an image rebuild. (Deferred: add a local dry-run harness over a
   tiny synthetic corpus once the demo is done.)

---

## Where we are / next steps

- ✅ medspaCy negation validated (5/5 ConText categories)
- ✅ chunking fixed (unique IDs), 14 tests pass
- ✅ 555,770 chunks embedded (gemini-embedding-001 @ 768)
- ✅ Run 8 in flight — both index fixes baked in, caching enabled for retries
- ⏳ §6 index build (brute-force + tree-AH) — **current**
- ⏳ §7 deploy index endpoint (~$68/mo — needs explicit user approval)
- ⏳ Validation trials (recall@k, cross-patient isolation R1, demo sanity)
- ⏳ §8-9 `rag_search` MCP tool, cohort rebuild, agent fusion, eval, UI

---

## Files touched today

- `projects/agent-harness/pipelines/components/build_index.py` — tree-AH leaf-node params
- `projects/agent-harness/pipelines/components/embed_chunks.py` — `/tmp` scratch + merge guard
- `projects/agent-harness/pipelines/rag_ingest_pipeline.py` — `enable_caching=True`
- Image rebuilt: `us-east1-docker.pkg.dev/trim-icon-498815-a0/readmission/rag-ingest:latest`
