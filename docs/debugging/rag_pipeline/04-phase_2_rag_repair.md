# Engineering Directive: Phase 2 — Update `rag_config.yaml`, Push Image, & Submit KFP `rag_ingest_pipeline`

## 0. Governance & Strict Boundaries
1. **Strict Scope:**
   - Update `corpus.demo.expected_vectors` in `services/mcp/rag_config.yaml` from `446` to `582` (plus any config test asserting that value).
   - Commit and push all verified changes so CI/CD builds and pushes the updated container image.
   - Programmatically submit the KFP pipeline (`services.mcp.pipelines.rag_ingest_pipeline`) once the container image with the new `chunking.py` is available, and verify the deployed index.

---

## Step 1: Update `expected_vectors` in `services/mcp/rag_config.yaml`
1. In `services/mcp/rag_config.yaml`, change `corpus.demo.expected_vectors` from `446` to `582`.
2. Run `python -m pytest tests -m "not live" -q` and confirm 100% pass.

---

## Step 2: Commit, Push, and Ensure Container Image Is Updated for KFP
Because KFP components (`chunk_notes`, `embed`, `eval_ingest`, `build_index`) execute inside the container image referenced by the pipeline, the image must include the updated `chunking.py` (`INDEX_SECTIONS` = 22 sections, headingless note support) before the KFP job runs:
1. Inspect how `rag_ingest_pipeline` resolves its base container image (e.g., Cloud Build / GitHub Actions / Artifact Registry tag).
2. Commit and push the changes, and ensure the updated container image is built/pushed so the KFP worker runs the new code (emitting 582 chunks) rather than the previous image (438 chunks).

---

## Step 3: Programmatically Submit the KFP Pipeline (`rag_ingest_pipeline`)
1. Submit the KFP pipeline for the `demo` corpus with `PREVIOUS_INGEST_URI` unset (so all 582 chunks are freshly embedded and gated by `eval_ingest`):
   ```bash
   RAG_CORPUS=demo .venv/bin/python -m services.mcp.pipelines.rag_ingest_pipeline submit