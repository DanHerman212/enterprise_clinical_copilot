#!/usr/bin/env bash
#
# submit_rag_pipeline.sh — compile & submit the RAG ingest pipeline to Vertex
# AI Pipelines. All data processing happens in the cloud (BigQuery / GCS /
# Vertex); nothing is stored on the submitting machine.
#
# Usage:
#   bash services/mcp/pipelines/submit_pipeline.sh
#
# PREREQUISITE: the RAG ingest image must be built first:
#   bash scripts/agent/build_rag_image.sh
#
# The corpus is chosen HERE, and this is the only place it is chosen for a
# submit. rag_config.yaml's `corpus.active` is the real MIMIC corpus, so a submit
# that does not name a corpus ingests MIMIC: on 2026-10-08 that is exactly how a
# demo re-ingest was submitted against mimiciv_note.discharge and 555,770 real
# vectors. The default below is demo; ask for mimic deliberately.
#
# Override by exporting first:
#   RAG_CORPUS=mimic            # the real ~556k-vector corpus (slow, priced)
#   PREVIOUS_INGEST_URI=""      # empty → fresh full embed
#   PREVIOUS_INGEST_URI=gs://…  # pin an explicit reuse artifact
#   PIPELINE_SA=...             # service account for pipeline execution
#
# PREVIOUS_INGEST_URI is deliberately NOT defaulted here. The pipeline derives a
# reuse artifact keyed by corpus AND by the fingerprint of the source tables
# (rag_ingest_pipeline.py::reuse_uri), so a changed corpus resolves to a
# different object and an absent one degrades to a full embed. Defaulting it to
# one fixed object — which this script did until 2026-10-08 — overrode that and
# made a two-month-old MIMIC ingest reachable from any submit, failing the embed
# step's count gate: the cached ingest was a superset of the current chunks, and
# the step asserts the two counts match. See embed_chunks.py:351.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"            # services/mcp
REPO_ROOT="$(cd "$HARNESS_DIR/../.." && pwd)"          # repo root
VENV_PY="$REPO_ROOT/.venv/bin/python"

export PROJECT_ID="${PROJECT_ID:-trim-icon-498815-a0}"
export REGION="${REGION:-us-east1}"
export PIPELINE_ROOT="${PIPELINE_ROOT:-gs://trim-icon-498815-a0-mlops/pipeline-root}"
export PIPELINE_SA="${PIPELINE_SA:-mlops-pipeline@trim-icon-498815-a0.iam.gserviceaccount.com}"
export RAG_IMAGE_URI="${RAG_IMAGE_URI:-us-east1-docker.pkg.dev/trim-icon-498815-a0/readmission/rag-ingest:latest}"
# Which corpus this submit ingests. rag_config.yaml selects the profile from this
# value, falling back to `active: mimic`, so leaving it unset is a silent choice.
export RAG_CORPUS="${RAG_CORPUS:-demo}"
# Forward the reuse source only when the caller set it — including when they set
# it to the empty string, which is the documented "full embed" switch. Unset here
# means the pipeline derives the fingerprint-keyed artifact itself.
if [ -n "${PREVIOUS_INGEST_URI+set}" ]; then
  export PREVIOUS_INGEST_URI
fi
export EMBED_WORKERS="${EMBED_WORKERS:-1}"

# The pipeline imports from TWO roots and needs both on the path:
#   pipelines.*     resolved from services/mcp (HARNESS_DIR, the cwd)
#   services.mcp.*  resolved from the repo root
# Only HARNESS_DIR was exported, so every submit died with
# ModuleNotFoundError: No module named 'services'. REPO_ROOT was computed and
# never used. The import that failed is chunk_notes.py's
# `from services.mcp.retrieval.chunking import INDEX_SECTIONS`.
export PYTHONPATH="$HARNESS_DIR:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "=== Submitting RAG ingest pipeline ==="
echo "  PROJECT_ID          : $PROJECT_ID"
echo "  PIPELINE_SA         : $PIPELINE_SA"
echo "  RAG_IMAGE_URI       : $RAG_IMAGE_URI"
echo "  RAG_CORPUS          : $RAG_CORPUS"
echo "  PREVIOUS_INGEST_URI : ${PREVIOUS_INGEST_URI-<unset: the pipeline derives a fingerprint-keyed one>}"
echo

cd "$HARNESS_DIR"
"$VENV_PY" pipelines/rag_ingest_pipeline.py submit
