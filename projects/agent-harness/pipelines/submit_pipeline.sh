#!/usr/bin/env bash
#
# submit_rag_pipeline.sh — compile & submit the RAG ingest pipeline to Vertex
# AI Pipelines. All data processing happens in the cloud (BigQuery / GCS /
# Vertex); nothing is stored on the submitting machine.
#
# Usage:
#   bash projects/agent-harness/pipelines/submit_pipeline.sh
#
# PREREQUISITE: the RAG ingest image must be built first:
#   bash projects/agent-harness/scripts/build_rag_image.sh
#
# Override by exporting first:
#   PREVIOUS_INGEST_URI=""      # empty → fresh full embed; default reuses the
#                               # previous ingest on GCS (cheap fix re-runs)
#   PIPELINE_SA=...             # service account for pipeline execution
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"            # projects/agent-harness
REPO_ROOT="$(cd "$HARNESS_DIR/../.." && pwd)"          # repo root
VENV_PY="$REPO_ROOT/.venv/bin/python"

export PROJECT_ID="${PROJECT_ID:-trim-icon-498815-a0}"
export REGION="${REGION:-us-east1}"
export PIPELINE_ROOT="${PIPELINE_ROOT:-gs://trim-icon-498815-a0-mlops/pipeline-root}"
export PIPELINE_SA="${PIPELINE_SA:-mlops-pipeline@trim-icon-498815-a0.iam.gserviceaccount.com}"
export RAG_IMAGE_URI="${RAG_IMAGE_URI:-us-east1-docker.pkg.dev/trim-icon-498815-a0/readmission/rag-ingest:latest}"
# Default: reuse the previous ingest on GCS so a fix re-embeds only changed chunks.
export PREVIOUS_INGEST_URI="${PREVIOUS_INGEST_URI:-gs://trim-icon-498815-a0-mlops/rag/embeddings/ingest/embed_ingest.jsonl.gz}"

export PYTHONPATH="$HARNESS_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "=== Submitting RAG ingest pipeline ==="
echo "  PROJECT_ID          : $PROJECT_ID"
echo "  PIPELINE_SA         : $PIPELINE_SA"
echo "  RAG_IMAGE_URI       : $RAG_IMAGE_URI"
echo "  PREVIOUS_INGEST_URI : $PREVIOUS_INGEST_URI"
echo

cd "$HARNESS_DIR"
"$VENV_PY" pipelines/rag_ingest_pipeline.py submit
