#!/usr/bin/env bash
#
# build_rag_image.sh — build & push the RAG ingest pipeline image to Artifact
# Registry via Cloud Build.
#
# Usage:
#   bash scripts/agent/build_rag_image.sh
#
# The image carries the recall component (`pipelines/recall_k.py`) that the deploy
# gate measures an index with, so a stale image is not cosmetic: on 2026-09-18 a
# `rag-ingest` image built 2026-08-21 predated the `empty_result_rate` metric, so
# the job could not report it and the gate refused a promotion over a metric that
# was never measured. Recall on that same run was 0.992 against a 0.90 floor, with
# no empty results at all. Rebuild this image whenever the components change.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The build context is services/mcp: the cloudbuild config, the Dockerfile and the
# pipeline components all live under services/mcp/pipelines, and the config reaches
# them as `pipelines/...` relative to that directory. This used to point at
# <repo>/scripts, where no such config exists — which is how the image went four
# weeks without a rebuild.
BUILD_CONTEXT="$(cd "$SCRIPT_DIR/../../services/mcp" && pwd)"

PROJECT_ID="${PROJECT_ID:-trim-icon-498815-a0}"
REGION="${REGION:-us-east1}"
REPO="${REPO:-readmission}"

echo "=== Building RAG ingest image ==="
echo "  PROJECT_ID : $PROJECT_ID"
echo "  REGION     : $REGION"
echo "  CONTEXT    : $BUILD_CONTEXT"
echo

if ! gcloud artifacts repositories describe "$REPO" \
      --project "$PROJECT_ID" --location "$REGION" >/dev/null 2>&1; then
  echo "Creating Artifact Registry repo '$REPO' in $REGION …"
  gcloud artifacts repositories create "$REPO" \
    --project "$PROJECT_ID" --repository-format=docker --location="$REGION"
fi

gcloud builds submit "$BUILD_CONTEXT" \
  --project "$PROJECT_ID" \
  --config "$BUILD_CONTEXT/pipelines/cloudbuild.yaml"
