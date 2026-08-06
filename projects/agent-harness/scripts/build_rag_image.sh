#!/usr/bin/env bash
#
# build_rag_image.sh — build & push the RAG ingest pipeline image to Artifact
# Registry via Cloud Build (build context is projects/agent-harness).
#
# Usage:
#   bash projects/agent-harness/scripts/build_rag_image.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"          # projects/agent-harness

PROJECT_ID="${PROJECT_ID:-trim-icon-498815-a0}"
REGION="${REGION:-us-east1}"
REPO="${REPO:-readmission}"

echo "=== Building RAG ingest image ==="
echo "  PROJECT_ID : $PROJECT_ID"
echo "  REGION     : $REGION"
echo

if ! gcloud artifacts repositories describe "$REPO" \
      --project "$PROJECT_ID" --location "$REGION" >/dev/null 2>&1; then
  echo "Creating Artifact Registry repo '$REPO' in $REGION …"
  gcloud artifacts repositories create "$REPO" \
    --project "$PROJECT_ID" --repository-format=docker --location="$REGION"
fi

gcloud builds submit "$HARNESS_DIR" \
  --project "$PROJECT_ID" \
  --config "$HARNESS_DIR/pipelines/cloudbuild.yaml"
