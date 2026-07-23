#!/usr/bin/env bash
#
# build_images.sh — build & push the training container image to Artifact
# Registry via Cloud Build. Creates the `readmission` Docker repo if it doesn't
# exist. Serving uses a Custom Prediction Routine image built separately by
# scripts/deploy_cpr.py (register_model only records it for provenance).
#
# Usage:
#   bash projects/mlops/scripts/build_images.sh [training]
#
# Default target is `training`. Override PROJECT_ID / REGION by exporting them.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLOPS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"            # projects/mlops

PROJECT_ID="${PROJECT_ID:-trim-icon-498815-a0}"
REGION="${REGION:-us-east1}"
REPO="${REPO:-readmission}"
TARGET="${1:-training}"

AR_HOST="${REGION}-docker.pkg.dev"

echo "=== Container build ==="
echo "  PROJECT_ID : $PROJECT_ID"
echo "  REGION     : $REGION"
echo "  AR repo    : ${AR_HOST}/${PROJECT_ID}/${REPO}"
echo "  Target     : $TARGET"
echo

# --- Ensure the Artifact Registry Docker repo exists -------------------------
if ! gcloud artifacts repositories describe "$REPO" \
      --project "$PROJECT_ID" --location "$REGION" >/dev/null 2>&1; then
  echo "Creating Artifact Registry repo '$REPO' in $REGION …"
  gcloud artifacts repositories create "$REPO" \
    --project "$PROJECT_ID" \
    --repository-format=docker \
    --location="$REGION" \
    --description="Readmission training & serving images"
else
  echo "Artifact Registry repo '$REPO' already exists."
fi
echo

# --- Build helpers -----------------------------------------------------------
build_training() {
  echo ">>> Building TRAINING image (pipelines/Dockerfile) …"
  gcloud builds submit "$MLOPS_DIR" \
    --project "$PROJECT_ID" \
    --config "$MLOPS_DIR/pipelines/cloudbuild.yaml"
  echo "    -> ${AR_HOST}/${PROJECT_ID}/${REPO}/training:latest"
}

case "$TARGET" in
  training) build_training ;;
  *) echo "Unknown target '$TARGET' (use: training)" >&2; exit 1 ;;
esac

echo
echo "Done. To use the pinned training image for pipeline steps (optional):"
echo "  export TRAINING_IMAGE_URI=${AR_HOST}/${PROJECT_ID}/${REPO}/training:latest"
