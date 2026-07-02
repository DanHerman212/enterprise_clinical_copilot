#!/usr/bin/env bash
#
# build_images.sh — build & push the training and serving container images to
# Artifact Registry via Cloud Build. Creates the `readmission` Docker repo if
# it doesn't exist.
#
# Usage:
#   bash projects/mlops/scripts/build_images.sh [training|serving|all]
#
# Default target is `all`. Override PROJECT_ID / REGION by exporting them.
#
# After a serving build, export the printed URI before submitting the pipeline:
#   export SERVING_IMAGE_URI=us-east1-docker.pkg.dev/<PROJECT>/readmission/serving:latest
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLOPS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"            # projects/mlops

PROJECT_ID="${PROJECT_ID:-trim-icon-498815-a0}"
REGION="${REGION:-us-east1}"
REPO="${REPO:-readmission}"
TARGET="${1:-all}"

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

build_serving() {
  echo ">>> Building SERVING image (pipelines/serving/Dockerfile) …"
  gcloud builds submit "$MLOPS_DIR" \
    --project "$PROJECT_ID" \
    --config "$MLOPS_DIR/pipelines/serving/cloudbuild.yaml"
  echo "    -> ${AR_HOST}/${PROJECT_ID}/${REPO}/serving:latest"
}

case "$TARGET" in
  training) build_training ;;
  serving)  build_serving ;;
  all)      build_training; build_serving ;;
  *) echo "Unknown target '$TARGET' (use: training | serving | all)" >&2; exit 1 ;;
esac

echo
echo "Done. To use the serving image in the pipeline:"
echo "  export SERVING_IMAGE_URI=${AR_HOST}/${PROJECT_ID}/${REPO}/serving:latest"
echo "To use the pinned training image for pipeline steps (optional):"
echo "  export TRAINING_IMAGE_URI=${AR_HOST}/${PROJECT_ID}/${REPO}/training:latest"
