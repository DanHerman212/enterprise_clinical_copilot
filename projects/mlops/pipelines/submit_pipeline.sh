#!/usr/bin/env bash
#
# submit_pipeline.sh — compile & submit the readmission training pipeline to
# Vertex AI Pipelines, associated with the `readmission-mlops` experiment.
#
# Usage:
#   bash projects/mlops/pipelines/submit_pipeline.sh
#
# PREREQUISITE: the training image must be built first (it bakes in the project
# source that the components import at runtime):
#   bash projects/mlops/scripts/build_images.sh all
#
# Override any value by exporting it first, e.g. a full run with serving image:
#   N_TRIALS=50 \
#   SERVING_IMAGE_URI=us-east1-docker.pkg.dev/trim-icon-498815-a0/readmission/serving:latest \
#     bash projects/mlops/pipelines/submit_pipeline.sh
set -euo pipefail

# --- Resolve paths relative to this script -----------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLOPS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"            # projects/mlops
REPO_ROOT="$(cd "$MLOPS_DIR/../.." && pwd)"          # repo root
VENV_PY="$REPO_ROOT/.venv/bin/python"

# --- Configuration (all overridable via env) ---------------------------------
export PROJECT_ID="${PROJECT_ID:-trim-icon-498815-a0}"
REGION="${REGION:-us-east1}"                         # pipeline is pinned to us-east1
export PIPELINE_ROOT="${PIPELINE_ROOT:-gs://trim-icon-498815-a0-mlops/pipeline-root}"
export N_TRIALS="${N_TRIALS:-5}"                     # dry-run default; use 50 for a full run
export SERVING_IMAGE_URI="${SERVING_IMAGE_URI:-}"    # empty -> register_model step will fail
export PIPELINE_SA="${PIPELINE_SA:-mlops-pipeline@trim-icon-498815-a0.iam.gserviceaccount.com}"
# REQUIRED: components import their helpers from source baked into this image.
export TRAINING_IMAGE_URI="${TRAINING_IMAGE_URI:-us-east1-docker.pkg.dev/trim-icon-498815-a0/readmission/training:latest}"

if [[ ! -x "$VENV_PY" ]]; then
  echo "ERROR: project venv not found at $VENV_PY" >&2
  echo "Create it and install kfp==2.16.1 + google-cloud-aiplatform first." >&2
  exit 1
fi

# So `import pipelines...` and `import src...` resolve.
export PYTHONPATH="$MLOPS_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "=== Submitting readmission training pipeline ==="
echo "  PROJECT_ID        : $PROJECT_ID"
echo "  REGION            : $REGION"
echo "  PIPELINE_ROOT     : $PIPELINE_ROOT"
echo "  N_TRIALS          : $N_TRIALS   (dry run = 5, full run = 50)"
echo "  PIPELINE_SA       : $PIPELINE_SA"
echo "  SERVING_IMAGE_URI : ${SERVING_IMAGE_URI:-<unset — register_model step will fail>}"
echo "  TRAINING_IMAGE_URI: $TRAINING_IMAGE_URI"
echo
"$VENV_PY" "$MLOPS_DIR/pipelines/training_pipeline.py" submit
