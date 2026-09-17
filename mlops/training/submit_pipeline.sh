#!/usr/bin/env bash
#
# submit_pipeline.sh — compile & submit the readmission training pipeline to
# Vertex AI Pipelines, associated with the `readmission-mlops` experiment.
#
# Usage:
#   bash mlops/training/submit_pipeline.sh
#
# PREREQUISITE: the training image must be built first (it bakes in the project
# source that the components import at runtime):
#   bash mlops/training/build_images.sh
#
# Override any value by exporting it first, e.g. a full run:
#   N_TRIALS=50 bash mlops/training/submit_pipeline.sh
#
# SERVING_IMAGE_URI is the container this run records on the provenance entry. It
# must be an immutable reference (`repo@sha256:...`), so when it is unset this
# script resolves the newest CPR image digest from Artifact Registry instead of
# recording the mutable `:latest` tag that used to be the default. Set it to
# override. The servable model is built and deployed separately by
# mlops/serving/deploy_cpr.py, which records the digest it deploys.
set -euo pipefail

# --- Resolve paths relative to this script -----------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLOPS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"            # mlops
REPO_ROOT="$(cd "$MLOPS_DIR/.." && pwd)"             # repo root
VENV_PY="$REPO_ROOT/.venv/bin/python"

# --- Configuration (all overridable via env) ---------------------------------
export PROJECT_ID="${PROJECT_ID:-trim-icon-498815-a0}"
REGION="${REGION:-us-east1}"                         # pipeline is pinned to us-east1
export PIPELINE_ROOT="${PIPELINE_ROOT:-gs://trim-icon-498815-a0-mlops/pipeline-root}"
export N_TRIALS="${N_TRIALS:-5}"                     # dry-run default; use 50 for a full run
export HPO_TIMEOUT="${HPO_TIMEOUT:-2700}"            # HPO wall-clock backstop (sec); 45 min default
# SERVING_IMAGE_URI is the container the registry entry records for this bundle.
# It must be an immutable reference, so it is resolved here as a DIGEST rather
# than passed as a tag: a tag can be re-pushed, and then the entry names bytes
# that never trained. It is not deploy-time truth either — mlops/serving/
# deploy_cpr.py resolves and records the digest of what it actually deploys.
CPR_IMAGE_REPO="${CPR_IMAGE_REPO:-${REGION}-docker.pkg.dev/${PROJECT_ID}/readmission/readmission-cpr}"
if [[ -z "${SERVING_IMAGE_URI:-}" ]]; then
  CPR_DIGEST="$(gcloud artifacts docker images list "$CPR_IMAGE_REPO" \
      --format="value(version)" --limit=1 --sort-by=~createTime 2>/dev/null || true)"
  if [[ -z "$CPR_DIGEST" ]]; then
    echo "ERROR: no CPR serving image found in $CPR_IMAGE_REPO." >&2
    echo "The registry entry must name the container that wraps this bundle, by" >&2
    echo "digest, and there is no \`:latest\` fallback. Build it first:" >&2
    echo "  python mlops/serving/deploy_cpr.py --build-only" >&2
    echo "or set SERVING_IMAGE_URI explicitly to a repo@sha256:... reference." >&2
    exit 1
  fi
  SERVING_IMAGE_URI="${CPR_IMAGE_REPO}@${CPR_DIGEST}"
fi
export SERVING_IMAGE_URI
export PIPELINE_SA="${PIPELINE_SA:-mlops-pipeline@trim-icon-498815-a0.iam.gserviceaccount.com}"
# REQUIRED: components import their helpers from source baked into this image, and
# the run must record WHICH image, because that is the only way the code that
# produced a model can be identified afterwards. So there is no `:latest` default
# and no guessing either: the image is named here or the submit stops.
if [[ -z "${TRAINING_IMAGE_URI:-}" ]]; then
  echo "ERROR: TRAINING_IMAGE_URI is not set." >&2
  echo "Name the training image this run should use — a build tag, not :latest:" >&2
  gcloud artifacts docker images list \
    "us-east1-docker.pkg.dev/${PROJECT_ID}/readmission/training" \
    --include-tags \
    --format="table[no-heading](tags,createTime.date('%Y-%m-%d %H:%M'))" 2>/dev/null \
    | sed 's/^/  /' >&2
  echo "  export TRAINING_IMAGE_URI=us-east1-docker.pkg.dev/${PROJECT_ID}/readmission/training:<tag>" >&2
  exit 1
fi
export TRAINING_IMAGE_URI
# Which revision of the code this run came from, recorded on the registry entry so
# a model can be traced back to the source that produced it. Resolved here rather
# than inside the run: the container has no git history. A modified worktree is
# reported as such instead of pretending to be the commit it sits on.
if [[ -z "${GIT_REVISION:-}" ]] && git rev-parse --git-dir >/dev/null 2>&1; then
  GIT_REVISION="$(git rev-parse --short HEAD)"
  [[ -n "$(git status --porcelain)" ]] && GIT_REVISION="${GIT_REVISION}-dirty"
fi
export GIT_REVISION="${GIT_REVISION:-}"

if [[ ! -x "$VENV_PY" ]]; then
  echo "ERROR: project venv not found at $VENV_PY" >&2
  echo "Create it and install kfp==2.16.1 + google-cloud-aiplatform first." >&2
  exit 1
fi

# So `import mlops.training...` and `import mlops.data...` resolve: the PARENT
# of the package goes on the path, not the package directory.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "=== Submitting readmission training pipeline ==="
echo "  PROJECT_ID        : $PROJECT_ID"
echo "  REGION            : $REGION"
echo "  PIPELINE_ROOT     : $PIPELINE_ROOT"
echo "  N_TRIALS          : $N_TRIALS   (dry run = 5, full run = 50)"
echo "  HPO_TIMEOUT       : ${HPO_TIMEOUT}s   (HPO wall-clock backstop)"
echo "  PIPELINE_SA       : $PIPELINE_SA"
echo "  SERVING_IMAGE_URI : $SERVING_IMAGE_URI"
echo "  TRAINING_IMAGE_URI: $TRAINING_IMAGE_URI"
echo "  GIT_REVISION      : ${GIT_REVISION:-<unset — the entry will record no revision>}"
echo
"$VENV_PY" "$MLOPS_DIR/training/training_pipeline.py" submit
