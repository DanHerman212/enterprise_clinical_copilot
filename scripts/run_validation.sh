#!/usr/bin/env bash
# Phase D — host-side runner for TFDV inside Docker.
#
# Builds (idempotent) and runs the linux/amd64 TFDV image, mounting the
# project's artifacts/validation/ directory for outputs and the user's
# Application Default Credentials for BigQuery access. The host venv stays
# on Python 3.12; this script is the only place that needs Docker.
#
# Usage:
#   scripts/run_validation.sh
#
# Outputs:
#   artifacts/validation/{schema.pbtxt, *_stats.pb, *_anomalies.pbtxt,
#                        *.html, summary.json}

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

IMAGE="enterprise-clinical-copilot/tfdv-runner:latest"
ARTIFACTS_DIR="$REPO_ROOT/artifacts/validation"
ADC_PATH="${HOME}/.config/gcloud/application_default_credentials.json"

# Source-of-truth config values (kept here to avoid importing the host venv).
BQ_PROJECT="${BQ_PROJECT:-enterprise-clinical-copilot}"
FEATURES_TABLE="${FEATURES_TABLE:-enterprise-clinical-copilot.readmission.cohort_features}"

if [[ ! -f "$ADC_PATH" ]]; then
  echo "ERROR: Application Default Credentials not found at $ADC_PATH" >&2
  echo "Run: gcloud auth application-default login" >&2
  exit 1
fi

mkdir -p "$ARTIFACTS_DIR"

echo "[run_validation] building image $IMAGE ..."
docker build \
  --platform=linux/amd64 \
  -t "$IMAGE" \
  "$REPO_ROOT/docker/validation"

echo "[run_validation] running container ..."
docker run --rm \
  --platform=linux/amd64 \
  -e BQ_PROJECT="$BQ_PROJECT" \
  -e FEATURES_TABLE="$FEATURES_TABLE" \
  -e GOOGLE_APPLICATION_CREDENTIALS=/gcp/adc.json \
  -e GOOGLE_CLOUD_PROJECT="$BQ_PROJECT" \
  -v "$ADC_PATH":/gcp/adc.json:ro \
  -v "$ARTIFACTS_DIR":/artifacts \
  "$IMAGE"

echo "[run_validation] artifacts written to $ARTIFACTS_DIR"
ls -la "$ARTIFACTS_DIR"
