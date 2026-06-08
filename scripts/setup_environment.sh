#!/usr/bin/env bash
#
# setup_environment.sh — one-time GCP bootstrap: APIs, IAM, storage. Idempotent.
# Reads config from the repo-root .env (see .env.example).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "${REPO_ROOT}/.env" ]] || { echo "Missing .env (copy .env.example)." >&2; exit 1; }
set -a; source "${REPO_ROOT}/.env"; set +a

: "${PROJECT_ID:?set PROJECT_ID in .env}"
REGION="${REGION:-us-central1}"
PIPELINE_SA_NAME="${PIPELINE_SA_NAME:-mlops-pipeline}"
GCS_BUCKET="${GCS_BUCKET:-${PROJECT_ID}-mlops}"
BQ_DATASET="${BQ_DATASET:-readmission}"
SA_EMAIL="${PIPELINE_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud config set project "${PROJECT_ID}" >/dev/null

echo "==> Enabling APIs"
gcloud services enable \
  aiplatform.googleapis.com artifactregistry.googleapis.com bigquery.googleapis.com \
  cloudbuild.googleapis.com cloudscheduler.googleapis.com dataform.googleapis.com \
  iam.googleapis.com pubsub.googleapis.com storage.googleapis.com

echo "==> Service account + IAM"
gcloud iam service-accounts describe "${SA_EMAIL}" >/dev/null 2>&1 || \
  gcloud iam service-accounts create "${PIPELINE_SA_NAME}" --display-name "MLOps pipeline"
for role in aiplatform.user artifactregistry.writer bigquery.dataEditor bigquery.jobUser \
            dataform.editor pubsub.editor storage.objectAdmin; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member "serviceAccount:${SA_EMAIL}" --role "roles/${role}" --condition=None --quiet >/dev/null
done

echo "==> Storage areas"
gcloud storage buckets describe "gs://${GCS_BUCKET}" >/dev/null 2>&1 || \
  gcloud storage buckets create "gs://${GCS_BUCKET}" --location "${REGION}" --uniform-bucket-level-access
bq --project_id="${PROJECT_ID}" show --dataset "${BQ_DATASET}" >/dev/null 2>&1 || \
  bq --project_id="${PROJECT_ID}" mk --dataset --location="${REGION}" "${BQ_DATASET}"

echo "Done."
