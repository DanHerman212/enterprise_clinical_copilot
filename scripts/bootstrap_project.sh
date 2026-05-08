#!/usr/bin/env bash
# bootstrap_project.sh — one-time GCP project setup for the
# enterprise_clinical_copilot MLOps pipeline.
#
# What this script does (idempotent — safe to re-run):
#   1. Enable the GCP APIs we need.
#   2. Create the MLOps service account.
#   3. Grant the service account the IAM roles it needs.
#   4. Create the GCS bucket used for Vertex staging + artifacts + TFDV stats.
#   5. (BigQuery dataset `readmission` is created by the notebook itself —
#      no action here.)
#
# Run from the repo root:
#   bash scripts/bootstrap_project.sh
#
# Prereqs:
#   - gcloud CLI installed and authenticated (`gcloud auth login`)
#   - You have Owner or equivalent on the project
#
# After this script:
#   - Application Default Credentials (`gcloud auth application-default
#     login`) are used by the notebook for now. The service account is
#     created but no key is downloaded; we'll attach it to Vertex jobs
#     via `--service-account` rather than handing keys around.

set -euo pipefail

# --- Config -----------------------------------------------------------
PROJECT_ID="enterprise-clinical-copilot"
VERTEX_REGION="us-east1"           # Vertex AI Experiments + Pipelines
BQ_LOCATION="US"                   # already set by the notebook (multi-region)
BUCKET_NAME="${PROJECT_ID}-mlops"  # gs://enterprise-clinical-copilot-mlops
BUCKET_LOCATION="${VERTEX_REGION}" # co-locate with Vertex to avoid egress

SA_NAME="clinical-copilot-mlops"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
SA_DISPLAY="Clinical Copilot MLOps"

# Roles the service account needs end-to-end (training, eval, registry,
# artifact storage, BigQuery reads/writes against the `readmission` dataset).
ROLES=(
  "roles/aiplatform.user"           # Vertex experiments, jobs, pipelines, registry
  "roles/storage.objectAdmin"       # read/write the MLOps bucket
  "roles/bigquery.dataEditor"       # read/write `readmission` dataset
  "roles/bigquery.jobUser"          # run query jobs
  "roles/artifactregistry.writer"   # push container images (Phase F)
  "roles/logging.logWriter"         # write logs from Vertex jobs
)

# APIs the pipeline depends on across all phases.
APIS=(
  "aiplatform.googleapis.com"
  "storage.googleapis.com"
  "bigquery.googleapis.com"
  "artifactregistry.googleapis.com"
  "cloudbuild.googleapis.com"
  "iamcredentials.googleapis.com"
  "logging.googleapis.com"
)

# --- Helpers ----------------------------------------------------------
say() { printf "\n\033[1;34m▶ %s\033[0m\n" "$*"; }
ok()  { printf "  \033[32m✓\033[0m %s\n" "$*"; }

# --- 0. Sanity ---------------------------------------------------------
say "Verifying gcloud is authenticated and pointed at ${PROJECT_ID}"
gcloud config set project "${PROJECT_ID}" >/dev/null
ACTIVE_ACCOUNT=$(gcloud config get-value account 2>/dev/null)
ok "active gcloud account: ${ACTIVE_ACCOUNT}"
ok "project:               ${PROJECT_ID}"
ok "Vertex region:         ${VERTEX_REGION}"
ok "BQ location:           ${BQ_LOCATION}"
ok "MLOps bucket:          gs://${BUCKET_NAME}  (${BUCKET_LOCATION})"

# --- 1. Enable APIs ----------------------------------------------------
say "Enabling APIs"
gcloud services enable "${APIS[@]}" --project="${PROJECT_ID}"
for api in "${APIS[@]}"; do ok "${api}"; done

# --- 2. Service account -----------------------------------------------
say "Ensuring service account ${SA_EMAIL}"
if gcloud iam service-accounts describe "${SA_EMAIL}" \
      --project="${PROJECT_ID}" >/dev/null 2>&1; then
  ok "already exists"
else
  gcloud iam service-accounts create "${SA_NAME}" \
      --display-name="${SA_DISPLAY}" \
      --project="${PROJECT_ID}"
  ok "created"
fi

# --- 3. IAM bindings ---------------------------------------------------
say "Granting IAM roles to ${SA_EMAIL}"
for role in "${ROLES[@]}"; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
      --member="serviceAccount:${SA_EMAIL}" \
      --role="${role}" \
      --condition=None \
      --quiet >/dev/null
  ok "${role}"
done

# --- 4. GCS bucket -----------------------------------------------------
say "Ensuring GCS bucket gs://${BUCKET_NAME}"
if gcloud storage buckets describe "gs://${BUCKET_NAME}" >/dev/null 2>&1; then
  ok "already exists"
else
  gcloud storage buckets create "gs://${BUCKET_NAME}" \
      --project="${PROJECT_ID}" \
      --location="${BUCKET_LOCATION}" \
      --uniform-bucket-level-access \
      --public-access-prevention
  ok "created"
fi

# Bucket layout (created lazily by Vertex/TFDV; documented here):
#   gs://${BUCKET_NAME}/staging/      Vertex AI staging (jobs, pipelines)
#   gs://${BUCKET_NAME}/artifacts/    model artifacts, eval reports
#   gs://${BUCKET_NAME}/tfdv/         baseline schema + stats reference
#   gs://${BUCKET_NAME}/pipelines/    KFP compiled pipeline specs

# --- 5. Summary --------------------------------------------------------
say "Done. Useful refs for the notebook:"
cat <<EOF

  PROJECT_ID      = "${PROJECT_ID}"
  VERTEX_REGION   = "${VERTEX_REGION}"
  GCS_BUCKET      = "gs://${BUCKET_NAME}"
  STAGING_BUCKET  = "gs://${BUCKET_NAME}/staging"
  ARTIFACTS_ROOT  = "gs://${BUCKET_NAME}/artifacts"
  SERVICE_ACCOUNT = "${SA_EMAIL}"

  BigQuery dataset \`readmission\` is built by the notebook (Cell 2).

EOF
