#!/usr/bin/env bash
# Deploy Langfuse's stateless half (web + worker) to Cloud Run.
#
# The data plane lives on the `langfuse-data` VM (see docker-compose.yml in this
# directory) and Postgres on the existing Cloud SQL instance. These two services
# are the only part that benefits from Cloud Run: they are stateless, they should
# scale to zero when nobody is looking, and they need to be reachable at a
# hostname.
#
# Connections:
#   * ClickHouse, Redis and MinIO are on the VM's private address, reached with
#     Direct VPC egress (`--vpc-egress=private-ranges-only`): private destinations
#     go over the VPC, everything else leaves directly. No VPC connector, so no
#     extra hop and no extra component to pay for.
#   * Postgres is the Cloud SQL socket, mounted with --set-cloudsql-instances.
#
# Secrets are passed with --set-secrets, never --set-env-vars: an environment
# variable is visible to anyone who can describe the revision, and these include
# the database password and the encryption key that protects stored credentials.
#
# Usage: bash infra/langfuse/deploy.sh
#
# The images are pulled from `docker.langfuse.com` through a remote repository
# in Artifact Registry (`langfuse`), not from the upstream registry: Cloud Run
# refuses an image whose host is not gcr.io, docker.io or Artifact Registry, and
# the pull-through cache is the supported way to bridge that. It also means the
# digest Cloud Run runs is one Artifact Registry has already seen.
set -euo pipefail

PROJECT=trim-icon-498815-a0
REGION=us-east1
VM_IP=10.142.0.3
SQL_CONNECTION="$PROJECT:$REGION:danielmherman-db"
# Mirrored from `docker.langfuse.com` by `mirror-images.yaml`, and referenced by
# DIGEST rather than by the `4` tag: a major tag is a moving target, and what a
# revision ran should be answerable a month later. Re-mirror and update these two
# values together — the build prints the digests.
WEB_IMAGE=us-east1-docker.pkg.dev/$PROJECT/langfuse-images/langfuse-web@sha256:5c0a19ef70e6d8a896150f9b23d1e3eecb4a9d4f206185199e870729a6ec9c89
WORKER_IMAGE=us-east1-docker.pkg.dev/$PROJECT/langfuse-images/langfuse-worker@sha256:a5b42c6194ee4434de90f18fcdc49cf00e63e3b6d7fda93f4816d0e7dd6bc73f
DOMAIN=observability.danielmherman.com
DNS_ZONE=danielmherman
SA=langfuse-sa@$PROJECT.iam.gserviceaccount.com
BUCKET=langfuse

echo "=== service account ==="
if gcloud iam service-accounts describe "$SA" --project "$PROJECT" >/dev/null 2>&1; then
  echo "  present: $SA"
else
  gcloud iam service-accounts create langfuse-sa --project "$PROJECT" \
    --display-name "Langfuse Cloud Run services" >/dev/null
  echo "  created: $SA"
fi

echo "=== MinIO bucket ==="
gcloud compute ssh langfuse-data --zone us-east1-b --project "$PROJECT" \
  --tunnel-through-iap --command "
    cd /opt/langfuse && set -a && . ./.env && set +a
    sudo docker exec langfuse-minio mc alias set local http://localhost:9100 \
      \"\$MINIO_ROOT_USER\" \"\$MINIO_ROOT_PASSWORD\" >/dev/null 2>&1 || true
    sudo docker exec langfuse-minio mc mb --ignore-existing local/$BUCKET
    sudo docker exec langfuse-minio mc ls local
  " 2>&1 | grep -vE "Warning:|To increase|please see|^\s*$" | tail -4

# Shared configuration. Everything here is non-secret; the secrets follow.
# The delimiter is `|`, NOT the `^@^` that the site's deploy uses: one of these
# values is an email address, and a delimiter that appears inside a value splits
# the list there — `..._EMAIL=dan@danielmherman.com` was read as two entries, and
# the service was refused with a usage error rather than a bad value.
COMMON_ENV="^|^NODE_ENV=production\
|TELEMETRY_ENABLED=false\
|CLICKHOUSE_URL=http://$VM_IP:8123\
|CLICKHOUSE_MIGRATION_URL=clickhouse://$VM_IP:9000\
|CLICKHOUSE_USER=default\
|CLICKHOUSE_CLUSTER_ENABLED=false\
|REDIS_HOST=$VM_IP\
|REDIS_PORT=6379\
|LANGFUSE_S3_EVENT_UPLOAD_BUCKET=$BUCKET\
|LANGFUSE_S3_EVENT_UPLOAD_REGION=auto\
|LANGFUSE_S3_EVENT_UPLOAD_ENDPOINT=http://$VM_IP:9100\
|LANGFUSE_S3_EVENT_UPLOAD_FORCE_PATH_STYLE=true\
|LANGFUSE_S3_EVENT_UPLOAD_PREFIX=events/\
|LANGFUSE_S3_MEDIA_UPLOAD_BUCKET=$BUCKET\
|LANGFUSE_S3_MEDIA_UPLOAD_REGION=auto\
|LANGFUSE_S3_MEDIA_UPLOAD_ENDPOINT=http://$VM_IP:9100\
|LANGFUSE_S3_MEDIA_UPLOAD_FORCE_PATH_STYLE=true\
|LANGFUSE_S3_MEDIA_UPLOAD_PREFIX=media/"

COMMON_SECRETS="DATABASE_URL=langfuse-database-url:latest\
,CLICKHOUSE_PASSWORD=langfuse-clickhouse-password:latest\
,REDIS_AUTH=langfuse-redis-auth:latest\
,LANGFUSE_S3_EVENT_UPLOAD_ACCESS_KEY_ID=langfuse-minio-user:latest\
,LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY=langfuse-minio-password:latest\
,LANGFUSE_S3_MEDIA_UPLOAD_ACCESS_KEY_ID=langfuse-minio-user:latest\
,LANGFUSE_S3_MEDIA_UPLOAD_SECRET_ACCESS_KEY=langfuse-minio-password:latest"

deploy() { # service image port cpu memory extra_env extra_secrets
  local svc=$1 image=$2 port=$3 cpu=$4 memory=$5 extra_env=$6 extra_secrets=$7
  echo "=== deploying $svc ==="
  local args=(
    run deploy "$svc"
    --image "$image"
    --region "$REGION"
    --project "$PROJECT"
    --service-account "$SA"
    --allow-unauthenticated
    --set-cloudsql-instances "$SQL_CONNECTION"
    --network default --subnet default --vpc-egress private-ranges-only
    --min-instances 0 --max-instances 2
    --cpu "$cpu" --memory "$memory"
    --set-env-vars "$COMMON_ENV$extra_env"
    --set-secrets "$COMMON_SECRETS$extra_secrets"
    --quiet
  )
  # The worker serves an HTTP health endpoint rather than a UI, so it needs its
  # port stated for the same reason: Cloud Run routes to a port, and a service
  # with no listening port is marked unhealthy however hard the worker is working.
  [ -n "$port" ] && args+=(--port "$port")
  gcloud "${args[@]}" 2>&1 | tail -3
}

deploy langfuse-worker "$WORKER_IMAGE" 3030 1 1Gi "" \
  ",SALT=langfuse-salt:latest\
,ENCRYPTION_KEY=langfuse-encryption-key:latest"

# 2 vCPU / 2Gi and an explicit heap ceiling, because 1Gi is not enough and the
# failure was not obvious from the outside: the revision reported Ready, then the
# container aborted with `FATAL ERROR: Reached heap limit` (signal 6) a few
# seconds later, so the service answered 503 while Cloud Run believed it healthy.
# The web image applies 438 Prisma migrations on boot and then runs Next.js; the
# heap ceiling is set below the container limit so Node fails its own allocation
# with a message rather than being killed by the kernel with none.
deploy langfuse-web "$WEB_IMAGE" 3000 2 2Gi \
  "|NODE_OPTIONS=--max-old-space-size=1536\
|NEXTAUTH_URL=https://$DOMAIN\
|LANGFUSE_INIT_ORG_ID=ecc-observability\
|LANGFUSE_INIT_ORG_NAME=Enterprise Clinical Copilot\
|LANGFUSE_INIT_PROJECT_ID=ecc-agent\
|LANGFUSE_INIT_PROJECT_NAME=Agent and evals\
|LANGFUSE_INIT_USER_EMAIL=dan@danielmherman.com\
|LANGFUSE_INIT_USER_NAME=Dan Herman" \
  ",NEXTAUTH_SECRET=langfuse-nextauth-secret:latest\
,SALT=langfuse-salt:latest\
,ENCRYPTION_KEY=langfuse-encryption-key:latest\
,LANGFUSE_INIT_USER_PASSWORD=langfuse-ui-password:latest\
,LANGFUSE_INIT_PROJECT_PUBLIC_KEY=langfuse-project-public-key:latest\
,LANGFUSE_INIT_PROJECT_SECRET_KEY=langfuse-project-secret-key:latest"

echo "=== reachable at ==="
gcloud run services describe langfuse-web --region "$REGION" --project "$PROJECT" \
  --format='value(status.url)'

# --- the front door ---------------------------------------------------------
# A domain mapping, not a load balancer: the site's apex and www already work
# this way (A records to Google's mapping addresses, www as a CNAME to
# ghs.googlehosted.com), so this reuses the mechanism the domain is already
# served by instead of introducing a second one. It also holds a trap worth
# naming: if the application's edge later moves to a global LB, do NOT add a
# wildcard record — it would capture this subdomain and send it to the LB, whose
# URL map has no host rule for it.
#
# The mapping requires the service to accept unauthenticated traffic, which it
# does (above). That also means the *.run.app URL remains publicly reachable; the
# gating for this UI is Langfuse's own authentication, and Cloud Armor or IAP at
# the edge if the domain is ever put behind the LB.
echo "=== domain mapping: $DOMAIN ==="
if gcloud beta run domain-mappings describe --domain "$DOMAIN" \
     --region "$REGION" --project "$PROJECT" >/dev/null 2>&1; then
  echo "  mapping exists"
else
  gcloud beta run domain-mappings create --service langfuse-web --domain "$DOMAIN" \
    --region "$REGION" --project "$PROJECT"
fi

# The record the mapping asks for, written to Cloud DNS by this script rather
# than left as a manual step: the certificate cannot be issued until it exists,
# and "waiting for a DNS record nobody added" is an outage that looks like a
# provisioning bug.
if gcloud dns record-sets describe "$DOMAIN." --zone "$DNS_ZONE" \
     --project "$PROJECT" --type CNAME >/dev/null 2>&1; then
  echo "  DNS record exists"
else
  gcloud dns record-sets create "$DOMAIN." --zone "$DNS_ZONE" --project "$PROJECT" \
    --type CNAME --ttl 300 --rrdatas "ghs.googlehosted.com."
fi
echo "  certificate provisioning continues in the background; check with:"
echo "  gcloud beta run domain-mappings describe --domain $DOMAIN --region $REGION \\"
echo "    --project $PROJECT --format='value(status.conditions[0].message)'"
