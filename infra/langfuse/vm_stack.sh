#!/usr/bin/env bash
# Bring the whole Langfuse data plane up on the `langfuse-data` VM.
#
# This is the stack that `deploy.sh` deliberately does not touch: ClickHouse,
# Redis, MinIO, the Cloud SQL auth proxy, and the worker. See docker-compose.yml
# for why each piece is here and, in the worker's case, why it is not on Cloud
# Run.
#
# Run it on a fresh VM or after changing docker-compose.yml. It is idempotent:
# compose converges the running set to the file, so re-running changes only what
# differs.
#
# What it needs from you first:
#   * the VM exists with a cloud-platform scope (a default-scope VM cannot mint
#     the token the proxy needs, and the failure reads like a permissions error)
#   * the VM's service account has roles/cloudsql.client on the instance
#   * the secrets named below exist in Secret Manager
#
# Usage: bash infra/langfuse/vm_stack.sh
set -euo pipefail

PROJECT=trim-icon-498815-a0
ZONE=us-east1-b
VM=langfuse-data
REMOTE=/opt/langfuse

# Staged 600 and removed on exit: this file holds the database password, the
# ClickHouse and Redis passwords, the MinIO keys, and the encryption key that
# protects stored provider credentials. It is the most sensitive file in the
# stack, so it is compiled in one place and never sits in a checkout.
STAGE=$(mktemp)
trap 'rm -f "$STAGE"' EXIT

val() { gcloud secrets versions access latest --secret="$1" --project "$PROJECT"; }

# The worker needs a Postgres URL it can reach over TCP; the web service reaches
# the same database over the Cloud SQL socket mounted by Cloud Run. Rather than
# store the password a second time — a second copy is a second thing to rotate,
# and rotations do not fail loudly — derive the worker's URL from the web's.
#
# postgresql://user:pass@localhost/dbname?host=/cloudsql/proj:region:inst&connection_limit=5
#   -> postgresql://user:pass@cloud-sql-proxy:5432/dbname?connection_limit=5&sslmode=disable
#
# Parsed rather than substituted, because the parameters after the socket path
# are real and must survive: a regex substitution dropped the `?` and left the
# tail glued to the database name, so the worker came up and asked Postgres for a
# database called `langfuse&connection_limit=5`. Everything else about its config
# was correct, which is why this is worth doing properly instead of cleverly.
#
# sslmode is explicit rather than left to libpq's `prefer`: this connection is
# plaintext across the compose network to the proxy, which encrypts the hop to
# Google. `prefer` would try TLS first against something with nothing to offer.
SOCKET_URL=$(val langfuse-database-url)
WORKER_URL=$(python3 -c '
import sys, urllib.parse as u
url = u.urlparse(sys.argv[1])
params = dict(u.parse_qsl(url.query))
params.pop("host", None)  # the socket path under /cloudsql exists only on Cloud Run
params.setdefault("sslmode", "disable")
netloc = url.netloc.rsplit("@", 1)[0] + "@cloud-sql-proxy:5432"
print(u.urlunparse(url._replace(netloc=netloc, query=u.urlencode(params))))
' "$SOCKET_URL")

case "$WORKER_URL" in
  *cloud-sql-proxy:5432*) ;;
  *) echo "ERROR: could not derive the worker database URL" >&2; exit 1 ;;
esac

umask 077
{
  echo "CLICKHOUSE_PASSWORD=$(val langfuse-clickhouse-password)"
  echo "REDIS_AUTH=$(val langfuse-redis-auth)"
  echo "MINIO_ROOT_USER=$(val langfuse-minio-user)"
  echo "MINIO_ROOT_PASSWORD=$(val langfuse-minio-password)"
  echo "WORKER_DATABASE_URL=$WORKER_URL"
  echo "SALT=$(val langfuse-salt)"
  echo "ENCRYPTION_KEY=$(val langfuse-encryption-key)"
} > "$STAGE"

echo "=== copying the stack definition and secrets ==="
gcloud compute scp --tunnel-through-iap --zone "$ZONE" --project "$PROJECT" \
  "$(dirname "$0")/docker-compose.yml" "$STAGE" "$VM:$REMOTE/" >/dev/null 2>&1

# The staging file has to land as `.env`: compose interpolates the
# project-directory .env and nothing else, so a differently-named file is not
# read and every `${VAR:?}` guard fires even though the values arrived intact.
gcloud compute ssh "$VM" --zone "$ZONE" --project "$PROJECT" --tunnel-through-iap --command "
  set -e
  cd $REMOTE
  mv -f '$(basename "$STAGE")' .env
  sudo chmod 600 .env
  sudo docker compose up -d --quiet-pull 2>&1 | tail -6
  echo '--- containers ---'
  sudo docker compose ps --format 'table {{.Name}}\t{{.Status}}'
" 2>&1 | grep -vE "Warning:|To increase|please see|^$" | tail -14

echo
echo "=== if the worker looked unhealthy, read its log first ==="
echo "gcloud compute ssh $VM --zone $ZONE --project $PROJECT --tunnel-through-iap \\"
echo "  --command 'sudo docker logs langfuse-worker 2>&1 | tail -30'"
