#!/usr/bin/env bash
#
# Self-heal the Langfuse data plane.
#
# Why this exists. On 2026-10-02 the host's OOM killer terminated the ClickHouse
# server. Docker did not mark the container exited: it stayed listed as "Up"
# while its healthcheck reported "cannot exec in a stopped state", so
# `restart: unless-stopped` never fired and nothing reported a problem for five
# days. The web tier runs ClickHouse migrations at startup and exits when they
# fail, which turned a dead database into a 503 at the front door.
#
# What it does. On every run it checks that each container is running and that
# Docker considers it healthy, and it probes ClickHouse for real — a container
# can satisfy a healthcheck and still be unable to answer a query. Anything that
# fails is recreated. It is deliberately blunt: recreating a container that is
# already broken costs seconds, whereas the failure it guards against cost five
# days of silence.
#
# Installed to /opt/langfuse/selfheal.sh, run by langfuse-selfheal.timer every
# five minutes. Source of truth is this file in the repository.

set -uo pipefail

cd /opt/langfuse || exit 1

LOG=/var/log/langfuse-selfheal.log
STARTUP_GRACE_SECONDS=90

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >>"$LOG"; }

# One run at a time, so a slow recreate cannot overlap the next tick.
exec 9>/var/lock/langfuse-selfheal.lock
flock -n 9 || exit 0

recreate() {
    local service="$1" reason="$2"
    log "RECREATE ${service} — ${reason}"
    docker compose up -d --force-recreate "$service" >>"$LOG" 2>&1
}

# Empty output means healthy; anything else is the reason to act.
container_fault() {
    local container="$1" running health
    if ! running=$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null); then
        echo "not present"
        return
    fi
    if [ "$running" != "true" ]; then
        echo "not running"
        return
    fi
    health=$(docker inspect -f \
        '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' \
        "$container" 2>/dev/null)
    if [ "$health" = "unhealthy" ]; then
        echo "unhealthy"
    fi
}

# True when the container has been up long enough that a failure to answer is a
# fault rather than a start in progress.
settled() {
    local container="$1" started now
    started=$(docker inspect -f '{{.State.StartedAt}}' "$container" 2>/dev/null) || return 1
    now=$(date -u +%s)
    [ $((now - $(date -u -d "$started" +%s))) -gt "$STARTUP_GRACE_SECONDS" ]
}

# service:container, for every component whose death breaks the stack. The Cloud
# SQL proxy is included because the web tier cannot reach Postgres without it.
for pair in clickhouse:langfuse-clickhouse \
            redis:langfuse-redis \
            minio:langfuse-minio \
            worker:langfuse-worker \
            cloud-sql-proxy:langfuse-cloud-sql-proxy; do
    service=${pair%%:*}
    container=${pair##*:}
    if reason=$(container_fault "$container") && [ -n "$reason" ]; then
        recreate "$service" "${container} is ${reason}"
    fi
done

if settled langfuse-clickhouse; then
    if ! curl -fsS --max-time 5 http://localhost:8123/ping >/dev/null 2>&1; then
        recreate clickhouse "ClickHouse did not answer its ping"
    fi
fi

exit 0
