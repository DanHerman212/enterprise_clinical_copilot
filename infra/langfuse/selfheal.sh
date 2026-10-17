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

# True while Docker is already doing something about this container.
#
# ClickHouse in a restart loop reports `Restarting` and keeps `Health: unhealthy`
# from its last probe; a container replaying its write-ahead log reports
# `health: starting`. Both are states Docker acts on by itself — `restart:
# unless-stopped` brings the process back, and the healthcheck decides when it is
# up — so a force-recreate here does not help, it interrupts. `up -d
# --force-recreate` is a stop/remove/start with a 10-second timeout, which throws
# away the replay in progress and restarts recovery from the beginning.
#
# Measured 2026-10-17: `RestartCount` read 123, and fifteen minutes later read 15
# for the same container. A counter cannot go down under `restart:
# unless-stopped`; it only resets when the container is replaced. That reset is
# this script acting on a container Docker was already recovering, every five
# minutes, which is why the loop never had a chance to finish.
#
# The 2026-10-02 case this script exists for is unaffected: there the container
# was `Running` with a failing healthcheck and Docker never restarted it. That is
# not "recovering", and it is still recreated.
docker_recovering() {
    local container="$1" restarting health
    restarting=$(docker inspect -f '{{.State.Restarting}}' "$container" 2>/dev/null) || return 1
    [ "$restarting" = "true" ] && return 0
    health=$(docker inspect -f \
        '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' \
        "$container" 2>/dev/null)
    [ "$health" = "starting" ]
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
    if docker_recovering "$container"; then
        # Logged rather than passed over in silence: a stack being restarted
        # every minute is a fault worth seeing in this log, even though acting
        # on it is the wrong response.
        log "SKIP ${container} — Docker is already recovering it"
        continue
    fi
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
