# Langfuse hardening — a durable fix for the observability platform

Status: **in progress — re-pinned 2026-10-16, paused again for the UX workshop.** Phases 0
and 2 are complete; Phases 1, 3, 4 and 5 remain. It was paused the first time on 2026-10-15 on
the assessment that "the platform is healthy as it stands". **That assessment was wrong.** The
2026-10-16 session — recorded at the foot of this document — found the store in a restart loop,
roughly one restart every 40 seconds. The resize bought headroom, not health, which is what the
section below already predicted it would do.

Written 2026-10-15. Supersedes nothing; extends the incident record in
`docs/architecture/layer-10-observability.md` and the notes in
`docs/production_hardening_pre_launch_checklist.md`.

---

## Where this stands — paused 2026-10-15

| Phase | State |
|---|---|
| 0 — Baseline | **Done.** Recorded below, measured rather than inferred. |
| 1 — Make the failure visible | **Partial.** 1.1 is diagnosed but not closed; 1.2 is not started and needs a decision. |
| 2 — Resize | **Done and verified.** The platform is healthy. |
| 3 — Stop the steady-state climb | Not started. |
| 4 — The two reliability bugs | Not started. |
| 5 — Verification | Not started; 5.4 is the acceptance test. |

**Phase 2, verified as applied rather than as written.** Host `7950 MB` (was 3924);
ClickHouse cgroup `4294967296 B` = 4 GB (was 2 GB); effective
`max_server_memory_usage` = `3000000000`, read back from `system.server_settings`
rather than trusted from the file; `OOMKilled=false`; all five containers `Up
(healthy)`; ClickHouse answering `SELECT count() FROM system.tables` (176); front door
`HTTP 200` in 0.33 s. The instance kept internal IP `10.142.0.3`, which the Cloud Run
services depend on.

**Phase 1.1, open.** The channel is enabled and the policy is correctly built: it uses
`REDUCE_COUNT_FALSE` over `uptime_check/check_passed`, so the `COMPARISON_GT 1.0` applies
to a count of *failures* and not to the metric itself. The only suspect is the channel,
whose `verificationStatus` is absent — which for an email channel means unverified, and
unverified channels do not deliver. The resize outage was a genuine induced failure and
is plausibly enough to breach a `> 1 failure in 300 s` threshold across a ~3-minute
window, but whether the check's probe period yields more than one sample in that window
has **not** been confirmed. So the outstanding observation is simply whether that email
arrived, and that needs a human to check an inbox.

**Phase 1.2, blocked on a decision.** The cheap implementation is ruled out: container
stdout does not reach Cloud Logging on this VM (only `GCEGuestAgent` and
`OSConfigAgent` do), so there is nothing to build a log-based metric over. Either the
check routes through a path that already exists, or the VM gains a log forwarder — and
the second is an unscoped change to the box.

**A bug found and fixed while executing Phase 2.** `vm_stack.sh` copied
`docker-compose.yml` but never the `clickhouse/config.d/memory.xml` it mounts, so the
ceiling could be edited in the repo and never reach the VM. Adding that copy exposed a
second problem: `clickhouse/` on the VM is root-owned while `$REMOTE` belongs to the
login user, so the direct copy failed with `Permission denied` *after* replacing the
compose file — leaving the VM momentarily at a 4 GB cgroup with a 1.6 GB server cap. The
script now stages the file and installs it with `sudo`, so the two limits move together.

---

## Why this is now on the critical path

The UX work is adding **answer rating**: a rating is stored in the chain and read back
in Langfuse. That changes what "fixed" means. The acceptance test is no longer
"the UI loads" — it is **"I can open a trace and read a score against it."**

This plan is therefore ordered so that the thing being fixed is also the thing that
tells you whether the fix worked. Observability first, then the change it observes.

---

## The symptom, and why it keeps misreading

Login works. Traces are empty. The front door returns **200**.

That 200 is the whole problem: the shell loads, so it reads as an application bug
rather than a starved process. Every incident so far has this shape, including
2026-10-13, when the user's description was "not working properly for whatever
reason". There is a reason, and it is measured.

---

## Root cause — measured, not inferred

- `langfuse-data` is an **`e2-medium` (2 vCPU, 4 GB)** in `us-east1-b`, running five
  containers: ClickHouse, Redis, MinIO, `langfuse-worker`, `cloud-sql-proxy`.
- **2026-10-13:** the kernel OOM-killed `clickhouse-server` **twice in 90 seconds**,
  `anon-rss ≈ 2.05 GB`. Source: `gcloud compute instances get-serial-port-output
  langfuse-data --port 1 | grep -i oom-kill`.
- The configured caps are **below what ClickHouse already uses**:
  - `infra/langfuse/clickhouse/config.d/memory.xml` → `max_server_memory_usage` = 1.6 GB
  - `infra/langfuse/docker-compose.yml` → `mem_limit: 2g`
- The 2026-10-07 `mem_limit: 2g` change converted a **VM-wide** OOM into a **contained**
  one. The VM survives, SSH survives, and `infra/langfuse/selfheal.sh` restarts
  ClickHouse within 5 minutes — which is exactly why it presents as intermittent
  rather than broken.
- **What drove the growth:** the 2026-09-30 evaluation run left ClickHouse a merge
  backlog. Its scheduler thread stalled for 34 minutes before the first kill.

---

## The correction that matters most

**Resizing the VM without raising the caps changes nothing.**

ClickHouse is killed by **its own cgroup cap**, not by the box. Move `e2-medium` to a
bigger machine with `mem_limit: 2g` still in place and ClickHouse is killed at 2 GB on
a bigger machine. The box and the caps must move **together**, and the caps are what
actually buy the headroom.

---

## Tasks, in execution order

### Phase 0 — Baseline, so "fixed" is measurable (10 min · free · no downtime)

| # | Task | Done when |
|---|---|---|
| 0.1 | Record the current OOM evidence: serial-port `oom-kill` count for the last 7 days, and peak ClickHouse RSS | The number exists in the log below |
| 0.2 | Record the config as-found: machine type, `max_server_memory_usage`, `mem_limit` | Written down, so a later "it's better" is comparable |

**Why first.** Without a before-number, "it stopped crashing" is an impression. With
one, it is a measurement.

**Recorded baseline, 2026-10-15 (0.1 and 0.2).**

| Fact | Value |
|---|---|
| Machine type | `e2-medium` — 2 shared vCPU, 3.9 GB |
| Swap | none |
| `max_server_memory_usage` | 1,600,000,000 B (1.6 GB) |
| `mem_limit` on the ClickHouse cgroup | 2 GB |
| Committed by the other four containers | ~1.9 GB of 3.9 GB (Redis, MinIO, worker, Cloud SQL proxy, and page cache) |
| `oom-kill` occurrences in serial output | 213, within a ~1 MB buffer |
| `oom-kill` lines carrying a syslog prefix | 76, all dated 2026-10-15 |
| Most recent kills | 12:53:23Z and 12:54:48Z, 85 seconds apart |
| Attribution | `constraint=CONSTRAINT_MEMCG`, `oom_memcg=/system.slice/docker-ee4bd9c1…scope`, `task=clickhouse-serv` |

Three things this baseline records, none of them inference. First, the loop is current
rather than historical: a probe taken during this work caught the container at
`Up 14 seconds (health: starting)`, so it was restarting on the order of every ten
minutes. Second, the serial buffer holds roughly one day at this rate, which is why
every syslog-prefixed kill is dated today — the buffer is being overwritten, not the
problem having started this morning. Third, and decisive: the kill is attributed to the
**container's memory cgroup**, not the host's, and the host already had about 1.9 GB
committed elsewhere against 3.9 GB total. The 2 GB cgroup cap therefore sat at what the
host could actually hand over, so ClickHouse was killed while still inside its own lower
1.6 GB limit. The cap was the binding constraint, not the workload.

### Phase 1 — Make the failure visible (15 min · free · no downtime)

| # | Task | Done when |
|---|---|---|
| 1.1 | **Verify the alert channel.** It exists, is enabled, and points at the right address, but its `verificationStatus` is absent from the API response — which for a Monitoring email channel means unverified, and unverified channels do not deliver | A test alert arrives |
| 1.2 | **Add a data-aware check.** The existing uptime check cannot see "200 but no data" — the signature of every incident here | A check that runs a trivial ClickHouse-backed query and alerts when it fails |

**Why before the resize.** These are free, they take minutes, and they are what make
the resize *observable*. Resize first and a regression during the change is invisible —
which is the failure mode that has already cost five silent days once.

**A correction, 2026-10-15.** The earlier version of 1.1 asserted that the verification
link "was never clicked, so alerts currently do not deliver at all." The channel state
is consistent with that but does not establish it: the field is absent from the API
response rather than set to `UNVERIFIED`, and absence is what the API returns when the
status is unset. The honest form is that the channel is *probably* undelivered and needs
a real test to settle. That test does not need to be synthetic — the Phase 2 resize
takes the front door down for several minutes, and the existing "Langfuse UI
unreachable" policy watches exactly that, so the outage doubles as the delivery test.

**One constraint found while preparing 1.2 (2026-10-15).** The obvious cheap check — a
log-based metric over `langfuse-web`'s ClickHouse connection errors — is not available:
this VM ships only `GCEGuestAgent` and `OSConfigAgent` logs to Cloud Logging, and
container stdout is not among them, because the Ops Agent was never installed. The check
must therefore either go through a path that already exists, or the VM must gain a log
forwarder. Installing one is a change to the box that this plan did not scope, so the
choice is deferred rather than guessed.

### Phase 2 — The resize: box **and** caps together (10 min · ≈10 min outage)

| # | Task | Done when |
|---|---|---|
| 2.1 | Raise the caps **in the repo first**, so config-as-code and the machine land together: `max_server_memory_usage` → ~3–4 GB, `mem_limit` → matching | Committed |
| 2.2 | Stop the instance — **a machine-type change requires the instance stopped** | `TERMINATED` |
| 2.3 | `e2-medium` → `e2-standard-2` (2 vCPU, 8 GB). Same vCPU count, so only the failing axis moves | Machine type confirmed |
| 2.4 | Start it; confirm all five containers are healthy and ClickHouse answers a query | Same checks as 1.2 |

**Ordering note.** 2.1 before 2.2 is deliberate: if the caps ship after the resize,
there is a window where the box is bigger and the kill still happens, and that window
will be blamed on the resize.

### Phase 3 — Stop the steady state climbing back (20 min · free · no downtime)

| # | Task | Done when |
|---|---|---|
| 3.1 | **Trace retention / TTL** in ClickHouse, so the store is bounded and long enough to be useful | A TTL exists and is documented |
| 3.2 | **Merge concurrency**: `background_pool_size`, `max_bytes_to_merge_at_once` | A backlog no longer produces a memory spike |

**Why this is not optional.** Phase 2 buys headroom; this is what stops the workload
consuming it again. Retention is the free lever, and it matters more now that scores
will accumulate alongside traces.

### Phase 4 — The two real reliability bugs (no downtime)

| # | Task | Done when |
|---|---|---|
| 4.1 | **`langfuse-web` exits(1) when ClickHouse migrations fail at startup.** This is what turns a ClickHouse blip into a front-door 503 | It retries and/or serves degraded instead of dying |
| 4.2 | **Docker never marked the killed container as exited** (2026-10-02: `Up (unhealthy)`, `cannot exec in a stopped state`, so `restart: unless-stopped` never fired). Decide: keep `selfheal.sh` as the mitigation, or replace it with a healthcheck that reflects death | A decision is recorded either way |

### Phase 5 — Verification, i.e. the definition of done

| # | Task | Done when |
|---|---|---|
| 5.1 | The OOM loop is gone over a real window under real load | No `oom-kill` in the serial output for that window |
| 5.2 | **A trace opens and shows its spans** — not "the UI loads" | Someone reads an actual trace |
| 5.3 | The alert fires on an **induced** failure (stop ClickHouse, watch the alert) | The alert arrives |
| 5.4 | **★ The acceptance test:** a rating written from the console is retrievable as a Langfuse score against its trace id | A score is read back next to its trace |

**5.4 is the point of the whole exercise.** `langfuse_trace_id` is already stored on
every turn (migration `0006_turn_langfuse_trace_id`), so the console-turn → trace link
exists today; what is missing is a trace store dependable enough to trust it.

---

## Rollback

- **Resize:** stop, set the machine type back, start. The box is not coupled to data.
- **Caps:** revert `memory.xml` and `docker-compose.yml`, recreate the ClickHouse
  container. Reverting to 1.6 GB / 2g restores the previous behaviour exactly, which is
  to be killed under load — so revert only alongside a resize-back.
- **`selfheal.sh` is untouched by this plan** and continues to work throughout.

---

## Cost

| Change | Cost | Confidence |
|---|---|---|
| `e2-medium` → `e2-standard-2` | **+$0.0335056/hr = +$0.80/day = +$24.46/month** | **Verified** against the Cloud Billing Catalog API, 2026-10-15 |

The rates, read from the Catalog rather than estimated. Service `6F81-5844-456A`
(Compute Engine), `currencyCode=USD`, `us-east1` confirmed present in `serviceRegions`
for both SKUs:

| SKU | Rate |
|---|---|
| `E2 Instance Core running in Americas` | $0.0218116 / vCPU-hour |
| `E2 Instance Ram running in Americas` | $0.0029235 / GB-hour |

| Machine type | Billed core-equivalents | RAM | Rate |
|---|---|---|---|
| `e2-medium` | 1.0 | 4 GB | $0.0335056/hr |
| `e2-standard-2` | 2.0 | 8 GB | $0.0670112/hr |

Shared-core types bill at a fractional core count — 0.25 for `e2-micro`, 0.5 for
`e2-small`, 1.0 for `e2-medium` — which is why a 2-vCPU machine prices as one core. The
derivation reproduces Google's published `e2-medium` figure of $0.03351 exactly, and that
agreement is what makes the decomposition trustworthy rather than merely plausible.

The earlier entry here read "on the order of +$1/day, estimate." The shape was right
and the figure was about $0.20/day high. Note also that `e2-standard-2` is exactly twice
`e2-medium` in both core-equivalents and RAM, so the delta equals the original cost:
doubling the box doubles the bill.

The standing decision is that the Langfuse stack stays up (~$1/day) and is not a
teardown candidate between phases. This change is a modest increase on a stack already
accepted as standing cost. It is included here so the number is visible, not because
it is in question.

---

## What this plan does **not** claim

Resizing alone is not the durable fix, and this document should not be read that way.
Two of the five items above (1.1, 1.2) are free and take minutes; two of them (4.1,
4.2) are reliability bugs that more memory only makes rarer. The resize is necessary
and it is the biggest single lever. It is not sufficient on its own.

---

## Session record — 2026-10-16: the loop is still live, and bounding the merge did not break it

**Verified on the VM, not inferred.**

| Fact | Value |
|---|---|
| Machine type | `e2-standard-2` — the Phase 2 resize is in place |
| Host memory at the time | 7 GB total, ~6 GB available — the host is not the constraint |
| `langfuse-clickhouse` `RestartCount` | 1101 at first reading; 41 more within the session |
| Restart interval | ~40–53 s |
| Kill signature | `constraint=CONSTRAINT_MEMCG`, `oom_memcg=...docker-06ba3f9a...scope`, `task=clickhouse-serv` |
| `anon-rss` at kill | 4.05–4.12 GB, against a 4 GB cgroup cap |
| What invokes the OOM killer | `MergeMutate` (once `TraceCollector`) |
| `docker inspect` OOM flag | `oom=false` *despite* the kernel kills — **Phase 4.2 confirmed**, not hypothesised |

**The resize did not close the loop, and this document predicted exactly that.** The kill
signature is identical to the 2026-10-15 baseline — same cgroup attribution, same process,
merges as the invoker — only at 4 GB instead of 2 GB. The ceiling has now been raised twice
and the workload grew to fill it both times. Phase 3 is not the follow-up to Phase 2; it is
the fix.

**Phase 3.2 attempted, and ineffective.** A merge throttle was added
(`clickhouse/config.d/merge_throttle.xml`): `background_pool_size=1` at the root, and
`max_bytes_to_merge_at_once=104857600` under `<merge_tree>`. Note the nesting — that setting is
a MergeTree *table* setting, so at the root of a drop-in it is ignored, with a log warning and no
error, which looks exactly like the file having had no effect. The file installs and ClickHouse
merges it (the container log names it), but the OOM loop continued unchanged, still with
`MergeMutate` as the invoker.

**Why it may not have helped — unestablished, and not guessed at.** A 100 MB merge should not
need 4 GB, and `MergeMutate` is the pool for mutations as well as merges. Whether the setting
took effect (read `system.merge_tree_settings`) and whether the OOM is a merge or a mutation
(read `system.merges.is_mutation`) were both attempted and neither could be answered: the
container refuses connections too often to query. **That is where the next session should
start**, before any further configuration change.

**Recommended pivot: stop bounding the merge and remove the backlog.** Phase 3.1 retention is
the lever that has not been tried, and Langfuse has its own mechanism for it (`CLICKHOUSE_TTL_DAYS`,
applied through its own migrations) — which needs no hand-written ClickHouse SQL and so does not
require catching a 53-second window. Pair it with pausing the worker, so ingestion stops adding
work while merges catch up. The variable name should be confirmed against this Langfuse version
before it is used.

**Option A — raising the cgroup cap to 6 GB — declined, with the arithmetic.** The host is 7 GB
and the other four containers hold about 1.9 GB, so 6 GB oversubscribes the box (7.9 GB > 7 GB)
and risks the host-wide OOM that `mem_limit` was introduced to contain. It also does not address
growth. If relief is wanted while retention lands, **5 GB** fits.

### `selfheal.sh` audit — 2026-10-16

- **Invocation:** systemd timer, `Type=oneshot`, `User=root`, `ExecStart=/opt/langfuse/selfheal.sh`;
source of truth is the repository file.
- **Concurrency: safe.** `flock -n 9` then exit — two runs cannot overlap.
- **Grace: partial.** `STARTUP_GRACE_SECONDS=90` guards the ClickHouse *probe* path only; the
`unhealthy` branch acts with no grace. In practice a restarting container reports
`health: starting`, which it does not flag.
- **Backoff: none.** A persisting fault is actioned again on every five-minute tick.
- **Hard kills: confirmed.** `docker compose up -d --force-recreate` is a stop/remove/start with a
10 s timeout, repeated against a ClickHouse replaying its write-ahead log.

Recommended and not yet made: have it skip ClickHouse while the container is `restarting` or
`health: starting`, since Docker is already recovering it in that state and a force-recreate only
interrupts the replay.

### Uncommitted at the moment of the pin

`clickhouse/config.d/merge_throttle.xml` (new), its mount in `docker-compose.yml`, and a change to
`vm_stack.sh` that installs **every** `config.d/*.xml` instead of naming `memory.xml`. That last
change is worth keeping whatever happens to the throttle: the scp already copied the whole
directory, and the hardcoded `install` line is precisely how `memory.xml` came to be copied but
never installed.
