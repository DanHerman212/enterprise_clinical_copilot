# Phase 6: Governance, Observability, Reliability & CI/CD Audit

**Status:** `PENDING (Runs After Phase 5)`  
**Target Layers:** Layer 8 (`08-conversational_governance.md`), Layer 9 (`09-evaluation_and_guardrails.md`), Layer 10 (`10-observability_and_audit.md`), Layer 11 (`11-reliability_and_runtime.md`), & Layer 12 (`12-delivery_ci_cd.md`)  
**Repositories in Scope:** `enterprise_clinical_copilot` & `danielmherman`  
**Objective:** Audit end-to-end conversational governance (TTL sweeps, turn ceilings, quota/refund atomicity), guardrail/evaluation integrity, W3C/X-Cloud-Trace-Context propagation and zero-PII structured logging, runtime concurrency/timeout bounds, and CI/CD deployment gates across both repositories.

---

## 0. Governance Protocol for This Spoke

* **Step A — The Audit Pass (Read-Only):**
  * Inspect the target files in Section 1 against `docs/architecture/08-conversational_governance.md` through `12-delivery_ci_cd.md`.
  * Record exact file paths, line numbers, findings, and any proposed surgical fixes in **Section 2 (Step A — Audit Findings)**.
  * **STOP.** Do NOT modify any application or test code during Step A. Await explicit Architect approval.
* **Step B — The Execution Pass (Write + Verify):**
  * After Architect approval of Step A, execute the approved changes, run all verification gates, and record the results in **Section 3 (Step B — Execution Record)**.

---

## 1. Step A Audit Scope & Questions to Answer

### 1.1 Layer 8 — Conversational Governance, Quota Atomicity & TTL Sweeps (`danielmherman/demo/` & `services/agent/`)
* **Architectural Standard (`08-conversational_governance.md`):**
  * The agent service is strictly stateless across turns; multi-turn continuity is replayed by the BFF (`turns`) and bounded by `MAX_REPLAY_TURNS` / `DEMO_CONVERSATION_TURN_CEILING`.
  * Expired conversations are refused at request time AND swept via a scheduled/management command (`sweep_expired_conversations` or equivalent).
  * Quota consumption (`DemoQuota.consume`) and refunds (`DemoQuota.refund`) must be atomic (`F()` expressions / row locks) and bounded by the daily refund cap when upstream compute was spent.
* **Audit Tasks:**
  1. Trace conversation lifecycle, turn-ceiling enforcement, TTL expiry checks, and the expiry sweep command in `danielmherman/demo/` (`models.py`, `conversations.py`, `views.py`, `management/commands/`).
  2. Trace how the agent validates and bounds replayed `turns` in `services/agent/contracts.py` and `services/agent/graph.py`.
  3. Identify any mismatch between the BFF's `DEMO_CONVERSATION_TURN_CEILING` and the agent's `MAX_REPLAY_TURNS` (or equivalent), or any missing governance tests.

### 1.2 Layer 9 & Layer 10 — Guardrails, Trace Propagation & Zero-PII Audit (`guardrail.py`, `observability.py`, `chain.py`, `demo/agent_client.py`, `demo/langfuse_scores.py`)
* **Architectural Standard (`09-evaluation_and_guardrails.md` & `10-observability_and_audit.md`):**
  * Every turn emits a single immutable execution record (`chain.record_execution`) containing model ID, code revision, guardrail flags, tool error codes, and trace ID—never raw patient note text or raw questions containing PHI in structured operational logs.
  * Trace IDs propagate end-to-end from the BFF (`demo/agent_client.py` → `X-Trace-Id` / `X-Cloud-Trace-Context`) into the agent (`http.py` / `turn.py` / `observability.py`) and back into `Turn.langfuse_trace_id` so clinician thumbs-up/down feedback (`demo/langfuse_scores.py`) attaches to the exact Langfuse trace.
* **Audit Tasks:**
  1. Trace the end-to-end path of `trace_id` and `langfuse_trace_id` from `demo/views.py` → `demo/agent_client.py` → `services/agent/http.py` → `turn.py` → `demo/conversations.py` → `demo/langfuse_scores.py`.
  2. Audit all `logger.*` calls in `services/agent/`, `services/mcp/`, and `danielmherman/demo/` to verify that raw clinical note passages or unredacted free-text questions are never written to operational log streams.
  3. Verify that Langfuse failures (`observability.py` and `demo/langfuse_scores.py`) are strictly non-blocking (fire-and-forget) and cannot fail a clinical turn.

### 1.3 Layer 11 & Layer 12 — Runtime Reliability & CI/CD Delivery Gates (`Dockerfile`, `.github/workflows/`, `cloudbuild.yaml` / deploy scripts)
* **Architectural Standard (`11-reliability_and_runtime.md` & `12-delivery_ci_cd.md`):**
  * Every external hop (Browser → BFF, BFF → Agent, Agent → Vertex AI, Agent → MCP, MCP → BigQuery/Vector Search/NCBI) has an explicit timeout and bounded concurrency/retry configuration.
  * Dockerfiles run as non-root users in production, pin deterministic dependency lockfiles, and CI workflows execute the contract/unit suites cleanly.
* **Audit Tasks:**
  1. Inventory all network hop timeouts and retry bounds across `danielmherman/demo/agent_client.py`, `services/agent/http.py` / `turn.py` / `model_runtime.py` / `mcp_client.py`, and `services/mcp/`.
  2. Inspect `Dockerfile` and `.github/workflows/` (or CI configs) in both repositories for non-root execution, stale script/file references (especially after Phases 1–5), or missing test gates.

---

## 2. Step A — Audit Findings & Proposed Surgical Plan (To Be Completed by Coding Agent)

**Audited:** 2026-10-27, read-only. Every claim below was read in source at the cited
line, or observed on the live service (`agent-00193-9wh`, `danielmherman-00176-pq8`).
No application or test code was modified.

**Context:** the prediction and RAG index endpoints were redeploying while this pass ran
(`/tmp/deploy_cpr.log`, `/tmp/deploy_rag.log`). Nothing in this audit depends on them.

### 2.1 Layer 8 — Conversational governance, quota atomicity, TTL sweeps

**Already correct — do not change:**

* **Quota is genuinely atomic.** `demo/models.py:98` (`consume`) and `:131` (`refund`)
  express every mutation as a single `UPDATE` with `F()` arithmetic, guarded by
  `period_start__lt` (rollover), `used__lt=F('daily_limit')` (grant), `used__gt=0`
  (no negative counter) and `refunds__lt=DEMO_DAILY_REFUND_CAP` (refund cap). The
  consume token is the period, so a refund cannot debit the wrong day (S1-07).
* **Expiry is refused at request time as well as swept.** `demo/views.py:165` returns
  `409 conversation_expired`, `:181` returns `409 conversation_full`,
  `demo/conversations.py:205` only reuses a thread with `expires_at__gt=now`.
* **The sweep is scheduled and running.** Cloud Run job `purge-conversations` is updated
  by `danielmherman/cloudbuild.yaml:160-190`, triggered by Cloud Scheduler job
  `purge-conversations-daily` (`0 3 * * *`); executions exist for 2026-10-25/26/27
  (07:00 UTC each day). Body: `demo/management/commands/purge_expired_conversations.py:47`.

**F-8.1 — Cross-repo replay bounds are duplicated literals (MEDIUM).**
`danielmherman/demo/conversations.py:30-32` declares
`MAX_REPLAYED_TURNS = 6`, `MAX_REPLAYED_TOOL_CALLS = 5`, `MAX_TURN_ANSWER_CHARS = 8000`.
The originals are `services/agent/contracts.py:214-215` plus the answer-length bound in
the same validator. Nothing links the two, and the agent refuses an over-long replay with
`413 too_many_turns` (`contracts.py:298`) — so raising the site's ceiling without the
agent's turns a working follow-up into a hard failure for every thread past the new bound.
The docstring at `conversations.py:338` says "the bound is the agent's", but only a comment
enforces it.
*Proposed fix:* expose the three bounds on the agent's health surface (they are constants,
not secrets) and add a site test asserting the Django literals equal what the agent
reports. No new runtime dependency: the test reads the same values the deploy already
serves.

**F-8.2 — The replay-window rule has no test (LOW).**
`demo/conversations.py:367-373` sends the FIRST turn plus the most recent five when a
conversation exceeds the bound; the size check and the anchor rule live only in that
comment. Ceiling is 20 (`settings.py:118`) and the replay bound is 6, so this path runs on
every thread past turn 6 — the common case for the demo's multi-turn traces.
*Proposed fix:* one test per branch (under the bound → all turns; over → anchor + last
five; never more than the bound), in `danielmherman/demo/tests.py`.

**F-8.3 — A capped refund is silent (LOW).**
`demo/models.py:150-160`: when `refunds` has reached `DEMO_DAILY_REFUND_CAP`, the
`spent=True` refund path updates zero rows and returns nothing. The clinician is correctly
not charged back, but there is no log line, so an account that is being refund-capped —
the S1-09 abuse signal — is invisible in operations.
*Proposed fix:* log a warning with the trace id when the capped update matches zero rows.

### 2.2 Layer 9 & 10 — Guardrails, trace propagation, zero-PII

**Already correct — do not change:**

* **One execution record per turn, and it carries no text.**
  `services/agent/chain.py:102-146` logs `trace`, `stages`, `duration_ms`, `outcome`,
  `finish_reason`, `filtered`, `served_model`, `tokens`, `tool_calls`, `tool_errors`,
  `guardrail_flags`, `langfuse_trace_id`, `follow_ups`, `intent`, `error`, `model`,
  `code_revision` — and deliberately not the question or the answer (documented at `:130`).
* **Operational logs carry identifiers and codes, never content.** Every `logger.*` call in
  `services/agent/`, `services/mcp/` and `danielmherman/demo/` was read (30 sites in
  `services/`, 12 in the site). Each interpolates a trace id, a tool name, an error code, a
  count or a duration. The two sites that interpolate a payload — `demo/views.py:643`
  (agent error frame) and `:956` (`AgentError` detail) — carry the error envelope, not note
  text or the question.
* **Trace identity propagates end to end.** `demo/agent_client.py:141-149` reads
  `X-Cloud-Trace-Context` and forwards it (`:186`, `:291`); the agent reads it in
  `services/agent/http.py` and puts `langfuse_trace_id` on the payload
  (`services/agent/turn.py`); the site stores it and `demo/langfuse_scores.py:55` scores
  that exact trace on thumbs-up/down.
* **Observability cannot fail a turn.** `services/agent/observability.py:305-321` guards
  flush (`except Exception`, rule 1) and documents that a request never flushes;
  every span/IO helper swallows its own failure (`:147`, `:178`, `:210`, `:216`, `:242`,
  `:302`). `demo/langfuse_scores.py:76-103` wraps the HTTP call in `try/except` and the
  endpoint answers `204` regardless.

**F-10.1 — The question/answer sink is undocumented in the standard (LOW).**
`services/agent/observability.py:182` (`set_trace_io`) deliberately writes the raw question
and the raw answer onto the Langfuse trace. That is the right call — it is the trace store,
self-hosted at `observability.danielmherman.com`, and without it every trace row is a dash —
but `10-observability_and_audit.md` states the zero-PII rule without naming this exception,
so a future reader can only conclude the code violates it.
*Proposed fix:* documentation only — state the exception, its destination and why it is
in scope, in Layer 10; no code change.

### 2.3 Layer 11 & 12 — Runtime reliability and delivery gates

**Already correct — do not change:**

* **Timeouts nest, and the nesting is enforced at startup.**
  `services/mcp/runtime.py:53-76` requires `MODEL_TIMEOUT_SECONDS (60) <
  MCP_TOOL_TIMEOUT_SECONDS (100) < ASK_TIMEOUT_SECONDS (110)` and raises at import time if
  not. Consumed by `model_runtime.py:45` (`timeout=`, plus `max_retries=3` at `:71`),
  `mcp_client.py:165`/`:214` (per-call read timeout) and `http.py:53`/`:232`.
  The NCBI hop is bounded at `services/mcp/config.py:162` (`NCBI_TIMEOUT_SECONDS = 10`,
  applied at `literature.py:342`).
* **BFF→agent is bounded twice.** `settings.py:101` `DEMO_AGENT_TIMEOUT = 120`, and a
  slot semaphore refuses rather than queues (`demo/agent_client.py:170`, `:274`).
* **Containers are non-root and pinned.** `services/agent/Dockerfile:36 USER 1000`,
  `services/mcp/Dockerfile:28 USER 1000`, `danielmherman/Dockerfile:62 USER app`; all three
  pin the base image by digest.

**F-11.1 — Agent concurrency is unpinned at the Cloud Run default (HIGH).**
The live service reports `containerConcurrency = 80`, `timeoutSeconds = 300`. A single turn
holds a request for up to `ASK_TIMEOUT_SECONDS` (110 s) while driving a Gemini stream and an
MCP call, and `services/agent/cloudbuild.yaml:69` sets only
`--min-instances 0 --max-instances 3 --memory 1Gi` — no `--concurrency`, no `--cpu`, no
`--timeout`. With `max-instances 3` the ceiling is 240 concurrent turns on 3 vCPU-equivalent
containers, which is the number the 100-user concurrency target will actually meet, and it
is not a number anybody chose.
*Proposed fix:* pin `--concurrency` and `--cpu` explicitly in the deploy config (a value
derived from the 110 s bound and the target concurrency, not a guess), and pin
`--timeout` just above `ASK_TIMEOUT_SECONDS` so a hung turn is recycled by the platform
rather than sitting in the 300 s window.

**F-12.1 — No deployment runs a test suite (HIGH).**
Neither repository has a `.github/workflows/` directory (verified: absent in both), and no
Cloud Build config executes a test: `services/agent/cloudbuild.yaml`,
`services/mcp/cloudbuild.yaml`, `evaluation/agent/cloudbuild.yaml`,
`mlops/training/cloudbuild.yaml` and `danielmherman/cloudbuild.yaml` contain no `pytest`,
no `npm test` and no `manage.py test` step. Every gate in this programme — 1,512 agent
tests, 292 Django tests, 84 JS tests — is executed by a human before a push. The trigger
fires on the push alone.
*Proposed fix:* add a test step ahead of the image build in both repos' deploy configs
(`python -m pytest tests -q` / `npm test` + `manage.py test`), failing the build before any
revision is created. This is the one finding here that changes the delivery contract, so it
is called out for explicit approval rather than assumed.

**F-12.2 — Deploy-time migrations are unordered against the traffic shift (LOW, informational).**
`danielmherman/cloudbuild.yaml` creates the revision with `--no-traffic`, then migrates,
then `update-traffic --to-latest`. That ordering is correct. What is absent is a
post-shift smoke check: if the new revision boots but fails on first request, traffic has
already moved.
*Proposed fix:* after `update-traffic`, curl a known-cheap endpoint and fail the build on a
non-200. No new infrastructure.

### 2.4 What the Architect is being asked to rule on

1. **F-12.1** — add test gates to Cloud Build. Changes the delivery contract; needs a
   decision, not a recommendation.
2. **F-11.1** — the concurrency number. I can derive a defensible value from the 110 s
   bound, but the target (“max concurrency 100 users”) is a product input I should not
   invent.
3. **F-8.1** — the cross-repo constant handshake: expose bounds on the health surface
   (my preference, test-enforced) versus a documented duplicate.

Everything else in 2.1–2.3 is local and low risk; I can execute it in Step B without
further input once this section is approved.


---

## 3. Step B — Execution Record

Executed 2026-10-27 against the six approved rulings. Every change below is in the
repository; none is in console state.

### 3.1 Executed Changes

**F-8.1 — the replay bounds are asserted equal on both sides of the wire.**
The three limits the site enforces before it sends a replay (`MAX_REPLAYED_TURNS`,
`MAX_REPLAYED_TOOL_CALLS`, `MAX_TURN_ANSWER_CHARS`) are now read by tests on both sides and
compared to `shared/contracts.py`. The agent's `tests/agent/test_wire_contract_parity.py`
parses the site module and asserts equality against `contracts.*`; the site's
`demo/test_wire_contract.py` imports its own bounds and asserts the same three numbers. Both
directions skip with a printed reason when the sibling checkout is absent, so neither build
can pass by reading nothing. The duplication remains — the site cannot import a Python
package it does not deploy — but a drift now fails a test instead of a turn.

**F-8.2 — the replay window is tested at its boundary.**
`demo/tests.py` gained three cases over `replay_turns`: a short conversation replays
verbatim and in order; a long conversation keeps the first turn as the anchor and then the
most recent turns; and no conversation of any length replays more than
`MAX_REPLAYED_TURNS`, with `None` replaying as empty. The anchor case is the one that
mattered: the window is not "the most recent N", and an off-by-one at the bound would have
turned every thread past turn six into a `413`.

**F-8.3 — a capped refund is no longer silent.**
`DemoQuota.refund` now returns the row count of the `UPDATE` and logs
`refund refused: daily refund cap of N reached period=… trace=…` when the cap is what
stopped it, re-reading the row first so a period that rolled over between the failed update
and the log does not raise an alarm about a cap that was never hit. `trace` was added to the
signature and is passed at all nine call sites in `demo/views.py`, so the line names the
request. Behaviour is unchanged: the refund is still refused.

**F-10.1 — the trace IO sink is documented as the Archive's exception.**
`docs/architecture/10-observability.md` §4 now states the invariant explicitly: the zero-PII
guarantee governs the Cloud Logging Archive, whose execution record carries no question or
answer field, and `set_trace_io` writes both into the self-hosted Langfuse trace on purpose.
The exception is bounded by tenancy, and a sink that would leave it is a Layer 11 decision.

**F-11.1 — the agent's runtime limits are pinned in the deploy config.**
`services/agent/cloudbuild.yaml` now passes `--cpu 2 --concurrency 20 --timeout 120`.
`--timeout 120` sits just above `ASK_TIMEOUT_SECONDS` (110, enforced at import in
`services/mcp/runtime.py`), so a wedged turn is recycled by the platform instead of holding a
request slot for the 300 s default. `--concurrency 20` across `--max-instances 3` is 60
simultaneous turns, derived from that 110 s bound; `--cpu 2` is what makes twenty concurrent
Gemini streams on one container defensible. The 240-simultaneous-turn ceiling nobody chose is
gone.

**F-12.1 — both deploys run their test suite before the image exists.**
`danielmherman/cloudbuild.yaml` gained `npm ci && npm test` and
`ENVIRONMENT=development python manage.py test` as its first two steps;
`services/agent/cloudbuild.yaml` gained `python -m pytest tests -q` against all three
requirement sets. A failing test stops the build before any image is pushed or revision
created, so production is untouched and there is nothing to roll back. The gate is a second
install of dependencies the Dockerfile installs anyway; that duplication buys testing what
ships rather than trusting that a human ran it.

**F-12.2 — the site is looked at after the traffic shift, and the agent's post-condition is
asserted.**
The site's build now curls `https://danielmherman.com/` after `update-traffic`, retrying five
times and failing the build on anything that is not 2xx or 3xx — a revision that boots but
fails its first real request is no longer discovered by a user. The agent's build asserts the
same intent by other means: the ruling's curl is not available on a
`--no-allow-unauthenticated` service whose invoker list does not include the build account, so
the step asserts the two facts the deploy can silently get wrong — the service is `Ready`, and
its template runs the image this build produced. Accepting Cloud Run's own 403 would have
been a check that cannot fail. Giving `cicd-deployer` `roles/run.invoker` would make a real
request possible; that widens who may call the agent and is left as a decision.

### 3.2 Verification Gates
| Gate | Result |
| :--- | :--- |
| `pytest tests` (`enterprise_clinical_copilot`) | 1513 passed, 1 skipped (251 s) |
| `python manage.py test` (`danielmherman`) | 298 passed (64 s) |
| `npm test` (`danielmherman`) | 84 passed |
| `npm run build` (`danielmherman`) | built in 62 ms, 816 kB / 196 kB gzip |
| `npx eslint static/js` (`danielmherman`) | clean |
| `python manage.py check --deploy` (`danielmherman`) | runs in the migrate job, unchanged |

### 3.3 Correction to the Step A finding

F-12.2's premise — that the site's ordering was correct and only the smoke check was
missing — holds. The Layer 8 investigation is a separate matter and is **not** closed by
these changes: `demo_conversation` was found empty twelve hours after a turn had been
answered, and the retention sweep is not the cause. The sweep filters `expires_at__lte=now`
only, the production TTL is 168 hours, and two tests now pin both facts
(`test_a_conversation_from_this_morning_survives_the_sweep`,
`test_deleting_a_patient_takes_their_conversations_with_it`). What remains is the second
test's mechanism: `Conversation.patient` is `on_delete=CASCADE`, and every deploy runs
`seed_demo_patients --prune`, so refreshing the cohort deletes the conversations of any
patient the artifact no longer carries — no expiry, no sweep, no log line. That is a policy
question and is raised rather than changed.
