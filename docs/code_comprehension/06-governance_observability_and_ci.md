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

*(Coding Agent: Populate findings for 1.1–1.3 here with exact file paths, line numbers, and proposed surgical fixes. Do NOT edit application or test code until the Architect approves this section.)*

---

## 3. Step B — Execution Record (Locked Until Step A Approval)

### 3.1 Executed Changes
* Pending Step A approval.

### 3.2 Verification Gates
| Gate | Result |
| :--- | :--- |
| `pytest tests` (`enterprise_clinical_copilot`) | *Pending* |
| `python manage.py test` (`danielmherman`) | *Pending* |
| `python manage.py check --deploy` (`danielmherman`) | *Pending* |