# Phase 3: Agent Orchestrator & Model Runtime Decomposition

**Status:** `STEP A — READ-ONLY AUDIT`  
**Target Layers:** Layer 3 (`docs/architecture/03-orchestrator.md`) & Layer 4 (`docs/architecture/04-model_runtime.md`)  
**Repositories in Scope:** `enterprise_clinical_copilot` (primary) & `danielmherman` (verify `demo/fixtures.py` import compatibility)  
**Objective:** Decompose the mixed-responsibility modules in `services/agent/` along strict Layer 3 (Orchestration) and Layer 4 (Model Runtime) boundaries—specifically extracting turn orchestration out of `http.py` into a new `turn.py`, isolating Layer 4 model runtime seams, evaluating micro-module consolidations, and modularizing `a2ui.py`—while keeping the flat package layout and 100% test compatibility.

---

## 0. Governance Protocol for This Spoke

* **Step A — The Audit Pass (Read-Only — CURRENT STEP):**
  * Inspect the target files in Section 1 against `docs/architecture/03-orchestrator.md` and `docs/architecture/04-model_runtime.md`.
  * Record exact file paths, line numbers, test `monkeypatch` targets, and a surgical refactoring plan in **Section 2 (Step A — Audit Findings)**.
  * **STOP.** Do NOT modify any application or test code during Step A. Await explicit Architect approval.
* **Step B — The Execution Pass (Write + Verify — LOCKED UNTIL STEP A APPROVED):**
  * After Architect approval of Step A, execute the approved changes, run all verification gates, and record the results in **Section 3 (Step B — Execution Record)**.

---

## 1. Step A Audit Scope & Questions to Answer

### 1.1 Track 3A — Extract Turn Orchestration (`turn.py`) from HTTP Transport (`http.py`)
* **Problem:** `services/agent/http.py` currently acts as both the Starlette HTTP/SSE transport layer (`/ask`, `/ask/stream`, `/health`, request parsing, SSE framing) and the turn orchestrator (`_run_chain`, `_compose_success`, post-model verification, and presentation assembly). Note: `turn.py` does not exist yet; the goal is to extract turn orchestration out of `http.py` into a new `services/agent/turn.py` module.
* **Audit Tasks:**
  1. Map `services/agent/http.py` by exact line ranges into (a) pure HTTP/SSE transport concerns vs. (b) turn orchestration concerns (`_run_chain`, `_compose_success`, etc.).
  2. List every symbol in `services/agent/http.py` imported or monkeypatched by `tests/` or `evaluation/`.
  3. Propose the exact boundary for creating `services/agent/turn.py` (including which re-exports or delegation shims in `http.py` must remain so existing tests and monkeypatches do not break).

### 1.2 Track 3B — Layer 4 Model Runtime Seam & Micro-Module Consolidation (`graph.py`, `model_turn.py`, `message_text.py`, `prompts.py`, `questions.py`, `chain.py`, `observability.py`)
* **Problem:**
  * Layer 4 (`04-model_runtime.md`) governs LLM client instantiation, prompt assembly, streaming token lookahead buffering, and response normalization (`model_turn.py`, `message_text.py`).
  * Several tiny two-file splits exist (`questions.py` vs. `prompts.py`, `message_text.py` vs. `model_turn.py`, and `chain.py` vs. `observability.py`).
* **Audit Tasks:**
  1. Locate every LLM client construction and model invocation across `services/agent/` (e.g., inside `graph.py`, `deterministic.py`, etc.) and document how Layer 3 (orchestration/tool policy) interacts with Layer 4 (model runtime, streaming lookahead buffer, finish-reason parsing).
  2. Audit the call sites and test imports of the three micro-module pairs:
     * `questions.py` + `prompts.py`
     * `message_text.py` + `model_turn.py`
     * `chain.py` + `observability.py`
  3. Recommend whether each pair should be consolidated (with re-exports if needed for tests) or kept separate, and whether any Layer 4 model-runtime logic inside `graph.py` or `http.py` should be extracted into a dedicated Layer 4 helper.

### 1.3 Track 3C — Presentation Composer Modularization (`services/agent/a2ui.py`)
* **Problem:** With `payload.py` deleted in Phase 2, `services/agent/a2ui.py` (~1,250+ lines) is now the sole presentation composer, but it mixes citation/reference resolution, clinical domain data shaping (risk influencers, comparative matrix, lab trends, medications), and A2UI v0.9 JSON envelope construction in one file.
* **Audit Tasks:**
  1. Map `services/agent/a2ui.py` by line range into its distinct functional responsibilities.
  2. Identify all external callers of `services/agent/a2ui.py` across `enterprise_clinical_copilot` and `danielmherman` (`demo/fixtures.py`).
  3. Propose whether and how to split `a2ui.py` into clean internal modules while keeping `a2ui.compose_presentation` (and any other externally imported functions) available at `services.agent.a2ui`.

---

## 2. Step A — Audit Findings & Proposed Surgical Plan (To Be Completed by Coding Agent)

*(Coding Agent: Populate findings for 1.1–1.3 here with exact file paths, line numbers, monkeypatch/import hazards, and your proposed surgical plan. Do NOT edit application or test code until the Architect approves this section.)*

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