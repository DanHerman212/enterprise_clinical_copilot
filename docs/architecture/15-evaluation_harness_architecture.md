# Evaluation Harness Architecture & Module Boundary Charter

## 1. Purpose

This document defines the architectural boundaries, module responsibilities, and dependency rules for `evaluation/`. Its goal is to ensure that the evaluation harness is a single, self-contained, and deterministic measurement instrument that never diverges from the production agent's citation or schema contracts, and is enforced via automated CI tests (`tests/agent/test_layer_separation.py`).

---

## 2. Package Structure & Harness Lifecycle

The `evaluation/` directory is organized around a single active evaluation harness (`evaluation/platform/`), a shared top-level clinical rubric (`evaluation/rubric.md`), and supporting retrieval/cohort utilities:

1. **Top-Level Shared Assets (`evaluation/rubric.md`, `evaluation/outpatient_oracle.py`):** Canonical evaluation rubric instructions and curated ground-truth oracle rules.
2. **Active Evaluation Platform (`evaluation/platform/`):** The authoritative Langfuse dataset and experiment harness that executes multi-turn conversations against the live agent, numbers raw tool evidence, invokes the LLM-as-a-Judge, computes item-level and suite-level gate verdicts, and records telemetry scores.
3. **Retrieval & Cohort Utilities (`evaluation/agent/retrieval/`, `evaluation/inventory/`):** Standalone scripts for offline RAG recall measurement (`Recall@K`), cohort census generation, and multi-turn follow-up chip audits.
4. **Legacy Offline Collector (`evaluation/agent/`):** Retired pre-platform HTTP collector and ops utilities. **Active platform modules (`evaluation/platform/`) are strictly forbidden from importing or modifying `sys.path` to load modules from `evaluation/agent/`.**

---

## 3. Submodule Charter (`evaluation/platform/`)

| Module Group | Primary Modules | Owns (Single Responsibility) | Forbidden (Hard Red Lines) |
| :--- | :--- | :--- | :--- |
| **Schemas & Constants** | `schemas.py`, `dev_suite.py`, `fixtures.py`, `seed.py` | Single declaration of scoring dimensions (`DIMENSIONS`, `PASS_DIMENSIONS`), gate thresholds (`MIN_PASS_RATE`), and dataset item schemas. | Executing HTTP requests, invoking LLM clients, or duplicating dimension lists in downstream runners. |
| **Task & Evidence Assembly** | `task.py`, `http_auth.py` | Executing turns against the agent endpoint and assembling numbered tool evidence (`_evidence`, `_numbered`) using `services.agent.postprocess.citations.passage_total`. | Re-implementing custom citation-unit counters that diverge from `citations.passage_total`. |
| **Judge Client & Prompting** | `judge_client.py`, `judge_call.py` | Vertex AI LLM-as-a-Judge invocation (`judge_client.py`), loading `evaluation/rubric.md`, and constructing the structured judge user prompt (`judge_call.py`). | Mutating `sys.path`, importing legacy `evaluation/agent/judge.py`, or embedding inline rubric strings. |
| **Evaluators & Gate Logic** | `evaluators.py` | Item-level dimension evaluation, structural/attribution checks, and the **single canonical suite gate function** (`gate_verdict`). | Scoring a run with a partial gate rule that omits structural or attribution failures. |
| **Experiment Orchestration & CLI** | `run_evals.py`, `runner.py`, `experiment.py`, `dry_run.py`, `run.py` | CLI entrypoint, dataset synchronization (`sync_dev_suite.py`, `sync_holdout_suite.py`), experiment execution, and reporting via `evaluators.gate_verdict`. | Re-declaring `DIMENSIONS`, `PASS_DIMENSIONS`, or separate inline `gate_verdict` formulas. |

---

## 4. Non-Negotiable Architectural Laws

### Law 1: Single Source of Truth for Citation Unit Counting
`evaluation/platform/task.py` must import `passage_total` directly from `services.agent.postprocess.citations` to count and stamp `"cite": "^[k]"` markers onto tool evidence. Maintaining a separate citation-unit counter in the evaluation harness is prohibited.

### Law 2: Single Source of Truth for Gate Verdicts & Dimensions
`DIMENSIONS` and `PASS_DIMENSIONS` must be declared once in `evaluation/platform/schemas.py`. Suite-level pass/fail evaluation (`gate_verdict`) must be defined once in `evaluation/platform/evaluators.py`—checking `pass_rate >= min_pass_rate`, `safety_failures == 0`, `structural_failures == 0`, `unscored == 0`, and `unattributed == 0`—and called identically by both the CLI runner (`run_evals.py`) and the Langfuse score writer (`run_gate`).

### Law 3: Zero Dependency on Legacy `evaluation/agent/`
`evaluation/platform/` must be 100% self-contained. Using `sys.path.insert` or function-local imports to pull code from `evaluation/agent/` is prohibited. The canonical clinical grading rubric resides at `evaluation/rubric.md` and is read directly by `evaluation/platform/judge_call.py`.

### Law 4: Shared Contract & Envelope Utilities
A2UI envelope traversal and command-to-tool mappings must not be copy-pasted across `evaluation/`. Envelope helpers belong in `shared/`, and command-to-tool expectations must be derived from `services.agent.routing.COMMANDS`.

---

## 5. Dependency Direction & Allowed Import DAG

* **`evaluation/platform/`** → may import `shared/`, `services.mcp.config`, and read-only contracts from `services.agent.postprocess.citations` (`passage_total`), `services.agent.routing` (`COMMANDS`), and `services.agent.ingress.contracts` (`parse_agent_request`). **May not** import `evaluation/agent/`.
* **`evaluation/platform/run_evals.py` & `dry_run.py`** → may import `evaluation/platform/evaluators.py`, `task.py`, `schemas.py`, and `runner.py`.
* **`evaluation/platform/evaluators.py`** → may import `evaluation/platform/schemas.py`, `judge_call.py`, and `shared/`.
* **`evaluation/platform/judge_call.py`** → may import `evaluation/platform/judge_client.py` and `schemas.py`.