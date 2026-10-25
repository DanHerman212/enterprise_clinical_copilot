# Master Architectural Alignment & Remediation Plan

**Reference Standard (The Law):** `docs/architecture/00-reference_architecture_for_agent_system.md` through `12-delivery_ci_cd.md`  
**Repositories in Scope:** `enterprise_clinical_copilot` (Agent, MCP, MLOps, Eval) & `danielmherman` (Django BFF, A2UI Frontend)

---

## 1. Core Engineering Principles

1. **Strict Layer Boundaries:** Any code that violates the *"Context in the Stack"* or *"What Layer X Refuses to Own"* sections in `docs/architecture/` is classified as architectural drift and must be remediated.
2. **No "God Files":** Transport, orchestration, model runtime execution, policy seams, and presentation assembly must live in single-responsibility modules.
3. **Single-Source-of-Truth Contracts:** Wire schemas, tool contracts, feature orders, and section vocabularies are defined once and validated at every trust boundary.
4. **Zero Legacy Dead Weight:** Superseded Phase 1 scripts, unused shims, and unreferenced bundles are removed prior to structural refactoring.

---

## 2. Two-Step Agent Protocol (Governance Rules)

Every phase is governed by its own Spoke document (`docs/remediation/phase-XX-<name>.md`) and must follow a strict two-step cadence:

* **Step A — The Audit Pass (Read-Only):** The coding agent inspects the target files against the specific Layer doc(s), lists exact file and line-number violations, and proposes a surgical file-split or boundary fix inside the Spoke document **without touching application code**.
* **Step B — The Execution Pass (Write + Verify):** Only after explicit Architect approval of Step A, the agent executes the refactor, runs the verification gates, and logs the test output in the Spoke document.

**Definition of Done (Gate to Close a Spoke):**
* `pytest` suite passes 100% in `enterprise_clinical_copilot`.
* `python manage.py test` and `python manage.py check --deploy` pass 100% in `danielmherman`.
* Zero regressions in contract or architectural boundary tests.

---

## 3. Phased Remediation Tracker

| Phase | Spoke Document | Target Layers | Primary Objective | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | `phase-01-dead-code-and-hygiene.md` | Layer 1 & Layer 12 | **Quarantine & Prune Legacy Artifacts:** Remove superseded Phase 1 JS files/bundles and dead Python shims/routes. | `COMPLETE` |
| **Phase 2** | `phase-02-contracts-and-boundaries.md` | Layer 1, Layer 2, Layer 3 & Layer 5 | **Harden Wire Contracts (`contracts.py` & BFF):** Enforce closed-set schemas, zero internal state leakage, and strict BFF vs. Agent separation. | `COMPLETE` |
| **Phase 3** | `03-orchestrator_and_model_runtime.md` | **Layer 3 & Layer 4** | **Decompose the Agent Spine (`http.py`), extracting the tool-budget loop into a new `turn.py`:**<br>• **Track 3A (Layer 3 Orchestration):** Separate HTTP transport (`http.py`), deterministic routing (`routing.py`), the tool-budget loop (currently inside `http.py`; to be extracted as a new module, `turn.py`), and presentation composition (`a2ui.py`).<br>• **Track 3B (Layer 4 Model Runtime):** Isolate all Vertex AI / Gemini SDK calls (primary synthesis + streaming lookahead buffer, `/meds` JSON extraction, and `follow_ups.py` generation), model pinning, and token/latency telemetry into a clean Layer 4 runtime seam. | `COMPLETE` |
| **Phase 4** | `phase-04-tools-models-and-data-isolation.md` | Layer 5, Layer 6 & Layer 7 | **Ground-Truth & Data Isolation Audit:** Verify `hadm_id` isolation (R1), deterministic degraded flags, and online/offline separation. | `PENDING` |
| **Phase 5** | `phase-05-frontend-islands-modularization.md` | Layer 1 (Frontend) | **Decompose `copilot_canvas.js` & Clean Shadow DOM:** Modularize the 1,040-line canvas script and replace Shadow-DOM piercing with a clean CustomEvent/method contract. | `PENDING` |
| **Phase 6** | `phase-06-governance-observability-and-ci.md` | Layers 8, 9, 10, 11 & 12 | **End-to-End Governance & CI Gate Verification:** Verify trace propagation, zero-PII structured logs, conversation sweeps, and immutable image digest gates in CI. | `PENDING` |
| **Phase 7** | `07-package_layout_reorganization.md` | Layer 3 & Layer 4 (`services/agent/`) | **6-Subpackage Layout & Compatibility Shims:** Reorganize flat `services/agent/` modules into the 6 domain subpackages (`ingress/`, `routing/`, `engine/`, `postprocess/`, `presentation/`, `telemetry/`) with `turn.py` at the root and re-export shims so all tests, `monkeypatch` targets, Docker entrypoints, and `danielmherman` fixture imports remain 100% intact. | `PENDING` |
---
