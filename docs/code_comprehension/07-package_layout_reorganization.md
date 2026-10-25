# Phase 7: Agent Package Layout Reorganization (6-Subpackage Structure)

**Status:** `PENDING (Runs After Phase 6)`  
**Target Layers:** Layer 3 (`docs/architecture/03-orchestrator.md`) & Layer 4 (`docs/architecture/04-model_runtime.md`)  
**Repositories in Scope:** `enterprise_clinical_copilot` & `danielmherman` (verify `demo/fixtures.py` and `demo/test_chips.py` bridge)  
**Objective:** Reorganize the flat `services/agent/` directory into the 6 domain subpackages defined in the Code Comprehension Guide (`ingress/`, `routing/`, `engine/`, `postprocess/`, `presentation/`, `telemetry/`) with `turn.py` at the package root, while preserving 100% backward compatibility for existing imports, `monkeypatch` targets, Cloud Run / Docker entrypoints, and reference architecture documentation.

---

## 0. Governance Protocol for This Spoke

* **Step A — The Audit Pass (Read-Only):**
  * Inspect the post-Phase-6 module list in `services/agent/` and map every file to its target subpackage in Section 1.
  * Audit every `import`, `from services.agent import ...`, `patch(...)`, and `monkeypatch.setattr(...)` across `tests/`, `evaluation/`, `Dockerfile`, and `danielmherman/demo/` to determine where flat root shims vs. direct import updates are required.
  * Record the exact move map and shim strategy in **Section 2 (Step A — Audit Findings)**.
  * **STOP.** Do NOT move or modify files during Step A. Await explicit Architect approval.
* **Step B — The Execution Pass (Write + Verify):**
  * After Architect approval of Step A, execute the package reorganization, verify all three gates, and record the results in **Section 3 (Step B — Execution Record)**.

---

## 1. Target 6-Subpackage Architecture (`services/agent/`)

Incorporating the modules created in Phase 3 (`turn.py`, `model_runtime.py`, `note_presentation.py`, `risk_canvas.py`) and the retirement of `payload.py` and `message_text.py`, the target package layout is:

```text
services/agent/
├── __init__.py
├── turn.py                        # Root Spine Orchestrator (extracted in Phase 3)
├── ingress/
│   ├── __init__.py
│   ├── http.py                    # Starlette routes (/ask, /ask/stream, /health)
│   ├── contracts.py               # Request & response wire schema validation
│   ├── served_admissions.py       # Cohort authorization whitelist
│   └── run.py                     # CLI client entrypoint
├── routing/
│   ├── __init__.py
│   ├── routing.py                 # Slash-command vs. ReAct + tool gating predicates
│   ├── questions.py               # User/command question wording templates
│   └── admission_context.py       # Pre-flight encounter header manifest
├── engine/
│   ├── __init__.py
│   ├── deterministic.py           # Slash-command fast path (/ehr, /meds, /labs, /risk, /pubmed)
│   ├── graph.py                   # LangGraph ReAct state machine + multi-turn replay
│   ├── model_runtime.py           # Layer 4 Vertex AI/Gemini client & streaming lookahead (from Phase 3)
│   ├── model_turn.py              # Layer 4 response & finish-reason parser (merged message_text.py)
│   ├── prompts.py                 # System prompt & driver-turn behavioral rules
│   └── mcp_client.py              # MCP tool bridge
├── postprocess/
│   ├── __init__.py
│   ├── guardrail.py               # Deterministic post-generation evidence checks
│   ├── citations.py               # Inline marker reconciliation & 1..K renumbering
│   └── followups.py               # Deterministic P1-P11 follow-up chip engine
├── presentation/
│   ├── __init__.py
│   ├── a2ui.py                    # Unified presentation composer entry point
│   ├── risk_canvas.py             # Risk, matrix, labs, meds & literature canvas builder (from Phase 3)
│   ├── note_presentation.py       # Citation/narrative surface resolution (from Phase 3)
│   ├── feature_labels.py          # Clinical ML feature naming
│   ├── admission_phenotypes.py    # Admission phenotype classification
│   └── literature_benchmarks.py   # Static published readmission cohort benchmarks
└── telemetry/
    ├── __init__.py
    ├── chain.py                   # Model pin, code revision & immutable execution record
    ├── observability.py           # Langfuse callback handler & span helpers
    └── stages.py                  # SSE interim progress labels