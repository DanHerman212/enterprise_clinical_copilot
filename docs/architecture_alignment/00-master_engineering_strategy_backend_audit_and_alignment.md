# Master Backend & Reasoning Engine Strategy

**Scope:** Clinical Reasoning Agent, Model Context Protocol Service, and Evaluation Harness  
**Methodology:** Audit → Inventory → Governance Lock → Hybrid Remediation  

---

## 1. Strategic Purpose

The Clinical Copilot backend powers multi-turn clinical reasoning, deterministic tool execution, citation verification, and structured UI surface composition. While the backend packages were previously organized into domain subdirectories and backed by automated test coverage, that earlier refactoring was executed without strict mechanical governance.

Live multi-turn testing across mixed clinical chart and medical literature workflows reveals two distinct categories of architectural drift within the backend:
1. **Structurally Sound Modules with Boundary Drift (Routing, Engine, Data Services, and Evaluation):** The core tool handlers, routing registries, and evaluation harness have clear architectural charters, but exhibit localized boundary leaks—such as fallback prose authored inside the execution engine and unconstrained tool availability during free-text literature queries.
2. **Entangled Cross-Layer State Reconstruction (Post-Processing and Presentation):** Rather than resolving turn evidence once, downstream post-processing and presentation layers independently scan raw tool outputs and conversation history to infer citable passages, literature articles, citation offsets, and active UI canvases. When a conversation pivots across patient chart notes and medical literature, these independent scanners disagree—producing truncated safety fallbacks, stripped citations, misaligned reference numbering, and unrequested clinical canvases bleeding into literature responses.

Our strategic objective is to execute a **Governed Backend Remediation Strategy**: auditing the codebase against our target architecture charters, locking boundaries with automated structural and contract gates, enforcing strict layer ownership across routing and engine execution, and replacing fragmented evidence reconstruction with a single canonical **Evidence Ledger**—all verified against our multi-turn evaluation harness.

---

## 2. Guiding Engineering Principles

### Principle 1: Separation of Concerns Across Six Architectural Layers
Maintainable clinical reasoning systems enforce strict boundaries between data retrieval, request routing, model execution, prose validation, UI packaging, and evaluation measurement. Every unit of backend code belongs to one of six horizontal layers:
1. **Data & Inference Layer (MCP):** Executes stateless clinical retrieval, literature search, readmission risk inference, and laboratory trend assembly with zero upward dependency on the reasoning agent and strict isolation from offline ingestion pipelines.
2. **Ingress & Routing Layer:** Validates wire contracts, resolves slash commands and routing predicates, and owns the canonical tool capability registries.
3. **Engine Execution Layer:** Orchestrates the multi-turn reasoning loop and deterministic command workflows, enforces tool-scope discipline, and returns raw model outputs, completion metadata, and tool payloads without mutating final prose or stripping citations.
4. **Post-Processing & Evidence Ledger Layer:** Owns all prose validation, stream-truncation and safety-filter fallback resolution, and the single canonical **Evidence Ledger** that maps raw clinical notes and literature records into sequential, non-colliding display citations.
5. **Presentation Layer:** Consumes settled prose and the canonical Evidence Ledger read-only to package source cards, narrative citation badges, and turn-scoped clinical canvases without altering citation indices or bleeding stale surfaces across turns.
6. **Telemetry & Evaluation Layer:** Records post-processed served outputs on root observability traces and grades multi-turn behavior using the exact same citation and schema contracts as production.

### Principle 2: Hybrid Remediation (Realign Leaking Boundaries, Redesign Fragmented Evidence State)
We match our engineering technique to the structural reality of each backend subsystem:
* **Realign Where Boundaries Drifted:** Where modules are structurally sound but violate layer ownership—such as fallback strings inside the execution engine or unconstrained tool binding during specialized queries—we relocate the misplaced logic to its designated layer without rewriting the surrounding subsystem.
* **Redesign Where State Is Reconstructed Repeatedly:** Where downstream layers repeatedly re-parse raw tool history and compute ad-hoc index offsets, we replace the scattered scanners with an immutable per-turn Evidence Ledger built once at the post-processing boundary and read uniformly by guardrails, presentation surfaces, and the evaluation harness.

### Principle 3: Mechanical Governance Over Automated Code Generation
When AI coding agents remediate backend architecture, relying on prompt instructions alone introduces the risk of "reward hacking"—where an agent weakens existing test assertions, introduces ad-hoc index patches, or creates cross-layer coupling simply to produce a passing test run. Execution is governed by three mechanical controls:
1. **Separation of Duties (Test Immutability):** Behavioral and contract test suites are locked as read-only whenever production code is being realigned or redesigned, unless explicitly authorized for a new architectural contract.
2. **Automated Structural Fitness Functions:** Layer dependency rules, single-owner registries, no-prose-mutation rules in the engine, and Evidence Ledger invariants are enforced by automated structural tests at every phase gate.
3. **Deterministic Phase Audits:** Each phase begins with a read-only inspection and concludes with an automated verification check confirming zero unauthorized test modifications, 100% test pass rates, and adherence to architectural boundaries before a commit is authorized.

---

## 3. Phased Roadmap

We execute this initiative across eight sequential phases, moving from discovery and inventory review through governed remediation and final evaluation sign-off.

* **Phase 1: Architecture Drift Audit**  
  Conduct a read-only inspection across the reasoning agent, data service, and evaluation harness to measure exact structural and behavioral drift against our target architecture charters.

* **Phase 2: Inventory Review & Dead-Code Catalog**  
  Catalog all backend modules, helper functions, legacy shims, and duplicate tool or citation selectors to establish a clean disposition map before modifying production code.

* **Phase 3: Baseline Equivalence & Governance Lock**  
  Expand automated structural fitness functions and multi-turn domain-transition contract tests, then lock the behavioral test suite as a read-only mechanical guardrail.

* **Phase 4: Routing & Engine Layer Remediation**  
  Enforce strict tool-scope discipline on free-text literature queries to prevent runaway chart and lab tool loops, and strip all fallback prose mutation out of the engine so it returns pure model outputs and completion status.

* **Phase 5: Post-Processing & Canonical Evidence Ledger**  
  Consolidate all stream-truncation and safety-filter fallback handling in the post-processing layer, and implement the single canonical Evidence Ledger to assign sequential, non-colliding citations across clinical notes and medical literature.

* **Phase 6: Presentation Layer & Surface Scoping Remediation**  
  Refactor the presentation layer to read exclusively from the settled Evidence Ledger without ad-hoc offset math, and gate visual clinical canvases strictly to the active turn's scope.

* **Phase 7: Data Service & Evaluation Harness Alignment**  
  Remediate any data-layer boundary violations identified in the audit and align the evaluation harness to consume the canonical Evidence Ledger contract.

* **Phase 8: Multi-Turn Stress Verification & Evaluation Sign-Off**  
  Execute the deterministic multi-turn cross-domain test matrix and run a full live evaluation pass to record the final release gate verdict.