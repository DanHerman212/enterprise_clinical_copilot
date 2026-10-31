# Master Backend & Reasoning Engine Strategy

**Scope:** Clinical Reasoning Agent, Model Context Protocol Service, and Evaluation Harness  
**Methodology:** Audit → Governance Lock → Hybrid Remediation → Verification  

---

## 1. Strategic Purpose

The Clinical Copilot backend powers multi-turn clinical reasoning, tool execution, citation verification, and structured UI composition. While the codebase is organized into domain packages, prior refactoring occurred without mechanical governance. 

An architectural review reveals two categories of structural drift within the reasoning agent:
1. **Localized Boundary Leaks (Routing and Engine):** Well-structured modules that exhibit isolated ownership violations, including duplicate tool registries, unconstrained tool binding during literature queries, and fallback prose selection inside the execution engine.
2. **Fragmented Evidence Reconstruction (Post-Processing and Presentation):** Downstream layers independently re-scan raw tool history to infer citations, index offsets, and active UI surfaces. Across multi-turn clinical and literature workflows, these independent scanners diverge—causing truncated safety responses, misaligned reference numbering, and unrelated clinical visuals bleeding across turns.

Our objective is to execute a **Governed Backend Remediation Strategy**: enforcing strict layer boundaries through automated structural tests first, relocating misplaced routing and engine logic, and replacing scattered evidence parsers with a single canonical **Evidence Ledger**—while preserving full evaluation parity.

---

## 2. Guiding Engineering Principles

### Principle 1: Separation of Concerns Across Six Architectural Layers
Every unit of backend code belongs to one of six horizontal layers with strict top-down dependencies:
1. **Data & Inference Layer:** Stateless clinical retrieval, literature search, and model inference with zero upward dependencies.
2. **Ingress & Routing Layer:** Wire validation, command dispatch, and canonical tool registries and intent profiles.
3. **Engine Execution Layer:** Workflow orchestration and tool execution that returns raw model outputs and completion metadata without mutating final prose.
4. **Post-Processing & Evidence Ledger Layer:** Prose validation, safety and truncation fallback resolution, and the single canonical Evidence Ledger for citation numbering.
5. **Presentation Layer:** Read-only packaging of settled prose and the Evidence Ledger into turn-scoped UI surfaces and source cards.
6. **Telemetry & Evaluation Layer:** Trace attribution of final served responses and multi-turn evaluation against production contracts.

### Principle 2: Hybrid Remediation
We match our engineering technique to the structural condition of each layer:
* **Realign Where Boundaries Drifted:** Relocate misplaced registries, tool-scoping rules, and fallback decisions to their designated owner layers without rewriting sound modules.
* **Redesign Where State Is Fragmented:** Replace redundant downstream history scanners and offset calculations with an immutable per-turn Evidence Ledger constructed once during post-processing.

### Principle 3: Mechanical Governance Over Automated Code Generation
To prevent AI coding agents from weakening tests or introducing cross-layer workarounds, execution is governed by three controls:
1. **Separation of Duties:** Structural fitness functions and behavioral contract tests are authored and locked before production code is modified. During implementation sprints, the test suite is strictly read-only.
2. **Automated Structural Fitness Functions:** Layer boundaries, single-owner registries, and Evidence Ledger invariants are enforced by automated AST and contract tests at every phase gate.
3. **Deterministic Phase Audits:** Each phase requires zero unauthorized test edits, 100% test pass rates, and clean boundary checks before advancing.

---

## 3. Phased Roadmap

We execute this initiative across five sequential phases:

* **Phase 1: Architecture Drift Audit**  
  Inspect the reasoning agent, data service, and evaluation harness against target architecture charters to catalog all boundary violations and state reconstruction defects.

* **Phase 2: Baseline Equivalence & Governance Lock**  
  Author automated structural fitness functions and behavioral contract tests covering all target architectural invariants, then lock the test suite as read-only for the implementation sprints.

* **Phase 3: Routing & Engine Layer Remediation**  
  Consolidate tool registries and intent-scoping rules into the routing layer, bound redundant tool loops, and strip prose mutation and fallback selection out of the execution engine.

* **Phase 4: Post-Processing & Presentation Layer Remediation**  
  Centralize fallback and stream-truncation resolution in post-processing, implement the canonical Evidence Ledger for unified citation numbering, and restrict presentation surfaces to turn-scoped inputs.

* **Phase 5: Multi-Turn Verification & Evaluation Sign-Off**  
  Verify full compliance across all locked structural and behavioral test suites and execute a live evaluation pass to sign off on the remediated backend.