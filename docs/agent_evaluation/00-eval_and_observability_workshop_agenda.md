# 00 — Evaluation & Observability Workshop Agenda
├── 00-eval_observability_workshop_agenda.md    # Master strategy, 2-tier architecture, &
phase roadmap

├── 01-langfuse_telemetry_and_trace_contract.md # Root trace schema, session grouping, & UI Saved Views

├── 02-harness_architecture_and_assertions.md   # Decoupled data, Turn-2 wire fix, & Suite 2 judge wiring

├── 03-rubric_design_and_judge_calibration.md   # Spec-aligned rubric, /risk widget rule, & Adaptive 2-of-3 Judge

├── 04-cicd_and_local_first_orchestration.md    # Local rehearsal loop vs. SHA-pinned Cloud Run Jobs

└── 04A-release_gate_certification_runbook.md   # Final 4-gate execution protocol & certification sign-off

## 1. Executive Summary & Core Philosophy
A production clinical AI system cannot be certified by reactive trial-and-error, nor can it rely on an uncalibrated LLM judge guessing whether the system behaved properly. 

Previously, our evaluation workflow suffered from three anti-patterns:
1. **Cloud Build as a Debugging Loop:** Test cases were baked into Docker images, turning minor data or prompt adjustments into 15-minute container rebuild cycles.
2. **Blurry Evaluation Boundaries:** Deterministic wire assertions, spec-versus-rubric conflicts, and stochastic LLM-as-a-Judge variance were conflated into a single pass/fail number.
3. **Headless Observability:** Langfuse was treated as a passive JSON dump rather than an interactive triage control plane, forcing engineers to grep raw `.jsonl` files in Cloud Storage to diagnose failures.

This workshop establishes a **programmatic, two-tier evaluation and observability architecture** from first principles. Every phase produces a deterministic contract, a fast local verification loop, and structured visibility in the Langfuse UI.

---
## 2. The Two-Tier Evaluation Architecture (Industry Standard)

In enterprise AI engineering, evaluation is strictly partitioned into two sequential layers. An LLM judge is never invoked until the deterministic perimeter passes 100%.

                 ┌──────────────────────────────────────────────┐
                 │        DEPLOYED AGENT (/ask Boundary)        │
                 └──────────────────────┬───────────────────────┘
                                        │
                                        ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│ TIER 1: DETERMINISTIC ENVELOPE & CONTRACT ASSERTIONS (Zero LLM Cost, 100% Required)   │
│  • Command Coverage: /ehr, /risk, /meds, /labs, /pubmed reach their exact tools       │
│  • Routing Integrity: Free-text positive routes vs. literature/affordance abstentions │
│  • Surface Hygiene: Zero raw JSON leaks, zero unresolved ^[ citation markers          │
│  • A2UI Contract: Expected visual components present; RiskInfluencers suppressed on OP│
│  • Boundary Fast-Fail: Mistyped commands (/ehrr, /) return HTTP 400 with 0 tool calls │
└───────────────────────────────────────────────────────────────────────────────────────┘
                                 (Must pass 100% to proceed)
                                            │
                                            ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│ TIER 2: CALIBRATED CLINICAL & SAFETY ADJUDICATION (Adaptive LLM-as-a-Judge)           │
│  • Spec-Aligned Rubric: 6 clinical dimensions (0–3 scale) + Adversarial policy checks │
│  • Modality-Aware Grading: Widget-only commands (/risk Turn 1) graded on data payload │
│  • Adaptive 2-of-3 Consensus: 1-sample fast pass at temp=0; 2 tie-breakers on FAIL    │
│  • Statistical Gate Alignment: Smoke gates vs. full-census confidence intervals       │
└───────────────────────────────────────────┬───────────────────────────────────────────┘
                                            │
                                            ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│ CONTROL PLANE: LANGFUSE OBSERVABILITY & GCS ARTIFACT SEALING                          │
│  • Root-Trace Taxonomy: sessionId, run:, cmd:, turn:, tools:, verdict                 │
│  • Score Rationales: Judge reasoning attached to score comments for 1-click UI hover  │
│  • Durable Archive: Immutable JSONL + Manifest sealed to gs://ecc-eval-artifacts/     │
└───────────────────────────────────────────────────────────────────────────────────────┘
## 3. Workshop Phases & Document Roadmap

We will execute this workshop across five focused modules, locking each layer before moving to the next:

### Phase 01: Langfuse Telemetry & Trace Contract (`01-langfuse_telemetry_and_trace_contract.md`)
* **Goal:** Turn `https://observability.danielmherman.com` from a chronological JSON firehose into a 1-click clinical triage dashboard.
* **Scope:**
  * Enforce root-trace metadata and tag schemas (`run:<tag>`, `hadm:<id>`, `cmd:<cmd>`, `turn:<1|2>`, `tools:<count>`, `verdict:<pass|fail>`).
  * Fix the Turn-2 `run_tag` drop in `collect_http.py`[cite: 13] and bind Turn 1 + Turn 2 under a unified `sessionId` (`{run_tag}:hadm_{hadm_id}:{cmd}`).
  * Attach judge flags and rationales directly to Langfuse Score `comment` fields.
  * Define the 3 persistent Langfuse UI Saved Views (*Safety & Gate Breaches*, *Turn-2 Evidence Misses*, *Grounding & Fidelity Drift*).

### Phase 02: Harness Architecture & Deterministic Assertions (`02-harness_architecture_and_assertions.md`)
* **Goal:** Eliminate silent data drops, fix file-routing bugs in the harness, and decouple test fixtures from container builds.
* **Scope:**
  * Repair `entrypoint.py` so `suite2_{tag}.jsonl` (including the 5 outpatient abstention cases) is passed to `judge.py` alongside `traces_clinical_{tag}.jsonl`[cite: 14] instead of being orphaned after structural checks.
  * Fix the `isinstance(v, int)` float-aggregation bug in `judge.py`[cite: 16] so consensus median scores (`float`) aggregate properly into the final report.
  * Decouple static case files (`suite2_cases.json`[cite: 23], `adversarial_cases.json`[cite: 22], `cases_82.json`) from requiring Docker rebuilds during iterative testing.

### Phase 03: Rubric Engineering & Judge Calibration (`03-rubric_design_and_judge_calibration.md`)
* **Goal:** Align the evaluation rubric with the product's actual UX specification and stabilize the LLM judge against stochastic flip-flopping and `429` quota exhaustion.
* **Scope:**
  * Codify authoritative display values (`probability_display`, `threshold_display`) so 1-decimal UI rounding is never penalized as a fidelity defect[cite: 7, 8, 9].
  * Codify the `/risk` Turn-1 Widget Rule: grade deterministic widget-only responses on their structured payload rather than penalizing empty narrative prose.
  * Replace unconditional 3-sample judging (and the broken 1-of-3 safety hair-trigger veto) with **Adaptive 2-of-3 Adjudication**: run 1 sample at `temperature=0`; only on `FAIL`, run 2 tie-breaker samples and take a strict 2-of-3 majority across all dimensions.

### Phase 04: Local-First Dev Loop & CI/CD Orchestration (`04-cicd_and_local_first_orchestration.md`)
* **Goal:** Stop using Cloud Build as a feedback loop. Establish a 3-minute laptop-to-Cloud-Run rehearsal workflow and a clean, SHA-pinned production job pipeline.
* **Scope:**
  * **Tier A (Inner Loop — Local CLI):** Run `pytest tests/agent/test_eval_contract.py` (1.2s) + local `collect_http.py`[cite: 13] / `judge.py`[cite: 16] against `$AGENT_URL` for rapid pilot verification with zero container builds.
  * **Tier B (Outer Loop — Cloud Run Jobs):** Enforce clean git-tree guards (`git status --porcelain`), strict commit-SHA image tagging, and automated trace cleanup on aborted preflights.

### Phase 04A: Release Gate Certification Runbook (`04A-release_gate_certification_runbook.md`)
* **Goal:** Execute the statistically calibrated 4-gate certification certification protocol to sign off the agent for public release.
* **Scope:**
  * **Gate 1 (Preflight & Route Coverage):** 100% reachability, 5/5 slash commands verified, SSE stream verified.
  * **Gate 2 (Adversarial, Modality Matrix & 3-Patient Smoke Pilot):** 20/20 Adversarial Pass, 54/54 Structural Assertions Pass, and $\ge 0.75$ Pilot Quality Pass (sized appropriately for $N=24$ variance).
  * **Gate 3 (Full 82-Patient Inpatient Census):** $\ge 0.90$ Clinical Pass Rate and $100\%$ Safety across 656 turns ($95\% \text{ CI} \approx \pm 2.3\%$), sealed to `gs://ecc-eval-artifacts/runs/golden-$SHA/` and certified in Langfuse.