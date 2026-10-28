# Phase 00: Master Strategy — Code Remediation, Observability & Evaluation

## 1. Executive Mission
The Enterprise Clinical Copilot is a flagship production demonstration designed to earn immediate technical trust from discerning evaluators. Each demo user operates within a 20-request budget inside a strictly patient-isolated, 7-day persistent workspace featuring **four distinct interaction surfaces**:
1. **Declarative Slash Commands:** One-click starter chips (`/ehr`, `/risk`, `/meds`, `/labs`, `/pubmed`) that render structured clinical UI cards.
2. **Interactive Risk-Driver Drill-Downs:** Clickable feature bars inside the Risk Visualizer card (e.g., *Oncology History*, *RDW*, *Prior Inpatient Days*) that populate targeted clinical follow-up questions in the composer.
3. **Proactive Follow-Up Chips:** Three context-aware suggested questions dynamically generated beneath every response (`followups.py`, modeled after OpenEvidence).
4. **Open-Ended Free-Form Inquiry:** Natural language and clinical shorthand questions typed directly by the user across multi-turn sessions.

**The Core Problem to Solve:**
During prior single-turn optimization sprints, the codebase was over-fitted to pass isolated 1-turn checks:
* Keyword interceptors were added at the routing layer that hijack conversational follow-ups into static command templates, dropping multi-turn conversation history.
* The conversational engine was stripped of tools and instructed to refuse medication and risk questions rather than reason over them in follow-up turns.
* Observability spans on the command path omit tool inputs and outputs, and the evaluation runner drops Turn 2+ trace IDs, blinding root-cause analysis.

**Strategic Mandate:**
We must first **audit and refactor the application codebase** to support stateful multi-turn interactions across all four UI surfaces, **restore complete trace observability**, and then validate the system using a **Two-Tier Evaluation Platform** fortified against reward hacking via a **Dev vs. Blind Holdout split**.

---

## 2. Core Architectural & Engineering Principles

### Principle 1: Re-Architecting for Stateful Multi-Turn Flow (De-Tuning Single-Turn Hacks)
* **Codebase Audit First:** The codebase must be audited and stripped of brittle single-turn workarounds before running further evaluations.
* **Explicit Commands Only on the Deterministic Path:** Only explicit `/slash` commands or starter chip clicks execute the stateless deterministic command pipeline.
* **Full Capability on the Conversational Path:** All typed questions, clicked `followups.py` chips, and clicked Risk-Driver bar queries must route to the stateful ReAct reasoning loop with full conversation history (`turns`) intact and full access to the clinical toolset (notes, labs, discharge medications, literature, and patient manifest).

### Principle 2: Complete Observability (Full Span I/O & Multi-Turn Traceability)
* **Full Span I/O Attribution:** Every trace span—across both commands and multi-turn conversations—must record its exact inputs, tool outputs, and routing decisions in Langfuse.
* **Multi-Turn Trace Linking:** The evaluation harness must capture and store the full trace ID for every turn in a multi-turn sequence (`Turn 1` through `Turn N`).

### Principle 3: Patient-Scoped Thread Isolation
* **One Patient Per Thread:** Conversation memory is strictly isolated to one patient per thread with a labeled 7-day retention policy. Clearing the patient pill and selecting a new patient isolates state into a separate conversation thread.

### Principle 4: Symmetric Gating & Cost Discipline
* **Zero-Cost Data Checks:** Never spend LLM judge credits testing basic data parsing across 82 patients when a deterministic Python test can verify all 87 served records for free.
* **Symmetric Behavioral Evaluation:** Never gate a multi-turn application on a single-turn canary.
* **Human-in-the-Loop Governance:** The engineering agent advises; the Architect authorizes. No evaluation run is started or aborted without explicit Architect approval.

### Principle 5: Anti-Overfitting & Holdout Generalization (No Reward Hacking)
* **Ban on Test-String Patching:** Engineering fixes must never hardcode evaluation prompt phrases, regex exceptions for specific test questions, or patient-specific overrides into routing or prompts. Fixes must be structural.
* **Dev vs. Blind Holdout Split:** Behavioral evaluation is divided into a **Dev Suite** (~25 scenarios for iterative engineering) and a **Blind Holdout Suite** (~15 scenarios on unseen patient records with paraphrased/shorthand clinical phrasing). The engineer is forbidden from inspecting or tuning against the Holdout Suite. A release is valid only when both suites pass without divergence.

---

## 3. The Two-Tier Evaluation Surface

### Tier 1: Deterministic Cohort Sweep (All 87 Served Records)
* **Goal:** Prove 100% data, manifest, and parser reliability across the entire patient cohort at zero LLM judge cost (<60 seconds runtime).
* **Coverage:** All 82 inpatient records, 5 outpatient records, and 2 unserved boundary records.
* **Verification:** Automated Python test suite asserting that every patient manifest hydrates cleanly, every underlying tool returns valid schemas, and outpatient/unserved guardrails hold across every record.

### Tier 2: Stratified Behavioral & Adversarial Matrix (Dev + Blind Holdout)
* **Goal:** Prove clinical accuracy, multi-turn context retention, UI affordance reliability, and adversarial robustness across realistic user sessions.
* **Axis A — Patient Archetypes (6 Dev Patients + 4 Unseen Holdout Patients):**
  1. Complex Inpatient with Medication Tapers (`90000015`)
  2. High-Risk Inpatient with Oncology History & Abnormal Labs (`90000108`)
  3. Inpatient with Empty Medication Section & Decimal Edge Cases (`90000106`)
  4. Inpatient with All Normal Discharge Labs (`90000105`)
  5. Inpatient with Irregular Note Formatting (`90000023`)
  6. Outpatient Encounter — Non-Admission Boundary (`90000051` / `90000073`)
  *(Plus 4 distinct Holdout patients mirroring these structural phenotypes)*
* **Axis B — The 7 Interaction & Adversarial Trajectories:**
  1. **Declarative Command Baseline (1-Turn):** Clicking `/ehr`, `/risk`, `/meds`, `/labs`, and `/pubmed`.
  2. **Dynamic UI Affordance Loop-Back (2-Turn):** Running a Turn 1 command with `follow_ups=True` or rendering a Risk Card, then feeding the *actual dynamically generated follow-up chip* or *clicked Risk-Driver bar query* (e.g., `"What is the oncology history for this patient?"`) into Turn 2.
  3. **Analytical Drill-Down & Clinical Shorthand (2-Turn):** Following a command with natural or shorthand clinical queries (*"any anticoagulants?"*, *"dc meds?"*, *"which lab was worst at discharge?"*).
  4. **Cross-Domain Multi-Turn Synthesis (3- to 5-Turn):** Exploring across multiple commands and free-text turns within one patient thread (e.g., `/risk` $\rightarrow$ `/labs` $\rightarrow$ synthesizing how lab trends explain the risk drivers).
  5. **Free-Text Cold Start (No Slash Commands):** Natural multi-turn inquiry starting from Turn 1 without using slash commands.
  6. **Adversarial — False Premise & Sycophancy Traps (Multi-Turn):** Asking leading questions mid-thread about non-existent abnormalities or unprescribed drugs to verify the agent refutes false premises rather than agreeing with the user.
  7. **Adversarial — Scope, Prescriptive & Tenant Boundaries:** Prompt injection, cross-patient `hadm_id` snooping mid-thread, treatment recommendation requests (*"Should we start a beta-blocker?"*), and inpatient queries on outpatient records.

---

## 4. Modular Phase Roadmap

Each phase is governed by its own standalone tactical specification document and must pass its acceptance gate before advancing:

* **Phase 00:** Master Strategy — Code Remediation, Observability & Evaluation *(Current Document)*
* **Phase 01:** Codebase Audit & Multi-Turn Routing Refactor *(Remove single-turn prompt/router hacks; enable stateful multi-turn handling for follow-ups, Risk Card driver clicks, and free-form questions)*
* **Phase 02:** Observability & Eval Harness Remediation *(Restore full Langfuse span I/O on all paths and full multi-turn trace ID capture in the runner)*
* **Phase 03:** Tier 1 Deterministic Cohort Sweep *(Build and run the zero-LLM test suite across all 87 records to lock data integrity)*
* **Phase 04:** Tier 2 Stratified Behavioral Evaluation — Dev Suite *(Build and execute the ~25-scenario Dev matrix across the 7 trajectories)*
* **Phase 05:** Tier 2 Blind Holdout Verification & Production Sign-Off *(Execute the unseen ~15-scenario Holdout suite to prove generalization, freeze the golden revision, and transition to the website rebuild)*