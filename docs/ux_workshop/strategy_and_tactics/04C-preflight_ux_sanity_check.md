# 04C-preflight_ux_audit_and_remediation.md

## Master Architectural Directive: Phase 4C Pre-Flight Pixel-by-Pixel UX Audit, Compaction & Telemetry Capture

### Executive Summary & Architectural Mandate
Before transitioning to automated Langfuse evaluation harnesses and scoring rubrics, the Enterprise Clinical Copilot must pass a strict **Pre-Flight Visual, Content, and Epistemic Sanity Check**. 

Automated metrics evaluate backend retrieval and ground-truth alignment, but they are blind to DOM-level presentation defects—such as raw, unformatted JSON leaking into prose, mutated Markdown syntax (`**`, broken headers, malformed `^[n]` citation tags), vertical UI bloat, and ungrounded clinical claims masquerading as medical guidance.

Furthermore, this directive establishes our core clinical guardrail: **The Copilot is strictly an evidence-retrieval, synthesis, and statistical risk-benchmarking engine focused exclusively on 30-day unplanned hospital readmission.** It is *never* to output autonomous "synthesized clinical guidance" or ungrounded medical advice. Every assertion on the screen must trace directly to an EHR section, a structured lab/medication record, the SHAP feature store, or a peer-reviewed PubMed cohort study.

This document instructs the engineering team to:
1. Execute immediate visual compactions and transport fixes.
2. Run a structured, multi-patient **Pixel-by-Pixel Clinical Simulation Gauntlet** in a live browser.
3. Capture the exact Turn Input/Output payloads, SSE frame traces, and DOM renderings across every turn, and compile a comprehensive **Pre-Flight UX Findings Report** for joint review before any automated evals begin.

---

### 1. Immediate Pre-Audit Engineering Fixes

Before running the audit matrix, complete these four foundational fixes so we are inspecting the intended production surface:

#### 1.1 Restore End-to-End Progressive Streaming (`views.py` & `copilot_canvas.js`)
* **Fix the Site Proxy Dispatch:** In `views.py`, replace any strict equality check on the `Accept` header with a substring match (`'text/event-stream' in request.headers.get('Accept', '')`) so the proxy never silently downgrades to a blocking `application/json` response.
* **Enforce Anti-Buffering Headers:** Ensure `StreamingHttpResponse` explicitly sets `Cache-Control: no-cache, no-transform` and `X-Accel-Buffering: no`.
* **Verify Stage Progression:** Confirm that the live cognitive stepper transitions smoothly across stages (never freezing on *"Sending the question"*) and that Stage 2 narrative tokens stream via `answer_delta` prior to mounting Stage 3 A2UI web components[cite: 14].

#### 1.2 Compact the Risk & Matrix Stack (`a2ui_clinical.js` / `demo_shell.css`)
When `/ehr` renders `RiskInfluencers` directly above `<a2ui-comparative-risk-matrix>`, the current layout suffers from excessive vertical footprint and visual redundancy[cite: 13]:
* **Compact `RiskInfluencers`:** Reduce vertical padding and card margins by ~30%. Tighten row spacing between individual SHAP driver bars (`admission type`, `RDW`, `lowest monocyte count`, etc.), reduce bar thickness from heavy blocks to sleek clinical indicators (`4px–6px` height), and tuck the `INCREASING RISK` / `DECREASING RISK` subheaders tightly under the top `9.0% BELOW THRESHOLD` header[cite: 13].
* **Eliminate the 4th Matrix Column (`SHARED RISK FACTORS`):** Completely remove the `SHARED RISK FACTORS` pill column from `<a2ui-comparative-risk-matrix>`[cite: 13]. Because the SHAP bars immediately above the table already visualize the patient's exact risk drivers and their directional magnitudes, repeating them as flat pills inside the table creates visual clutter[cite: 13]. The matrix must render cleanly as a tight 3-column table:
  `SOURCE` | `READMISSION RATE` | `COHORT`[cite: 13]

#### 1.3 Eradicate Raw JSON Leaks & Mutated Markdown
* **Prose Sanitization Audit:** Inspect both deterministic formatters (`deterministic.py` / `routing.py`) and the client-side Markdown/citation parser (`copilot_canvas.js`) for rendering leaks:
  * **No Raw JSON in Prose:** Tool payloads, dictionary literals, or raw array strings must never bleed into `.cp-narrative` text blocks.
  * **No Mutated Markdown:** Fix broken bold tags (`**Hospital course**` rendering with literal asterisks), orphaned list markers, or half-parsed citation brackets (`^[1]` or `[PMID: ...]`) caused by streaming chunk splits or guardrail regex replacements (`guardrail.py`).
  * **Strict Block Typography:** Verify that the four `/ehr` narrative blocks (*Hospital Course*, *Discharge Diagnosis*, *Discharge Instructions*, *Readmission Risk & Trajectory*) render with clean structural hierarchy and intact inline citation pills[cite: 14].

#### 1.4 Replace "Synthesized Clinical Guidance" with Chart-Derived Evidence (Phase 3.2 Remediation)
* **Retire Ungrounded Guidance:** In Phase 3.2 (clicking suggested follow-up rows or medication drill-downs), the model must **never** generate general medical advice or ungrounded clinical recommendations[cite: 14].
* **Enforce Chart-Derived Extraction:** Clicking a clinical follow-up row (or running `/meds` / `/labs`) must strictly summarize what is documented in the patient's MIMIC-IV discharge record (with `[n]` section citations) paired with the deterministic `MedsGrid` or `LabTrendCard`[cite: 14].
* **Unhandled Slash Command Guard:** Ensure mistyped slash commands (e.g., `/drugs`) never hang the composer in a pending state. Route `/drug` and `/drugs` as aliases to `/meds`, and ensure any unrecognized `/command` returns a clean system notice and immediately releases the composer lock.

---

### 2. Known Architectural Boundary (Document in Repo / README)

Add an explicit architectural note documenting the boundary between our global ML model and phenotype-stratified literature cohorts:
* **Global Risk Model vs. Phenotype-Specific Literature Cohorts:** The Enterprise Clinical Copilot's proprietary XGBoost predictor (`readmission-final`) is trained as a unified, cross-cohort inpatient risk model (outperforming the clinical HOSPITAL score benchmark on AUCPR)[cite: 13, 14]. In contrast, peer-reviewed literature benchmarks in `<a2ui-comparative-risk-matrix>` report 30-day readmission rates within specific clinical phenotypes (e.g., *Primary TKA, n=3,015* at `3.4%` vs. *Acute Decompensated Heart Failure*). While comparing an individual's global ML risk score against their admission phenotype's published baseline provides essential clinical context at discharge, our post-launch MLOps roadmap includes training **phenotype-stratified sub-models** so model calibration and cohort baseline granularity match 1:1.

---

### 3. The Pixel-by-Pixel Clinical Simulation Protocol

Do **not** rely solely on headless unit tests. Execute the following 4-Archetype, Multi-Turn Simulation Protocol through the **live browser UI** (`a2ui_console.html`) against the live agent.

For **every single turn** below, the engineer must capture and inspect:
1. **The Raw SSE Event Stream & Timing:** Did `stage` events progress cleanly? Did `answer_delta` frames stream progressively, or arrive as a single burst?
2. **The Settled Prose (`.cp-narrative`):** Copy the exact rendered text. Flag any raw JSON fragments, mutated Markdown tokens, stripped numbers (e.g., missing percentages or lab units), or claims lacking an inline citation pill.
3. **The Shadow DOM Web Components:** Inspect the rendered state of `<a2ui-risk-influencers>`, `<a2ui-comparative-risk-matrix>`, `<a2ui-lab-trend>`, `<a2ui-meds-grid>`, and `<a2ui-literature-list>`. Verify column alignment, compact spacing, and absence of console errors.
4. **The Epistemic & Guardrail State:** Record the turn scope badge (`Patient Chart Synthesis` vs. `General Medical Evidence`), whether a `Reconciled with medical record` badge appeared, and whether the `⚡ Routed to /...` pill rendered on free-text turns.

---

### 4. The 4-Archetype Audit Matrix (14 Turns Total)

#### Archetype A: The Orthopedic Golden Path — Cynthia Petrov (`#90000015`, Right TKA)
* **Turn A1 (`/ehr`):**
  * *Check:* Progressive streaming of the 4 narrative sections; inline citations `[1]–[4]`; compacted `RiskInfluencers` widget (`9.0%` vs `11.0%` threshold)[cite: 13]; 3-column `<a2ui-comparative-risk-matrix>` showing Row 0 (`9.0%`) pinned above the *Journal of Arthroplasty* TKA benchmark (`3.4%`) with **no** 4th "Shared Risk Factors" column[cite: 13].
* **Turn A2 (Click `red cell distribution width (RDW)` bar in `RiskInfluencers`):**
  * *Check:* Contextual drill-down dispatch[cite: 13, 14]; narrative preserves `12.0%` to `15.6%` without guardrail stripping or false safety-swap badges; `<a2ui-lab-trend>` renders cleanly below the prose[cite: 14].
* **Turn A3 (Free-text: `"What did her hemoglobin and sodium do during this admission?"`):**
  * *Check:* Free-text intent router maps to `/labs` (`⚡ Routed to /labs` pill visible); zero tool-call recursion loops; anchored BigQuery values (`hemoglobin_min: 8.3 g/dL`, `sodium: 132–139 mEq/L`) reported accurately with clean Markdown formatting.
* **Turn A4 (`/meds`):**
  * *Check:* Renders chart-grounded discharge medication summary + structured `<a2ui-meds-grid>`[cite: 14]. Zero ungrounded prescribing advice.
* **Turn A5 (Click a Suggested Follow-Up Question Row from Turn A1):**
  * *Check:* Verify Phase 3.2 compliance—response is 100% grounded in the chart or curated evidence with inline citations[cite: 14]. Flag any ungrounded "clinical guidance" or raw JSON/Markdown defects.

#### Archetype B: High-Risk Medical Phenotype — Select 1 Validated `cardiac` (CHF) or `copd` Patient from the 87-Cohort
* **Turn B1 (`/ehr`):**
  * *Check:* Verify phenotype resolver maps the admission to `cardiac` or `copd`; verify `RiskInfluencers` displays the patient's high-risk state (`ABOVE THRESHOLD` styling palette); verify the 3-column matrix loads the matched CHF (`37495280`) or COPD (`36528962`, `36701677`, etc.) 30-day cohort rows.
* **Turn B2 (`/labs`):**
  * *Check:* First-class `/labs` slash command execution; verify sparklines/trends and narrative formatting.
* **Turn B3 (Free-text: `"Show me the discharge medications and dosing for this admission"`):**
  * *Check:* Intent routing to `/meds` (`⚡ Routed to /meds` pill); verify `<a2ui-meds-grid>` rendering and prose hygiene.

#### Archetype C: Cross-Condition Guardrail & Matrix Suppression — Stay on Cynthia Petrov (`#90000015`)
* **Turn C1 (`/pubmed congestive heart failure 30-day readmission`):**
  * *Check:* Because Cynthia Petrov is an `orthopedic` admission, querying CHF must **never** inject her `9.0%` surgical risk score into a CHF comparative matrix. It must render `<a2ui-literature-list>` cards badged `General Medical Evidence`.
* **Turn C2 (`/pubmed total knee arthroplasty readmission risk factors`):**
  * *Check:* Phenotype matches active patient (`orthopedic`); verify clean citation rendering and zero duplicate/broken matrix layouts.

#### Archetype D: Policy Fallback & Edge-Case Resilience
* **Turn D1 (Select 1 `obstetric` or `psychiatry` patient — Excluded by Policy — and run `/ehr`):**
  * *Check:* Narrative and `RiskInfluencers` render normally. Because the bucket is marked non-matrix-eligible by policy, verify whether `<a2ui-comparative-risk-matrix>` gracefully omits literature comparison rows (or suppresses the table cleanly) without throwing a JS error or rendering an empty broken grid.
* **Turn D2 (Select 1 of the 3 `outpatient / non-admission` notes and run `/ehr`):**
  * *Check:* Verify how the UI handles a non-admission record—no bogus readmission matrix offered, clean fallback narrative.
* **Turn D3 (Type invalid command `/drugs` in the composer):**
  * *Check:* Verify alias routing to `/meds` (or clean error recovery) and confirm the composer input never locks up.

---

### 5. Required Deliverable: `04C-ux_audit_findings_report.md`

Upon executing the 14 turns above, do **not** push to production yet. Generate and present a structured engineering report (`04C-ux_audit_findings_report.md`) containing:

1. **Turn-by-Turn Ledger:** A structured table and verbatim log of all 14 turns showing:
   * `Turn ID` | `Patient (hadm_id & Phenotype)` | `Input` | `SSE Stream Verified? (First/Last Delta ms)` | `Routed Pill / Scope Badge` | `A2UI Components Mounted` | `Visual / Prose / Markdown Defects Observed`.
2. **Verbatim Prose & DOM Snapshots:** Include the exact rendered text and component properties for Turns A1, A2, A5, B1, C1, and D1 so we can inspect every sentence, citation pill, and number together.
3. **Defect Triage List:** Categorize every anomaly discovered during the 14-turn walkthrough into:
   * **Category 1 (Formatting & DOM Polish):** Mutated Markdown, leaked JSON strings, CSS spacing/alignment flaws, shadow DOM glitches.
   * **Category 2 (Epistemic & Guardrail Violations):** Any sentence lacking a citation, any hallucinated "clinical guidance," or any false-positive guardrail stripping.
   * **Category 3 (Orchestration & Transport):** Any streaming stalls, missing stage transitions, or routing misses.

We will review `04C-ux_audit_findings_report.md` together, sign off on the final surgical fixes, build the `/demo/` static fixture from the verified Golden Path output, and only then advance to the automated evaluation harness.