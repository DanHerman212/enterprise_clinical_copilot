# 04B-ux_orchestration_remediation.md

## Master Architectural Directive: Phase 4B UX Orchestration, Phenotype Alignment & Idempotent Routing

### Executive Summary & Problem Statement
Recent verification of Phase 3 (`LabTrendCard`) and Phase 2 (`<a2ui-comparative-risk-matrix>`) exposed critical orchestration, clinical alignment, and guardrail regressions that prevent a predictable, clinician-grade Golden Path:

1. **Pathophysiology Mismatch in Risk Matrix:** Running `/pubmed` or comparative risk queries on an orthopedic surgical patient (e.g., Cynthia Petrov, admitted for *Right Total Knee Replacement*) either fails to render the matrix on orthopedic queries or renders the matrix populated with *Congestive Heart Failure* literature cohorts when CHF is queried. Pairing an orthopedic patient's 9.0% readmission score against heart failure benchmarks breaks clinical integrity.
2. **Fragmented Entry Points & Discovery Friction:** Triggering the `<a2ui-comparative-risk-matrix>` currently requires guessing brittle `/pubmed` query strings that happen to match a 3-condition static lookup table (CHF, COPD, Lumbar Stenosis). Clinicians should not have to hunt for the killer feature or run three separate turns to see the chart summary, the ML risk drivers, and the peer-reviewed cohort comparison.
3. **Guardrail Channel Collision (The RDW `%` Stripping Bug):** On lab trajectory turns, `verify_risk_numbers` scans prose for `%` symbols and strips `RDW` percentages (`12.0%` to `15.6%`) because they do not match the patient's 30-day readmission risk percentage (`9.0%`). This produces corrupted prose (*"The RDW increased from to, moving into the high range"*) and triggers a false-positive `Reconciled with medical record` safety swap badge.
4. **Free-Text Recursion Blocker & Missing `/labs` Macro:** Hiding `/labs` from the top-level slash bar forced users into free-text queries (*"what are the lab trajectories for this patient?"*), which can loop until hitting the graph recursion limit (`MAX_TOOL_CALLS_PER_TURN` applied per-invocation rather than per-turn) or bypass progressive token streaming.
5. **Loss of Progressive Prose Streaming:** Synthesis turns across deterministic and multi-tool paths are arriving as blocked terminal payloads rather than streaming progressively via `answer_delta` ahead of Stage 3 A2UI component mounting.

---

### Remediation Pillars & Engineering Specifications

#### 1. Unify `/ehr` into the Master Clinical Briefing (Golden Path Anchor)
Eliminate the artificial separation between chart summary, SHAP risk drivers, and comparative literature benchmarking. When a clinician runs `/ehr`, deliver the complete inpatient briefing in a single progressive turn:

* **Stage 1 (Live Cognitive Stepper):** *Analyzing clinical chart → Scoring 30-day readmission model & matching cohort literature → Verifying clinical evidence*.
* **Stage 2 (Streaming Clinical Narrative):** Stream the 4-block clinical synthesis (*Hospital Course*, *Discharge Diagnosis*, *Discharge Instructions*, *Readmission Risk & Trajectory* — including a 1–2 sentence risk delta comparison against the matched peer-reviewed cohort rate) via `answer_delta` with inline citation pills.
* **Stage 3 (Unified Risk & Comparative Evidence Stack):**
  1. Mount `<a2ui-risk-influencers>` (Risk Chip + interactive SHAP driver bars).
  2. Immediately beneath (or unified within the risk surface), mount `<a2ui-comparative-risk-matrix>` where:
     * **Row 0:** Pinned Enterprise Clinical Copilot prediction (`probability_display`, `threshold_display`, model provenance, and top matched SHAP drivers).
     * **Rows 1..N:** Peer-reviewed 30-day readmission cohorts dynamically matched to **this patient's primary admission phenotype** (never mismatched to an unrelated condition).
* **Stage 4 (Provenance & Next Steps):** Render the collapsible **References** accordion drawer (containing both EHR note sections and matched PubMed cohort citations) and the 3 clickable **Follow-Up Questions** rows.

---

#### 2. Universal Admission Phenotype Taxonomy Across the 87-Patient Cohort
Replace the brittle 3-condition lookup with a deterministic **Admission Phenotype Classifier** covering the demo cohort so every valid patient admission maps to clinically appropriate 30-day readmission literature:

1. **Define Core Inpatient Phenotype Buckets (`literature_benchmarks.py`):**
   Expand curated 30-day readmission benchmark records (2–3 peer-reviewed studies per bucket with `pmid`, `journal`, `year`, `reported_rate`, `cohort_summary`, `evidence_tier`, and `risk_factor_keywords`) across the primary clinical categories present in the 87-patient cohort:
   * `orthopedic_arthroplasty` — *Total Joint Arthroplasty / Orthopedic Surgery* (e.g., Cynthia Petrov `#90000015`; benchmark 30-day readmission rates ~3.5%–4.8%, e.g., JBJS / JAAOS / JAMA Surgery cohorts).
   * `heart_failure_cardiorenal` — *Congestive Heart Failure / Acute Decompensated HF / Cardiorenal* (existing JACC / Circulation cohorts).
   * `copd_respiratory` — *COPD Exacerbation / Acute Respiratory Failure* (existing cohorts).
   * `spine_neurosurgery` — *Lumbar Stenosis / Spine & Neurosurgery* (existing cohorts).
   * `sepsis_infectious` — *Sepsis / Severe Infection / Pneumonia*.
   * `gi_hepatobiliary_surgery` — *GI Bleed / Hepatobiliary / General Abdominal Surgery*.
   * `general_medicine_metabolic` — *General Inpatient Medicine / Metabolic / Renal*.

2. **Deterministic Admission-to-Phenotype Mapping:**
   * Build a deterministic resolver `resolve_admission_phenotype(hadm_id, discharge_note, diagnosis_features)` that inspects the admission's primary discharge diagnosis / surgical procedure headers and maps each of the 87 cohort admissions to its canonical phenotype key.
   * **Cohort Hygiene Audit:** Run an audit script across all 87 admissions to verify that every `hadm_id` resolves cleanly to a phenotype and has sufficient EHR note density. Flag any corrupted or empty records so they can be excluded from the patient selector whitelist if necessary.

3. **Condition-Locked Matrix Guardrail:**
   * `<a2ui-comparative-risk-matrix>` Row 0 (the patient's ML risk score) may **only** be paired with literature rows matching the active patient's resolved phenotype (or an explicitly queried comorbidity present in the patient's active problem list).
   * If a user runs an open-ended `/pubmed <unrelated condition>` query (e.g., asking about CHF while viewing an orthopedic patient without HF), render standard `<a2ui-literature-list>` evidence cards badged `General Medical Evidence`—do **not** inject the patient's unrelated index readmission score into a mismatched condition matrix.

---

#### 3. First-Class Slash Macros + Idempotent Free-Text Intent Routing
Clinicians rely on `/slash` commands for zero-latency cognitive ergonomics, while evaluators and casual users frequently type natural language. Both paths must be idempotent and converge on the same deterministic handlers.

1. **First-Class Composer Slash Bar (`a2ui_console.html` / `copilot_canvas.js`):**
   Expose four primary deterministic macros in the `/` autocomplete menu and composer hints:
   * `/ehr` — *Comprehensive chart synthesis, readmission risk & cohort matrix*
   * `/labs` — *Longitudinal inpatient lab trajectories & sparklines*
   * `/meds` — *Discharge medication reconciliation grid*
   * `/pubmed` — *Targeted clinical literature evidence search*
   *(Keep `/risk` aliased to `/ehr` in backend routing so existing tests and muscle memory remain unbroken.)*

2. **Idempotent Free-Text Intent Router (`routing.py` / `graph.py`):**
   * Before entering an unconstrained ReAct tool loop on free-text input, classify high-confidence core intents (`ehr_summary`, `lab_trends`, `med_reconciliation`, `risk_comparison`):
     * Queries asking for lab trends/trajectories/panels (e.g., *"what are the lab trajectories for this patient?"* or clicking a lab SHAP driver bar) route directly to the deterministic `/labs` pipeline (`get_lab_trends` + `<a2ui-lab-trend>`).
     * Queries asking for medication lists/doses route directly to `/meds`.
     * Queries asking for overall readmission risk, chart summary, or cohort comparison route directly to `/ehr`.
   * **Discreet Macro Discovery Pill:** When a free-text query is routed into a deterministic slash handler, include `"routed_command": "/labs"` (or `/ehr`, `/meds`) in `Turn.presentation` and render a subtle, muted pill in the turn header next to the scope badge:
     `⚡ Routed to /labs` (with hover tooltip: *"Tip: Type /labs for direct command access"*).

3. **Fix Per-Turn Tool Call Budget (`graph.py`):**
   * Fix `MAX_TOOL_CALLS_PER_TURN` so it counts total `ToolMessage` instances accumulated across the **entire turn** rather than resetting on each `_execute_tool_calls` invocation.
   * Deduplicate identical `(tool_name, args)` calls within the same turn so an open-ended ReAct turn can never loop 6 times on `get_lab_trends` and hit `GraphRecursionError`.

---

#### 4. Explicit Evidence-Channel Vocabulary in Guardrails (`guardrail.py`)
Eliminate the recurring pattern where adding a new tool channel causes previously written regex guardrails to strip valid clinical numbers and trigger false-positive safety swaps:

1. **Formalize `TurnEvidenceChannels`:**
   Pass a unified structured evidence context into post-generation guardrails containing all grounded tool outputs for the active turn:
   * `notes_text` (from `rag_search_sections`)
   * `prediction_payload` (from `predict_readmission`)
   * `literature_text` & `benchmark_rates` (from `search_literature` / `literature_benchmarks`)
   * `lab_trends_payload` (from `get_lab_trends`)
   * `medications_payload` (from `extract_medications`)

2. **Fix the RDW `%` Stripping Defect in `verify_risk_numbers`:**
   * Extract all numeric values from `lab_trends_payload`—specifically analyte values whose unit is `%` (such as `RDW`: `11.8%`, `12.0%`, `15.6%`, plus reference bounds `11.5%–14.5%`)—and add them to the allowed percentage whitelist in `verify_risk_numbers`.
   * Add a regression test verifying that *"The RDW increased from 12.0% to 15.6%, moving into the high range"* passes `verify_risk_numbers` with zero modifications and `swap_mode = "none"`.

3. **Fix Turn Scope & Badge Suppression on `/labs` and Drill-Downs:**
   * Ensure turns grounded in `get_lab_trends` are recognized as valid `Patient Chart Synthesis` turns and do not raise a bogus `Visible Safety Swap` (`Reconciled with medical record`) when no ungrounded claims were removed.

---

#### 5. Restore End-to-End Progressive Streaming
Ensure that our streaming architecture (§1.1–§1.4 of `04-advanced_ux_new_directives.md`) operates visibly across `/ehr`, `/labs`, `/pubmed`, and free-text turns:

1. **Stream Narrative Tokens Before Mounting Stage 3 Surfaces:**
   * Verify that deterministic synthesis paths (`/ehr`, `/labs`, `/pubmed`) stream LLM tokens live via `astream` -> `answer_delta` SSE frames rather than awaiting blocking `ainvoke` completion.
   * On the client (`copilot_canvas.js`), ensure Stage 2 prose drains smoothly via the character pacing queue (`~45–50 chars/sec`) *before* Stage 3 reveals `<a2ui-risk-influencers>`, `<a2ui-comparative-risk-matrix>`, or `<a2ui-lab-trend>`, and *before* Stage 4 reveals the References drawer and Follow-Up rows.
2. **Preserve Stream on Clean Guardrail Pass:**
   * When `guardrail.py` makes zero content modifications (or only silent whitespace/citation normalization), `prose_settled` must use `"silent"` reconciliation so the paced streaming animation completes naturally without clobbering the DOM mid-stream.

---

### Execution Sequence & Verification Gate
Please execute these items in the following order before resuming Phase 3.2 / Phase 4 (`/demo/` sandbox):

1. **Step 1 (Guardrails & Loop Bound):** Fix `MAX_TOOL_CALLS_PER_TURN` across iterations, deduplicate repeated tool calls, and add `lab_trends_payload` (`RDW %`) to `verify_risk_numbers` so `"from 12.0% to 15.6%"` renders intact with no false safety badge.
2. **Step 2 (Phenotype Mapping & Cohort Audit):** Expand `literature_benchmarks.py` to cover the core inpatient phenotypes (starting with `orthopedic_arthroplasty` for Cynthia Petrov `#90000015`) and map all 87 cohort admissions to their primary phenotype. Report the cohort distribution (and any sparse `hadm_id`s that should be hidden).
3. **Step 3 (Unified `/ehr` + `/labs` Macro + Free-Text Routing):** Wire `/ehr` to render the unified narrative + `RiskInfluencers` + phenotype-matched `<a2ui-comparative-risk-matrix>`. Expose `/labs` in the slash menu and wire free-text intent routing with the `⚡ Routed to /...` indicator.
4. **Step 4 (Streaming Verification):** Verify in the live browser on Cynthia Petrov (`#90000015`) that `/ehr` streams prose smoothly, renders her 9.0% risk against *Orthopedic / Total Knee Arthroplasty* 30-day readmission cohorts, and clicking the RDW driver bar (or running `/labs` or asking for lab trajectories in free text) streams a clean narrative with intact `12.0%` → `15.6%` figures followed by `<a2ui-lab-trend>`.