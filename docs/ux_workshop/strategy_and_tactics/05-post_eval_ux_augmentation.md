# Phase 05: Proactive UX Augmentation — Deterministic Phenotype & Entity-Slotted Follow-Up Engine (`followups.py`)

**Status:** Active — Section 1 Authorized (Gate 2 Paused Pending Completion)  
**Objective:** Replace unconstrained, probabilistic LLM follow-up question generation in `services/agent/followups.py` with a deterministic, zero-LLM **Clinical Phenotype & Entity-Slotted Follow-Up Engine** (Architecture B). Every suggested follow-up chip must be dynamically tailored to the active patient's verified chart entities and guaranteed to be 100% answerable from grounded record evidence.

---

## Architectural Guardrails & Anti-Overfitting Rules
1. **Zero Patient-ID Hardcoding (`hadm_id` Agnostic):** The phenotype selector and entity slot-filler must operate strictly on runtime chart signals (retrieved note passages, medication text, laboratory panels, readmission risk payload, and encounter type). Never hardcode rules by `hadm_id`.
2. **Holdout Isolation Preserved:** Do not inspect, reference, or test against any Holdout record IDs during design or unit testing. All verification in this document must use the 7 Dev Cohort archetype records (`90000015`, `90000108`, `90000106`, `90000105`, `90000023`, `90000051`, `90000073`) or synthetic unit test fixtures.
3. **100% Answerability Contract:** A follow-up template may only be selected if the exact data required to answer that question is verified as present in the patient's record. Never suggest a question whose answer is absent from the chart.
4. **Zero LLM Calls / Zero Added Latency on Chip Generation:** Follow-up generation must execute deterministically in memory without invoking an LLM generation call.

---

## Section 1: Read-Only Audit of `followups.py` & Runtime Context Availability

### 1.1 Objective
Before designing the deterministic taxonomy and slot-extraction rules in Section 2, perform a read-only architectural audit of how `followups.py` is currently called, what data structures are already in memory at the end of Turn 1, and what existing deterministic extraction logic in the repository can be reused.

### 1.2 Audit Tasks (Read-Only — Zero Code Changes)
Inspect `services/agent/` and `evaluation/inventory/` and answer the following four questions in a concise **Section 1 Audit Report**:

1. **Current Call Sites & Signature of `followups.py`:**
   - Where is `followups.py` invoked across `deterministic.py`, `orchestrator.py`, or route handlers when `follow_ups=True`?
   - What arguments are currently passed into the follow-up generator, and what is its exact return contract (e.g., `list[str]`, max count, SSE event or envelope field)?
   - How is it currently traced in Langfuse (span name, input/output attributes)?

2. **In-Memory Data Available at Each Call Site:**
   - For each Turn 1 execution path (`/ehr`, `/meds`, `/labs`, `/risk`, `/pubmed`, and free-text ReAct turns), list the exact tool payloads or state objects already in memory when `followups.py` is called (e.g., `rag_search_sections` passages, `get_medications` text, `get_lab_results` rows, `predict_readmission_risk` manifest/drivers, `hadm_id`, `OUTPATIENT_HADM_IDS`).
   - Identify if any Turn 1 path lacks direct in-memory access to the patient's note/lab/risk signals and whether a lightweight MCP/store lookup or cached patient-state helper is needed so `followups.py` has complete chart-state visibility regardless of which command ran on Turn 1.

3. **Reusable Entity & Phenotype Extraction Logic:**
   - Inspect `evaluation/inventory/dev_fact_sheet.py` (and any regex/entity parsers used to build `cohort_provenance_inventory.json`, such as `anticoagulant`, `beta_blocker`, `diuretic`, `taper`, abnormal lab ranking, and section detection).
   - Report which deterministic extraction patterns already exist that can be promoted/adapted into `services/agent/followups.py` (plus additional entity extractors for specific drug names, procedure/condition phrases, follow-up providers, or return precautions).

4. **Turn-2 Routing Compatibility:**
   - When a user clicks a follow-up chip on Turn 2, that string is sent as a free-text user message into the orchestrator/ReAct loop.
   - Confirm how the orchestrator routes free-text follow-up questions on Turn 2 (e.g., which tools the ReAct agent calls for medication, lab, risk, or discharge-instruction questions) so our Section 2 chip templates align cleanly with the ReAct agent's tool-calling behavior.

### 1.3 Deliverable & Stop Gate
Present the **Section 1 Audit Report** in chat covering items 1–4 above. **Make zero code edits in Section 1 and STOP for Architect review before Section 2.**

---

## Section 2: Deterministic Taxonomy, Entity Slot-Filling & Implementation

### 2.1 Architectural Decisions from Section 1 Audit
Based on `05a-section1_followups_audit.md`:
1. **Pure In-Memory Context (Approach 2.2a — Zero Extra Network Calls):** Pass `state` (containing `command`, `command_payload`, `tool_calls`, `replayed_tool_calls`) and `outpatient` into `followups.suggest()` from `http.py:_compose_success`. Do not add network/MCP lookups inside `followups.py`.
2. **Reuse Verified Service & Inventory Extractors (§3.1):**
   - Promote the brand-inclusive `DRUG_CLASSES` regexes (`anticoagulant`, `beta_blocker`, `diuretic`, `taper`) and lab excursion helpers (`_excursion`, `_discharge_point`) from `evaluation/inventory/dev_fact_sheet.py` into `services/agent/followups.py`.
   - Reuse `deterministic.py::parse_medications` (or `state["command_payload"]` when `command == "/meds"`) to extract verified medication names, doses, and frequencies.
   - Add lightweight, deterministic regex extractors over note passages for:
     - **Monitoring / lab-check cues** (e.g., `INR`, `Monday and Thursday`, `sliding scale`).
     - **Procedure / device / red-flag cues** (e.g., `repeat ERCP`, `stent occlusion`, `fever`, `groin`, `lifting`, `clear liquids`, `shower`).
     - **Follow-up appointment cues** (e.g., `Dr. [A-Z]+`, `clinic`, `Urology`, `primary care`, date/week intervals).
3. **Strict Turn-2 Routing Compatibility (§4.1):**
   - **Never** include literature-trigger words (`evidence`, `study`, `studies`, `trial`, `trials`, `meta analysis`, `systematic review`, `pubmed`, `guideline`, `guidelines`, `efficacy`, `effectiveness`, `benchmark`) in any `patient_chart` follow-up chip.
   - Any chip addressing readmission risk drivers **must** include `"why"`, `"readmission risk"`, and the driver concept so `routing.wants_risk_driver(question, intent)` unlocks `predict_readmission` on Turn 2, and **must never** ask for a bare numeric probability.
   - Every chip must be **single-scope and atomic** (≤ 200 characters, asking about 1–2 verified entities from the chart) so Turn 2 can answer it completely without rubric ambiguity.
4. **Langfuse Observability (§1.4 Fix):**
   - Record the generated `follow_ups` list onto the active trace/observation metadata or output in `_compose_success` / `followups.suggest` so every chip list is visible in Langfuse.

---

### 2.2 Curated Clinical Taxonomy & Priority Catalog

When `followups.suggest()` runs on a non-empty `patient_chart` response (`outpatient=False` unless noted), evaluate candidate templates in deterministic priority order based on the signals present in `state["tool_calls"]` (plus `replayed_tool_calls`), and return up to `3` distinct chips (`FOLLOW_UP_COUNT = 3`):

| Priority | Domain / Rule ID | Required Verified Signal in `state` | Slot-Filled Question Template (≤ 200 chars, Atomic Scope) |
|---|---|---|---|
| **P1** | `MED_TAPER` | Taper cue + drug name detected in med/note text (e.g., `oxycodone SR`) | `"What is the dosing and taper schedule for {taper_drug} at discharge?"` |
| **P2** | `MED_ANTICOAG` | Anticoagulant drug detected (`Coumadin`, `warfarin`, `Lovenox`, etc.) | `"What is the discharge dosing and monitoring plan for {anticoag_drug}?"` |
| **P3** | `CARE_RED_FLAGS` | Return-precaution / restriction / stent / wound / diet cue detected in note/med text | `"What post-discharge precautions, activity limits, or warning signs are documented for {precaution_topic}?"` *(or `"What post-discharge activity restrictions, diet instructions, or warning signs are documented?"`)* |
| **P4** | `CARE_FOLLOWUP` | Follow-up provider/clinic cue detected in note/med text (e.g., `Dr. A`, `Urology`, `office`) | `"What outpatient follow-up appointments or clinic visits are scheduled after discharge?"` |
| **P5** | `MED_SPECIFIC_REGIMEN` | Verified parsed medications (`>= 2` drugs) when `P1`/`P2` did not both fire | `"What are the discharge dosing instructions for {drug_1} and {drug_2}?"` |
| **P6** | `RISK_TOP_DRIVER` | `predict_readmission` in `state` with `decision == "above"` (or top increasing driver present) | `"Why does {top_driver_label} increase this patient's readmission risk driver profile?"` *(must satisfy `routing.asks_why_risk_driver`)* |
| **P7** | `LAB_ABNORMAL_TOP` | `get_lab_trends` in `state` with `>= 1` abnormal discharge analyte | `"How does the abnormal discharge {worst_lab_name} ({worst_lab_val} {worst_lab_unit}) compare to its reference range and trend?"` |
| **P8** | `LAB_ALL_NORMAL` | `get_lab_trends` in `state` with `0` abnormal discharge analytes | `"What were the discharge values for {lab_1} and {lab_2} across the normal laboratory panels?"` |
| **P9** | `NOTE_DIAGNOSIS_COURSE` | `discharge_diagnosis` or `brief_hospital_course` or `history_of_present_illness` present (when Turn 1 was not `/ehr`) | `"What were the primary diagnosis and hospital course documented in the clinical note?"` |
| **P10** | `MED_EMPTY_FALLBACK` | Turn 1 showed empty discharge medications (`0` passages), and note/risk/lab signals exist | Surface `P6` (`RISK_TOP_DRIVER`) or `P9` (`NOTE_DIAGNOSIS_COURSE`) — **never** offer a medication chip when medications are absent. |
| **P11** | `LITERATURE_SCOPE` | `scope == "literature"` (`/pubmed`) and articles returned | Deterministic clinical-topic follow-up derived from the search query/article titles (≤ 200 chars). |

#### Command-Specific De-Duplication Rules
- Avoid proposing a chip that merely repeats the exact command Turn 1 just executed (e.g., if Turn 1 was `/ehr`, prioritize specific drill-downs `P1`–`P6` rather than generic `P9` overview).
- If `outpatient=True` and the turn returned the outpatient abstention/notice without clinical passages, return `[]` (or only lab chips if `get_lab_trends` is in `state`).
- Maintain existing contract guarantees: `async suggest(...) -> list[str]`, max 3 items, each ≤ 200 chars, returns `[]` when `presentation["answer"]` is empty (so `/risk` continues to return `[]`).

---

### 2.3 Deliverables & Verification Gate (Zero Live LLM Calls)

1. **Implement Deterministic `services/agent/followups.py` & Wire `http.py`:**
   - Replace the LLM call in `services/agent/followups.py` with the deterministic phenotype selector and slot-filler.
   - Thread `state` (and `outpatient`) into `followups.suggest()` in `services/agent/http.py:_compose_success` and record the generated `follow_ups` in Langfuse trace/span metadata.
2. **Unit Test Suite (`tests/agent/test_followups.py`):**
   - Test every priority rule (`P1`–`P11`), empty-answer short-circuit (`/risk`), empty-med suppression (`90000106`), outpatient handling (`90000051`), literature-keyword exclusion (`not routing.asks_for_literature(chip)` for all chart chips), and risk-driver routing compatibility (`routing.wants_risk_driver(chip, None)` for `P6`).
3. **Dev Cohort Verification Matrix:**
   - Run a zero-LLM verification script across the 7 Dev Archetypes (`90000015`, `90000108`, `90000106`, `90000105`, `90000023`, `90000051`, `90000073`) showing the exact chips emitted after `/ehr`, `/meds`, and `/labs` (and confirming `dev_t2_03` and `dev_t2_04` Turn-1 chips).
   - **STOP after presenting the Verification Matrix and unit test results. Do not deploy or run Gate 2 until the Architect reviews the generated chips.**

---

## Section 3: Production Deployment, Canary Verification (`3/3`) & Manual Production UX Sign-Off

### 3.1 Verification Matrix Sign-Off
The Section 2 implementation and Dev Cohort Verification Matrix are approved as built:
1. **`_plan_text()` Scoping:** Restricting `P1`–`P5` (`MED_TAPER`, `MED_ANTICOAG`, `CARE_RED_FLAGS`, `CARE_FOLLOWUP`, `MED_SPECIFIC_REGIMEN`) to discharge-plan passages (`discharge_medications`, `discharge_instructions`, `discharge_disposition`) eliminated inpatient-drug (`90000105` subcut heparin) and inpatient-consult (`90000106`) false positives.
2. **Execution Record Telemetry:** Recording `follow_ups` on `chain.record_execution(..., follow_ups=[...])` keyed by `langfuse_trace_id` is approved, avoiding orphan post-span traces in `_compose_success`.

### 3.2 Pre-Deploy Dataset Check (`dev_t2_03` & `dev_t2_04`)
Before pushing and deploying:
1. Inspect `dev_t2_03` (`/ehr` $\rightarrow$ `{{turn_1.follow_ups[0]}}`) and `dev_t2_04` (`/meds` $\rightarrow$ `{{turn_1.follow_ups[0]}}`) in `evaluation/datasets/dev_suite.jsonl`.
2. Ensure their `expected_output` / judge reference context includes the verified ground-truth facts for `90000015`'s `oxycodone SR` taper (`20 mg q12h x 4 days then 10 mg q12h x 4 days`) and discharge medications so the judge has complete ground truth for Turn 2. If `dev_suite.jsonl` is updated, re-sync `ecc-tier2-dev-suite` to Langfuse.

### 3.3 Deploy to Production & Run Gate 1 Canary (`3/3` Required)
1. Commit and push the Section 2 changes to trigger `deploy-agent-on-push` and verify the new Cloud Run revision is live and serving traffic.
2. Run the Gate 1 Canary (`--canary`: `dev_t1_01`, `dev_t2_03`, `dev_t7_01`) against the new production revision.
3. Confirm that **`3/3` canary items pass** (`verdict = 1`, zero structural failures) and that `dev_t2_03` deterministically resolves `follow_ups[0]` to *"What is the dosing and taper schedule for oxycodone SR at discharge?"* and passes all 6 rubric dimensions.

### 3.4 STOP GATE: Manual Production Demo Testing (Do Not Run Gate 2 Yet)
Once the new revision is deployed and the `3/3` Canary passes:
1. Report the live Cloud Run revision name, Canary Run 4 telemetry table (`3/3`), and a quick test checklist of suggested patients/commands (`90000015`, `90000108`, `90000106`) for the Architect to test directly in the production web UI.
2. **STOP and wait for the Architect's manual production UX sign-off before executing the full 26-scenario Gate 2 evaluation suite.**

---

---

## Section 4: Sprint 1 — Multi-Turn Follow-Up De-Duplication & Current-Turn Scoping

### 4.1 Manual Production UX Findings (`agent-00173-9pv`)
Manual multi-turn testing on patient `90000015` (`/ehr` $\rightarrow$ click `"What is the dosing and taper schedule for oxycodone SR at discharge?"` $\rightarrow$ ask `"how long do patients typically need pain meds after surgery?"`) exposed two UX defects in how `followups.suggest()` handles multi-turn state:
1. **Self-Echo / Prior-Turn Repetition:** Clicking `"What is the dosing and taper schedule for oxycodone SR at discharge?"` on Turn 2 caused Turn 2 (and Turn 3) to re-render the exact same `"What is the dosing and taper schedule for oxycodone SR at discharge?"` chip back to the user.
2. **Replayed Tool-Call Bleed on `literature` / Free-Text Turns:** Because `followups.suggest()` inspects `state["replayed_tool_calls"]` identically to `state["tool_calls"]`, Turn 3 (a `scope == "literature"` turn where PubMed returned `0` records) fell back to Turn 1's replayed `/ehr` payload and repeated the exact same 3 `/ehr` chips (`oxycodone SR`, `Coumadin`, `outpatient follow-up`).

### 4.2 Required Fixes (`services/agent/followups.py` & `services/agent/http.py`)

1. **Conversation History De-Duplication (Never Repeat an Asked Question):**
   - Pass prior turn user questions (`turns` from the request / `state`) alongside the current `question` into `followups.suggest()`.
   - Normalize all previously asked user questions in the thread (plus the current turn's `question`) via casefolding, punctuation stripping, and whitespace collapsing.
   - Exclude any candidate follow-up chip that matches a question already asked in the current turn or any prior turn in the conversation thread.

2. **Strict Scope Boundary on `scope == "literature"` Turns:**
   - When `presentation.get("scope") == "literature"`, `followups.suggest()` must evaluate **only** `P11` (`LITERATURE_SCOPE` from current-turn `search_literature` articles).
   - If `search_literature` returned `0` articles on a `literature`-scoped turn, return `[]` immediately. **Never** fall back to `state["replayed_tool_calls"]` to emit `patient_chart` chips on a `literature`-scoped turn.

3. **Current-Turn Priority Before Replayed Fallback (`patient_chart` Scope):**
   - On `patient_chart` turns, evaluate candidate templates against **current-turn `state["tool_calls"]` first** (excluding any chip already asked in conversation history).
   - Only if current-turn `state["tool_calls"]` produces fewer than `FOLLOW_UP_COUNT` (`3`) unasked chips, fill remaining slots from `state["replayed_tool_calls"]` (again strictly excluding any question already asked in the thread).
   - Ensure that as a user clicks through chips across Turns 2, 3, and 4, each clicked chip is consumed and replaced by the next available unasked candidate in the taxonomy (e.g., `P5` `MED_SPECIFIC_REGIMEN`, `P6` `RISK_TOP_DRIVER`, `P9` `NOTE_DIAGNOSIS_COURSE`) until no unasked verified chips remain.

### 4.3 Unit Tests (`tests/agent/test_followups.py`)
Add unit tests covering:
- Clicking `follow_ups[0]` on Turn 2 never re-emits that question in Turn 2's `follow_ups` list.
- Multi-turn progression (`Turn 1` $\rightarrow$ `Turn 2` $\rightarrow$ `Turn 3`) consumes previously asked questions and advances down the taxonomy without repeating prior user turns.
- A `scope == "literature"` turn with `0` PubMed articles and populated `replayed_tool_calls` returns `[]` (zero chart-chip bleed).

### 4.4 Deploy & STOP for Manual Production UI Verification (Do Not Run Canary or Gate 2)
1. Run the local unit test suite and confirm 100% pass.
2. Commit and push the Section 4 changes to deploy the new Cloud Run revision.
3. **STOP IMMEDIATELY AFTER DEPLOYMENT.** Report the live Cloud Run revision name and wait for the Architect to manually verify multi-turn chip progression in the production UI. **Do NOT execute Canary or Gate 2 in Sprint 1.**
---

---

---

## Section 5.5: Fix Turn 3+ Chip Rehydration & `P6` (`RISK_TOP_DRIVER`) Turn-2 Grounding/Chip Leak

### 5.5.1 Manual UI Observations (`agent-00176-ngw`)
1. **Turn 3+ Chip Rehydration:** On `90000015` (`/ehr` $\rightarrow$ click Chip 1 $\rightarrow$ click Chip 2), Chip 1 is properly excluded on Turn 2, but **rehydrates (reappears) in the follow-up chip set after Turn 3**.
2. **`P6` (`RISK_TOP_DRIVER`) Uncited ML Prose & Unrelated Lab Chip Leak:** On `90000078`, `/ehr` rendered 1 chip (`"Why does prior inpatient days increase this patient's readmission risk driver profile?"`). Clicking it produced:
   - Raw float formatting (`"16.0417 days"`) and ungrounded ML mechanics prose (*"contributing to the logit-space risk score. The model uses this historical utilization metric as a statistical indicator..."*) with **zero `^[n]` citations** to the patient's chart passages.
   - An unrelated **Labs follow-up chip** on Turn 2 (because the ReAct loop called `get_lab_trends` during a non-lab risk-driver turn, causing `followups.py` to fire `P7`/`P8`).

### 5.5.2 Required Remediations

1. **Fix Turn 3+ Chip Rehydration Across Full Thread History (`http.py` / `contracts.py` / `questions.py`):**
   - Trace `_asked_questions` across a 4-turn browser session (`Turn 1` $\rightarrow$ `Turn 2` $\rightarrow$ `Turn 3` $\rightarrow$ `Turn 4`).
   - Ensure `questions.own_words()` normalization (stripping `"For admission <hadm_id>."` and any command/prompt wrappers) is applied to **every historical `turn.question` in `request.turns`**, not only the current turn's `own_words`.
   - Verify that no question asked on Turn 1, 2, or 3 can ever reappear on Turn 3, Turn 4, or later turns.

2. **Fix `P6` (`RISK_TOP_DRIVER`) Turn-2 Grounding, Citations & Float Formatting (`graph.py` / `prompts.py` / `routing.py`):**
   - **Tool Scoping on Risk-Driver Turns:** When `routing.wants_risk_driver(question, intent)` is true and the asked driver is **not** a laboratory analyte, do not call `get_lab_trends` (restrict tools to `predict_readmission` + note retrieval `rag_search_sections`/`rag_search`) so the ReAct loop does not pollute `state["tool_calls"]` with unrequested lab panels or emit an unrelated `P7`/`P8` lab chip.
   - **Chart-Cited Explanation (No ML Jargon):** In the synthesis instructions for risk-driver questions, forbid ML/statistical jargon (*"logit-space"*, *"statistical indicator"*), format numeric feature values cleanly (e.g., `16 days` or `16.0 days`, never raw 4-decimal floats like `16.0417`), and require the answer to cite the replayed/retrieved EHR note passages (`^[n]`) that document the relevant clinical background (or explicitly state with an inline `^[n]` citation to the

---

## Section 6: Final Canary 5 Verification (`3/3`) & Gate 2 Full Suite Execution

### 6.1 Section 5 & 5.5 Sign-Off
The `agent-00180-k8n` remediations (`questions.words_of()` multi-turn history normalization, `routing.asks_about_a_lab_analyte()` tool scoping, `admission_context._display()` numeric-string formatting, and chart-cited risk-driver synthesis; `1349 passed, 1 skipped`) are approved.

### 6.2 Step 1: Execute Gate 1 Canary (Run 5 — `3/3` Required)
1. Run the Gate 1 Canary (`--canary`: `dev_t1_01`, `dev_t2_03`, `dev_t7_01`) against live Cloud Run revision `agent-00180-k8n`.
2. Verify that **all 3 scenarios pass (`3 / 3`, `verdict = 1`, zero structural failures)**:
   - `dev_t1_01`: All dimensions `>= 2` (confirming `judge_call.py` published `sources` alignment resolves the prior `citation: 1` evaluator artifact).
   - `dev_t2_03`: Turn 1 deterministically selects the `oxycodone SR` taper chip, and Turn 2 passes with verified inline citations (`citation >= 2`, `responsiveness >= 2`).
   - `dev_t7_01`: Passes outpatient boundary checks (`verdict = 1`).
3. If any Canary item fails, **STOP immediately** and report the trace diagnostic without launching Gate 2.

### 6.3 Step 2: Execute Gate 2 (Full 26-Scenario Dev Suite Run)
Immediately upon confirming `3 / 3` on Canary Run 5:
1. Execute the full 26-scenario evaluation run against `ecc-tier2-dev-suite` on `agent-00180-k8n`.
2. Write the complete scorecard to `evaluation/results/phase04_dev_scorecard.md` and present the summary in chat, including:
   - Serving Cloud Run revision (`agent-00180-k8n`) and Langfuse dataset run ID.
   - Overall pass rate (`X / 26`) and mean dimension scores (`fidelity`, `grounding`, `citation`, `responsiveness`, `safety`, `context`).
   - Per-trajectory pass breakdown across Trajectories 1–7 (`T1`–`T7`).
   - Full diagnostic table for any failing scenario, strictly classified as either an **Evaluator Artifact** or an **Application Defect** (keeping `services/agent/` strictly frozen after the run pending Architect review).