# Phase 03: Tier 1 Deterministic Cohort Sweep & Cohort Inventory

## Section 1: Scope & Frozen-Code Guardrails

### 1.1 Objective
Execute **Tier 1** of the Two-Tier Evaluation Surface: a zero-LLM, deterministic Python verification sweep across all **87 records** (82 served inpatient records, 5 outpatient records, and 2 unserved boundary records) with a target runtime of **<60 seconds** and **$0 LLM judge cost**. In addition, generate a factual **Cohort Provenance Inventory** from the underlying data store to verify the exact chart profiles of our candidate Phase 04 (Dev) and Phase 05 (Holdout) patient archetypes.

### 1.2 Strict Read-Only Codebase Freeze
To prevent evaluation work from introducing regressions into the verified production application or engaging in reward hacking, the following strict boundaries apply throughout Phase 03:

1. **Application & UI Freeze (Read-Only):**
   - `ECC` runtime packages (`services/agent/src/`, `services/mcp/src/`) are **strictly frozen**.
   - The `danielmherman` repository (`demo/`, `static/`, `templates/`) is **strictly frozen**.
   - Do **not** modify routing logic, prompts, tool schemas, `a2ui` presentation builders, or frontend JavaScript during Phase 03.
2. **Permitted Write Scope:**
   - New or updated deterministic test files under `tests/` (e.g., `tests/evaluation/test_tier1_cohort_sweep.py`).
   - Read-only cohort inspection/inventory scripts or artifacts under `evaluation/`.
3. **Fail-Stop Escalation Rule:**
   - If the Tier 1 deterministic sweep uncovers a genuine parser or schema failure on any of the 87 records, **do not patch `services/agent` or `services/mcp` autonomously**. Report the exact `hadm_id`, stack trace, and failing field, and halt for Architect sign-off.
4. **Verification Gate:**
   - Before marking Section 1 / Phase 03 tasks complete, verify via `git status` and `git diff` that zero files under `services/agent/src/`, `services/mcp/src/`, or `danielmherman` have been modified.

## Section 2: Cohort Discovery, Manifest Hydration & Boundary Sweep (All 87 Records)

### 2.1 Objective
Programmatically enumerate the exact served cohort from the codebase/data store (identifying which `hadm_id`s are the 82 inpatient admissions vs. the 5 outpatient encounters, plus 2 unserved boundary IDs), and verify via `pytest` that patient context hydration (`<admission_metadata>`) and encounter scope guardrails operate deterministically across all **87 records** without invoking an LLM.

### 2.2 Step 1 — Programmatic Cohort Discovery (No Hardcoded Assumptions)
Do not guess which `hadm_id` is inpatient vs. outpatient. Inspect the repository's cohort source of truth (e.g., `services/mcp`, patient manifest fixtures/tables, or scope-resolution helpers in `services/agent`) to dynamically load:
1. **`INPATIENT_HADM_IDS`:** The complete list of served inpatient `hadm_id`s (expected count: **82**).
2. **`OUTPATIENT_HADM_IDS`:** The complete list of served outpatient / non-admission `hadm_id`s (expected count: **5**, including known examples such as `90000051` and `90000073`).
3. **`UNSERVED_HADM_IDS`:** Two deterministic boundary test IDs outside the served cohort (e.g., `"99999999"` and `"00000000"` / `"invalid_hadm_id"`).

*Note: If the actual counts in the data store differ from 82 inpatient / 5 outpatient, report the exact counts and `hadm_id` lists found.*

### 2.3 Step 2 — Test Implementation (`tests/evaluation/test_tier1_cohort_sweep.py`)
Create `tests/evaluation/test_tier1_cohort_sweep.py` with parametrized `pytest` checks covering all three groups:

1. **Cohort Partition Assertion:**
   - Assert that `INPATIENT_HADM_IDS` and `OUTPATIENT_HADM_IDS` are disjoint sets and match the served cohort inventory.
2. **Inpatient Manifest Hydration (All 82 Inpatient `hadm_id`s):**
   - Parametrize across every ID in `INPATIENT_HADM_IDS`.
   - Assert that each resolves as a valid inpatient admission and that its manifest / `<admission_metadata>` hydrates cleanly without raising exceptions or returning empty/null required structural fields.
3. **Outpatient Encounter Boundary (All 5 Outpatient `hadm_id`s):**
   - Parametrize across every ID in `OUTPATIENT_HADM_IDS`.
   - Assert that each is deterministically classified as an outpatient / non-inpatient encounter and triggers the deterministic outpatient refusal (`verified=False`, no leaked inpatient chart data) when queried on inpatient pathways.
4. **Unserved / Unknown Boundary (2 Unserved `hadm_id`s):**
   - Parametrize across `UNSERVED_HADM_IDS`.
   - Assert that both trigger a clean, deterministic unserved-patient refusal without unhandled exceptions.

### 2.4 Execution & Deliverable for Section 2 Sign-Off
1. Run `pytest tests/evaluation/test_tier1_cohort_sweep.py` and confirm 0 LLM calls are made.
2. Output a brief **Section 2 Execution Report** containing:
   - Exact count and list of `OUTPATIENT_HADM_IDS` discovered.
   - Exact count of `INPATIENT_HADM_IDS` discovered and verified.
   - `pytest` pass/fail summary and execution time.
   - Output of `git status` confirming zero modifications to `services/agent/src/`, `services/mcp/src/`, or `danielmherman`.
3. **STOP and wait for Architect sign-off before proceeding to Section 3.**

## Section 3: Deterministic Tool & Parser Sweep (All 82 Inpatients + 5 Outpatients)

### 3.1 Objective
Extend `tests/evaluation/test_tier1_cohort_sweep.py` to verify that every underlying patient-scoped MCP tool and deterministic payload parser (`predict_readmission`, `get_lab_trends` / discharge labs, `discharge_medications`, and clinical note retrieval) executes cleanly across all **82 inpatient admissions** and **5 outpatient encounters** without unhandled exceptions, malformed schemas, or LLM calls.

*(Note on Section 2 Finding #1: Keep `services/agent/` and `services/mcp/` strictly frozen. Log the `/risk` primary-tool outpatient behavior as Tracked Finding #1 for Architect review after Phase 03 completes.)*

### 3.2 Test Implementation (`tests/evaluation/test_tier1_cohort_sweep.py`)
Add parametrized, zero-LLM tests across `INPATIENT_HADM_IDS` (82) and `OUTPATIENT_HADM_IDS` (5) covering:

1. **Readmission Risk Tool & Payload Builder (`predict_readmission` / `/risk` payload):**
   - Across all **82 inpatient IDs**, assert that `predict_readmission` returns a valid response schema (numeric risk probability/score in `[0, 1]`, non-empty risk tier/band, and a structured list of feature drivers).
   - Assert that the deterministic `/risk` command payload and `a2ui` canvas builder construct a valid `RiskCard` without raising validation errors.
2. **Discharge Medications Tool & Parser (`discharge_medications` / `/meds` payload):**
   - Across all **82 inpatient IDs**, invoke the medication retrieval/extraction pipeline (`discharge_medications` and deterministic `/meds` payload builder).
   - Assert that every record returns a valid schema (either a populated medication list with valid row schemas OR a clean, structured empty-medication state for records with no discharge medications, such as `90000106`) without raising parser or regex exceptions.
   - Verify that when medications are present, the cited source passage (`^[1]`) aligns with the consolidated discharge medications section (Phase 02.5 regression invariant).
3. **Laboratory Tool & Parser (`get_lab_trends` / `/labs` payload):**
   - Across all **82 inpatient IDs**, invoke the laboratory retrieval and deterministic `/labs` payload builder.
   - Assert that every record returns a valid lab schema (handling decimals, reference ranges, abnormal flags, and all-normal discharge lab records such as `90000105`) without raising exceptions.
4. **Clinical Note Retrieval (`/ehr` / note passages):**
   - Across all **82 inpatient IDs** and **5 outpatient IDs**, verify that clinical note retrieval returns non-empty note chunks with valid citation metadata (`note_id` / section headers) without parser crashes on irregular note formatting (such as `90000023`).

### 3.3 Execution & Deliverable for Section 3 Sign-Off
1. Run `pytest tests/evaluation/test_tier1_cohort_sweep.py` and confirm **0 LLM calls** are made and total runtime remains **<60 seconds**.
2. Output a brief **Section 3 Execution Report** containing:
   - Total test count and runtime for `test_tier1_cohort_sweep.py`.
   - Summary of tool/parser pass rates across the 82 inpatients and 5 outpatients (explicitly noting any records with empty meds, zero abnormal labs, or schema quirks discovered during the sweep).
   - Confirmation via `git status` that `services/agent/`, `services/mcp/`, and `danielmherman` remain untouched.
3. **STOP and wait for Architect sign-off before proceeding to Section 4 (Cohort Provenance Inventory).**

---
## Section 4: Ground-Truth Cohort Provenance Inventory (Dev & Holdout Selection)

### 4.1 Objective
Generate a factual, zero-LLM **Cohort Provenance Inventory** from the underlying data store to:
1. Verify the exact chart contents of the **6 Dev Patient Archetypes** listed in Section 3 of the Master Strategy (`90000015`, `90000108`, `90000106`, `90000105`, `90000023`, and outpatients `90000051` / `90000073`).
2. Identify **4 Unseen Holdout Patient Candidates** from the remaining inpatient pool (plus 1 holdout outpatient from `90000028`, `90000040`, `90000054`) that mirror the structural phenotypes of the Dev cohort without overlapping IDs.

### 4.2 Inventory Generation (`evaluation/cohort_provenance_inventory.json` & Summary Report)
Write a read-only inspection script or test helper (reusing the concurrency-6 transport settings from Section 3) that extracts and saves the ground-truth profile for the 6 Dev patients and scans the cohort to recommend 4 Holdout inpatient matches:

For each candidate record, capture:
1. **Demographics & Manifest (`get_admission_profile`):** `age`, `gender`, `discharge_location`, length of stay / prior inpatient days, and oncology flag/history.
2. **Readmission Risk (`predict_readmission`):** Risk score/probability, decision tier, and top 5 feature drivers (`name`, `contribution`, `direction`, and underlying feature value).
3. **Discharge Medications (`discharge_medications`):** Whether the consolidated `discharge_medications` passage is empty or populated, character length, and raw snippet of the medication text (noting tapers, anticoagulants, or empty sections).
4. **Laboratory Trends (`get_lab_trends`):** Total lab panels returned, count of abnormal discharge flags vs. normal flags, any analytes with `decimals > 0`, and top abnormal labs at discharge.
5. **Clinical Note Sections (`rag_search_sections`):** List of available section keys returned (and whether it is one of the 3 heading-less notes from TF-2: `90000026`, `90000033`, `90000087`).

### 4.3 Specific Provenance Checks to Report
1. **Verify the 6 Dev Patients:** Confirm whether the actual database contents of `90000015`, `90000108`, `90000106`, `90000105`, and `90000023` match their Master Strategy archetype descriptions, and report any discrepancies.
2. **Recommend 4 Holdout Inpatients + 1 Holdout Outpatient:** From the remaining served records, identify 4 inpatient `hadm_id`s (and 1 outpatient `hadm_id`) that provide clean structural counterparts for Phase 05 Blind Holdout testing:
   - **Holdout Archetype 1:** Complex inpatient with rich discharge medications / tapers.
   - **Holdout Archetype 2:** High-risk inpatient with oncology history & abnormal labs.
   - **Holdout Archetype 3:** Inpatient with empty/minimal discharge medications OR all-normal discharge labs.
   - **Holdout Archetype 4:** Inpatient with irregular/sparse note sections (e.g., one of `90000026`, `90000033`, `90000087` or similar structural edge case).
   - **Holdout Archetype 5 (Outpatient Boundary):** One of `90000028`, `90000040`, or `90000054` (outpatient with note sections present per TF-3).

### 4.4 Execution & Deliverable for Phase 03 Completion
1. Generate the inventory artifact under `evaluation/` and ensure zero LLM calls are made.
2. Present a clear **Section 4 Provenance & Archetype Report** in chat summarizing:
   - The verified ground-truth profile of each of the 6 Dev patients.
   - The recommended 4 Holdout inpatient IDs (+ 1 Holdout outpatient ID) and their factual profiles.
   - Final `git status` check confirming `services/agent/`, `services/mcp/`, and `danielmherman` remain untouched.
3. **STOP and wait for Architect sign-off to close Phase 03.**

---
## Section 5: Phase 03 Close-Out — TF-1 Surgical Remediation & Locked Cohort Matrix

### 5.1 Architect Authorization (Scoped Unfreeze for TF-1 Only)
Following review of the Section 3 & Section 4 findings, the Architect authorizes a **strictly scoped code change** to resolve **TF-1 (`/risk` not gated on outpatient status)** before closing Phase 03 and re-locking the codebase for Phase 04.

- **Permitted Write Scope for TF-1:**
  - `services/agent/routing.py` and/or `services/agent/deterministic.py` (solely to gate `/risk` / `predict_readmission` on `OUTPATIENT_HADM_IDS`).
  - `tests/evaluation/test_tier1_cohort_sweep.py` (and corresponding unit tests in `tests/`) to assert the updated outpatient boundary rule.
- **All Other Paths Remain Frozen:**
  - Do **not** modify `services/mcp/`, `danielmherman`, or unrelated agent modules (`TF-2` through `TF-5` remain deferred/Tier 2 concerns).

### 5.2 TF-1 Remediation Specification
1. **Outpatient Gate on `/risk` & `predict_readmission`:**
   - Ensure that when `hadm_id in OUTPATIENT_HADM_IDS` (`90000028, 90000040, 90000051, 90000054, 90000073`), executing `/risk` does **not** invoke `predict_readmission` as a primary tool or render an inpatient readmission `RiskInfluencers` card for a non-admission visit.
   - Instead, `/risk` on an outpatient encounter must deterministically trigger the outpatient scope refusal / notice (`outpatient=True`, `verified=False`, no `RiskInfluencers` widget, no hallucinated inpatient readmission score), consistent with `/ehr` outpatient handling.
2. **Update Tier 1 Sweep Assertions (`tests/evaluation/test_tier1_cohort_sweep.py`):**
   - Update the Section 2 / Section 3 outpatient assertions to verify that across all 5 `OUTPATIENT_HADM_IDS`, both `/ehr` and `/risk` enforce the outpatient boundary and suppress `predict_readmission`.


### 5.3 Execution & Deliverable for Phase 03 Final Sign-Off
1. Implement the `TF-1` fix and run:
   - `pytest tests/evaluation/test_tier1_cohort_sweep.py`
   - Full repository `pytest tests` suite to verify zero regressions across the existing 1,100+ unit tests.
2. Output a concise **Section 5 Close-Out Report** confirming:
   - `TF-1` is resolved and verified across all 5 outpatient records.
   - Full test suite pass counts.
   - `git diff --stat` showing only the authorized `TF-1` files and test updates were touched, and the codebase is now re-frozen for Phase 04.

---

## Tracked Findings (remediation deferred — logged for Architect review after Phase 03)

Nothing below was fixed during Phase 03: `services/agent/` and `services/mcp/` are frozen,
so each finding is recorded with its evidence and left for a later phase.

### TF-1 — `/risk` was not gated on outpatient status — **RESOLVED (Section 5, 2026-10-23)**
`routing.OUTPATIENT_HADM_IDS` dropped `predict_readmission` only where it was a
*Supplementary* call (`routing.supplementary_arguments`). On `/risk` the estimate is the
**primary** tool, and `deterministic`'s `outpatient` flag was derived from supplementary
calls, so a `/risk` turn on an outpatient record was treated as an ordinary risk turn and
scored a record with no admission. Measured: `90000051` returned 0.2916, above threshold,
for a speech-therapy note.

Fixed in `routing.refuses_readmission_score` + `routing.OUTPATIENT_RISK_REFUSAL`, checked
at the top of `deterministic.run_command` before any tool is resolved. `/risk` on an
outpatient now makes no tool call, returns a deterministic refusal, draws no
`RiskInfluencers` card, and reports `verified=False`. Verified across all five outpatient
records by `tests/evaluation/test_tier1_cohort_sweep.py` and
`tests/agent/test_outpatient_risk_gate.py` (20 tests).

### TF-2 — three inpatient notes carry no discharge-summary headings
`90000026` (1,153 chars), `90000033` (2,346), `90000087` (4,153). The note exists and
retrieval reads it; it contains none of the four `SUMMARY_SECTIONS`, so
`rag_search_sections` answers with a structured empty and a sentence saying so.
Consequence: `/ehr` on these three can only answer through the record-wide `rag_search`
fallback, which embeds the query — a model call, and therefore outside Tier 1.

### TF-3 — outpatient note section coverage is uneven
`90000028` and `90000054` return `history_of_present_illness`; `90000040` returns
`brief_hospital_course`; `90000051` and `90000073` return nothing. An outpatient note that
returns `brief_hospital_course` can have a hospital course written about a visit — the
`OUTPATIENT_ENCOUNTER_NOTICE` is what prevents it, and it is a prompt instruction rather
than a structural guard.

### TF-4 — Tier 1 runtime exceeds the 60-second budget (measured 92–147s)
Not a data problem. Every tool call builds its own gRPC channel and re-fetches an OAuth
token; past ~8 concurrent refreshes `oauth2.googleapis.com` intermittently answers
`SSL: UNEXPECTED_EOF_WHILE_READING`, which surfaces as `ServiceUnavailable` and looks like
a code defect. The sweep runs at concurrency 6 with a three-attempt retry on transport
failures only (never on a tool's own error payload), which is stable at the cost of wall
time. Fixing this properly means reusing clients inside `services/mcp` — frozen, so it
needs sign-off.

### TF-5 — the medication extraction model is not exercised by Tier 1
`/meds` parses the *model's* extraction JSON with `parse_medications`. Tier 1 is zero-LLM,
so the sweep exercises retrieval, consolidation, the narrative's both branches and the
`^[1]` resolution against the consolidated passage — but not the extraction itself. That
half remains a Tier 2 concern.