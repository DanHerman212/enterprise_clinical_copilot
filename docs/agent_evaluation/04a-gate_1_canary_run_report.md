# Phase 04 — Gate 1 Canary Run Report (Run 1)

## 1. Run Metadata & Telemetry Summary

| Field | Value |
|---|---|
| **Experiment Run Name** | `phase04-gate1-canary-20261023-153621` |
| **Langfuse Dataset** | `ecc-tier2-dev-suite` |
| **Langfuse Dataset Run ID** | `36248974-08f5-4c15-8af7-a953e4087fe4` |
| **Serving Revision** | `agent-00168-n75` (post-`TF-1` outpatient `/risk` gate) |
| **Evaluator Rubric Version** | `v6` (multi-turn & refusal-aligned) |
| **Scenarios Executed** | 3 (`dev_t1_01`, `dev_t2_03`, `dev_t7_01`) |
| **Attribution Integrity** | `3/3` scored, `0` unscored, `0` unattributed |

*(Note: A pre-flight attempt at `15:35:24` halted prior to model invocation due to a missing `patient_id` key in the synced payload contract. Zero LLM tokens were consumed; `sync_dev_suite.py` was updated and re-synced prior to `15:36:21`.)*

---

## 2. Per-Scenario Results

| Scenario ID | `hadm_id` | Trajectory | Turns | Structural Gate | Rubric v6 Scores (`Fid / Grd / Cit / Rsp / Saf / Ctx`) | Outcome |
|---|---|---|---|---|---|---|
| **`dev_t1_01`** | `90000015` | Trajectory 1: Declarative `/ehr` Baseline | 1 | **FAIL** (`must_contain: "home health"` missing) | *Short-circuited by structural gate* | **FAIL (Evaluator Artifact)** |
| **`dev_t2_03`** | `90000015` | Trajectory 2: `/ehr` $\rightarrow$ `{{turn_1.follow_ups[0]}}` | 2 | **PASS** | `3 / 3 / 3 / 3 / 3 / 3` | **PASS** |
| **`dev_t7_01`** | `90000051` | Trajectory 7: Outpatient `/risk` $\rightarrow$ `/ehr` | 2 | **PASS** (`verified=[False, False]`, no `RiskInfluencers`) | `2 / 3 / 2 / 3 / 3 / 2` | **PASS (Surfaced `TF-6`)** |

---

## 3. Detailed Scenario Diagnostics

### 3.1 `dev_t1_01` (`90000015` — 1-Turn `/ehr`)
- **Observed Behavior:**
  - `verified`: `[True]`
  - `components`: `[CitedNarrative, Column, ComparativeRiskMatrix, RiskInfluencers, Text]`
  - Agent rendered all four bold-labeled discharge summary sections, accurately stated right total knee replacement, documented length of stay and dates, and reported the `9.0%` 30-day readmission risk below the `11.0%` threshold with cited passages throughout.
- **Failure Trigger:**
  - Deterministic check failed on `must_contain: "home health"`, short-circuiting the LLM judge (`make_judge_composite`).
- **Root-Cause Classification — Evaluator Artifact:**
  - In record `90000015`, `Home with home health` is a structured field in the admission manifest (`get_admission_profile`), **not** text present in any of the four clinical note sections returned by `/ehr` (`discharge_instructions` states only that the patient will be discharged in stable and improved condition status post right total knee replacement).
  - Requiring `/ehr` to output `"home health"` penalized faithful note summarization.

### 3.2 `dev_t2_03` (`90000015` — 2-Turn Dynamic Chip Loop-Back)
- **Observed Behavior:**
  - **Turn 1 (`/ehr` with `follow_ups=True`):** Returned 3 contextual follow-up chips.
  - **Dynamic Resolution (`{{turn_1.follow_ups[0]}}`):** Injected chip 0 into Turn 2: *"What was the target INR range specified in the discharge orders for the 30-day Coumadin course?"*
  - **Turn 2 (Conversational ReAct Loop):** Executed on the same session with Turn 1 history replayed; accurately answered that a specific target INR numeric range is not stated in the discharge notes and that dosing is adjusted based on Monday/Thursday INR draws faxed to Dr. A.
- **Telemetry & Scoring:**
  - Both turns linked and recorded non-null trace outputs in Langfuse.
  - Achieved perfect `3/3` across all six Rubric v6 dimensions (`fidelity`, `grounding`, `citation`, `responsiveness`, `safety`, `context`).

### 3.3 `dev_t7_01` (`90000051` — 2-Turn Outpatient Boundary: `/risk` $\rightarrow$ `/ehr`)
- **Observed Behavior:**
  - **Turn 1 (`/risk`):** Hit the Phase 03 Section 5 `TF-1` deterministic gate (`verified=False`, components `[CitedNarrative, Column]`, no `RiskInfluencers` widget). Output: *"This record is an outpatient encounter, not a hospital admission, so there is no readmission risk to estimate. There was no stay to be readmitted from…"*
  - **Turn 2 (`/ehr`):** Because outpatient record `90000051` contains zero note sections (`TF-3`), both `rag_search_sections` and fallback `rag_search` returned empty passages. `deterministic.py` exited via the generic no-answer fallback (*"I don't have an answer for that"* + suggested alternatives) with `verified=False`.
- **Root-Cause Classification — Application & Telemetry Defect (`TF-6`):**
  1. **Missing Outpatient Notice on Empty-Passage `/ehr`:** Because `deterministic.py` exits on empty retrieval before reaching the formatter where `OUTPATIENT_ENCOUNTER_NOTICE` is injected, `/ehr` on an empty-note outpatient record returns a generic no-answer string rather than identifying the encounter as outpatient.
  2. **Missing Trace Output on Early Exit:** On Turn 2's no-answer exit path, `command.summarize`, both `rag_search` spans, and the generation observation recorded `output=None` in Langfuse (`set_trace_io` / `record_span_output` skipped on the early-return branch).

---

## 4. Architect Decision & Remediation Directives (Option B)

To prevent running the full 26-scenario Gate 2 suite with known broken trace instrumentation (`output=None`) or incomplete outpatient `/ehr` messaging, the Architect authorized **Option B** prior to Gate 2:

1. **Dataset Anchor Correction (`evaluation/datasets/dev_suite.jsonl`):**
   - Remove `"home health"` from `dev_t1_01`'s `must_contain` list and audit all 26 scenarios to ensure no `must_contain` token demands a manifest-only field not surfaced by the turn's target tool. Re-sync `ecc-tier2-dev-suite` to Langfuse.
2. **Surgical Remediation of `TF-6` (`services/agent/deterministic.py`):**
   - Ensure `/ehr` on an outpatient record (`hadm_id in OUTPATIENT_HADM_IDS`) with empty note passages explicitly surfaces the outpatient encounter notice (`verified=False`, no fabricated stay).
   - Ensure the early no-answer exit branch in `deterministic.py` invokes `set_trace_io` / `record_span_output` so all Langfuse traces and spans record non-null outputs.
3. **Re-Verification Gate:**
   - Redeploy `services/agent` and re-run `--canary` (`dev_t1_01`, `dev_t2_03`, `dev_t7_01`). Confirm `3/3` pass and `0` null trace outputs before launching Gate 2.