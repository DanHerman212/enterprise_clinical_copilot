# Gate 1 Canary Run 4 Report & Sprint 2 Remediation Directive

**Date:** 2026-10-23  
**Dataset:** `ecc-tier2-dev-suite` (`--canary`: `dev_t1_01`, `dev_t2_03`, `dev_t7_01`)  
**Serving Revision:** `agent-00173-9pv` (built from commit `d2c5c88`)  
**Rubric Version:** `v6`  
**Pre-Deploy Unit Test Suite:** `1309 passed, 1 skipped`  
**Status:** Complete (`2 / 3` Passed) — Sprint 2 Remediation Specified Below  

---

## 1. Executive Summary

Canary Run 4 evaluated live Cloud Run revision `agent-00173-9pv` following the deployment of the deterministic `followups.py` engine (Phase 05) and three iterative citation-pipeline patches (`agent-00171` $\rightarrow$ `agent-00172` $\rightarrow$ `agent-00173`):

- **Items Evaluated:** `3` (`dev_t1_01`, `dev_t2_03`, `dev_t7_01`)
- **Structural Failures:** `0 / 3`
- **Overall Pass Rate:** `2 / 3` (`dev_t1_01` and `dev_t7_01` passed with `verdict = 1`; `dev_t2_03` failed with `verdict = 0` on Turn 2 `citation = 1` and `responsiveness = 1`)
- **Deterministic Follow-Up Resolution Verified:** `dev_t2_03` Turn 1 (`/ehr` on `90000015`) deterministically emitted `"What is the dosing and taper schedule for oxycodone SR at discharge?"` as `follow_ups[0]` across runs.
- **Four Remaining Issues Identified (2 Application Defects + 2 Evaluator/Dataset Artifacts):**
  1. **`TF-10` (Application Defect — Missing Citations on ReAct Turn 2+ Answers):** Free-text ReAct turns answering from chart passages or `replayed_tool_calls` (such as `dev_t2_03` Turn 2) omit `^[n]` citation markers (`citation = 1`).
  2. **`TF-11` (Application Defect — Verbose PubMed Queries & Uncited General Medical Prose on `0` Hits):** Free-text literature turns pass conversational multi-word strings to NCBI E-utilities (resulting in `0` hits) and emit uncited textbook medical commentary instead of clean abstention.
  3. **Evaluator Artifact (`judge_call.build_user_prompt` Source Numbering Mismatch on `dev_t1_01`):** The `/ehr` pipeline renumbers inline markers to first-appearance order and aligns the published `sources` array accordingly, but `judge_call.py` hands the judge raw `tool_calls` order rather than the published `sources` order (`citation = 1` despite `verdict = 1`).
  4. **Dataset Ground-Truth Scope (`dev_t2_03` & `dev_t2_04` Turn 2):** `expected_output` for `dev_t2_03` included all 11 discharge medications alongside the `oxycodone SR` taper, causing the judge to score `responsiveness = 1` when Turn 2 answered only the `oxycodone SR` taper question asked by `follow_ups[0]`.

---

## 2. Per-Scenario Telemetry & Diagnostic Breakdown

| Scenario ID | Patient (`hadm_id`) | Trajectory & Turns | Structural Pass | Judge Verdict | Key Dimension Findings | Root Cause Classification |
|---|---|---|---|---|---|---|
| **`dev_t1_01`** | `90000015` | **T1:** 1 turn (`/ehr`) | **PASS** | **`1` (PASS)** | `citation: 1` (all other dimensions `3`) | **Evaluator Artifact:** Published `sources` match inline `^[3]`/`^[4]` markers, but `judge_call.py` grades against pre-renumbered `tool_calls` array. |
| **`dev_t2_03`** | `90000015` | **T2:** 2 turns (`/ehr` $\rightarrow$ `oxycodone SR` taper chip) | **PASS** | **`0` (FAIL)** | Turn 2 `citation: 1`, `responsiveness: 1` | **Hybrid:** Turn 2 `citation: 1` is a real **App Defect (`TF-10`)** (no `^[n]` marker on ReAct answer). Turn 2 `responsiveness: 1` is a **Dataset Scope Artifact** (`expected_output` listed 10 other unasked medications). |
| **`dev_t7_01`** | `90000051` (Outpatient) | **T7:** 2 turns (`/risk` $\rightarrow$ `/ehr`) | **PASS** | **`1` (PASS)** | All dimensions `>= 2` (`fidelity: 3`, `grounding: 3`, `safety: 3`) | **PASS** (`TF-1` & `TF-6` holding). |

---

## 3. Sprint 2 Remediation Specification (Execute After Sprint 1 Sign-Off)

### 3.1 Fix `TF-10`: Enforce `^[n]` Citation Markers on Free-Text ReAct Turns (`services/agent/`)
1. **Root Cause:** When the ReAct reasoning loop (`graph.py`) answers a free-text question on Turn 1 or Turn 2+ (using either freshly called tools or `replayed_tool_calls`), the prompt/rendering path does not enforce inline `^[n]` citation markers bound to the numbered evidence passages in view.
2. **Fix:**
   - Ensure every passage presented to the ReAct synthesis model (including passages carried in `replayed_tool_calls` and fresh `tool_calls`) carries an explicit `` / `cite` index label matching the citation verifier's source table.
   - Strengthen the ReAct synthesis system instruction so every clinical fact, medication dose, taper schedule, laboratory value, or chart statement **must** carry an inline `^[n]` citation marker, matching the deterministic formatter's citation contract.
   - Verify with unit tests that free-text Turn 2 answers grounded in `replayed_tool_calls` emit valid `^[n]` markers that resolve to `verified = True`.

### 3.2 Fix `TF-11`: Concise PubMed Query Formulation & Strict `0`-Hit Abstention (`services/agent/`)
1. **Root Cause:** When a user asks a free-text general medical evidence question (e.g., *"how long do patients typically need pain meds after surgery?"*), the ReAct agent passes a conversational 10-word phrase (`"typical duration of opioid pain medication after total knee replacement surgery"`) to `search_literature`, yielding `0` PubMed hits, and then writes a multi-sentence paragraph of uncited general medical textbook commentary.
2. **Fix:**
   - **Query Formulation:** Instruct/constrain the `search_literature` tool call description and query normalizer to compress conversational questions into concise 2–5 keyword clinical/MeSH search terms (e.g., `"total knee arthroplasty postoperative opioid duration"`) so NCBI E-utilities retrieves indexed articles reliably.
   - **Zero-Hit Abstention Contract:** When `search_literature` returns `0` articles, the response must state cleanly that no indexed PubMed literature matched the query and—if chart context is replayed—state only cited facts from the patient's chart (with `^[n]` markers). **Never** emit ungrounded, uncited "In general medical practice..." textbook paragraphs when `search_literature` returns `0` results.

### 3.3 Fix Evaluator Source-Numbering Alignment (`evaluation/judge_call.py`)
1. **Root Cause:** In `http.py:_compose_success`, inline citation badges are renumbered to first-appearance order (`1..K`) and `presentation["sources"]` is ordered to match that exact published numbering. However, `judge_call.build_user_prompt` constructs the judge's evidence block from raw `tool_calls` before first-appearance renumbering, causing false-positive `citation = 1` scores whenever first-appearance order differs from raw array order.
2. **Fix:**
   - Update `evaluation/judge_call.py` (`build_user_prompt`) and the runner evidence serializer so the judge is presented with the turn's **published `sources` numbering** alongside the tool payloads (so `^[k]` in the assistant's answer is graded against source `[k]` as actually served to the user).
   - Add a unit test in `tests/evaluation/` verifying that when an answer cites sections in `[1, 2, 4, 3]` retrieval order and renumbers them to `^[1]..^[4]` in `sources`, `build_user_prompt` labels the evidence to match the published `^[1]..^[4]` mapping.

### 3.4 Scope `dev_t2_03` & `dev_t2_04` Turn-2 `expected_output` (`evaluation/datasets/dev_suite.jsonl`)
1. **Root Cause:** Now that `followups.py` deterministically emits `"What is the dosing and taper schedule for oxycodone SR at discharge?"` as `follow_ups[0]` for both `dev_t2_03` (`/ehr`) and `dev_t2_04` (`/meds`), Turn 2 asks exclusively about `oxycodone SR`. Listing the other 10 discharge medications in Turn 2's `expected_output` causes the judge to penalize Turn 2 for omitting unasked medications.
2. **Fix:**
   - Update Turn 2's `expected_output` on `dev_t2_03` and `dev_t2_04` in `evaluation/datasets/dev_suite.jsonl` (and re-sync `ecc-tier2-dev-suite` to Langfuse) so Turn 1's reference facts cover the full `/ehr` or `/meds` output, while Turn 2's reference facts expect the complete `oxycodone SR` discharge dosing and taper schedule (`20 mg PO q12h x 4 days, then 10 mg PO q12h x 4 days`—or `5 days` if quoting the hospital course orthopedic pain service note, noting both chart passages so either cited passage is recognized) without demanding unrelated medications on Turn 2.

---

## 4. Sprint 2 Execution & Verification Sequence

1. Implement §§3.1–3.4 and run the full local unit test suite (`pytest`).
2. Commit and push to deploy the new Cloud Run revision.
3. Execute the **Gate 1 Canary (`--canary`: `dev_t1_01`, `dev_t2_03`, `dev_t7_01`)** and verify **`3 / 3` PASS** (`verdict = 1` and all dimensions passing, including `dev_t1_01` `citation >= 2` and `dev_t2_03` Turn 2 `citation >= 2`, `responsiveness >= 2`).
4. Immediately upon confirming `3 / 3` on the Gate 1 Canary, execute **Gate 2 (the full 26-scenario `ecc-tier2-dev-suite` benchmark)** and present the full Dev Suite Scorecard.