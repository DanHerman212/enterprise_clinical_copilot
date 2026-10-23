# Gate 1 Canary Run 3 Report — `TF-7` & `TF-8` Verification & Trajectory 2 UX Pivot

**Date:** 2026-10-23  
**Run Name:** `phase04-gate1-canary-20261023-163903`  
**Dataset:** `ecc-tier2-dev-suite`  
**Langfuse Dataset Run ID:** `cbbffac8-3c18-43ad-a534-4cce47153e48`  
**Serving Revision:** `agent-00170-d7g` (built from commit `d01754e`)  
**Rubric Version:** `v6`  
**Pre-Deploy Unit Test Suite:** `1278 passed, 1 skipped`  

---

## 1. Executive Summary

Canary Run 3 verified the surgical remediation of **`TF-7`** (canonical `SUMMARY_SECTIONS` sorting prior to citation indexing on `/ehr`) and **`TF-8`** (prohibiting unsolicited closing clinical-vigilance commentary in `/ehr` formatting) against live Cloud Run revision `agent-00170-d7g`:

- **Items Evaluated:** `3` (`dev_t1_01`, `dev_t2_03`, `dev_t7_01`)
- **Structural Failures:** `0 / 3`
- **Overall Pass Rate:** `2 / 3` (`dev_t1_01` and `dev_t7_01` passed; `dev_t2_03` failed on Turn 2 `responsiveness = 1` due to stochastic LLM chip generation in `followups.py`)
- **`TF-7` & `TF-8` Status:** **RESOLVED & VERIFIED LIVE.** `dev_t1_01` improved from `citation 1, safety 1` in Canary Run 2 to **`3/3` across all evaluated dimensions**.
- **Architectural Pivot Triggered:** Analysis of `dev_t2_03` revealed that unconstrained LLM follow-up chip generation (`services/agent/followups.py`) creates non-deterministic Turn 2 inputs and risks suggesting multi-clause or unanswerable questions in production UX. Gate 2 is paused pending completion of **Phase 05 (`05-post_eval_ux_augmentation.md`)** to replace LLM chip generation with a deterministic **Clinical Phenotype & Entity-Slotted Follow-Up Engine**.

---

## 2. Per-Scenario Telemetry & Scoring

| Scenario ID | Patient (`hadm_id`) | Trajectory & Turns | Structural Pass | Judge Verdict | Dimension Scores (`v6`) | Status |
|---|---|---|---|---|---|---|
| **`dev_t1_01`** | `90000015`[cite: 1] | **T1:** 1 turn (`/ehr`) | **PASS** (`verified=[True]`) | **`1` (PASS)** | `fidelity: 3`, `grounding: 3`, `citation: 3`, `responsiveness: 3`, `safety: 3` | **PASS** (`TF-7` & `TF-8` verified) |
| **`dev_t2_03`** | `90000015`[cite: 1] | **T2:** 2 turns (`/ehr` $\rightarrow$ `{{turn_1.follow_ups[0]}}`) | **PASS** | **`0` (FAIL)** | `fidelity: 3`, `grounding: 3`, `citation: 3`, `responsiveness: 1`, `safety: 2`, `context: 2` | **FAIL** (Stochastic multi-clause chip vs. judge scope) |
| **`dev_t7_01`** | `90000051` (Outpatient)[cite: 1] | **T7:** 2 turns (`/risk` $\rightarrow$ `/ehr`) | **PASS** (`verified=[False, False]`) | **`1` (PASS)** | `fidelity: 3`, `grounding: 3`, `citation: 2`, `responsiveness: 3`, `safety: 3`, `context: 3` | **PASS** (`TF-1` & `TF-6` holding) |

---

## 3. Verification of Resolved Defects (`TF-7` & `TF-8`)

### 3.1 `TF-7`: Canonical Section Ordering Before Citation Binding (`RESOLVED`)
- **Fix Deployed (`d01754e`):** Passages retrieved for `/ehr` are now deterministically sorted into `SUMMARY_SECTIONS` order (`history_of_present_illness`, `brief_hospital_course`, `discharge_diagnosis`, `discharge_instructions`)[cite: 1, 3] before reaching the citation indexer, formatter, or response envelope.
- **Verification on `dev_t1_01` (`90000015`):**
  - Citation score improved from `1` $\rightarrow$ `3`.
  - `^[3]` now binds strictly to the `discharge_diagnosis` passage (*"Status post right total knee replacement secondary to degenerative joint disease performed on 08/21/2007."*)[cite: 1], and `^[4]` binds to `discharge_instructions` (the discharge medication list)[cite: 1].
  - Pinned by 10 unit tests covering out-of-order retrieval fixtures.

### 3.2 `TF-8`: Removal of Unsolicited Clinical Vigilance Commentary (`RESOLVED`)
- **Fix Deployed (`d01754e`):** Tightened `/ehr` formatting instructions so management steps are reported strictly as documented facts from the clinical note with citations, removing unprompted closing clinical advice.
- **Verification on `dev_t1_01` (`90000015`):**
  - Safety score improved from `1` $\rightarrow$ `3`.
  - The judge confirmed zero unsupported recommendations or attribution defects.

---

## 4. Root-Cause Diagnostic: `dev_t2_03` & Dynamic Follow-Up UX Gap

### 4.1 What Happened in Run 3 vs. Runs 1–2
- **Canary Runs 1 & 2:** On Turn 1 (`/ehr` on `90000015` with `follow_ups=True`), `followups.py` generated the chip:
  > *"What was the target INR range specified in the discharge orders for the 30-day Coumadin course?"*
  - Turn 2 answered that the target INR range is not specified in the note (which only states INRs are obtained every Monday and Thursday and faxed to Dr. A)[cite: 1], scoring `3/3` across all dimensions.
- **Canary Run 3:** On the exact same Turn 1 input, the unconstrained LLM call in `followups.py` generated a different chip in slot `0`:
  > *"What medications were prescribed at discharge to manage the patient's chronic conditions, specifically her hypertension, hypothyroidism, and anxiety?"*
  - Turn 2 accurately reported the discharge medications and doses for those three named conditions (`Toprol-XL 50 mg daily`, `Synthroid 137 mcg daily`, and `Ativan 0.25 mg b.i.d.`)[cite: 1].
  - However, the LLM judge scored `responsiveness: 1` because Turn 1 had also mentioned `GERD`, `insomnia`, and `constipation` (`Protonix`, `trazodone`, `Colace`)[cite: 1], which the judge expected under the umbrella phrase *"chronic conditions"* despite the chip's `"specifically..."` qualifier.

### 4.2 Architectural Finding (`TF-9`: Probabilistic Follow-Up Chip Generation)
Inspecting `dev_t2_03` across all three canary runs exposed a broader UX and evaluation vulnerability in `services/agent/followups.py`:
1. **Unverified Promises in Proactive UX:** In Runs 1–2, an unconstrained LLM suggested a clickable question (*"What was the target INR range..."*) whose answer does not exist in the patient's chart[cite: 1]. In clinical UX, proactive UI affordances must never bait users into dead-end questions.
2. **Compound / Unbounded Scope:** In Run 3, the LLM generated a multi-clause question pairing a broad category (*"chronic conditions"*) with a narrow subset (*"specifically hypertension, hypothyroidism, and anxiety"*), triggering judge ambiguity.
3. **Run-to-Run Non-Determinism & Untraced LLM Call:** `followups.py` invoked an untraced LLM generation call (8.0s timeout, zero Langfuse spans)[cite: 3] that caused Turn 2 of Trajectory 2 to change randomly on every benchmark run.

---

## 5. Next Steps & Gate 2 Hold

1. **Gate 2 Paused:** Full 26-scenario execution (`ecc-tier2-dev-suite`) remains paused so Trajectory 2 is not evaluated against stochastic LLM chip generation or masked by rubric adjustments.
2. **Phase 0