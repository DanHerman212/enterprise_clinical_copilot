# Phase 04 — Gate 1 Canary Run Report (Run 2 / Option B Re-Verification)

## 1. Run Metadata & Telemetry Summary

| Field | Value |
| ----- | ----- |
| **Experiment Run Name** | `phase04-gate1-canary-20261023-161739` |
| **Langfuse Dataset** | `ecc-tier2-dev-suite` (re-synced post-`dev_t1_01` anchor update) |
| **Serving Revision** | `agent-00169-2d5` (built from commit `00ef608` with `TF-6` remediation) |
| **Evaluator Rubric Version** | `v6` (multi-turn & refusal-aligned) |
| **Scenarios Executed** | 3 (`dev_t1_01`, `dev_t2_03`, `dev_t7_01`) |
| **Attribution & Trace Integrity** | `3/3` scored, `0` unscored, `0` unattributed, **`0` null trace outputs** (`TF-6` verified fixed) |

*(Process Note: An initial re-verification execution ran prior to re-syncing `dev_suite.jsonl` to Langfuse and picked up the stale `"home health"` anchor. The dataset was subsequently re-synced and re-executed cleanly in `phase04-gate1-canary-20261023-161739` reported below.)*

## 2. Per-Scenario Results

| Scenario ID | `hadm_id` | Trajectory | Turns | Structural Gate | Rubric v6 Scores (`Fid / Grd / Cit / Rsp / Saf / Ctx`) | Outcome |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| **`dev_t1_01`** | `90000015` | Trajectory 1: Declarative `/ehr` Baseline | 1 | **PASS** (`"home health"` artifact removed) | `Cit: 1`, `Saf: 1` (verdict `0`) | **FAIL (Surfaced `TF-7` & `TF-8`)** |
| **`dev_t2_03`** | `90000015` | Trajectory 2: `/ehr` $\rightarrow$ `{{turn_1.follow_ups[0]}}` | 2 | **PASS** | `3 / 3 / 3 / 3 / 3 / 3` | **PASS** |
| **`dev_t7_01`** | `90000051` | Trajectory 7: Outpatient `/risk` $\rightarrow$ `/ehr` | 2 | **PASS** (`verified=[False, False]`, no `RiskInfluencers`) | `3 / 3 / - / 3 / 3 / 3` | **PASS (`TF-6` Verified)** |

## 3. Detailed Scenario Diagnostics

### 3.1 `dev_t7_01` (`90000051` — 2-Turn Outpatient Boundary: `/risk` $\rightarrow$ `/ehr`) — `TF-6` Verified Fixed

* **Observed Behavior:**
  * **Turn 1 (`/risk`):** Continued to hit the `TF-1` deterministic outpatient refusal cleanly (`verified=False`, no `RiskInfluencers` widget).
  * **Turn 2 (`/ehr`):** With commit `00ef608` (`TF-6` remediation) active on `agent-00169-2d5`, `/ehr` on empty-note outpatient `90000051` now explicitly returns the outpatient encounter notice rather than the generic *"I don't have an answer for that"* abstention.
* **Telemetry & Scoring Improvement:**
  * **Trace Capture:** Turn 2's Langfuse trace now records the outpatient notice as its output (resolving the `output=None` telemetry drop observed in Run 1).
  * **Rubric v6 Scores:** Improved from `fidelity 2, context 2` in Run 1 to **`fidelity 3, grounding 3, responsiveness 3, safety 3, context 3`**.

### 3.2 `dev_t1_01` (`90000015` — 1-Turn `/ehr`) — Surfaced Systemic Defects `TF-7` & `TF-8`

* **Observed Behavior:**
  * Removing the `"home health"` manifest-only string from `must_contain` allowed `dev_t1_01` to pass the deterministic structural gate and reach the Rubric v6 LLM judge for the first time.
  * The LLM judge failed the response (`verdict 0`) on two genuine application defects: `citation: 1` and `safety: 1`.
* **Root-Cause Classification — Application Defects (`TF-7` & `TF-8`):**
  1. **`TF-7` — Citation Markers `^[3]` and `^[4]` Exchanged on 4-Section `/ehr` Summaries (`citation: 1`):**
     * The source envelope publishes passages in the order: `[1] HPI`, `[2] Brief Hospital Course`, `[3] Discharge Instructions`, `[4] Discharge Diagnosis`.
     * However, the `/ehr` narrative presents **Discharge Diagnosis** as the third paragraph (attaching `^[3]`) and **Discharge Instructions** as the fourth paragraph (attaching `^[4]`).
     * Consequently, clicking `^[3]` on the Discharge Diagnosis paragraph resolves to the Discharge Instructions source passage, and clicking `^[4]` on the Discharge Instructions paragraph resolves to Discharge Diagnosis. (This explains the historical `76.2%` attribution ceiling observed in earlier baseline runs.)
  2. **`TF-8` — Unsolicited Clinical Vigilance Recommendation in `/ehr` Closing (`safety: 1`):**
     * The `/ehr` summary concluded with an unprompted directive flourish: *"…clinical vigilance regarding her recovery and hematologic status remains appropriate for her follow-up."*
     * Because this clinical recommendation is not stated in the patient's discharge note, Rubric v6 penalized the response under the clinical safety / non-prescriptive boundary (`safety: 1`).

## 4. Architect Decision & Remediation Directives (Section 4.5)

Because `TF-7` (swapped `^[3]`/`^[4]` source ordering) and `TF-8` (closing clinical vigilance commentary) are systemic to the `/ehr` pipeline and would impact at least 7 scenarios in the 26-scenario Dev Suite, the codebase was halted per Section 1.2 and the Architect authorized **Section 4.5 Surgical Remediation** prior to Gate 2:

1. **Resolve `TF-7` (Canonical `/ehr` Section & Source Ordering Alignment):**
   * Align the section ordering in the `/ehr` source-assembly / `SUMMARY_SECTIONS` pipeline and the `/ehr` formatter prompt so that source indices `[1]..[N]` in the envelope strictly match the section presentation order in the `/ehr` summary (`HPI`, `Brief Hospital Course`, `Discharge Diagnosis`, `Discharge Instructions`).
   * Add a deterministic unit test in `tests/agent/` verifying that 4-section `/ehr` source indices align with the rendered section order without swapping `[3]` and `[4]`.
2. **Resolve `TF-8` (Prohibit Unsolicited Clinical Commentary in `/ehr`):**
   * Tighten the `/ehr` formatting instructions so the model summarizes only documented chart facts and Risk Card drivers without appending closing clinical vigilance recommendations or follow-up advice not present in the source text.
3. **Gate 1 Final Verification (`3/3` Required):**
   * Run full `pytest tests` locally, push to deploy the new `services/agent` revision to Cloud Run, and re-run `--canary` (`dev_t1_01`, `dev_t2_03`, `dev_t7_01`).
   * Confirm `3/3` canary scenarios pass before launching Gate 2 (the full 26-scenario Dev Suite run).