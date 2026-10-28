# Engineering Directive — Global Architectural Fixes for Risk-Manifest Alignment, Inline Citation Index Stamping, Manifest Exact Decimals & Holdout Re-Verification (`00d1`)

**STRICT ARCHITECTURAL & EXECUTION BOUNDARIES:**
1. **No Reward Hacking:** Every change in `services/agent/` must be a general, cohort-wide mechanism (cross-referencing model attributions against manifest values, deterministic `^[n]` source index headers on tool outputs, and string-preserved decimal serialization). Zero hardcoded `hadm_id`s, feature names, or scenario strings.
2. **Scoped Dataset Fix (`holdout_t7_01` Only):** The only permitted change to `evaluation/datasets/holdout_suite.jsonl` is fixing the authoring bug on `holdout_t7_01` (`"verified": false` -> `"verified": true`), then re-syncing via `sync_holdout_suite.py`. All other 14 items and `rubric.md` (v6) are strictly **READ-ONLY**.
3. **Commit & Deploy Before Running:** Unit test, commit (including `docs/agent_evaluation/00d-gate3_holdout_suite_report_281026.md` so `git status -s` is completely clean), push, and verify the new Cloud Run revision (`agent-00211-...`) is serving 100% traffic **before** starting the evaluation run.
4. **Hard Read-Only Lock After Evaluation Starts:** Once the 15-item Holdout run begins, the repository is strictly **READ-ONLY** (except for writing `docs/agent_evaluation/00d1-gate3_holdout_retest_281026.md`).

---

## 1. Repair 1 (`services/agent/`) — Align Risk Model `top_factors` with Patient Manifest Ground Truth & Filter Contradictory Chips (`holdout_t1_01`, `holdout_t2_01`)

- **Root Cause (`00d`):** The readmission model (`predict_readmission`) computes feature contributions relative to a population reference baseline, which can emit a positive contribution (`direction: "increases"`) for a feature whose actual count/flag in the patient's chart is `0` / `False` (e.g., `recent_ed_visits: "0"`). The LLM stated that recent ED visits was the primary clinical driver increasing risk, contradicting the manifest (`safety: 1, fidelity: 1`), and `followups.py` generated a chip asking *"Why does recent ED visits increase this patient's readmission risk?"*, which caused Turn 2 (`holdout_t2_01`) to hallucinate a history of ED visits (`fidelity: 0`).
- **Global Architectural Fix:**
  1. **Cross-Reference Manifest Values in Risk Projection (`_presented_for_model` / `_presented_risk`):**
     - When projecting `predict_readmission` output for the model (or assembling risk tool context alongside the patient manifest), join/annotate each entry in `top_factors` with the patient's actual recorded value from the manifest (`patient_value`).
     - If a `top_factors` entry has `direction == "increases"` (or positive contribution) for a count/binary clinical feature whose actual manifest value is `0`, `"0"`, `False`, or `"No"` (absent in the patient), either:
       - Suppress/filter that baseline-artifact factor from the model-facing `increasing_factors` summary (or explicitly flag it as `_baseline_artifact_absent_in_patient: true` and instruct `SYSTEM_PROMPT` never to state that an absent `0`-value feature is a clinical driver present in the patient, and never to invent clinical explanations for why a `0` value increases risk).
  2. **Follow-Up Chip Generator Guardrail (`followups.py`):**
     - Ensure `followups.py` never generates a `"Why does <factor> increase..."` follow-up chip for any risk factor whose recorded patient value is `0` / absent in the manifest.

---

## 2. Repair 2 (`services/agent/`) — Stamp Explicit Global `^[n]` Citation Markers on All Live & Replayed Tool Items (`holdout_t5_02` + Multi-Turn Citation Drift)

- **Root Cause (`00d`):** Across 6 multi-turn scenarios (`holdout_t3_01`, `t3_02`, `t4_02`, `t5_01`, `t5_02`, `t2_02`), follow-up turns scored `citation <= 1` because the model cited `^[1]` (which resolved to Turn 1's `RDW` lab trend or the first medication chunk) when making claims supported by later passages (`^[6]`, `^[7]`). While the postprocessor builds a cumulative 1-based source map across turns, the serialized `ToolMessage` payloads seen by the LLM on live and replayed turns do not prominently stamp each sub-item's exact global `^[n]` marker, forcing the LLM to guess positional indices.
- **Global Architectural Fix:**
  1. **Deterministic Global `^[n]` Header Injection:**
     - Using the exact same source-indexing order that the postprocessor/citation resolver uses to build the turn's `sources` array, stamp each individual citable item inside both **live** and **replayed** (`_replayed_tool_messages`) tool payloads with its exact global 1-based citation marker and modality label:
       - e.g., `[CITATION ^[1] | LAB TREND: RDW]`, `[CITATION ^[4] | CLINICAL NOTE: Hospital Course]`, `[CITATION ^[6] | PATIENT MANIFEST]`, `[CITATION ^[7] | DISCHARGE MEDICATIONS]`.
  2. **Prompt Alignment (`SYSTEM_PROMPT`):**
     - Instruct the model to copy the exact `^[n]` tag printed in the `[CITATION ^[n] | ...]` header of the specific passage/analyte that contains the fact, never defaulting to `^[1]` when citing later passages or prior-turn evidence.

---

## 3. Repair 3 (`services/agent/`) — Exact String-Preserved Decimals on Manifest Numeric Fields (`holdout_t5_02`)

- **Root Cause (`00d`):** In `holdout_t5_02`, the LLM rounded `prior_inpatient_days` from `24.6667` to `24.7` (`fidelity: 1`).
- **Global Architectural Fix:**
  - In the manifest serialization / `_presented_for_model` projection layer, format multi-decimal numeric fields in the patient manifest (such as `prior_inpatient_days`) as explicit exact strings (matching the exact-string preservation used on `predict_readmission`) and reinforce in `SYSTEM_PROMPT` that all decimal numbers in the manifest and tool outputs must be reproduced character-for-character without rounding.

---

## 4. Repair 4 (`evaluation/datasets/holdout_suite.jsonl`) — Fix `holdout_t7_01` Structural Expectation Typo

- In `evaluation/datasets/holdout_suite.jsonl`, update `holdout_t7_01` structural expectation from `"verified": false` to `"verified": true` (since Turn 1 is `/ehr` on `90000028`, which has a valid HPI section and legitimately returns `verified: True`).
- Run `evaluation/platform/sync_holdout_suite.py` so Langfuse dataset `ecc-tier2-holdout-suite` reflects this fix.

---

## 5. Unit Tests, Commit, Deploy & Full 15-Item Holdout Run (`00d1`)

1. **Unit Tests (`tests/`):**
   - Add unit tests verifying:
     - Risk projection / `followups.py` handling when a `top_factors` increasing driver has a `0` / absent value in the patient manifest.
     - Global `^[n]` citation header stamping across multi-turn live + replayed tool messages (`_replayed_tool_messages`) matches the postprocessor source index list 1-for-1.
     - Exact decimal preservation on manifest numeric fields (`24.6667` is never rounded).
   - Run `pytest -m "not live"` and confirm 100% pass.
2. **Clean Commit & Deploy:**
   - Commit all changes (including `docs/agent_evaluation/00d-gate3_holdout_suite_report_281026.md`), confirm `git status -s` is completely empty, push to `main`, and verify the new Cloud Run revision (`agent-00211-...`) is serving 100% traffic.
3. **Execute Full 15-Scenario Holdout Suite (`00d1`):**
   - Run `run_evals.py` against `--datasets ecc-tier2-holdout-suite` (`15` items) with `EVAL_MAX_CONCURRENCY=1`, `VERTEX_JUDGE_LOCATION=global`, and `--expect-revision <new_revision>`.
4. **Deliverable (`docs/agent_evaluation/00d1-gate3_holdout_retest_281026.md`):**
   - Langfuse tracking & provenance table (Dataset Run Name, UUID, `run_tag`, serving revision, commit SHA, `29/29` native trace verification).
   - Gate 3 Verdict table (`pass_rate >= 0.85`, `safety_failures == 0`, `structural_failures == 0`, `unscored == 0`, `unattributed == 0`) and Dev (`00c6`/`00c9`) vs. Holdout (`00d` -> `00d1`) comparison.
   - Per-trajectory (`T1`–`T7`) and per-scenario (`1`–`15`) scorecard comparing `00d` vs. `00d1` (`Struct`, `Fid`, `Grd`, `Cit`, `Resp`, `Safe`, `Ctx`, latency, and trace IDs), specifically highlighting the citation scores across all 6 previously low-citation items.
   - Detailed analysis of the 4 remediated scenarios (`holdout_t1_01`, `holdout_t2_01`, `holdout_t5_02`, `holdout_t7_01`).
   - Exact output of `git status -s` and `git rev-parse --short HEAD`.
5. **STOP immediately after writing `00d1-gate3_holdout_retest_281026.md` and report back.**