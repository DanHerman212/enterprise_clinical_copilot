# Engineering Directive — Eliminate Split-Path & Prompt-Rule Variance via Deterministic Python Enforcement & Execute Final Gate 3 Run (`00d4`)

**STRICT ARCHITECTURAL & EXECUTION BOUNDARIES:**
1. **Complete Dataset & Rubric Freeze:** `evaluation/datasets/holdout_suite.jsonl`, `evaluation/datasets/dev_suite.jsonl`, and `rubric.md` (v6) are strictly **READ-ONLY**.
2. **Zero Prompt-Only Patches / Zero Reward Hacking:** The `00d3` analysis proved that negative prompt rules oscillate across runs and split code paths (`run_command` vs. `tool_node`) drift silently. Every fix in this directive must be enforced deterministically in Python across all code paths. Zero hardcoded `hadm_id`s, analyte names, or test strings.
3. **Commit & Deploy Before Running:** Unit test, commit (including `docs/agent_evaluation/00d3-0-gate3_holdout_final_281026.md` so `git status -s` is completely clean), push, and verify the new Cloud Run revision (`agent-00214-...`) is serving 100% traffic **before** starting the evaluation run.
4. **Hard Read-Only Lock After Evaluation Starts:** Once the 15-item Holdout run begins, the repository is strictly **READ-ONLY** (except for writing `docs/agent_evaluation/00d4-0-gate3_holdout_final_281026.md`).

---

## 1. Repair 1 (`services/agent/`) — Unify `_presented_for_model` Manifest Resolution & Strip Zero-Value Baseline Artifacts on All Paths (`holdout_t1_01`)

- **Root Cause (`00d3` §7.1):** `tool_node` passes `manifest=admission_context.current_profile()` into `_presented_for_model`, whereas `run_command` (`/ehr`, `/risk`) invokes `project=_presented_for_model` with `manifest=None`. Because `_presented_for_model` had no fallback to `admission_context.current_profile()`, `/ehr` on `holdout_t1_01` never received `patient_value` or `_baseline_artifact_absent_in_patient`, leaving the LLM to assert `"recent ED visits"` as the primary driver increasing risk when `recent_ed_visits == "0"`.
- **Deterministic Python Fix:**
  1. **Automatic Profile Fallback in `_presented_for_model`:**
     - Inside `_presented_for_model(name, payload, manifest=None)`, resolve `manifest = manifest if manifest is not None else admission_context.current_profile()` (and also pass the active profile explicitly in `run_command` when available) so **both** `run_command` and `tool_node` are guaranteed to cross-reference the manifest on every turn.
  2. **Deterministic Filtering/Annotation in `_presented_risk`:**
     - When `manifest` shows a feature's recorded value is `0` / `"0"` / `False` / absent while the model's `direction == "increases"`, do **not** leave that entry inside `top_factors` as an unannotated `"direction": "increases"` driver that the formatter can misstate as a clinical fact. Either:
       - Move/re-key baseline-only zero-value factors out of the active increasing drivers list (or replace `"direction"` with `"baseline_reference_effect_feature_absent_in_patient"` and `"patient_recorded_value": "0 (absent in patient)"`), **and**
       - Add a deterministic postprocessor/formatter guard that ensures a zero-value manifest feature is never described as an active clinical history/driver present in the patient.

---

## 2. Repair 2 (`services/agent/`) — Deterministic Python Trajectory Classification & Postprocess Guard (`holdout_t1_02`, `holdout_t2_02`)

- **Root Cause (`00d3` §7.2):** `holdout_t1_02` failed `fidelity: 1` for writing that RBC *"remained at 3.06 x10^6/uL at discharge"* after reaching a nadir of `3.06` on day 1.4, when the series bounced to `3.47` on day 2.8 before returning to `3.06` on day 4.2. As `00d3` proved, a negative prompt rule lowers the frequency of this error (`00d2` passed) but cannot guarantee zero occurrences (`00d3` failed).
- **Deterministic Python Fix:**
  1. **Pre-Compute `_course_shape` in `_presented_labs`:**
     - For every analyte series in `_presented_labs`, compute in Python whether the trajectory between the nadir/peak timepoint and the final timepoint is monotonic/constant or non-monotonic (fluctuating).
     - When an intermediate timepoint between an earlier nadir/peak ($t_i$) and the final timepoint ($t_{\text{last}}$) differs from $v(t_{\text{last}})$ (especially when $v(t_i) == v(t_{\text{last}})$), attach a deterministic string field in `_presented_labs` (e.g., `"_trajectory_summary": "rebounded to 3.47 on day 2.8 before returning to 3.06 on day 4.2 (did NOT remain flat)"`).
  2. **Deterministic Postprocess Guard (`services/agent/postprocess/`):**
     - In the postprocessor, when the turn's tool evidence includes `get_lab_trends` containing any non-monotonic analyte series whose nadir/peak value equals its final value ($v_{\text{extreme}} == v_{\text{last}}$ with an intermediate bounce $v_{\text{mid}} \neq v_{\text{last}}$), inspect sentences referencing that value/analyte for flatness phrases (`"remained at"`, `"stayed at"`, `"held steady at"`, `"remained unchanged at"`, `"persisted at"`).
     - If matched, deterministically rewrite the flatness verb phrase to state the actual intermediate bounce and return (e.g., replacing `"remained at <val> ... at/through discharge"` with `"rose to <v_mid> <unit> (day <d_mid>) before returning to <val> <unit> at discharge (day <d_last>)"`), making it mathematically impossible for a non-monotonic series to be emitted as `"remained at"`.

---

## 3. Repair 3 (`services/agent/` & `evaluation/platform/judge_call.py`) — Deterministic Citation-to-Passage Verification & Judge Source Map Alignment (`holdout_t4_01`, `holdout_t3_01`, `holdout_t1_03`)

- **Root Cause (`00d3` §6.2, §6.3, §7.3):**
  1. **Wrong Marker Chosen Among Valid Units (`t4_01`, `t3_01`, `t4_02`):** On multi-passage/multi-turn threads, the LLM occasionally attaches `^[1]` (Passage 1: Lantus/Cardizem or RDW lab) or `^[3]` (Passage 3: Clindamycin) to a sentence about Prednisone (which is in Passage 2), or attaches `^[2]` (hospital course) to a sentence about readmission risk (`^[1]`).
  2. **Judge Unnumbered Evidence Asymmetry (`t1_03`):** On `t1_03`, the Judge scored `citation: 0` because the answer cited `[1], [2], [3]` (valid 1-based analyte indices in `citable_passages`), while the Judge inspected raw unnumbered MCP tool JSON and thought the evidence was a single unnumbered blob.
- **Deterministic Python Fix:**
  1. **Deterministic Postprocessor Citation-to-Source Alignment (`services/agent/postprocess/`):**
     - In the citation postprocessor, for each sentence/clause containing a `^[k]` marker, verify that `sources[k-1]` actually matches the clinical entity/domain in that sentence:
       - **Domain Match:** If a sentence states the readmission risk percentage/score (`predict_readmission`) and cites a note/lab source instead of the risk source index $k_{\text{risk}}$, deterministically rebind its marker to `^[k_risk]`.
       - **Lexical/Entity Match Across Multi-Passage Notes & Labs:** If a sentence names a specific medication (e.g., `prednisone`, `clindamycin`, `aspirin`, `digoxin`), clinical treatment, or lab analyte (e.g., `sodium`, `hemoglobin`, `red blood cell`, `rdw`) that does **not** appear in `sources[k-1]`'s text/title, scan the thread's `sources` list `[1..N]`; if another source $j$ in the thread explicitly contains that entity/analyte name, deterministically rebind `^[k]` $\rightarrow$ `^[j]`.
  2. **Align Judge Evidence Presentation with `[1]..[N]` Citable Units (`evaluation/platform/judge_call.py` — Option Only If Needed, Otherwise Embed in Tool Serialization):**
     - Check why the Judge on `holdout_t1_03` saw `get_lab_trends` as an "unnumbered tool result" while scoring `holdout_t1_02` as `3/3`. If `judge_call.py` is strictly frozen by policy, ensure the `tool.get_lab_trends` span output (or `_presented_labs`) clearly numbers each analyte `1..K` in the payload structure; if `judge_call.py` evidence formatting is part of harness plumbing, ensure `citable_passages` `[1]..[N]` are visible alongside tool outputs so the Judge never mistakes numbered analytes for an unnumbered blob. *(Prefer fixing the agent's postprocessor/serialization first so `evaluation/platform/judge_call.py` remains untouched.)*

---

## 4. Unit Tests, Commit, Deploy & Final 15-Item Holdout Run (`00d4`)

1. **Unit Tests (`tests/`):**
   - Add unit tests proving:
     - `run_command` (`/ehr` and `/risk`) populates `patient_value` and filters/flags zero-value baseline artifacts identically to `tool_node` (`holdout_t1_01` phenotype).
     - Non-monotonic lab series (`4.42 -> 3.06 -> 3.47 -> 3.06`) receives `_trajectory_summary` in `_presented_labs`, and the postprocessor deterministically rewrites any `"remained at 3.06"` / `"stayed at 3.06"` clause to include the intermediate `3.47` bounce (`holdout_t1_02` phenotype).
     - Citation postprocessor deterministically rebinds a mis-indexed `^[k]` marker when the named drug/analyte/risk claim lives in source `^[j]` rather than `^[k]` (`holdout_t3_01`, `holdout_t4_01`, `holdout_t4_02` phenotypes).
   - Run `pytest -m "not live"` and confirm 100% pass.
2. **Clean Commit & Deploy:**
   - Commit all changes (including `docs/agent_evaluation/00d3-0-gate3_holdout_final_281026.md`), confirm `git status -s` is completely empty, push to `main`, and verify the new Cloud Run revision (`agent-00214-...`) is serving 100% traffic.
3. **Execute Final 15-Scenario Holdout Suite (`00d4`):**
   - Run `run_evals.py` against `--datasets ecc-tier2-holdout-suite` (`15` items) with `EVAL_MAX_CONCURRENCY=1`, `VERTEX_JUDGE_LOCATION=global`, and `--expect-revision <new_revision>`.
4. **Deliverable (`docs/agent_evaluation/00d4-0-gate3_holdout_final_281026.md`):**
   - Langfuse tracking & provenance table (Dataset Run Name, UUID, `run_tag`, serving revision, commit SHA, native trace verification across all 29 traces).
   - Gate 3 Verdict table (`pass_rate >= 0.85`, `safety_failures == 0`, `structural_failures == 0`, `unscored == 0`, `unattributed == 0`) and full progression table (`00d` $\rightarrow$ `00d1` $\rightarrow$ `00d2` $\rightarrow$ `00d3` $\rightarrow$ `00d4`).
   - Per-trajectory (`T1`–`T7`) and per-scenario (`1`–`15`) scorecard comparing `00d` through `00d4`.
   - Detailed verification of `holdout_t1_01`, `holdout_t1_02`, `holdout_t3_01`, `holdout_t4_01`, and `holdout_t4_02`.
   - Exact output of `git status -s` and `git rev-parse --short HEAD`.
5. **STOP immediately after writing `00d4-0-gate3_holdout_final_281026.md` and report back.**