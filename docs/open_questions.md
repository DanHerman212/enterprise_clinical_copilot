# Open Questions & Ambiguities — MIMIC-IV 30-Day Readmission Pipeline

Living document. Each item is a decision required before (or during) the data
representation layer is finalized. Resolve in place by replacing the **Status**
line with `RESOLVED — <decision> — <date>` and adding a short rationale.

Source of record: [MIMIC-IV Readmission Prediction Research.txt](../MIMIC-IV%20Readmission%20Prediction%20Research.txt)

---

## 1. MIMIC-IV version & BigQuery dataset names
- **Why it matters:** Dataset paths and a few schema details differ between
  v2.2 and v3.1 (e.g. `physionet-data.mimiciv_hosp` vs
  `physionet-data.mimiciv_3_1_hosp`). Every downstream SQL reference depends
  on this.
- **Options:** v2.2 · v3.1 · both (pin one for training, validate on the other).
- **Status:** RESOLVED — v3.1 — 2026-05-04. Confirmed available datasets:
  `mimiciv_3_1_hosp`, `mimiciv_3_1_icu`, `mimiciv_3_1_derived`,
  `mimiciv_ed` (unversioned), `mimiciv_note` (unversioned).

## 2. Exact `admission_type` membership for acute vs planned
- **Why it matters:** The outline lists a partial set
  (`URGENT`, `EMERGENCY`, `EW EMER.` as acute; `ELECTIVE`, `OBSERVATION`,
  `SURGICAL SAME DAY ADMISSION` as planned). MIMIC-IV actually contains
  additional values such as `DIRECT EMER.`, `DIRECT OBSERVATION`,
  `OBSERVATION ADMIT`, `AMBULATORY OBSERVATION`. Mis-classifying these
  shifts label prevalence and pollutes the positive class.
- **Decision needed:** Final allow-list for the **positive label trigger**
  (acute readmission) and the explicit **exclusion list** (planned).
- **Status:** RESOLVED — 2026-05-04.
  - **Acute (positive trigger):** `URGENT`, `EMERGENCY`, `EW EMER.`,
    `DIRECT EMER.`, `DIRECT OBSERVATION`, `EU OBSERVATION`,
    `OBSERVATION ADMIT`, `AMBULATORY OBSERVATION`.
  - **Planned (excluded as triggers):** `ELECTIVE`,
    `SURGICAL SAME DAY ADMISSION`.

## 3. "First eligible index admission per patient" — selection order
- **Why it matters:** Two valid interpretations produce different cohorts:
  - (a) **Filter then rank:** apply all inclusion/exclusion rules, then take
    the earliest remaining admission per `subject_id`.
  - (b) **Rank then filter:** take each patient's first-ever admission, then
    drop the patient entirely if it fails any rule.
  Option (a) yields more patients; option (b) is more conservative against
  selection bias.
- **Status:** RESOLVED — (a) FILTER then RANK — 2026-05-04. Apply all
  exclusions to every admission per `subject_id`, then take the earliest
  surviving admission as the index. Aligns with standard healthcare quality
  reporting logic and preserves more viable patients.

## 4. Lookback windows under MIMIC-IV's de-identified timeline
- **Why it matters:** MIMIC-IV dates are randomly shifted per patient
  (`anchor_year_group`). The shift is *consistent within a patient*, so
  intra-patient `DATE_DIFF` is valid for prior-365-day admission counts and
  prior-180-day ED counts. No real calendar reasoning is possible.
- **Decision needed:** Confirm we treat the per-patient shifted timeline as
  continuous (standard practice) and document this assumption next to every
  rolling-window feature.
- **Status:** RESOLVED — ACCEPT — 2026-05-04. MIMIC-IV applies a single
  consistent random date shift per `subject_id`, so intra-patient
  `DATE_DIFF` is exact. Document this assumption next to every rolling-window
  feature.

## 5. ED visit source — `mimiciv_ed` module vs proxy
- **Why it matters:** LACE's `E` component and several utilization features
  require ED visit counts. The `mimiciv_ed` module is separately credentialed
  on PhysioNet.
- **Options:**
  - Use `mimiciv_ed.edstays` for true ED encounters (preferred).
  - Proxy via `admissions.edregtime IS NOT NULL` if ED module is unavailable.
- **Status:** RESOLVED — use `mimiciv_ed.edstays` — 2026-05-04. Module access
  confirmed in project `physionet-data`.

## 6. Prediction time anchor — at discharge vs at admission
- **Why it matters:** The HOSPITAL score and CMS HRRP framing both anchor at
  **discharge**. All point-in-time-correct features must therefore use data
  strictly before `dischtime`. An admission-time anchor would change the
  feature set materially (no last-24h vitals, no last labs pre-discharge).
- **Status:** RESOLVED — AT DISCHARGE (`dischtime`) — 2026-05-04. Matches
  HOSPITAL score / CMS HRRP framing and preserves last-24h vitals and last
  pre-discharge labs as predictive signal.

## 7. `mimiciv_note` access for unstructured text branch
- **Why it matters:** Discharge summaries drive the multimodal AUROC lift
  (~0.75–0.79 per outline). The notes module requires separate PhysioNet
  credentialing.
- **Decision needed:** Confirm access; if unavailable, the text branch is
  deferred and the v1 model is tabular-only.
- **Status:** RESOLVED — access available — 2026-05-04. Text branch is in
  scope for v2; v1 remains tabular-only to keep the first pipeline minimal.

## 8. Censoring policy for transfers
- **Why it matters:** Patients discharged to `OTHER FACILITY` / `ACUTE HOSPITAL`
  are excluded from the cohort (right-censored, unobservable). Confirm we do
  *not* attempt to include them with a survival/competing-risk treatment in v1.
- **Status:** RESOLVED — EXCLUDE in v1 — 2026-05-04. Patients discharged to
  `OTHER FACILITY` / `ACUTE HOSPITAL` leave the BIDMC network and their
  30-day outcome is unobservable; including them would inject false negatives.
  Survival / competing-risk treatment deferred to v2.

## 9. Missingness policy for labs/vitals
- **Why it matters:** Outline calls for **median imputation + boolean
  missingness indicators**. We need to fix:
  - Imputation computed on the **training fold only** (no leakage).
  - Indicator column naming convention (`<feat>__is_missing`).
  - Where this lives: BigQuery view vs Vertex pipeline component.
- **Status:** STRUCTURALLY RESOLVED — 2026-05-22. Split into two stages:
  *policy* decided in Phase E (E6) per surviving feature with the chosen
  policy recorded in `docs/feature_shortlist_v2.md`; *execution* via
  `src/imputation.py` (`sklearn.compose.ColumnTransformer`) fit on the
  training fold only and refit per CV fold (Phase F1). **No imputation in
  BigQuery / `cohort_features`** — the table stays raw with NaNs.
  Indicator-column convention: `<feat>__is_missing`. The actual per-feature
  policy table is still pending E6 execution.

## 10. Cross-validation / split strategy
- **Why it matters:** Outline warns about "hierarchical bias and heavily
  correlated sequential records." Even with one row per patient, we should
  confirm: split by `subject_id` (group-aware), stratified on the label,
  with a held-out temporal slice for drift evaluation.
- **Status:** RESOLVED — group-by `subject_id`, chronological 70/15/15 —
  2026-05-04. Splits ordered by index `admittime` to simulate prospective
  deployment; no patient appears in more than one fold.

## 11. Time anchor for vitals (last-24h window)
- **Why it matters:** Issue #6 fixed the default prediction-time anchor at
  hospital `dischtime`. Applying that literally to vitals is wrong in
  practice: `mimiciv_3_1_derived.vitalsign` is the ICU-only pivot of
  `chartevents`, and most ICU patients are stepped down to a floor before
  hospital discharge. A `[dischtime − 24h, dischtime]` window therefore
  matches ~3 % of the cohort even though ~17 % had an ICU stay.
- **Status:** RESOLVED — anchor vitals to `icustays.outtime` — 2026-05-21.
  Window is `[icu_outtime − 24h, icu_outtime]` with a safety guard
  `icu_outtime <= dischtime`. For admissions with multiple ICU stays, events
  from any stay's last 24 h are aggregated; `*_last_24h` resolves to the
  most recent reading overall (i.e. before final ICU discharge for the
  admission). Coverage rose from 0.027 → 0.157, tracking the cohort's
  severity (ICU) coverage of 0.174. Documented in
  `feature_engineering.ipynb` Family 5 markdown.

---

## Phase E / F modeling decisions (opened 2026-05-22)
_These were implicit until Phases E (EDA + feature selection) and F (modeling)
were formalized in `production_workflow_plan.md`. Resolve in order; #12–#15
block modeling, the rest can settle during E._

## 12. Class-imbalance handling
- **Why it matters:** Train prevalence is 18.6 %. Choice affects loss
  surface, calibration, and downstream threshold tuning.
- **Options:**
  - `class_weight='balanced'` (LR) / `scale_pos_weight = neg/pos` (GBM).
  - Threshold tuning post-fit on a naturally-trained model (preferred by
    most modern guidance — preserves calibration).
  - Focal loss (overkill at this prevalence).
  - **No oversampling** — patient-grouped data makes SMOTE leak across
    folds and distort calibration.
- **Status:** OPEN. Default proposal: train naturally, calibrate, tune
  threshold. Revisit only if val PR-AUC underperforms.

## 13. Probability calibration
- **Why it matters:** Clinical utility depends on calibrated probabilities,
  not just ranking. Tree models are typically miscalibrated.
- **Options:** Platt (sigmoid) · isotonic · `CalibratedClassifierCV` with
  cross-fit on `train` (avoids val contamination).
- **Status:** OPEN. Default proposal: isotonic via `CalibratedClassifierCV(method='isotonic', cv=GroupKFold)` on train only; validate calibration on val (Brier, ECE, reliability curve).

## 14. Operating-threshold policy
- **Why it matters:** Drives the demo harness UX and the
  precision/recall/NNT trade-off shown to clinicians.
- **Options:** maximize F1 on val · fix recall at e.g. 0.60 (CMS-style
  sensitivity floor) · cost-weighted (requires a $$ assumption) ·
  Youden's J.
- **Status:** OPEN. Default proposal: fix recall at 0.50–0.60 on val and
  report the implied precision and NNT; publish a decision-curve plot
  spanning thresholds rather than committing to one number.

## 15. Headline metric bundle
- **Why it matters:** AUROC alone is uninformative at this prevalence and
  unconvincing to clinical reviewers.
- **Options:** AUROC · PR-AUC · Brier · ECE · decision-curve net benefit
  at the chosen threshold · sensitivity / specificity / PPV / NNT.
- **Status:** OPEN. Default proposal: **PR-AUC primary**; AUROC, Brier,
  ECE, calibration plot, and decision-curve net benefit secondary.
  Reported on `test` only after model selection on `val` is frozen.

## 16. Categorical encoding strategy
- **Why it matters:** LightGBM's native `categorical_feature` handling
  produces different splits than one-hot and is *not* interchangeable.
  LR baseline still needs one-hot.
- **Status:** OPEN. Default proposal: one-hot for LR; native categorical
  for LightGBM, with explicit `categorical_feature` list declared in code.

## 17. High-cardinality categorical bucketing
- **Why it matters:** `admit_provider_id`, `language` long tail, fine-grained
  `marital_status` or `discharge_location` codes blow up dimensionality and
  invite overfitting.
- **Status:** OPEN. Default proposal: floor at 1 % frequency on train;
  collapse the residual into `_other`. Fit the bucket map on train only.

## 18. Numeric winsorization / clipping
- **Why it matters:** E4 clinical-plausibility review will surface ranges
  (HR > 250, temp < 30 °C, etc.). Decision: drop those rows, set to NaN
  and route through the imputer, or winsorize to the plausible bound.
- **Status:** OPEN. Default proposal: clip to a clinician-defensible
  range, with the bounds fit on train and applied unchanged to
  val/test/demo.

## 19. Cohort-version retention policy
- **Why it matters:** When `cohort_features_v2` is cut in Phase E, do we
  keep `v1` for reproducibility, drop it, or alias?
- **Status:** OPEN. Default proposal: retain `v1` as an immutable table for
  audit; new work targets `v2`. `FEATURES_VERSION` in `src/config.py` is
  the single source of truth.

## 20. Reproducibility envelope
- **Why it matters:** A real pipeline needs deterministic splits and
  model fits across re-runs.
- **Status:** OPEN. Default proposal: fixed `RANDOM_SEED = 42` in
  `src/config.py`; pinned `lightgbm` / `xgboost` / `scikit-learn` versions
  in `requirements.txt`; the SQL hash of the assembly query and the
  `FEATURES_VERSION` recorded in `src.tracking` for every run.

## 21. Phase D CI gate failure policy
- **Why it matters:** Evidently threshold 0.1 currently gives 0/74 drifted.
  We need an explicit rule for what counts as a hard-fail vs warn.
- **Status:** OPEN. Default proposal:
  - Hard-fail: any change to the schema (new/missing column, dtype change);
    label-prevalence drift > 0.02 absolute; >5 % of features drifted; any
    single feature with drift score > 0.25.
  - Warn (no block): 1–5 % of features drifted at threshold 0.1; vitals
    coverage swing > 5 percentage points.

## 22. Text branch (`mimiciv_note`) sequencing
- **Why it matters:** #7 marked the text branch as in scope for **v2**.
  Need to decide whether v2 starts immediately after Phase H or is parked
  until v1 is deployed.
- **Status:** OPEN. Default proposal: park until v1 is end-to-end
  reproducible; revisit after Phase H demo harness ships.

---

## Resolution log
_Append entries here as items move from OPEN → RESOLVED._
