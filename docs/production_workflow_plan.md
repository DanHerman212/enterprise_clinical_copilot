# Production Workflow Plan — MIMIC-IV 30-day Readmission

Authoritative plan for moving from the validated data representation
(cohort + label + LACE + HOSPITAL baselines) into a production MLOps
workflow on Google Cloud (Gemini Enterprise / Agent Platform; SDK
surface unchanged from Vertex AI).

## Refinements over the prior plan

1. **Four splits, not three.** Add a `demo` bucket alongside
   train/val/test. The demo patients are the "live patient population"
   for a Clinical Copilot UI that calls the deployed endpoint —
   producing prediction-drift and outcome-accuracy signal against
   ground-truth labels.
2. **Experiment tracking lives in `src/`.** Reusable from notebook now
   and from a containerized Pipelines step later. Built on
   `google-cloud-aiplatform`; service rebrand is irrelevant.
3. **Missingness, drift, and feature contract handled by a validation
   step.** Evidently report after feature engineering drives per-feature
   imputation policy and acts as the CI-gate baseline. HOSPITAL
   "missing = 0" stays as-is for the baseline.
4. **Notebook forks at feature engineering.** Splits and tracking
   stay in `readmission_pipeline.ipynb`. Feature engineering and model
   development each get their own notebook.
5. **No Feature Store yet.** Defer until an online-serving requirement
   exists. BigQuery views port cleanly when that day comes.

## Step-by-step plan

### Phase A — Splits (this notebook)
- A1. Markdown: rationale for hash-based patient-level splits, four
  buckets, leakage discipline.
- A2. Build `readmission.cohort_splits`:
  - One row per distinct `subject_id`.
  - `bucket = ABS(MOD(FARM_FINGERPRINT(CAST(subject_id AS STRING)), 1000))`.
  - `bucket < 5` → `demo` (~0.5%)
  - `5 ≤ bucket < 705` → `train` (70%)
  - `705 ≤ bucket < 855` → `val` (15%)
  - `855 ≤ bucket < 1000` → `test` (~14.5%)
  - Cluster by `subject_id`. Diagnostics: per-split patient/admission
    counts, prevalence; assert zero patients span splits.
- A3. Re-evaluate LACE/HOSPITAL on the `test` split alone — locked-in
  discriminative floor.

### Phase B — Experiment-tracking scaffold
- B1. `src/tracking.py`: thin wrapper around `aiplatform.init`,
  `start_run`, `log_params`, `log_metrics`.
- B2. `src/config.py`: project, region, dataset, table FQNs, MIMIC
  version, git SHA helper.
- B3. Notebook cell: open single experiment `readmission-30d`, log
  run `baseline-v1` with cohort SHA, n_train/val/test/demo, prevalence
  per split, LACE/HOSPITAL test AUROC + AUPRC.

### Phase C — Feature engineering (`feature_engineering.ipynb`)
- C1. Markdown schema table — ~26 features across demographics,
  index-stay administrative, comorbidities, prior utilization, last
  24h vitals, last pre-discharge labs.
- C2. One SQL cell per family → BigQuery view, keyed
  `(subject_id, hadm_id)`. Each cell ends with a leakage diagnostic
  (max source `charttime` vs row `dischtime`).
- C3. Final cell: assemble `readmission.cohort_features` =
  `cohort_labeled ⨝ cohort_splits ⨝ all feature views`. Cluster by
  `subject_id`. One row = one `hadm_id` with label, split, and full
  feature vector.

### Phase D — Data validation
- D1. **Evidently OSS** over `cohort_features` (train as reference; val
  and test as current). `DataSummaryPreset` for schema/dtype/missingness/
  descriptive stats; `DataDriftPreset` for per-column drift (Wasserstein
  normed for numerics, Jensen-Shannon for categoricals, threshold 0.1).
- D2. Artifacts archived under `artifacts/validation/<RID>/`:
  interactive HTML reports + full Evidently JSON dumps + a condensed
  `summary.json` (drift counts, top-N by drift score) suitable as a
  future CI / promotion gate.
- D3. Imputation policy decisions are deferred to **Phase E (E6)** —
  the missingness rates surfaced here are inputs to that decision, not
  acted on yet. `cohort_features` remains raw with NaNs.

### Phase E — EDA + feature selection (`feature_selection.ipynb`)
Deliberately fused with EDA: classical EDA's univariate work is already
covered by Evidently's `DataSummaryPreset` in Phase D, so this notebook
focuses on the questions Evidently structurally cannot answer
(relationships with the label, feature-feature redundancy, clinical
plausibility). Output is a defended shortlist that downstream modeling
operates on.
- E1. **Re-use Phase D output.** Load the latest
  `artifacts/validation/<RID>/summary.json` rather than recomputing
  univariate stats.
- E2. **Bivariate vs label.** Per-feature univariate AUROC and mutual
  information; distribution-by-label overlays for the top-N.
- E3. **Redundancy.** Correlation heatmap grouped by family;
  hierarchical clustering on `|ρ|` to collapse the `*_mean/min/max/last`
  vital clusters; VIF for the residual numerics.
- E4. **Clinical plausibility / outliers.** Targeted panel for
  physiologic ranges (HR, BP, SpO2, temperature, labs); decisions
  recorded as keep / winsorize / drop with rationale.
- E5. **Multivariate signal.** LightGBM baseline trained on `train`,
  evaluated on `val`. Permutation importance + SHAP summary. Leakage
  audit on the top-20 by univariate AUROC (any single-feature
  AUROC > 0.75 gets scrutinized).
- E6. **Missingness policy.** For each kept feature, decide one of:
  *drop* (rate too high to be useful), *impute-median* (numeric MCAR-ish),
  *impute-mode* (categorical MCAR-ish), *keep-as-NaN* (tree models
  handle natively), or *missing-indicator + impute* (MNAR — e.g.,
  vitals/labs missing = patient not in ICU; the absence is itself
  signal). Recorded as a `missingness_policy` column in
  `feature_shortlist_v2.md`. **No imputation happens in BigQuery** —
  `cohort_features` stays raw with NaNs; the fit is deferred to F1
  (train-only) to avoid leakage.
- E7. **Output.** `docs/feature_shortlist_v2.md` listing kept /
  dropped / winsorized features with one-line justification each and
  the chosen `missingness_policy`, and
  `readmission.cohort_features_v2` rebuilt (or projected) accordingly.
  `FEATURES_VERSION` bumped to `v2`. Phase D re-run against `v2` to
  confirm the contract is still clean.

### Phase F — Modeling (`model_development.ipynb`)
- F1. **Imputer pipeline.** Build a `sklearn.compose.ColumnTransformer`
  from the E6 policy table: median imputers for the MCAR numerics,
  most-frequent for MCAR categoricals, `MissingIndicator + SimpleImputer`
  for MNAR features, identity passthrough for tree-native NaN columns.
  Fit on `train` only; transform applied to `val`/`test`/`demo`.
  Encoded as `src/imputation.py` (created here, not in Phase G).
- F2. Logistic regression (interpretable baseline) → gradient boosting
  (XGBoost / LightGBM) on the shortlisted feature set from Phase E.
- F3. All runs logged via `src.tracking`.
- F4. `GroupKFold(groups=subject_id)` for hyperparameter CV within
  train. Imputer + scaler refit per fold to keep CV honest.
- F5. Final scoring on `test`. `demo` untouched.

### Phase G — Refactor & containerize
- Extract: `src/data.py`, `src/features.py`, `src/train.py`,
  `src/evaluate.py`, `src/predict.py`. (`src/imputation.py` already
  exists from Phase F.)
- Dockerfile + `kfp` pipeline: build features → validate → select →
  train → evaluate → register → (optionally) deploy.

### Phase H — Demo simulation harness (separate app)
- Reads `demo` patients, attaches synthetic name/photo, calls the
  deployed endpoint, logs predictions and eventual outcomes for
  drift/accuracy dashboards.

## Settled decisions (2026-05-08)

| # | Decision                                           |
|---|----------------------------------------------------|
| 1 | Demo split = ~0.5% of patients.                    |
| 2 | Single experiment `readmission-30d`, run-naming convention `<family>-v<n>`. |
| 3 | Data validation tool: **Evidently OSS** (TFDV dropped — py3.12 wheels unavailable across local arm64, Docker+Rosetta, and Colab Enterprise). |
| 4 | `src/` layout deferred to Phase F **except** `src/config.py` and `src/tracking.py`, which are created in Phase B because every later run (notebook, pipeline step, demo app) shares them. |
| 5 | Splits + tracking in existing notebook; fork at features. |
| 6 | No GCS bucket yet — provision via a project-bootstrap script that also enables APIs, creates a service account, and provisions BigQuery + GCS resources. (To be authored before Phase B.) |
| 7 | No Feature Store yet — revisit when online serving is required. |
| 8 | EDA is **not** a standalone phase. Univariate EDA is subsumed by Phase D (Evidently's `DataSummaryPreset`); the remaining EDA work (bivariate vs label, redundancy, clinical plausibility) is fused with feature selection in Phase E to avoid duplicated histograms and to keep the audit trail in one place. |
| 9 | Missingness handling is split: **policy** is decided in Phase E (E6) per feature, recorded in `feature_shortlist_v2.md`; **execution** lives in `src/imputation.py` and is fit on `train` only in Phase F (F1). No imputation is performed in BigQuery / `cohort_features`. |

## Open / deferred items
- Project-bootstrap script (APIs, IAM, GCS bucket) — design before
  Phase B.
- Open question #9 (missingness imputation policy) — closed by Phase
  E (E6) output (per-feature policy table) + Phase F (F1) implementation
  in `src/imputation.py`.
