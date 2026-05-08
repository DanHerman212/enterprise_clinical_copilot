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
3. **Missingness handled by a validation step.** TFDV report after
   feature engineering drives per-feature imputation policy. HOSPITAL
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
- D1. TFDV over `cohort_features`: schema, stats, missingness, drift
  reference saved to GCS.
- D2. Imputation policy decisions encoded in `src/imputation.py`
  (created in Phase F).

### Phase E — Modeling (`model_development.ipynb`)
- Logistic regression (interpretable baseline) → gradient boosting
  (XGBoost / LightGBM).
- All runs logged via `src.tracking`.
- Train-only fit for imputers/scalers.
- `GroupKFold(groups=subject_id)` for hyperparameter CV within train.
- Final scoring on `test`. `demo` untouched.

### Phase F — Refactor & containerize
- Extract: `src/data.py`, `src/features.py`, `src/imputation.py`,
  `src/train.py`, `src/evaluate.py`, `src/predict.py`.
- Dockerfile + `kfp` pipeline: build features → validate → train →
  evaluate → register → (optionally) deploy.

### Phase G — Demo simulation harness (separate app)
- Reads `demo` patients, attaches synthetic name/photo, calls the
  deployed endpoint, logs predictions and eventual outcomes for
  drift/accuracy dashboards.

## Settled decisions (2026-05-08)

| # | Decision                                           |
|---|----------------------------------------------------|
| 1 | Demo split = ~0.5% of patients.                    |
| 2 | Single experiment `readmission-30d`, run-naming convention `<family>-v<n>`. |
| 3 | Data validation tool: TFDV.                        |
| 4 | `src/` layout deferred to Phase F **except** `src/config.py` and `src/tracking.py`, which are created in Phase B because every later run (notebook, pipeline step, demo app) shares them. |
| 5 | Splits + tracking in existing notebook; fork at features. |
| 6 | No GCS bucket yet — provision via a project-bootstrap script that also enables APIs, creates a service account, and provisions BigQuery + GCS resources. (To be authored before Phase B.) |
| 7 | No Feature Store yet — revisit when online serving is required. |

## Open / deferred items
- Project-bootstrap script (APIs, IAM, GCS bucket) — design before
  Phase B.
- Open question #9 (missingness imputation policy) — closed by Phase
  D output.
