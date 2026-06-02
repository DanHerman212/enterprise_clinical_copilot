# Readmission Risk — ML Workflow

Predict the probability that a discharged patient is readmitted within 30 days, using MIMIC-IV.

## Phase 1 — Prediction Task

**1. Task.** Binary probabilistic classification: predict the probability of unplanned hospital readmission within 30 days of discharge. Time of prediction $t_0$ = index admission `dischtime`; the model may only use data with timestamp $\le t_0$.

**2. Cohort & exclusions.** Spine: `mimiciv_hosp.admissions` (not `icustays`). Inclusion: adult patients (age $\ge$ 18 at `admittime`) with valid non-null `admittime` and `dischtime`. Exclude index admissions with:
- Hospital mortality (`hospital_expire_flag = 1`).
- Transfer to another acute care facility.
- Left against medical advice (AMA), via `discharge_location`.
- Elective / planned procedures.
- Short stays — length of stay < 24h (observational / routine / administrative anomalies).
- Terminal / hospice discharges, via `discharge_location`.

**3. Label.** Binary: `1` if a qualifying readmission occurs ≤ 30 days after `dischtime`, else `0`. Construction rules:
- **Temporal-leakage prevention** — only data available at/before `dischtime` may inform a prediction.
- **Overlapping-stay filter** — merge administrative transfers / contiguous stays so they are not counted as readmissions.
- **Right-censoring** — exclude any index admission whose `dischtime` is within 30 days of the absolute maximum date in the entire MIMIC-IV dataset.

**4. Evaluation metric.** Single metric: **PR-AUC** (average precision). Applied to a common-sense baseline and improved by the ML model.

## Phase 2 — Data Representation

**Goal.** Attach a feature vector to each Phase 1 index admission, producing the modeling matrix `[hadm_id, subject_id, features…, label, split]`.

**Constraints (carried from Phase 1).**
- **Temporal cutoff** — every feature uses only data available at/before `dischtime` ($t_0$). Vitals/labs are aggregated over a rolling window before discharge (default: final 12–48h).
- **Leakage warning** — comorbidity indices (CCI / Elixhauser) are computed from **prior** admissions' diagnoses only (`admittime` strictly < index `admittime`), excluding index-admission ICD codes. First-ever admission → historical comorbidity scores = 0.

**Sourcing tiers.** _Derived_ = ready-made aggregation in `mimiciv_derived`; _Custom_ = bespoke SQL/aggregation required.

| Feature Category | Specific Features | Original MIMIC-IV Source | Derived Table Mapping | Tier | Predictive Rationale / Notes |
|---|---|---|---|---|---|
| Demographics | Age, Gender, Ethnicity/Race | `patients`, `admissions` | `icustay_detail` | Derived | Age has a non-linear relationship with physiological reserve. Race/ethnicity often proxy structural healthcare disparities. |
| Administrative / SES | Insurance Type (Medicare, Medicaid, Private), Marital Status | `admissions` | `icustay_detail` | Derived | Insurance is a potent SES proxy; correlates with medication adherence and outpatient follow-up access. |
| Encounter Context | Admission Type (Urgent, Emergent), Admission Location, Discharge Location | `admissions` | `icustay_detail` | Derived | Discharge location (home vs. SNF) strongly stratifies baseline dependency and immediate readmission vulnerability. |
| Historical Utilization | Prior Admission Count, Prior ED Visits | `admissions`, `edstays` | — | Custom | Aggregated historically by `subject_id`. Velocity of historical utilization is among the strongest readmission predictors. |
| Cardiovascular Vitals | Heart Rate, Systolic/Diastolic BP, Mean Arterial Pressure (MAP) | `chartevents` | `pivoted_vital`, `vital_first_day` | Derived | Persistent tachycardia/hypotension at discharge indicates unresolved shock or inadequate resuscitation. Extract min/max/mean/std/last over rolling window (final 12–48h). |
| Respiratory Vitals | Respiratory Rate, Oxygen Saturation (SpO2) | `chartevents` | `pivoted_vital`, `vital_first_day` | Derived | Hypoxia and tachypnea are sentinel signs of cardiopulmonary decompensation. SpO2 often a top-tier SHAP predictor. |
| Neurological & Other Vitals | Glasgow Coma Scale (GCS), Core Temperature | `chartevents` | `pivoted_gcs`, `pivoted_vital` | Derived | GCS quantifies neurological impairment / aspiration risk. Temperature deviations signal unresolved SIRS or infection. |
| Hematology (CBC) | Hemoglobin, Hematocrit, WBC, Platelets, RBC, RDW | `labevents`, `d_labitems` | `pivoted_lab`, `labs_first_day` | Derived | Anemia limits oxygen delivery. RDW/RBC correlate with 90-day readmission. WBC shows U-shaped risk curves. |
| Renal Function | BUN, Creatinine, eGFR, BUN-to-Creatinine Ratio | `labevents` | `pivoted_lab` | Derived | Indicators of renal perfusion. BUN:Creatinine ratio flags intravascular congestion — highly predictive in heart failure. |
| Metabolic & Electrolytes | Glucose, Sodium, Potassium, Chloride, Calcium, Bicarbonate, Anion Gap, Lactate | `labevents` | `pivoted_lab`, `pivoted_bg` | Derived | Electrolyte derangements prompt dysrhythmias. Lactate clearance failure precedes collapse. Glucose variability is prognostic. |
| Hepatic & Coagulation | AST, ALT, Bilirubin, Albumin, PT, PTT, INR | `labevents` | `pivoted_lab` | Derived | Coagulopathies elevate hemorrhagic risk. Hypoalbuminemia reflects malnutrition / chronic inflammation. |
| Clinical Risk Indices | Charlson (CCI), Elixhauser, SOFA, SAPS II | `diagnoses_icd` (historical) | `charlson`, `elixhauser_ahrq`, `sofa` | Derived | Standardized indices distill histories into risk scores; CCI/Elixhauser are paramount 30-day predictors. **Leakage:** compute from prior admissions only, excluding index-admission ICD codes. |
| Interventions | Mechanical Ventilation, Vasopressors (Epinephrine, Dopamine), Fluids | `procedureevents`, `inputevents` | `ventilation_durations`, `vasopressor_durations`, `pivoted_uo` | Derived | Prolonged ventilation correlates with deconditioning. Cumulative fluid balance tracks homeostatic disruption / pulmonary edema. |
| Polypharmacy & Complexity | Total Active Medication Count, Medication Regimen Complexity Index (MRCI) | `prescriptions` | — | Custom | High active-med count raises drug-interaction, non-adherence, and post-discharge ADE probability. |
| High-Risk Medication Classes | Antithrombotics, Insulin, Opioids, Loop Diuretics | `prescriptions`, `inputevents` | — | Custom | Filtered via NDC/ATC codes. Narrow therapeutic windows; consistently implicated in medication-related readmissions. |
| Potentially Inappropriate Meds | Benzodiazepines, Anticonvulsants, Antidepressants, PPIs, Psychoanaleptics | `prescriptions` | — | Custom | Filtered via Beers Criteria. Independently elevate fall risk, cognitive impairment, or physiological instability. |
| Therapeutic Dynamics | In-hospital Med Changes, Dose Titrations, Route (IV vs. Oral/Enteral), Standardized Exposure Duration | `prescriptions`, `inputevents` | — | Custom | Discontinuation/modification/initiation correlates with instability and care-transition failure. IV delivery indicates acute instability; exposure duration quantifies therapeutic intensity. |
| Unstructured Clinical Narratives | Chief Complaint, Discharge Summaries | `edstays`, `triage`, `note` | — | Custom (NLP/Embeddings) | Text encapsulates diagnostic uncertainty, social discharge barriers, and clinical nuance. Chief-complaint embeddings are dense predictors; discharge summaries carry continuity-of-care context. |

### ICU vs. Hospital-Wide Strategy

**Spine.** Build features off the `admissions`-based cohort spine so no "floor" patient is orphaned. Do not use the ICU-only `derived` schema as the spine.

**A. Hospital-Wide Backbone (available for every patient).**
- **Demographics & encounter:** `admissions`, `patients`.
- **Historical comorbidities (strict leakage control):** CCI / Elixhauser from `diagnoses_icd`, joining only codes whose `admittime` is strictly prior to the index `admittime`. First-ever admission → scores = 0.
- **Medications:** aggregated from `prescriptions` (hospital-wide pharmacy).
- **Labs:** aggregated from `labevents`, windowed to the last 48h before `dischtime`.

**B. ICU-Specific Enhancements (only for ICU patients).**
- **Indicator flag:** `has_icu_stay` = 1 if the index `hadm_id` appears in `icustays`, else 0.
- **Conditional joins:** left-join derived tables (`vital_first_day`, `sofa`, etc.) onto the spine via `icustay_id`.

**Missingness strategy.**
- **Domain-driven:** impute ICU severity scores (SOFA, SAPS) = 0 for non-ICU patients (normal baseline / no acute organ failure). Do **not** use median imputation (it assigns false ICU-level severity to stable floor patients).
- **Algorithmic:** leave derived vitals as raw `NaN` and feed them to a gradient-boosted tree (XGBoost / LightGBM). Combined with `has_icu_stay`, the model learns that this missingness is informative ("stable enough to stay on the general ward").

### Dataset Splits

Partition by **patient** (`subject_id`) before EDA/validation so no patient's admissions span multiple subsets (~223,452 patients / ~546k admissions). **Five disjoint splits**, all carved from the same deterministic patient hash so that no `subject_id` ever appears in more than one split.
- **Method:** `ABS(MOD(FARM_FINGERPRINT(CAST(subject_id AS STRING)), 10000))` → a deterministic, reproducible bucket (0–9999) per patient; all of a patient's admissions share one bucket.
- **Demo (disjoint partition, carved first):** buckets 0–44 ≈ 0.45% (fraction 0.0045) ≈ ~1,000 patients. Held out from train/val/test so evaluation metrics stay pure and the notes-filtered demo doesn't bias the test set. Powers the application/agent demo (prediction endpoint + agentic RAG over discharge notes).
- **Production test (disjoint partition, carved second):** buckets 45–94 ≈ 0.5% ≈ ~1,100 patients (≈ ~200 positive readmissions at the ~18% base rate). This is the **production holdout** — reserved now and **never touched by any model** during Phase 3. It is consumed only at deployment, to confirm that live endpoint predictions match ground-truth labels (acceptance test + drift / skew / accuracy monitoring). Sizing rationale: confidence on ranking metrics (AUPRC/AUROC) is driven by the *count of positive events*, not total rows; ~200 positives yields ≈ ±3% bootstrap CIs — tight enough to trust the pre-production acceptance gate while costing the train/test splits almost nothing.
- **Remainder (buckets 95–9999) split 70/15/15 by patient:** train / validation / test.
- Proportions apply to patients; per-split row counts are approximately proportional (~2.44 admissions/patient).
- **Demo selection** additionally requires available unstructured data (see below) so the RAG demo has notes to read.
- **Diagnostics (carved once, asserted every run):** per-split patient and admission counts, per-split readmission prevalence (expect ~18% across all five), and a hard assertion that the five buckets are mutually exclusive and exhaustive (zero patients span splits).

### Unstructured Data Inventory

Cursory coverage scan to confirm per-patient unstructured data for the agentic RAG demo and to validate the narrative features:
- **Discharge summaries** — `mimiciv_note.discharge`.
- **Radiology reports** — `mimiciv_note.radiology`.
- **ED chief complaint** — `mimiciv_ed.triage` / `edstays`.
- Report coverage (% of `hadm_id` / `subject_id` per source); note MIMIC-IV has no nursing/physician progress notes and coverage is partial. Prioritize demo patients who have a discharge summary.

### Exploratory Data Analysis (EDA)

Purpose-driven discovery on the **training split**; its output is a data profile plus a set of expectations that feed validation and Phase 3. Answers four questions:
- **Validity / plausibility** — ranges, units, impossible values → value-range & type expectations.
- **Missingness structure** — confirm non-ICU rows are ~100% missing on derived vitals/SOFA and that `has_icu_stay` explains it → per-feature missingness bounds; validates the imputation policy.
- **Signal for readmission** — target-stratified distributions, base-rate check, leakage smell-tests (suspiciously predictive features) → feature shortlist + leakage red-flags for Phase 3.
- **Cohort conformance** — class balance, per-exclusion row counts, demographic sanity → Table-1 characterization.

_Output:_ training-set data profile + expectations (ranges, missingness bounds, schema, base rate), reused as the validation reference.

### Data Validation (Evidently AI)

Operationalizes EDA's expectations as a repeatable, pass/fail gate run before Phase 3 (and reused in Phase 5 for production drift).
- **Schema/type checks** — columns, dtypes, categorical levels match spec.
- **Value-range & missingness tests** — thresholds discovered in EDA.
- **Split consistency** — train vs. val vs. test distribution similarity (catches bad splits / leakage).
- **Reference snapshot** — persist the training profile as the baseline reused for Phase 5 production-drift monitoring.

_Gate:_ validation must pass before feature matrix proceeds to modeling.

## Phase 3 — Model Training

**Goal.** Establish a clinical common-sense floor, then train a model with statistical power that **beats that floor on PR-AUC (average precision)**. Three artifacts are produced and compared head-to-head on the locked `test` split: (1) the **HOSPITAL score** clinical baseline, (2) an **untuned XGBoost** benchmark, (3) a **hyperparameter-tuned XGBoost** candidate. Every artifact is logged to the experiment tracker.

**Scope.** This phase delivers an orchestrated **Vertex AI / Agent-Platform training pipeline** that ends at a registered, evaluated model carrying a baseline-gate verdict. The actual real-time endpoint, production monitoring/drift, and live traffic are **out of scope** here and handled in Phase 4 / Phase 5. The `demo` and `production_test` splits are **not touched** in this phase.

### 3.0 — Split amendment (prerequisite)

Phase 2's split definition is extended from four buckets to **five** (see _Dataset Splits_): `train` / `val` / `test` / `demo` / `production_test`. The `production_test` holdout (~1,100 patients, ~200 positives) is carved **now** so it is provably unseen later, but it is reserved for Phase 4/5 and never enters any Phase 3 fit or evaluation. Re-run split diagnostics and assert mutual exclusivity before proceeding.

### 3.1 — Common-sense baseline: the HOSPITAL score (`test` split only)

**Why HOSPITAL.** A published, externally validated risk equation for 30-day readmission, computed entirely from data available **at discharge** — no training, no leakage. It is the number every learned model must beat. The trivial floor (base-rate AP ≈ prevalence ≈ 0.18) is reported alongside it so the value-add story is unambiguous: random = prevalence, HOSPITAL = clinical floor, ML = must beat both.

**Components (ordinal 0–13).**

| Letter | Component | Points | Source |
|---|---|---|---|
| **H** | Hemoglobin < 12 g/dL at discharge | 1 | Reuse — `*_last` hematology feature |
| **O** | Discharge from an **Oncology** service | 2 | Supplemental — `services` |
| **S** | Sodium < 135 mEq/L at discharge | 1 | Reuse — `*_last` chemistry feature |
| **P** | Any **procedure** during the index admission | 1 | Supplemental — `procedures_icd` |
| **I** | Index admission is **non-elective** (urgent/emergent) | 1 | Reuse — `admission_type` feature |
| **T** | Admissions in the **previous 365 days**: 0–1 → 0, 2–5 → 2, > 5 → 5 | 0 / 2 / 5 | Supplemental — 365-day window |
| **AL** | Length of stay **≥ 5 days** | 2 | Derive — `dischtime` − `admittime` |

Risk bands (for the secondary operating-point report only): **0–4** low, **5–6** intermediate, **≥ 7** high.

**Data sourcing — reuse + one supplemental query.** Four of seven components come straight from the engineered `test`-split feature matrix; the remaining three (O, P, T) are not currently engineered features and are fetched by **one supplemental BigQuery query keyed on the test `(subject_id, hadm_id)`**, then left-joined onto the test matrix. No fully separate dataset is built. The supplemental query inherits the same temporal-cutoff discipline ($\le$ `dischtime`); the 365-day prior-admission window (T) reuses the strict leakage rule from Phase 1/2 (source `admittime` strictly < index `admittime`).

**Scoring metric — continuous, not thresholded.** Average precision requires a **continuous ranking score**. The raw **0–13 HOSPITAL ordinal** is fed directly into `average_precision_score` (PR-AUC). The literature threshold (≥ 7 = high risk) is **not** used to compute AP — binarizing first would collapse the PR curve to a single degenerate point. The threshold is reported only as a **secondary operating point** (sensitivity / PPV / specificity / NPV at the ≥ 7 cut). Add a bootstrap 95% CI on the AP. Log the result as a tracked run.

### 3.2 — Benchmark model: untuned XGBoost (default params)

An out-of-the-box XGBoost classifier (default hyperparameters, native NaN handling per the Phase 2 missingness policy) trained on `train`, evaluated on `test`. This establishes "what a competent model with zero tuning achieves" — the second number the tuned candidate must improve upon, and a cheap leakage smell-test (a default model scoring suspiciously near-perfect signals a leak). Log as a tracked run.

### 3.3 — Candidate model: Optuna-tuned XGBoost

**Search engine.** Hyperparameter optimization uses **Optuna** (TPE sampler + Hyperband/median pruning) executing **inside a single Vertex / Agent-Platform custom job** — one container, all trials in-process. Optuna is chosen over the managed Vertex HPO tuning job (Vizier engine) because tabular XGBoost trials are cheap and numerous: the managed product's per-trial job provisioning overhead would dominate runtime with no accuracy benefit, whereas Optuna's in-process trials + aggressive pruning minimize compute, and the identical code runs locally in a notebook and unchanged in the container.

**Validation protocol.** `GroupKFold(groups=subject_id)` **within `train`** so no patient leaks across folds; the held-out `test` split is never seen during the search. XGBoost early stopping is paired with Optuna pruning to terminate weak trials early. Seed the study for reproducibility.

**Search space (initial).** `n_estimators`, `max_depth`, `learning_rate`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `gamma`, and `scale_pos_weight` (class imbalance, ~18% prevalence). CV objective = mean fold AUPRC.

**Refit & evaluate.** Retrain on the **full `train`** split with the best parameters, then score **once** on `test`. Log the study, the best trial's parameters, and the final test metrics as a tracked run.

### 3.4 — Model comparison & selection

All three artifacts are compared on the **same locked `test` split**:
- **Headline:** PR-AUC (average precision) — the Phase 1 evaluation metric. Prevalence is ~18%, so AUPRC is the decision metric.
- **Secondary:** AUROC, calibration (reliability curve + Brier score), and a chosen operating threshold with its confusion-matrix metrics.
- **Confidence:** bootstrap 95% CIs on each model's AUPRC, plus a **paired bootstrap** test of (tuned XGBoost − HOSPITAL) and (tuned − default) so the improvement over the floor is shown to exclude 0.

**Promotion criterion.** The candidate is accepted only if its test-AUPRC CI lies **above** the HOSPITAL floor and the paired-bootstrap difference vs HOSPITAL excludes 0.

### 3.4a — Pipeline architecture: two pipelines, a baseline gate, manual promotion

Phase 3 is delivered as a **Vertex AI / Agent-Platform pipeline**, deliberately kept **separate from deployment**:

- **Training pipeline (this phase).** Steps: data extraction → preprocessing → train default XGBoost (§3.2) → Optuna HPO (§3.3) → refit on best params → evaluate vs the HOSPITAL floor (§3.4) → **register the model + emit a promotion-decision artifact**. Re-run on every experiment; low-privilege (BigQuery + training compute only).
- **Deployment pipeline (Phase 4, separate).** Consumes a specific *chosen* registered model version: deploy to endpoint → acceptance test on `production_test` → enable monitoring → manual traffic promotion. Triggered deliberately by a person, not by every training run.

**The baseline gate (built now, controls a tag — not deployment).** A `dsl.Condition` step runs the §3.4 promotion criterion (candidate test-AUPRC CI above the HOSPITAL floor **and** paired-bootstrap difference excludes 0). On pass, the pipeline **stamps the registered model version with a `promotable` label** in the Model Registry and records the verdict in experiment tracking. It does **not** deploy. A human reviews `promotable` versions — checking calibration, operating-point PPV, and subgroup fairness, which are not yet automatable into a single boolean — and manually triggers the deployment pipeline on the version they choose.

**Rationale.** Beating HOSPITAL on AUPRC is *necessary but not sufficient* for a clinical model, so a human sign-off stays between training and go-live. Separating the pipelines keeps experimental training runs fast and low-privilege, and lets the `production_test` acceptance test sit naturally in the deploy pipeline. The gate **logic** is identical to what a future fully-automated continuous-training loop would use; maturing from "gate the tag" to "gate the deployment" then becomes a one-line change rather than a rearchitecture.

### 3.5 — Experiment tracking contract

Tracking runs on the **GCP Agent Platform** experiment tracker (Vertex AI Experiments; SDK surface unchanged from Vertex AI), single experiment **`readmission-30d`**, one run per artifact (`hospital-baseline`, `xgb-default`, `xgb-tuned`). Each run logs:
- **Params** — cohort/feature version, split sizes & prevalence per split, random seed, git SHA, model hyperparameters (or HOSPITAL component definitions).
- **Metrics** — test AUPRC (headline) + bootstrap CI, AUROC, Brier, secondary operating-point metrics.
- **Artifacts** — the supplemental-query SQL (baseline), Optuna study object / importance plots (tuned), reliability and PR curves.

### 3.6 — Correctness checks

Lightweight, fast assertions guarding the most error-prone steps (not yet the full pipeline CI suite, which lands with deployment in Phase 4):
- **HOSPITAL computation** — spot-check component scores against the published rubric on a handful of hand-verified admissions; assert score ∈ [0, 13].
- **Metric integrity** — assert AP is computed on the continuous score (PR curve has multiple distinct points, not a single degenerate one); assert the base-rate AP ≈ prevalence.
- **Leakage guards** — assert the `test` split is never referenced inside the Optuna CV loop; assert the supplemental 365-day window excludes the index admission.

## Phase 4 — Production Deployment (GCP)
_TBD — deploy the trained model to production on GCP._

## Phase 5 — Monitoring & Correctness
_TBD — monitor accuracy in production and adjust for correctness._
