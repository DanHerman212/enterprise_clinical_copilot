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
_TBD — train a model with statistical power that beats the common-sense baseline on PR-AUC._

## Phase 4 — Production Deployment (GCP)
_TBD — deploy the trained model to production on GCP._

## Phase 5 — Monitoring & Correctness
_TBD — monitor accuracy in production and adjust for correctness._
