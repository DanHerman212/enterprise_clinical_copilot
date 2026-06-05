 # Readmission Risk — ML Workflow

## Table of Contents

- [Phase 1 — Prediction Task](#phase-1--prediction-task)
  - [1. Inputs (Features and Entities)](#1-inputs-features-and-entities)
  - [2. Output (Target Variable)](#2-output-target-variable)
  - [3. Time Horizon and Wait Period](#3-time-horizon-and-wait-period)
  - [4. Baseline Heuristic](#4-baseline-heuristic)
- [Phase 2 — Data Representation](#phase-2--data-representation)
  - [1. Cohort Definition](#1-cohort-definition)
  - [2. Data Splitting](#2-data-splitting)
  - [3. Core Features](#3-core-features)
  - [4. Data Transformation Strategies](#4-data-transformation-strategies)
  - [5. Feature Selection Process](#5-feature-selection-process)
  - [6. Data Validation](#6-data-validation)
- [Phase 3 — Model Training](#phase-3--model-training)
  - [1. Common-Sense Baseline](#1-common-sense-baseline)
  - [2. Benchmark Model](#2-benchmark-model)
  - [3. Hyperparameter Optimization](#3-hyperparameter-optimization)
  - [4. Final Training](#4-final-training)
  - [5. Interpretability](#5-interpretability)
  - [6. Fairness](#6-fairness)
- [Phase 4 — Production Deployment](#phase-4--production-deployment)
  - [1. Serving Artifact](#1-serving-artifact)
  - [2. Endpoint Deployment](#2-endpoint-deployment)
  - [3. Production Validation](#3-production-validation)
  - [4. Promotion](#4-promotion)
- [Phase 5 — Monitoring & Correctness](#phase-5--monitoring--correctness)
  - [1. Holdout Evaluation Set](#1-holdout-evaluation-set)
  - [2. Input Monitoring](#2-input-monitoring)
  - [3. Outcome Monitoring](#3-outcome-monitoring)
  - [4. Retraining Signal](#4-retraining-signal)

## Phase 1 — Prediction Task

### 1. Inputs (Features and Entities)

#### Entity

Individual patients admitted to Beth Israel Deaconess Medical Center, including their complete electronic health record (EHR).

#### Features

_Placeholder — to be back-filled after the feature selection process is finalized._

### 2. Output (Target Variable)

Classification identifying whether a patient has high or low readmission risk.

### 3. Time Horizon and Wait Period

#### Forecast Origin

The prediction is made at discharge time.

#### Prediction Horizon

Readmission risk is predicted for the 30 days following discharge.

#### Wait Time

0 days (no blackout). The 30-day horizon opens immediately at discharge. Administrative transfers and contiguous stays are excluded in the label definition, so the horizon itself requires no gap period.

### 4. Baseline Heuristic

The HOSPITAL risk equation, measured with AUCPR (average precision). This is the baseline the machine learning model must beat to be considered valid.

---

## Phase 2 — Data Representation

### 1. Cohort Definition

#### Inclusion

- Adult inpatient admissions at Beth Israel Deaconess Medical Center recorded in MIMIC-IV (2008–2019).
- Patient discharged alive.

#### Exclusion

- Death during the index admission (readmission is impossible).
- Length of stay under one day (not a true inpatient admission).
- Administrative transfers and contiguous stays (counted as one episode, not a readmission).
- Planned follow-up visits (scheduled readmissions are not unplanned readmission events).

### 2. Data Splitting

Once the cohort is final, split it into five disjoint groups: training, validation, testing, production endpoint test, and a demo holdout set. Assignment is deterministic — `FARM_FINGERPRINT(subject_id)` taken modulo the bucket count — so the split is stable across runs and reproducible by anyone. Splitting on `subject_id` (not admission) keeps every admission for a given patient in a single group, preventing leakage of patient-specific signal across the train/validation/test boundary.

### 3. Core Features

| Feature Group | Individual Features | MIMIC-IV Table | Description |
|---|---|---|---|
| Demographics & Admin | Age, gender, marital status, language, ethnicity, admission type, insurance, discharge location | `patients`, `admissions` | Baseline profile capturing physical vulnerability and socioeconomic resources. |
| Historical Utilization | Prior admission count, total prior inpatient days, recent ED visits, index length of stay | `admissions`, `edstays` | Past healthcare use — the strongest signal of chronic illness and recurring risk. |
| Structured Clinical Codes | ICD-9/10 diagnosis codes, procedure codes, total condition count | `diagnoses_icd`, `procedures_icd` | What conditions the patient has and which interventions they received. |
| Medications | Drug name, dose, route, distinct-drug count (polypharmacy), high-risk drug flags | `prescriptions` | Treatment complexity and risk of adverse drug events after discharge. |
| Physiology & Labs | Heart rate, blood pressure, glucose, red blood cell count, RDW, monocytes | `labevents`, `chartevents` | Physical stability of the patient in the period before discharge. |
| Unstructured Text Notes | Discharge summaries, radiology reports | `discharge`, `radiology` | Clinician reasoning and human context — frailty, confusion, social barriers. |

### 4. Data Transformation Strategies

_Placeholder — to be filled after the full feature set is reviewed._

### 5. Feature Selection Process

| Feature Selection Type | Tools Used | Description of Method |
|---|---|---|
| Filter | Spearman correlation, Chi-squared, information gain | Runs basic statistical tests before any model trains to check whether a feature has a measurable link to readmission. Cheapest and broadest; drops obvious dead weight but cannot see feature interactions. |
| Embedded (linear) | LASSO regression (L1 penalty) | Selects features while training a linear model, shrinking weak features' coefficients to zero. Clean and interpretable but assumes roughly linear effects and is sensitive to scaling and collinearity. |
| Embedded (tree) | Random Forest, XGBoost, LightGBM | Ranks features by how well they split readmitted from non-readmitted patients while the model trains. Handles non-linear, threshold-driven signals and mixed feature types natively. |
| Wrapper (recursive) | Recursive Feature Elimination (RFE) | Repeatedly trains the model, removes the weakest features each round, and retrains until accuracy stops improving. Precise but expensive and prone to overfitting the selection. |
| Wrapper (Boruta) | Boruta | Keeps only features that consistently outperform randomized "shadow" copies of themselves. A defensible final confirmation pass; safer than RFE but still computationally heavy. |

### 6. Data Validation

Define the data contract for the constructed dataset — schema, allowed value ranges, null policy, and the reference distribution baseline. Evidently AI generates the data-quality and drift report that checks each new dataset against this baseline. The contract is defined here once and executed as a blocking gate inside the ML pipeline: a breach fails the run before training. The same baseline is the reference used by Phase 5 input monitoring at serving time.

---

## Phase 3 — Model Training

Training will be initiated through automated ML pipelines on Gemini Agent Platform Pipelines with the full experiment and metadata tracking suite.

### 1. Common-Sense Baseline

Compute the HOSPITAL score for every patient in the cohort, use that score to predict readmission risk, then evaluate the predictions with AUCPR. The resulting value is the common-sense baseline metric the machine learning model must beat.

### 2. Benchmark Model

Train an XGBoost model with default parameters and measure performance with AUCPR. This benchmark establishes the model's starting point above the common-sense baseline.

### 3. Hyperparameter Optimization

Run hyperparameter optimization with Optuna, selecting the parameter set that maximizes AUCPR.

### 4. Final Training

Train the final XGBoost model using the best parameters from optimization. Save the final model as an artifact in the model registry.

### 5. Interpretability

Run SHAP on the final model to attribute each feature's contribution to the predictions. Produce global importance to confirm the model relies on clinically sensible signals, and per-patient explanations to support clinician review. These explanations are also surfaced at serving time (see Phase 5).

### 6. Fairness

Evaluate the final model across demographic subgroups (e.g., race, ethnicity, age, gender, insurance type). For each subgroup, compute Negative Predictive Value (NPV) and Positive Predictive Value (PPV) and compare them across groups. Large gaps indicate the model is more reliable for some populations than others. The model passes the fairness check only when NPV and PPV hold within an acceptable tolerance across all subgroups.

---

## Phase 4 — Production Deployment

### 1. Serving Artifact

During the training pipeline run, save the model together with its preprocessing and postprocessing logic as a single serving artifact in the model registry. Bundling the transforms with the model guarantees training and serving apply identical logic, preventing training/serving skew.

### 2. Endpoint Deployment

Build the server container from the registered artifact and deploy it to a real-time inference endpoint. Configure Vertex AI Explainable AI with Sampled Shapley so the endpoint returns per-prediction feature attributions alongside each risk score, giving clinicians the top features driving an individual patient's prediction.

### 3. Production Validation

Run the locked Phase 3 holdout test set through the live endpoint and confirm two things: prediction parity with the offline results (catches serving bugs and skew), and AUCPR that clears the HOSPITAL baseline (confirms model quality).

### 4. Promotion

Mark the validated model version as production-ready in the registry once both checks pass.

---

## Phase 5 — Monitoring & Correctness

Because the readmission label only matures 30 days after discharge, monitoring runs on two clocks: input monitoring detects change immediately at serving time, while outcome monitoring measures model erosion once ground-truth labels arrive. A breach on either track emits a single retraining signal.

### 1. Holdout Evaluation Set

A fixed cohort of ~1,000 patients with known outcomes serves as the monitoring population. Their inputs, predictions, and Sampled Shapley attributions are logged at serving time, and their matured labels support outcome evaluation without waiting on a live event feed.

### 2. Input Monitoring

Vertex AI Model Monitoring compares live serving inputs against the registered training baseline to detect feature skew/drift and attribution drift, using per-feature statistical-distance thresholds. This is immediate and requires no labels — it answers whether the inputs or the model's reasoning have shifted away from training conditions.

### 3. Outcome Monitoring

A scheduled evaluation job joins matured labels to the stored predictions and recomputes AUCPR against the HOSPITAL baseline (model erosion), the label base rate against training prevalence (label drift), and treats confirmed input/attribution drift as an early proxy for concept drift. This track is lagged by the 30-day label horizon.

### 4. Retraining Signal

A threshold breach on either track publishes a message to a retraining-signal topic identifying the change in inputs or the erosion in correctness. The message is the deliverable: a downstream subscriber would trigger retraining, but that automation is decoupled and out of scope for the demo.

---
