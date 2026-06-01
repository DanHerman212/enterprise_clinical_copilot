# Readmission Risk — ML Workflow

Predict the probability that a discharged patient is readmitted within 30 days, using MIMIC-IV.

## Phase 1 — Prediction Task

**1. Task.** Binary probabilistic classification: predict the probability of unplanned hospital readmission within 30 days of discharge.

**2. Cohort & exclusions.** Source: `mimiciv_hosp.admissions`, all patients. Exclude index admissions with:
- Hospital mortality (in-hospital death).
- Transfer to another acute care facility.
- Left against medical advice (AMA).
- Elective / planned procedures.

**3. Label.** Binary: `1` if a qualifying readmission occurs ≤ 30 days after `dischtime`, else `0`. Construction rules:
- **Temporal-leakage prevention** — only data available at/before `dischtime` may inform a prediction.
- **Overlapping-stay filter** — merge administrative transfers / contiguous stays so they are not counted as readmissions.
- **Right-censoring** — exclude any index admission whose `dischtime` is within 30 days of the absolute maximum date in the entire MIMIC-IV dataset.

**4. Evaluation metric.** Single metric: **PR-AUC** (average precision). Applied to a common-sense baseline and improved by the ML model.

## Phase 2 — Data Representation
_TBD — build the feature representation from the Phase 1 cohort and label._

## Phase 3 — Model Training
_TBD — train a model with statistical power that beats the common-sense baseline on PR-AUC._

## Phase 4 — Production Deployment (GCP)
_TBD — deploy the trained model to production on GCP._

## Phase 5 — Monitoring & Correctness
_TBD — monitor accuracy in production and adjust for correctness._
