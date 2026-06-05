 # Readmission Risk — ML Workflow

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

## Phase 2 — Data Representation

## Phase 3 — Model Training

## Phase 4 — Production Deployment

## Phase 5 — Monitoring & Correctness
