# MLOps — Architecture

The **HOW (design)** for the readmission-risk MLOps project: the exact components and services behind each workflow phase. See the master architecture for shared foundation; see [workflow.md](workflow.md) for the WHAT.

## Diagram

```mermaid
flowchart TB
    subgraph P2["Phase 2 — Data Representation"]
        direction LR
        BQ["BigQuery<br/>(MIMIC-IV sources)"]
        DF["Dataform<br/>(ELT: cohort → features → mart)"]
        SPLIT["Deterministic Split<br/>(FARM_FINGERPRINT on subject_id)"]
        EV["Evidently AI<br/>(data quality + drift gate)"]
        BQ --> DF --> SPLIT --> EV
    end

    subgraph P3["Phase 3 — Model Training (Vertex AI Pipelines)"]
        direction TB
        LOAD["load-data<br/>(train + val + test)"]
        IMPUTE["impute<br/>(ColumnTransformer, fit train only)"]

        subgraph FS["Feature Selection"]
            TIER1["Tier 1: filter, LASSO, LGBM<br/>(parallel)"]
            AGG1["aggregate-tier1"]
            TIER2["Tier 2: RFE, Boruta<br/>(parallel, on reduced set)"]
            AGG2["aggregate-final<br/>(vote across 5 methods)"]
            TIER1 --> AGG1 --> TIER2 --> AGG2
        end

        HOSPITAL["hospital-score<br/>(AUCPR baseline)"]
        BENCH["benchmark-xgboost<br/>(default params)"]
        HPO["optuna-hpo<br/>(maximize AUCPR)"]
        TRAIN["train-final<br/>(best params)"]
        FS_STORE["create-feature-store<br/>(Vertex Feature Store)"]
        SHAP["shap-explain<br/>(global + per-patient)"]
        FAIRNESS["fairness-audit<br/>(NPV/PPV by subgroup)"]
        REG["register-model<br/>(model + transforms +<br/>feature store → registry)"]

        LOAD --> IMPUTE
        IMPUTE --> FS
        IMPUTE --> HOSPITAL
        FS --> BENCH
        FS --> FS_STORE
        HOSPITAL --> BENCH
        BENCH --> HPO --> TRAIN
        TRAIN --> SHAP
        TRAIN --> FAIRNESS
        SHAP --> REG
        FAIRNESS --> REG
        FS_STORE --> REG
    end

    subgraph P4["Phase 4 — Production Deployment"]
        direction LR
        ART["Serving Artifact<br/>(model + imputer + scaler<br/>+ feature list + feature store config)"]
        EP["Vertex Endpoint"]
        XAI["Explainable AI<br/>(Sampled Shapley)"]
        ART --> EP --> XAI
    end

    subgraph P5["Phase 5 — Monitoring & Correctness"]
        direction LR
        MON["Vertex Model Monitoring<br/>(input + attribution drift)"]
        EVAL["Scheduled Eval Job<br/>(AUCPR on matured labels)"]
        PS["Pub/Sub<br/>(retraining signal)"]
        MON --> PS
        EVAL --> PS
    end

    P2 --> P3
    P3 --> P4
    P4 --> P5
    P5 -.retraining signal.-> P3
```

## Components by Phase

### Phase 2 — Data Representation

BigQuery stores the mirrored MIMIC-IV source tables. Dataform executes the ELT DAG: `sources → staging (cohort, split) → features → features_clean → analytics_dataset`. The split uses `FARM_FINGERPRINT(subject_id)` for deterministic, patient-level assignment. Built-in Dataform assertions enforce non-null keys, valid split names, and disjoint patient groups. Evidently AI runs as a post-build gate, comparing current data distributions against a reference baseline and blocking the pipeline if drift or quality thresholds are breached.

### Phase 3 — Model Training

Runs as a Vertex AI Pipeline. The DAG proceeds through: data load → imputation (ColumnTransformer fit on train only) → feature selection (5 methods in two parallel tiers, vote-aggregated) in parallel with the HOSPITAL clinical baseline. Once the feature shortlist is determined, the pipeline forks: the training chain (benchmark XGBoost gated to beat HOSPITAL → Optuna HPO with TPE sampler and median pruner → final XGBoost training → SHAP interpretability → fairness audit) runs in parallel with Vertex AI Feature Store creation (registers the selected features and ingests the offline values for online serving). Both forks converge at model registration, where the model, imputer, scaler, feature list, and feature store reference are bundled into a single serving artifact.

### Phase 4 — Production Deployment

_Placeholder — serving artifact, Vertex endpoint, Explainable AI (Sampled Shapley) config._

### Phase 5 — Monitoring & Correctness

_Placeholder — Vertex Model Monitoring, scheduled evaluation job, Pub/Sub retraining signal._

---
