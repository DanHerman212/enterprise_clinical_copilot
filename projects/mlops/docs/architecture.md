# MLOps — Architecture

The **HOW (design)** for the readmission-risk MLOps project: the exact components and services behind each workflow phase. See the master architecture for shared foundation; see [workflow.md](workflow.md) for the WHAT.

## Diagram

```mermaid
flowchart TB
    subgraph P2["Phase 2 — Data Representation"]
        BQ["BigQuery<br/>(source tables)"]
        DF["Dataform<br/>(ELT)"]
        SPLIT["Deterministic Split<br/>(FARM_FINGERPRINT)"]
        EV["Evidently AI<br/>(data validation gate)"]
        BQ --> DF --> SPLIT --> EV
    end

    subgraph P3["Phase 3 — Model Training"]
        PIPE["Vertex AI Pipelines"]
        HPO["Optuna<br/>(HPO)"]
        SHAP["SHAP<br/>(interpretability + fairness)"]
        REG["Model Registry"]
        PIPE --> HPO --> SHAP --> REG
    end

    subgraph P4["Phase 4 — Production Deployment"]
        ART["Serving Artifact<br/>(model + transforms)"]
        EP["Vertex Endpoint"]
        XAI["Explainable AI<br/>(Sampled Shapley)"]
        ART --> EP --> XAI
    end

    subgraph P5["Phase 5 — Monitoring & Correctness"]
        MON["Vertex Model Monitoring<br/>(input + attribution drift)"]
        EVAL["Scheduled Eval Job<br/>(AUCPR vs baseline)"]
        PS["Pub/Sub<br/>(retraining signal)"]
        MON --> PS
        EVAL --> PS
    end

    EV --> PIPE
    REG --> ART
    XAI --> MON
    XAI --> EVAL
```

## Components by Phase

### Phase 2 — Data Representation

_Placeholder — BigQuery datasets, Dataform ELT, the split implementation, Evidently validation._

### Phase 3 — Model Training

_Placeholder — Vertex AI Pipelines, Optuna HPO, model registry, SHAP analysis._

### Phase 4 — Production Deployment

_Placeholder — serving artifact, Vertex endpoint, Explainable AI (Sampled Shapley) config._

### Phase 5 — Monitoring & Correctness

_Placeholder — Vertex Model Monitoring, scheduled evaluation job, Pub/Sub retraining signal._

---
