# Definitions — Dataform ELT

The semantic layer: a Dataform project that turns raw MIMIC-IV tables into the
analysis-ready, encoded dataset the model trains on. This is the single source of truth for
cohort, split, and feature logic.

## DAG

```mermaid
flowchart LR
    subgraph SRC["sources"]
        ADM["admissions"]
        PAT["patients"]
        LAB["labevents"]
        DIA["diagnoses_icd"]
        PRC["procedures_icd"]
        RX["prescriptions"]
        ED["edstays"]
    end
    subgraph STG["staging"]
        COH["cohort"]
        SPL["cohort_split"]
    end
    subgraph FEA["features"]
        F1["feat_demographics"]
        F2["feat_utilization"]
        F3["feat_codes"]
        F4["feat_medications"]
        F5["feat_labs"]
        F6["feat_oncology"]
        FEAT["features"]
        CLEAN["features_clean"]
    end
    subgraph MAR["marts"]
        AN["analytics_dataset"]
        ENC["analytics_dataset_encoded"]
    end
    SRC --> STG --> FEA
    F1 --> FEAT
    F2 --> FEAT
    F3 --> FEAT
    F4 --> FEAT
    F5 --> FEAT
    F6 --> FEAT
    FEAT --> CLEAN --> AN --> ENC
```

## Structure

| Directory | Purpose |
|---|---|
| `sources/` | Declarations for the mirrored MIMIC-IV BigQuery tables |
| `staging/` | `cohort.sqlx` (inclusion/exclusion), `cohort_split.sqlx` (deterministic split) |
| `features/` | `feat_*.sqlx` — one view per feature family; `features.sqlx` + `features_clean.sqlx` |
| `marts/` | `analytics_dataset.sqlx` (analysis-ready) + `analytics_dataset_encoded.sqlx` (one-hot, for training) |
| `assertions/` | `split_is_disjoint.sqlx` — blocking split-integrity gate |
| `baselines/` | `hospital_score.sqlx` — the HOSPITAL clinical baseline |

## Key guarantees

- **Cohort** is inpatient-only (observation/same-day/elective stays excluded), discharged
  alive, length of stay ≥ 1 day.
- **Split** is deterministic and patient-level (`FARM_FINGERPRINT(subject_id)`), so a
  patient never spans train/test — enforced by the `split_is_disjoint` assertion.

## Running

```bash
npx --yes @dataform/cli@3.0.0 install
npx --yes @dataform/cli@3.0.0 run
```

See [`docs/data-pipeline.md`](../docs/data-pipeline.md) for the full data-representation
story.
