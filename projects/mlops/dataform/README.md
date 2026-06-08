# Dataform — Data Representation (Phase 2)

Builds the model-ready readmission dataset as a BigQuery DAG. Each `.sqlx`
defines one table/view; `ref()` calls wire the dependency order, so Dataform
sequences the build — no manual ordering.

## Layers

```
definitions/
  sources/    declarations of raw MIMIC-IV tables (no logic)
  staging/    cohort filter -> deterministic split
  features/   feature engineering -> missingness handling
  marts/      final model-ready table the training pipeline reads
  assertions/ data-contract checks (schema, ranges, null policy)
```

## Build order (derived from ref())

```
sources -> cohort -> cohort_split -> features -> features_clean -> analytics_dataset
```

The split happens **before** feature engineering and missingness so any
train-derived statistic (e.g. an imputation median) is computed from training
rows only — never validation/test. This is the core leakage guard.

## Status

Skeleton only — models contain placeholder SQL. No real logic yet, nothing
has been run.
