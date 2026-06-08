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

## Configuration

`dataform.json` holds only **non-secret** config (the public PhysioNet source
IDs, output schema, and location).

`defaultDatabase` is intentionally left as a placeholder (`your-gcp-project-id`)
— the real GCP project ID is **never committed**. Because this runs on
GCP-managed Dataform, set the project per environment via a **compilation
override** on the release/workspace configuration (Dataform console → release
configuration → compilation overrides → Google Cloud project). The override
lives in GCP, not git, mirroring the repo's `.env` convention for the shell
scripts.

## Running this yourself

Nothing in this repo hardcodes a personal GCP project, so a new user supplies
their own in three places. Prerequisite first, then config:

1. **MIMIC-IV access (the gatekeeper).** Complete PhysioNet credentialing for
   MIMIC-IV v3.1 and link your Google account to the `physionet-data` BigQuery
   project. Without this, no query against the source tables will run.
2. **Authenticate locally** (for the prototyping notebook):
   ```bash
   gcloud auth application-default login
   gcloud auth application-default set-quota-project <your-gcp-project>
   ```
3. **Set your project for the notebook.** Copy `.env.example` -> `.env` at the
   repo root and set `PROJECT_ID=<your-gcp-project>`. The notebook reads this
   (or a `PROJECT_ID` env var) for its billing project — `.env` is gitignored.
4. **Set your project for Dataform.** In the Dataform console, set a
   **compilation override** → Google Cloud project = `<your-gcp-project>`
   (schema suffix / table prefix left blank). This replaces the
   `your-gcp-project-id` placeholder in `dataform.json` at compile time.

The notebook (step 2–3) and Dataform (step 4) are independent: the notebook
talks to BigQuery directly, Dataform writes tables via its override.

## Status

Skeleton only — models contain placeholder SQL. No real logic yet, nothing
has been run.
