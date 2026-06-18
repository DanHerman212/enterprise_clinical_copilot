# Feature Selection Results — June 15, 2026

## What was done

Five selection methods voted on 74 features from the readmission dataset. Features with 3+ votes were kept, features with exactly 2 were flagged for review, and features with 0–1 were dropped.

## Results

| Decision | Count |
|---|---|
| **Keep** | 20 |
| Review | 21 |
| Drop | 33 |

## Kept features (20)

| Feature | Votes (of 5) |
|---|---|
| prior_admission_count | 5 |
| prior_inpatient_days | 5 |
| medication_order_count | 5 |
| hemoglobin_last | 5 |
| admission_type | 4 |
| discharge_location | 4 |
| recent_ed_visits | 4 |
| medication_count | 4 |
| rdw_max | 4 |
| rdw_min | 4 |
| hemoglobin_max | 4 |
| sodium_min | 4 |
| oncology_flag | 4 |
| age | 3 |
| index_los_days | 3 |
| diagnosis_count | 3 |
| rbc_last | 3 |
| glucose_last | 3 |
| glucose_max | 3 |
| monocytes_min | 3 |

## The five methods

1. **Filter** — statistical correlation with the readmission label
2. **LASSO** — linear model that shrinks weak features to zero (val AUCPR: 0.36)
3. **LightGBM** — tree model that ranks features by importance (val AUCPR: 0.40)
4. **RFE** — recursively removes the weakest feature until accuracy drops
5. **Boruta** — keeps features that beat randomized shadow copies

## How to review

**Artifacts** (local):
```
projects/mlops/artifacts/feature_selection/20260615t140808/
  feature_shortlist.csv     ← final decisions
  filter_importance.csv     ← statistical scores
  lasso_coefficients.csv    ← linear model weights
  lgbm_gain.csv             ← tree importance
  rfe_ranking.csv           ← elimination order
  boruta_decisions.csv      ← shadow-feature verdicts
```

**Experiment** (Vertex AI):
Open Vertex AI Experiments → `readmission-mlops` (us-east1). Each method has its own run with metrics and parameters.

## HOSPITAL baseline (for context)

AUCPR = 0.3325. The LightGBM on all 74 features achieved 0.4001 — already above the baseline.
