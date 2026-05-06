# Session summary — cohort & baselines

## What we did
1. **Built v1 cohort** (`cohort_admissions`) — one row per patient (first eligible stay only). 179K rows, 12.1% prevalence.
2. **Built label, LACE, HOSPITAL, combined baselines** — evaluated with sklearn AUROC/AUPRC.
3. **Got suspect results**: LACE 0.608 / 0.169, HOSPITAL 0.622 / 0.183 — below the published 0.60–0.68 band.
4. **Diagnosed** via component diagnostics: LACE-E mean ≈ 0.05, HOSPITAL-A fire rate ≈ 0.7%. The "first stay only" rule mathematically forces every prior-utilization signal to zero.
5. **Rebuilt v2 cohort** as one row per *eligible admission* (no per-patient collapse). Added explicit leakage discipline: train/val/test splits must be `GroupKFold` on `subject_id`.
6. **Reran** label, LACE, HOSPITAL, baselines.

## v1 → v2 results

|                       | v1 (first stay) | v2 (all admissions) |
|-----------------------|-----------------|---------------------|
| Rows                  | 179,224         | 406,958             |
| Distinct patients     | 179,224         | 179,224             |
| Mean admits / patient | 1.00            | 2.27                |
| Prevalence            | 12.13%          | 18.63%              |
| LACE-E mean           | 0.05            | 0.50                |
| HOSPITAL-A fire rate  | 0.7%            | 24.3%               |
| LACE AUROC / AUPRC    | 0.608 / 0.169   | **0.648 / 0.282**   |
| HOSPITAL AUROC / AUPRC| 0.622 / 0.183   | **0.678 / 0.320**   |

## What we learned
- **The cohort rule is part of the model.** "First stay only" looks clean but deletes the strongest predictor in the readmission literature (prior utilization) by construction.
- **Diagnose at the component level, not just the score.** Near-zero LACE-E / HOSPITAL-A immediately localized the defect; the headline AUROC alone wouldn't have.
- **Patient-level leakage must be handled at the split, not the cohort.** Keeping all admissions + `GroupKFold(subject_id)` is the correct factoring.
- **HOSPITAL > LACE** on MIMIC-IV at ~0.68 vs ~0.65 AUROC — matches published reference results. This is now our discriminative floor: an ML model must clear ~0.70 AUROC / ~0.34 AUPRC to claim real lift.
- **Saturated components are not bugs.** LACE-A / HOSPITAL-I near-100% reflects BIDMC's acute case mix, not a coding error.
