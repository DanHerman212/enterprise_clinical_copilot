# `baseline-v1` — interpretation

Companion to the `baseline-v1` run in the `readmission-30d` Vertex AI
experiment. Defines the floor that future model runs must clear.

## Numbers

Cohort: MIMIC-IV v3.1, acute admissions only, 30-day all-cause unplanned
readmission. Splits are deterministic on `subject_id`. Test
`n = 59,792`, prevalence `0.191`. Val `n = 60,874`, prevalence `0.185`.

| Scorer   | Test AUROC | Test AUPRC | Val AUROC | Val AUPRC |
|----------|-----------:|-----------:|----------:|----------:|
| LACE     | 0.647      | 0.289      | 0.644     | 0.281     |
| HOSPITAL | **0.683**  | **0.331**  | **0.675** | **0.315** |
| chance   | 0.500      | 0.191      | 0.500     | 0.185     |

Clinical-cutoff screening metrics (test split):

| Scorer   | Cutoff | Sens  | Spec  | PPV   | NPV   | Flag-rate |
|----------|-------:|------:|------:|------:|------:|----------:|
| LACE     | ≥ 10   | 0.674 | 0.542 | 0.258 | 0.875 | 0.499     |
| HOSPITAL | ≥ 7    | 0.295 | 0.901 | 0.414 | 0.844 | 0.136     |

## What the numbers say

- **HOSPITAL is the headline baseline.** It strictly dominates LACE on
  AUROC and AUPRC on both splits.
- **AUPRC lift over chance is modest.** HOSPITAL ≈ 1.73×, LACE ≈ 1.51×.
  Real headroom for a learned model.
- **The two cutoffs are different operating points, not comparable
  side-by-side.** LACE ≥ 10 is a wide-net rule (flags ~50% at PPV 0.26);
  HOSPITAL ≥ 7 is a narrow-net rule (flags ~14% at PPV 0.41). Use AUROC
  and AUPRC for ranking comparisons; use the cutoff metrics only at
  matched flag-rates.
- **Calibration is unmeasured.** LACE and HOSPITAL are ordinal scores,
  not probabilities. Calibration metrics will be added when the first
  probabilistic model is logged.

## Bar for `<model>-v1` runs

To claim a real win over HOSPITAL on the test split:

1. Beat HOSPITAL on AUROC and AUPRC — target roughly **≥ 0.70 AUROC**
   and **≥ 0.35 AUPRC**.
2. At a flag-rate matched to `hospital_test_flag_rate ≈ 0.136`,
   match or improve **sensitivity (0.295)** and **PPV (0.414)**.
3. Show better calibration once calibration is in scope.

Improving (1) without (2) means more discriminative on paper, no better
at the bedside — reported, not celebrated.
