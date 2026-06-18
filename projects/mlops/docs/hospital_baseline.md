# HOSPITAL Score Baseline — Computation & Validation

**Version:** 1.0  
**Date:** 2026-06-15  
**Experiment:** `readmission-mlops` (us-east1)  
**Result:** AUCPR = 0.3325

---

## 1. Purpose

This document describes, in complete and auditable detail, how the HOSPITAL score is computed for every patient in the MIMIC-IV cohort and how its predictive performance is measured. The HOSPITAL score serves as the **clinical baseline** — the floor that any machine learning model must beat to be considered valid for 30-day readmission prediction.

---

## 2. What is the HOSPITAL Score?

The HOSPITAL score is a validated clinical risk score for 30-day potentially avoidable hospital readmission, published by Donzé et al. in *JAMA Internal Medicine* (2013;173(8):632–638). It assigns points across 7 clinical variables measured during a hospital stay. The total score (0–13) stratifies patients into low, intermediate, and high risk tiers.

**Reference:** Donzé J, Aujesky D, Williams D, Schnipper JL. Potentially avoidable 30-day hospital readmissions in medical patients. *JAMA Intern Med.* 2013;173(8):632-638.

---

## 3. The Seven Variables

| Variable | Acronym Letter | Clinical Threshold | Points |
|---|---|---|---|
| Hemoglobin at discharge | **H** | < 12 g/dL | +1 |
| Discharged from oncology service | **O** | Yes | +2 |
| Sodium at discharge | **S** | < 135 mEq/L | +1 |
| Procedure performed during stay | **P** | Any ICD-coded procedure | +1 |
| Index admission type | **I** | Urgent or emergent | +1 |
| Hospital admissions in previous year | **T** | 0–1 = 0 pts, 2–5 = +2 pts, >5 = +5 pts | 0 / +2 / +5 |
| Length of stay | **A** | ≥ 5 days | +2 |

**Maximum possible score:** 13 points.  
**Risk tiers:** Low (0–4), Intermediate (5–6), High (≥7).

---

## 4. Data Sources

All variables are derived from MIMIC-IV v3.1, mirrored to the project BigQuery instance (`trim-icon-498815-a0`). The source tables are:

| MIMIC-IV Table | Content | Used For |
|---|---|---|
| `mimiciv_3_1_hosp.admissions` | Index encounters, admission type, discharge time | I, T, A |
| `mimiciv_3_1_hosp.patients` | Demographics | (join key) |
| `mimiciv_3_1_hosp.labevents` | Laboratory measurements | H, S |
| `mimiciv_3_1_hosp.diagnoses_icd` | ICD diagnosis codes | O |
| `mimiciv_3_1_hosp.procedures_icd` | ICD procedure codes | P |

The cohort is defined by `staging/cohort.sqlx` and split by `staging/cohort_split.sqlx`. Only adult patients discharged alive with length of stay ≥ 1 day and no elective/same-day admissions are included. The split is deterministic: `FARM_FINGERPRINT(subject_id) % 100`.

---

## 5. Variable Definitions — Exact Logic

### 5.1 H — Hemoglobin at Discharge

**MIMIC-IV itemid:** `51222` (Hemoglobin, Blood).  
**Source table:** `labevents`, filtered to `charttime` within the index admission window (`admittime ≤ charttime ≤ dischtime`) and `valuenum > 0`.

The **last** hemoglobin value before discharge is used:

```sql
ARRAY_AGG(valuenum IGNORE NULLS ORDER BY charttime DESC LIMIT 1)[SAFE_OFFSET(0)]
AS hemoglobin_last
```

**Scoring:** `hemoglobin_last < 12` → +1 point. `hemoglobin_last ≥ 12` or `NULL` → 0 points.

### 5.2 O — Oncology Service

MIMIC-IV does not record the admitting service directly. Oncology status is approximated by checking whether **any** billed ICD diagnosis code for the index admission falls in the neoplasm chapter:

| Coding System | Code Range | Regex |
|---|---|---|
| ICD-9 | 140–239 | `^1[4-9]` or `^2[0-3]` |
| ICD-10 | C00–D49 | `^C` or `^D[0-4]` |

```sql
REGEXP_CONTAINS(icd_code, r'^1[4-9]')
OR REGEXP_CONTAINS(icd_code, r'^2[0-3]')
OR REGEXP_CONTAINS(icd_code, r'^C')
OR REGEXP_CONTAINS(icd_code, r'^D[0-4]')
```

**Scoring:** Any neoplasm code present → +2 points. None → 0 points.

### 5.3 S — Sodium at Discharge

**MIMIC-IV itemid:** `50983` (Sodium, Blood).  
**Source table:** `labevents`, same in-stay window and quality filter as hemoglobin.

```sql
ARRAY_AGG(valuenum IGNORE NULLS ORDER BY charttime DESC LIMIT 1)[SAFE_OFFSET(0)]
AS sodium_last
```

**Scoring:** `sodium_last < 135` → +1 point. `sodium_last ≥ 135` or `NULL` → 0 points.

### 5.4 P — Procedure During Stay

**Source:** `procedures_icd`, counted per `hadm_id`. Any procedure code present counts:

```sql
COUNT(DISTINCT icd_code) > 0 AS has_procedure
```

**Scoring:** `has_procedure = TRUE` → +1 point. `FALSE` → 0 points.

### 5.5 I — Index Admission Type

**Source:** `admissions.admission_type`. Only `EMERGENCY` and `URGENT` count as non-elective. Elective (`ELECTIVE`, `SURGICAL SAME DAY ADMISSION`) and observation (`EU OBSERVATION`, `OBSERVATION ADMIT`) types receive 0 points, consistent with the original HOSPITAL derivation cohort of medical inpatients.

```sql
admission_type IN ('EMERGENCY', 'URGENT')
```

**Scoring:** Urgent or emergent → +1 point. All other types → 0 points.

### 5.6 T — Prior Admissions in 12 Months

**Source:** `admissions`, counted for the same patient (`subject_id`) with `admittime` strictly before the index `admittime` and within the preceding 12 months:

```sql
a.admittime < c.admittime
AND a.admittime >= TIMESTAMP_SUB(c.admittime, INTERVAL 12 MONTH)
```

**Scoring (bucketed, not a literal count):**

| Prior Admissions | Points |
|---|---|
| 0–1 | 0 |
| 2–5 | +2 |
| >5 | +5 |

This bucketing matches the original HOSPITAL score definition. A patient with exactly 2 prior admissions in the past year receives +2 points, not 2 points.

### 5.7 A — Length of Stay ≥ 5 Days

**Source:** `admissions.admittime` and `admissions.dischtime`:

```sql
TIMESTAMP_DIFF(dischtime, admittime, HOUR) / 24.0 AS index_los_days
```

**Scoring:** `index_los_days ≥ 5` → +2 points. `< 5` → 0 points.

---

## 6. Handling of Missing Laboratory Values

Hemoglobin is unmeasured in ~10.2% of admissions; sodium in ~13.8%. These missing values arise because the physician did not order the test — typically because the patient appeared clinically stable with respect to that metric.

**Policy:** Missing labs are scored as **0 points** (normal). This follows standard clinical practice for the HOSPITAL score: if a doctor does not order a discharge lab, there is no clinical suspicion of abnormality, and the score correctly assigns no risk points for that component.

No imputation is performed. The raw `NULL` values are preserved in the output table for auditability. A downstream analysis confirmed that missingness rates are consistent across train/validation/test splits (~10–14%), ruling out systematic bias.

---

## 7. Total Score and Risk Tiers

The 7 component scores are summed to produce the total HOSPITAL score (0–13):

```sql
points_h + points_o + points_s + points_p
  + points_i + points_t + points_a AS hospital_score
```

Risk tiers are assigned as:

| Total Score | Risk Tier | Interpretation |
|---|---|---|
| 0–4 | Low | ~5.8% risk of potentially avoidable readmission (literature) |
| 5–6 | Intermediate | Elevated risk |
| ≥7 | High | Substantially elevated risk |

---

## 8. Score Distribution (Train Split)

| Risk Tier | Patients | % of Cohort |
|---|---|---|
| Low (0–4) | 183,498 | 74.1% |
| Intermediate (5–6) | 43,663 | 17.6% |
| High (≥7) | 20,526 | 8.3% |

---

## 9. AUCPR — How the Baseline is Measured

### 9.1 Why AUCPR?

Readmission is an imbalanced classification problem — only ~20.9% of patients are readmitted within 30 days. In imbalanced settings, AUROC (area under the ROC curve) can be misleadingly optimistic because it rewards correct ranking of the majority (negative) class. AUCPR (area under the precision-recall curve) focuses on the positive (readmitted) class and is the recommended metric for this task.

### 9.2 Computation

The HOSPITAL score produces an integer risk score $s \in [0, 13]$ for each patient. This score is used directly as the prediction — a higher score indicates higher predicted risk. No thresholding or calibration is applied.

For each patient $i$:
- $y_i \in \{0, 1\}$ — ground truth (1 = readmitted within 30 days)
- $s_i \in [0, 13]$ — HOSPITAL score

The precision-recall curve is constructed by varying the decision threshold $t$ across all unique score values. At each $t$:

$$\text{Precision}(t) = \frac{|\{i : s_i \geq t \land y_i = 1\}|}{|\{i : s_i \geq t\}|}$$

$$\text{Recall}(t) = \frac{|\{i : s_i \geq t \land y_i = 1\}|}{|\{i : y_i = 1\}|}$$

AUCPR is the area under this curve, computed via `sklearn.metrics.average_precision_score(y_true, y_score)`.

### 9.3 Interpretation

| Value | Meaning |
|---|---|
| 0.209 | Random classifier (guessing "readmitted" at the base rate) |
| **0.3325** | **HOSPITAL score (this baseline)** |
| 1.0 | Perfect prediction |

The HOSPITAL score achieves AUCPR = 0.3325, representing a 59% improvement over random (0.3325 / 0.2094 ≈ 1.59). This is the floor that any machine learning model must exceed.

---

## 10. Implementation Reference

| Artifact | Location |
|---|---|
| Lab features (hemoglobin, sodium) | `definitions/features/feat_labs.sqlx` |
| Oncology flag | `definitions/features/feat_oncology.sqlx` |
| Feature assembly | `definitions/features/features.sqlx` |
| HOSPITAL score table | `definitions/baselines/hospital_score.sqlx` |
| Baseline computation script | `projects/mlops/scripts/compute_hospital_baseline.py` |
| Experiment tracking | Vertex AI Experiments → `readmission-mlops` (us-east1) |
| Local summary | `projects/mlops/artifacts/hospital_baseline.json` |

### Reproducibility

To recompute the baseline:

1. Build the Dataform pipeline (materializes `readmission.hospital_score`)
2. Run `python projects/mlops/scripts/compute_hospital_baseline.py`

The script queries the train split only, computes AUCPR, and logs the result to Vertex AI Experiments. The experiment name (`readmission-mlops`) is reused by all downstream pipeline runs for side-by-side comparison.
