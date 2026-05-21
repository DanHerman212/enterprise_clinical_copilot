# `baseline-v1` — interpretation

Reference companion to the `baseline-v1` run logged under the
`readmission-30d` Vertex AI experiment. Defines the discriminative
floor that every subsequent model run (`logreg-v1`, `gbt-v1`, …) is
judged against. No figures yet; visualization is deferred.

## Setup, in one paragraph

Cohort: 30-day all-cause unplanned readmission, MIMIC-IV v3.1,
acute admissions only, one row per eligible `hadm_id`. Splits are
deterministic on `subject_id` via
`ABS(MOD(FARM_FINGERPRINT(CAST(subject_id AS STRING)), 1000))`,
yielding train (70%) / val (15%) / test (~14.5%) / demo (~0.5%);
no patient appears in more than one split. The headline numbers
below come from the held-out **test** split. Reference values from
the v2 cohort-wide evaluation (session_2026-05-06): LACE 0.648 AUROC
/ 0.282 AUPRC; HOSPITAL 0.678 AUROC / 0.320 AUPRC; prevalence ≈ 0.19.
The test-split run reproduces these to within sampling noise.

## What the metrics say

**HOSPITAL beats LACE on every metric in this cohort.** AUROC,
AUPRC, and the screening characteristics at the literature cutoff
all favour HOSPITAL. Going forward, **HOSPITAL is the headline
baseline**; LACE is reported alongside but is not the bar to clear.

**The discriminative floor is modest but non-trivial.** AUROC ≈ 0.68
puts HOSPITAL well above chance (0.50) and squarely inside the
0.60–0.68 band reported in the external-validation literature for
these scores. AUPRC lift over chance is ≈ 1.7× (0.32 vs 0.19) —
honest, but small, and the right number to focus on at this
prevalence.

**There is real headroom for a learned model.** A useful production
model needs to clear roughly **0.70 AUROC / 0.34 AUPRC** on the test
split to claim genuine lift over the deployed rules. Anything below
that is rediscovering the same signal LACE/HOSPITAL already encode
by hand.

**Clinical-cutoff metrics matter as much as AUROC.** The literature
cutoffs (LACE ≥ 10; HOSPITAL ≥ 7) are how these scores are actually
deployed. Sensitivity, PPV, and flag-rate at those cutoffs define a
specific bedside operating point, and any new model should be
benchmarked against them at a comparable flag-rate — not just on
ranking metrics. A model that improves AUROC but hits a lower PPV at
the same flag-rate is not an operational improvement.

**LACE and HOSPITAL are ordinal scores, not probabilities.** Their
calibration is poor at the high end (high scores under-estimate true
risk). A learned model with proper probability calibration should
out-perform on net benefit, not only on AUROC.

## How later runs use this

Every later run logs the same metric keys
(`{model}_test_auroc`, `{model}_test_auprc`,
`{model}_test_{sensitivity,specificity,ppv,npv,flag_rate}` at a
chosen operating point, plus `*_val_*` counterparts). To claim a
real win over the rules baseline, a new run must:

1. Beat HOSPITAL on `*_test_auroc` and `*_test_auprc`.
2. At a flag-rate matched to HOSPITAL's `hospital_test_flag_rate`,
   match or improve sensitivity *and* PPV.
3. Show better calibration (deferred — calibration metrics will be
   added when the first probabilistic model is logged).

If criterion 1 is met but 2 fails, the model is more discriminative
on paper but no better at the bedside; that is reported, not
celebrated.
