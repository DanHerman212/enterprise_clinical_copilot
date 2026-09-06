# Evaluation

Evaluation is split into two deliberately separate tiers: the **model's quantitative
correctness** (AUCPR, calibration, fairness) and the **agent's semantic quality**
(faithfulness, groundedness). Conflating them is how agent projects lose credibility — the
agent is judged only on whether its narrative is supported by what the tools actually
returned, never on re-litigating the model.

## 1. Model — quantitative

Measured on the held-out test split. AUCPR is the headline metric because the task is
heavily imbalanced (~14.7% positive).

| Metric | Value |
|---|---|
| Test AUCPR | **0.328** |
| HOSPITAL baseline AUCPR | 0.251 |
| Default-param XGBoost benchmark | 0.309 |
| Optuna best (validation) | 0.323 |
| Test AUROC | 0.736 |
| Brier score | 0.112 |

The model clears the HOSPITAL clinical baseline and the default-parameter benchmark; the
gap from benchmark (0.309) to final (0.328) is the contribution of hyperparameter search
(Optuna, TPE sampler, median pruner).

### Operating point (threshold 0.11)

| | |
|---|---|
| Precision | 0.220 |
| Recall (TPR) | 0.815 |
| Specificity | 0.507 |
| NPV | 0.942 |
| Fβ (β = 2) | 0.535 |
| Net benefit | 0.066 |

The threshold is chosen to weight recall (β = 2) — missing a readmission is costlier than a
false flag — and validated with decision-curve analysis.

### Calibration

The endpoint returns calibrated probabilities (Brier 0.112); the decision layer threshold is
applied on top of the calibrated score, not a raw margin.

## 2. HOSPITAL baseline

The baseline is not quoted from the literature — it is **implemented as a pipeline
component** (the seven HOSPITAL variables: Hemoglobin, Oncology, Sodium, Procedure, Index
admission type, prior admissions, Length of stay) and evaluated on the same cohort, split,
and metric as the model. That makes the comparison apples-to-apples: HOSPITAL scores 0.251
AUCPR on the same test patients the model scores 0.328.

Full derivation in `mlops/docs/hospital_baseline.md`.

## 3. Feature selection

Two stages, tracked as experiments under `readmission-mlops`:

1. **Five-method vote** (filter, LASSO, LightGBM gain, RFE, Boruta) over the initial 74
   candidates.
2. **XGBoost gain → grouped-CV RFE → parsimony** on the reduced set, producing the final
   **49 features / 23 parent groups**.

The surviving set is broader than HOSPITAL's seven variables — richer utilization (prior
inpatient days), additional labs (RDW, monocytes), and medication complexity — which is
where the AUCPR edge comes from (see [data-and-model.md](data-and-model.md)).

## 4. Agent — LLM-as-judge

The agent narrative is scored by Gemini against a versioned rubric
(`evaluation/agent/rubric.md`): five dimensions (faithfulness, groundedness,
citation accuracy, clinical sensibility, safety), each 0–3, passing at ≥ 2. A case passes
iff faithfulness, groundedness, and safety all pass.

| Run | Traces | Pass rate | Notes |
|---|---|---|---|
| Golden set (100 held-out patients × 3 prompts) | 300 | **95%** (285/300) | 0 agent errors; 3 safety failures |
| Demo cohort (108 patients × 3 prompts) | 324 | **97.2%** (315/324) | 0 agent errors; 2 safety failures |

Dimension-level (golden set): faithfulness 96%, groundedness 98.7%, citation 99.7%,
clinical 99%, safety 99%.

## 5. Retrieval

Retrieval is evaluated on the 108-patient demo cohort with ground truth parsed from the
notes themselves:

| Path | Metric | Result |
|---|---|---|
| Summary sections (deterministic) | section recall | **100%** |
| Free-text index (embedding) | recall@5 | **100%** |
| Free-text index | recall@1 | 84.7% (meds) / 93.8% (course) |

recall@5 is 100% and the intended section is never absent from the top-5, so the agent
always receives the right evidence; the display layer resolves the exact section
deterministically. Full report: `services/docs/retrieval_eval_2026-08-24.md`.

## 6. Fairness audit

Error-rate parity at threshold 0.11 (n = 30,340) — **REVIEW**:

- Overall TPR 0.815, FPR 0.493.
- TPR gaps: insurance 0.338, race 0.184, age 0.110, gender 0.041.
- FPR gaps: age 0.334, race 0.328, insurance 0.327, gender 0.091.

PPV/NPV are reported for context only (they are prevalence-sensitive, not the parity
signal). The audit is a diagnostic yellow light — it flags subgroups for human review and
does not block registration.

## 7. Gaps & planned work

- **Feature engineering (v2):** quantify, feature-by-feature, how much each non-HOSPITAL
  signal moves AUCPR — turning the SHAP evidence into a controlled ablation.
- **Bias mitigation (v2):** reduce the insurance/race TPR gaps (e.g., subgroup-aware
  thresholds, reweighting) and re-audit.
- **Retrieval on real MIMIC notes:** current retrieval eval covers the 108-patient demo
  cohort; extend to the ~34k-note corpus.
- **Langfuse observability:** migrate eval scoring to the v4 OTel-native path so scores and
  traces live together (see `services/docs/langfuse_v4_migration_plan.md`).
