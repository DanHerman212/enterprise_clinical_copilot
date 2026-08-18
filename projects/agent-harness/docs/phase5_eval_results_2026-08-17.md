# Phase 5 Agent Evaluation — Golden Re-run Results (2026-08-17)

_Companion to `phase5_eval_results.md` (baseline) and the remediation workflow.
This captures ONLY the evaluation results of the golden re-run on the fixed agent._

## Setup

- **Golden set:** 100 test-split patients × 3 prompts (risk / meds / summarize) = 300 traces
- **Agent under test:** rebuilt with P4 deterministic guardrails + hardened prompt
  (deployed revision `agent-00015-j59`; mcp-server revision `mcp-server-00006-mhk` with the
  meds-retrieval + source-card-title fixes)
- **Judge:** LLM-as-judge, v2 full-evidence (EVIDENCE_CAP 120000, PER_PASSAGE_CAP 20000),
  judged from scratch — **0 already judged, 300 to go** (baseline `judged.jsonl` was moved
  aside so the stale-result trap could not fire)
- **Rows:** 300 collected, 300 scored, 0 agent errors

## Overall verdict

| Metric | Baseline (2026-08-14) | Re-run (2026-08-17) | Delta |
|---|---|---|---|
| **Pass rate** | 88.6% (265/299) | **92.7% (278/300)** | **+4.1 pts** |
| Safety failures | 16 | **16** | 0 |
| Agent errors | — | 0 | — |

## By prompt (pass / total)

| Prompt | Baseline | Re-run | Delta |
|---|---|---|---|
| risk | 89/99 | **98/100** | +9 |
| meds | 92/99 | **94/100** | +2 |
| summarize | 84/99 | **86/100** | +2 |

## By dimension (pass rate)

| Dimension | Baseline | Re-run | Delta |
|---|---|---|---|
| faithfulness | 91% | **93%** | +2 |
| groundedness | 90% | **94%** | +4 |
| citation | 96% | **97%** | +1 |
| clinical | 96% | **98%** | +2 |
| safety | 95% | **95%** | 0 |

## Safety failures (16 remaining)

No longer the baseline's invented-age / fabrication class. The 16 remaining failures are
now a narrower **medication-detail fidelity** class, e.g.:

- Omitted specific dosage for Oxycodone in discharge medications
- Medication frequency contradicts the detailed prescription (e.g. Q24H vs every 12 hours)
- Levofloxacin duration incorrect (7 days vs 6 days in evidence)
- Discharge medications list incomplete (meds omitted)
- Acetaminophen dosage inconsistency across sections (1000 mg vs 500 mg)

These are consistent with the known P4 guardrail limitation: dose/freq checks are
token-global (they catch a dose/freq absent everywhere, but not a per-med error like
"Simvastatin should be DAILY, not BID" when another med in the note is BID).

## Interpretation

- **Real improvement confirmed:** +4.1 pts overall, driven mostly by the risk prompt (+9)
  — the prompt hardening + guardrails working on the assessment path. Meds +2 and
  summarize +2 reflect the meds-retrieval fix and hardening.
- **Gate not yet green on safety:** 16 safety failures remain (plan ship gate: safety
  pass ≥ 95% met, but invented-med count = 0 and wrong-citation count = 0 are the
  remaining criteria to verify). The remaining failures are med-fidelity gaps, not
  fabrication — a narrower, more fixable class than the baseline.

## Artifacts

- `traces.jsonl` (2026-08-17) — 300 raw traces
- `judged.jsonl` (2026-08-17) — 300 judge scores (baseline preserved as `judged_baseline_aug14.jsonl`)
- `golden_report.json` (2026-08-17) — aggregate report
- `collect_p5.log`, `judge_p5.log` — run logs
