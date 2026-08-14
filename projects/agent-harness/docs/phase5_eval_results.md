# Phase 5 — Agent Evaluation Gate: Results so far

_Date: 2026-08-14 · Status: IN PROGRESS (deterministic + live tiers green; golden-set
rubric pending)._

## Context
The demo remains **closed** until this gate passes. Endpoints were torn down at
EOD 2026-08-13 and **redeployed in parallel** on 2026-08-14:

- **Predict** → `readmission-endpoint` (`5171797032825782272`), CPR model deployed
- **RAG** → `readmission-rag-index` (`5244349407096209408`), `rag_tree_ah` on index
  `2371299135438454784`

## 1. Endpoint verification ✅
- **Predict smoke** (`smoke_test.py 20924467`): `Risk (probability) = 0.1314`
  (exact expected fixture value), decision 1 (threshold 0.12).
- **`integration_test_live.py`** (real MCP tools, independently verified via BigQuery
  note→hadm mapping): **R1+ PASS, R1 PASS, ML PASS** (predict `20724182` → 0.194512).

## 2. Tier 1 — deterministic tool tests ✅
`pytest projects/agent-harness/tests/test_tier1.py` → **4 passed** (26.3s).
Tool routing, known-good value `0.1314`, response schema, graceful unknown-id error.

## 3. Tier 2 — agent faithfulness (local stdio) ✅
`pytest projects/agent-harness/tests/test_agent_local.py` → **6 passed** (27.0s).
Agent calls the tool and reports the exact number (no answer-from-memory).

## 4. Tier 2 — agent faithfulness (vs deployed MCP over HTTP) ✅
`MCP_TRANSPORT=http MCP_URL=https://mcp-server-jamycsjjzq-ue.a.run.app
pytest projects/agent-harness/tests/test_agent_local.py` → **6 passed** (39.1s).
Same assertions against the live service.

## 5. Live retrieval validation — `validate_rag.py` ✅
- `r1` (cross-patient isolation, B's text restricted to A): **PASS**
- `r1_positive` (A's own text restricted to A): **PASS** (6 returned, 0 leaked)
- `sanity` (3 real clinical queries): **PASS** (5 neighbors each, all mapped to real notes)

> **Code fix:** `validate_rag.py` hardcoded a deleted endpoint ID. Now reads
> `INDEX_ENDPOINT_ID` env (default = current endpoint). Commit `73214ba`.

## 6. Quantitative holdout pass (deployed model, full test split) ✅
`eval/quant.py` scored the **full test split (49,103 rows)** with the deployed
endpoint and computed threshold-free + operating-point metrics:

| Metric | Value |
|---|---|
| Rows | 49,103 |
| Readmission rate | 0.2091 |
| **AUCPR** | **0.4098** (HOSPITAL baseline **0.3325** → **+0.077** ✅) |
| AUROC | 0.7117 |
| Brier | 0.1482 |
| Precision @ 0.12 threshold | 0.2592 |
| Recall @ 0.12 threshold | 0.8921 |

> **Decision (2026-08-14):** the deployed model clears the HOSPITAL baseline on an
> honest 49k holdout, consistent with validation (~0.40). **No HPO re-run needed
> for the demo gate.** A proper full-HPO re-run (inspect config → run → redeploy →
> re-eval) is tracked as a **post-demo / pre-Phase-7** follow-up.

> **Two holdouts exist:** `split_name='test'` = 49,103 rows (quantitative metrics
> above). `split_name='demo'` = 3,402 rows (the demo-shaped holdout) — the
> **agent narrative sample is drawn from the `demo` split**, since those are the
> patient profiles the demo actually shows.

## 7. Golden-set rubric (LLM-as-judge) — IN PROGRESS
Decoupled approach: full-holdout ML metrics (above, quantitative) + sampled
(weighted, calibrated bands) generative agent eval (qualitative). Langfuse
self-host (standard: Cloud Run + Cloud SQL + Memorystore + GCS) being stood up
for trace review + rubric-score attachment. Results to be appended when complete.

## Re-run commands
```bash
# endpoint verification
.venv/bin/python projects/mlops/scripts/smoke_test.py 20924467
cd projects/agent-harness && ../.venv/bin/python scripts/integration_test_live.py
# gate tiers
.venv/bin/python -m pytest projects/agent-harness/tests/test_tier1.py -q
.venv/bin/python -m pytest projects/agent-harness/tests/test_agent_local.py -q
MCP_TRANSPORT=http MCP_URL=https://mcp-server-jamycsjjzq-ue.a.run.app \
  .venv/bin/python -m pytest projects/agent-harness/tests/test_agent_local.py -q
# retrieval
cd projects/agent-harness && ../.venv/bin/python scripts/validate_rag.py
```
