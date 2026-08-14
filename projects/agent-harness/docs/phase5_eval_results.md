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

## 6. Golden-set rubric (LLM-as-judge) — PENDING
Design in progress. Decoupled approach: full-holdout ML metrics (quantitative) +
sampled (50–100, stratified) generative agent eval (qualitative). Results to be
appended here when complete.

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
