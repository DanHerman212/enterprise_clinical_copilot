# Engineering Directive — Eval Checkpoint 1: Zero-LLM Pre-Flight Audit & Dataset Alignment (`c7d116b`)

**Target Commits:** `enterprise_clinical_copilot` @ `c7d116b` | `danielmherman` @ `03830fb`
**Execution Rule:** Strictly $0 LLM cost. Do NOT run the live LLM evaluation suite or judge calls in this step.

## 1. Objective
Following the Phase 7 AST package reorganization of `services/agent/` (`ingress`, `routing`, `engine`, `postprocess`, `presentation`, `telemetry`), execute a deterministic pre-flight audit of the evaluation harness, dataset contract, and structural output guards before we launch Canary Run 5.

---

## 2. Pre-Flight Verification Tasks

### Task 1: Evaluation Harness Import & Schema Check
1. Run `pytest tests/evaluation/` and verify that all evaluation runner modules (`evaluation/run_dev_suite.py`, `evaluation/judge_call.py`, structural validators, etc.) import cleanly against the refactored `services/agent/` package hierarchy with zero stale module references.
2. Verify which Cloud Run revision of `agent` is currently serving 100% traffic and confirm it matches commit `c7d116b`.

### Task 2: Audit `dev_suite.jsonl` for Legacy `/risk` Commands & Contract Drift
1. Inspect `evaluation/datasets/dev_suite.jsonl` across all 26 scenarios (`dev_t1_01` through `dev_t7_03`).
2. Check whether any turn input still invokes the retired `/risk` slash command (e.g., `dev_t1_02`, `dev_t2_01`, `dev_t2_02`, `dev_t4_01`, `dev_t6_03`, `dev_t7_01`) or expects legacy `LiteratureList` components on `/pubmed` (`dev_t1_06`) where `references` drawer rows are now emitted.
3. If any legacy `/risk` inputs or pre-Phase-F component expectations remain in `dev_suite.jsonl` (or the structural validator), list the exact scenario IDs and required updates, align `dev_suite.jsonl` + `test_dev_suite_schema.py`, and re-sync `ecc-tier2-dev-suite` to Langfuse.

### Task 3: Verify Structural Guards for `evidence_found` & `predict_readmission` Replay
1. Check the refactored `services/agent/` subpackages to confirm whether:
   - `predict_readmission` is included in `ONCE_PER_TURN` and `ONCE_PER_TRANSCRIPT` (and suppressed from re-rendering `RiskInfluencers` / `ComparativeRiskMatrix` on free-text Turn 2+ follow-ups when already in `replayed_tool_calls`).
   - Standalone `{"evidence_found": true|false}` sentinel output is stripped and guarded against returning an empty/raw JSON answer on manifest-backed risk-driver follow-ups (such as `"Why does discharge destination increase this patient's readmission risk?"`).
   - The deterministic structural validator in `evaluation/` automatically flags any turn whose `answer` contains `"evidence_found"`, `"For admission 900"`, or `"Search PubMed for"`.
2. If any of those three guards was not yet applied during the AST refactor, implement them cleanly in their proper AST subpackages (`engine`, `postprocess`, `presentation`, and `evaluation/`) with unit tests.

---

## 3. Deliverable
Run `pytest` across the full test suite (`tests/agent/` and `tests/evaluation/`) and report back in <= 35 lines:
1. Active Cloud Run revision for `c7d116b`.
2. `dev_suite.jsonl` alignment status (and whether Langfuse `ecc-tier2-dev-suite` was updated).
3. Confirmation of the 3 structural guards + full `pytest` pass count.
**STOP after reporting — wait for Architect sign-off before running `--canary`.**