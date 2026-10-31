# Phase 5 Directive: Final Verification, Evaluation Dry-Run & Sign-Off

## Execution Governance (ALL CODE & TESTS STRICTLY READ-ONLY)
- **Zero Code or Test Edits:** Do NOT modify any `.py` file under `services/`, `tests/`, or `evaluation/`.
- **Single Markdown Deliverable:** Write the final sign-off report to `docs/architecture_alignment/05a-final_verification_and_signoff_report.md`.

## Verification & Sign-Off Tasks

1. **Full Test & Charter Gate Suite:**
   - Run `pytest tests/agent/test_layer_separation.py tests/agent/test_architecture_remediation_contracts.py`
   - Run `pytest tests -m "not live"`
   - Run `pytest tests/mcp` and `pytest tests/evaluation` (or the equivalent MCP and evaluation test suites in `tests/`) to confirm Charters 13, 14, and 15 are 100% green together.

2. **Evaluation Harness Dry-Run Verification:**
   - Run the offline evaluation dry-run (`python3 -m evaluation.platform.dry_run` or the repo's standard dry-run command) to verify that `evaluation/platform/task.py` and `evaluation/platform/evaluators.py` execute cleanly against `citations.MARKER_PATTERN` and `EvidenceLedger` with zero regressions.

3. **Git Working Tree & Diff Audit:**
   - Run `git status --short` and `git diff --stat` to record the exact final footprint of the remediation across `services/`, `evaluation/`, `tests/`, and `docs/`.

4. **Publish `05a-final_verification_and_signoff_report.md`:**
   - Summarize the before/after state of all 8 Charter 13 findings from `01a-audit_report.md`, the 4 behavioral contracts, the final test counts, and the evaluation dry-run result so the branch is ready to commit and push.