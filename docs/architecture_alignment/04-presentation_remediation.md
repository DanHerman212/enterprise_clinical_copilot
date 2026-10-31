# Phase 3 Test Alignment & Phase 4 Directive: Post-Processing, Presentation & Evaluation Remediation

## Step 1 — Authorized Surgical Update to Two Legacy Tests (From Phase 3 Escalation)
You are explicitly authorized to make the two minimal legacy test updates identified in `03a-routing_and_engine_remediation_report.md` §4:
1. `tests/agent/test_freetext_boundary.py:165-172`: Update the expected literature toolset assertion to `routing.LITERATURE_QUESTION_TOOLS`.
2. `tests/agent/test_progress_stream.py:223` (`test_a_refused_call_announces_nothing`): Update the 5-call fixture to reach `MAX_TOOL_CALLS_PER_TURN = 5` using distinct tool calls within `PER_TOOL_CALL_LIMIT = 2` (so the 6th call is still refused by the turn budget and announces nothing).
- **No other file under `tests/` may be modified.**

## Step 2 — Phase 4 Objective (Turn All Remaining 6 Gates Green)
Remediate `services/agent/postprocess/`, `services/agent/presentation/`, `services/agent/turn.py`, and `evaluation/` (for L1-4 regex consolidation only) so that all 3 remaining AST gates (`test_law1_*`) and all 3 remaining Behavioral Contracts (`Contract 3` ×2 and `Contract 4` ×1) pass with zero regressions across the entire suite.

## Required Production Changes for Phase 4

### 1. Charter 13 Law 1 & Contract 3 — Canonical `EvidenceLedger` in `postprocess/citations.py`
- **Construct `EvidenceLedger` in `postprocess/citations.py`:**
  - Define `EvidenceLedger` in `postprocess/citations.py` as the single source of truth for a turn's resolved citations, passage counts, and literature numbering offset.
  - Eliminate all prose citation renumbering (`renumber_citations` / `citation_remap`) and citable-unit counting (`len(citable_passages(...))`) from `presentation/a2ui.py` (turning `test_law1_presentation_never_transforms_a_citation_marker` and `test_law1_presentation_never_counts_citable_units` green).
  - Ensure `compose_presentation` in `presentation/a2ui.py` serves the settled `answer` unchanged so the prose recorded on Langfuse in `turn.py` (`set_trace_io`) matches the served HTTP/A2UI answer byte-for-byte.
  - Keep `citations.passage_total` positional integrity intact for `evaluation/platform/task.py`.

### 2. Charter 13 Law 1 (L1-4) — Consolidate Inline Citation Regexes in `evaluation/`
- Replace the raw inline marker regexes in `evaluation/platform/evaluators.py:207` and `evaluation/inventory/multiturn_chip_audit.py:93` with imports/helpers from `services/agent/postprocess/citations.py` (turning `test_law1_no_module_compiles_an_inline_marker_regex` green).

### 3. Contract 4 — Turn-Scoped `presentation/canvas.py` Surface Resolution
- Update `presentation/canvas.py` and `presentation/a2ui.py` so a literature turn (`literature` payload present / literature view active) resolves cleanly to the literature surface without appending an unrequested `LabTrendCard` (`labs` sparkline table) or pulling conversation-wide `rag_payload` fallback cards from earlier turns.

## Verification Gate
End your execution by running and reporting:
1. `git diff --name-only tests/` (must list ONLY `tests/agent/test_freetext_boundary.py` and `tests/agent/test_progress_stream.py`)
2. `pytest tests/agent/test_layer_separation.py` (must be 27 passed, 0 failed)
3. `pytest tests/agent/test_architecture_remediation_contracts.py` (must be 9 passed, 0 failed)
4. `pytest tests -m "not live"` (must be 0 failed across the entire repository)