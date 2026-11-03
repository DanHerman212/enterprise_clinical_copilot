## NON-NEGOTIABLE GOVERNANCE & ANTI-HACKING LAWS

1. **Strict Scope Enforcement (`Authorized Scope`):**
   - You may ONLY create or modify files explicitly listed in the current prompt's `Authorized Scope`.
   - Modifying any file outside `Authorized Scope` is a hard failure.

2. **Test & Evaluation Harness Immutability:**
   - NEVER modify `evaluation/platform/` (`task.py`, `evaluators.py`, `judge_call.py`) or `tests/agent/test_layer_separation.py` during feature implementation.
   - NEVER weaken, delete, mock out, or alter existing test assertions to make a failing test pass. Fix the production implementation instead.

3. **Zero Hardcoded Clinical Data or Un-Traced UI Shortcuts:**
   - NEVER hardcode PMIDs, clinical study rows, readmission rates, phenotype lookup tables, or `hadm_id` mappings inside `presentation/`, `engine/`, `routing/`, or `mcp/`.
   - Every literature citation or benchmark row rendered in `presentation/` MUST originate from a traced `search_literature` tool call executed in `state["tool_calls"]` during that turn.

4. **Six-Subpackage Layer Purity (`services/agent/`):**
   - `presentation/` (`a2ui.py`, `canvas.py`, `domain.py`) is a pure, stateless projection of settled engine state. It must never perform domain classification, fabricate citations, or mutate `sources` with un-traced data.
   - `telemetry/` (`set_trace_io`) must capture the final settled response and composed presentation envelope so Langfuse traces 100% match the UI output.