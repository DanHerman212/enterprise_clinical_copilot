# Phase 02: Observability & Evaluation Harness Remediation

**System:** Enterprise Clinical Copilot (`services/agent`, `evaluation/platform`, `danielmherman` site)  
**Parent Strategy:** `00-eval_master_strategy.md` (Principle 2 & Principle 4)  
**Prerequisite:** Phase 01 Closed (`9985f4b` / `ca84c87`)  
**Execution Rule:** Zero live LLM evaluation suite runs are permitted during Phase 02. All changes must be verified locally via `pytest` and `manage.py test` unit tests.

---

## 1. Objective

In previous evaluation runs (`census_first10_dump.json`), root-cause analysis was severely hindered by observability blind spots:
1. **Null Root Span I/O:** Custom root spans (`command.<wire>` and `free_text.turn<N>`) recorded `input: null, output: null` because `observability.set_trace_io()` only updated the top-level trace table row (`set_current_trace_io`) and never updated the active root span observation object itself.
2. **Null Child Tool Span I/O:** On the deterministic command path (`graph._execute_tool_calls` with `traced=True`), `with observability.observation(f"tool.{call['name']}"):` never received `input=arguments` or `output=payload`. Consequently, retrieved note passages, lab trends, prediction payloads, and tool errors were absent from Langfuse spans.
3. **Dropped Turn 2+ Trace IDs in the Evaluation Runner:** Multi-turn evaluation runs stored only `turn1_trace` in the run artifact and printed an 8-character prefix (`t2=xxxxxxxx`) to stdout, making it impossible to programmatically inspect or export Turn 2+ traces and spans.
4. **Pre-Existing Website Fixture Test Mismatch:** In the `danielmherman` website repo, `test_fixture_mode_answers_the_literature_chip_from_captured_records` fails because the fixture's literature query string does not match the patient's resolved phenotype under the Phase 4B condition-lock (`routing.comparison_applies` / `a2ui._draws_matrix`).

**Phase 02 Goal:** Achieve 100% span I/O attribution across all command and conversational paths, capture full trace IDs across every turn in multi-turn evaluations, and resolve the pre-existing fixture test failure so both repositories are 100% green before Phase 03.

---

## 2. Tactical Engineering Tasks

### Task 1: Root Span Input/Output Attribution (`services/agent/observability.py` & `services/agent/graph.py`)

* **Problem:** Calling `langfuse.get_client().set_current_trace_io(input=..., output=...)` populates the trace list columns in Langfuse, but leaves the root observation span (`root_span`) with `input: null, output: null` when exported or inspected in the span tree.
* **Required Refactor:**
  1. Update `observability.set_trace_io(user_input: str | None, output: str | None, span: Any = None) -> None` so that in addition to calling `set_current_trace_io`, if `span is not None`, it calls `span.update(input=(user_input or None), output=(output or None))`.
  2. In `graph.ask`, pass `root_span` into `observability.set_trace_io(...)` on both the `command.<question_kind>` path and the `free_text.turn<N>` path.
  3. In `http._unserved_admission`, call `observability.set_trace_io(parsed.get("question") or "", ADMISSION_NOT_FOUND, span=span)` inside the `turn.refused` span block so refused turns also carry full input/output attribution.
  4. Verify that for `/risk` (which produces a widget and no prose), `command_trace_output(command_state)` continues to populate `output` with the JSON summary of `command_payload` / `predict_readmission` so `/risk` root spans never record `output: null`.

### Task 2: Child Tool Span Input/Output Attribution (`services/agent/observability.py` & `services/agent/graph.py`)

* **Problem:** In `graph._execute_tool_calls`, the command path executes tools inside `with observability.observation(f"tool.{call['name']}"):` without recording tool arguments or responses on the span.
* **Required Refactor:**
  1. Update `observability.observation(name: str, *, as_type: str = "span", trace_metadata: dict | None = None, input: Any = None)` so that if `input is not None`, it passes `input=input` to `client.start_as_current_observation(...)` (or calls `span.update(input=input)` on entry).
  2. Add a helper `observability.record_span_output(span: Any, output: Any, *, is_error: bool = False) -> None` (governed by Rule 1: best-effort, never raises) that calls `span.update(output=output, level="ERROR" if is_error else None)`.
  3. In `graph._execute_tool_calls`, when `traced=True`:
     * Open the child span with `with observability.observation(f"tool.{call['name']}", input=arguments) as tool_span:`.
     * Immediately after `payload = await toolbox.call(call["name"], arguments)`, call `observability.record_span_output(tool_span, payload, is_error=bool(isinstance(payload, dict) and payload.get("error")))`.

### Task 3: Full Multi-Turn Trace ID Capture & Export in Evaluation Harness (`evaluation/platform/`)

* **Problem:** The evaluation runner only stores `turn1_trace` in its output artifacts and logs an 8-character prefix for Turn 2+, preventing automated inspection of Turn 2+ spans and scores.
* **Required Refactor:**
  1. Audit `evaluation/platform/` (`collect.py`, `run_evals.py`, and any associated result/export modules).
  2. For every item executed (whether 1-turn or N-turn), capture and store the full 32-character `langfuse_trace_id` for **every turn** in an ordered list field `turn_trace_ids: list[str]`, alongside `turn1_trace` and `turn2_trace` (populated with the full trace ID of Turn 2 when a 2nd turn runs) for backward compatibility.
  3. Ensure the evaluation runner's `unattributed` counter increments if **any** turn in a multi-turn item returns an empty `langfuse_trace_id`.
  4. Ensure any trace dump / inspection helper used to export evaluation results fetches and attaches spans (including `input` and `output`) for **all** trace IDs in `turn_trace_ids`, labeled by turn index (`turn: 1`, `turn: 2`, etc.).

### Task 4: Resolve Pre-Existing Website Fixture Test Failure (`danielmherman` repo)

* **Problem:** `manage.py test demo` reported 1 pre-existing failure in `test_fixture_mode_answers_the_literature_chip_from_captured_records` because the fixture's literature query no longer resolves to the admission's phenotype under the Phase 4B condition-lock, causing `_comparative_matrix` to return `None`.
* **Required Refactor:**
  1. Inspect `test_fixture_mode_answers_the_literature_chip_from_captured_records` and the demo fixture data in the `danielmherman` repository.
  2. Align the fixture's literature query/condition with the fixture admission's resolved phenotype in `admission_phenotypes.py` (or update the fixture test expectation if the fixture intentionally queries a non-matching condition) so the fixture accurately reflects the Phase 4B condition-matched comparative matrix contract.

---

## 3. Anti-Reward-Hacking & Safety Constraints

1. **Rule 1 Preservation (`observability.py`):** Every Langfuse call (`start_as_current_observation`, `span.update`, `set_current_trace_io`) must remain wrapped in `try...except Exception` so that a telemetry failure or serialization quirk can **never** break or delay a clinical answer.
2. **No Clinical Logic Changes:** Phase 02 touches only observability instrumentation, evaluation harness trace tracking, and the website fixture alignment. Do not alter prompt instructions or routing predicates.

---

## 4. Phase 02 Verification Gate (Local Zero-Cost Tests Only)

Before Phase 02 is marked complete:
1. **Agent Unit Tests (`pytest`):**
   * Add unit tests in `tests/agent/test_observability.py` verifying that:
     * `set_trace_io` updates both the client trace I/O and the passed `root_span` (`input` and `output`).
     * `_execute_tool_calls(..., traced=True)` passes `input=arguments` and records `output=payload` (and `level="ERROR"` when `payload` contains `"error"`) on the `tool.<name>` span.
2. **Evaluation Harness Unit Tests (`pytest`):**
   * Add or update unit tests in `tests/evaluation/` verifying that a multi-turn evaluation item records full trace IDs for all turns in `turn_trace_ids` and `turn2_trace`, and flags `unattributed` if any turn's trace ID is missing.
3. **Website Test Suite (`manage.py test demo`):**
   * Verify all tests in `manage.py test demo` pass with **0 failures**.
4. **Reporting:**
   * Commit changes locally. Do **not** deploy or run any live LLM evaluations. Report the commit hashes and test counts in <=40 lines and await Architect approval for **Phase 03 (Tier 1 Deterministic Cohort Sweep)**.