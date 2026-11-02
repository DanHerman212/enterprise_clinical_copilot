# Stage Gate 2 Directive: Wire Live Literature Comparator Pipeline into `/ehr`, Remove Static Benchmark Table, and Close Langfuse Trace Boundary

## Governance & Anti-Hardcoding Mandate
- First, save your Stage Gate 1E report to `docs/debugging/05-stage1e_live_literature_extraction_report.md`.
- **Keep `evaluation/` and `tests/agent/test_layer_separation.py` completely untouched (read-only).**
- **ZERO HARDCODED CLINICAL DATA OR BUCKETS:** No PMIDs, study rows, readmission rates, phenotype buckets, or `hadm_id` lookup maps may remain in `presentation/` or be moved to `routing/`, `engine/`, or `mcp/`.
- **Six-Subpackage Layer Purity (`services/agent/`):**
  - `presentation/` (`a2ui.py`, `canvas.py`, `domain.py`) must be a pure, stateless projection of settled engine state. It must never classify clinical notes, select literature rows from constants, or inject un-traced citations into `sources`.
  - Every literature citation in `response["sources"]` and `response["references"]` must originate from a traced `search_literature` tool call recorded in `state["tool_calls"]`.

## Authorized Scope
- `services/agent/routing/routing.py`
- `services/agent/engine/deterministic.py`
- `services/agent/engine/prompts.py`
- `services/agent/presentation/domain.py`
- `services/agent/presentation/canvas.py`
- `services/agent/presentation/a2ui.py`
- `services/agent/turn.py`
- `services/agent/telemetry/observability.py`
- Unit test files under `tests/agent/` (excluding `tests/agent/test_layer_separation.py`)

---

## Required Architectural Changes

### 1. Close Procedure-Invasiveness Compatibility Gap (`prompts.py` & `deterministic.py`)
In Stage Gate 1E, admission `90000081` (managed with bedside thoracentesis) accepted a surgical thoracic decortication cohort (`PMID 37065558`).
- Update `LITERATURE_COMPARATOR_PROMPT` (and deterministic verification if applicable) so that when an admission's index procedure is medical management or a minimally invasive/bedside percutaneous procedure (e.g., thoracentesis, paracentesis, catheterization, endoscopy), cohorts restricted to major open/thoracoscopic surgical operations (e.g., surgical decortication, lobectomy, open resection) are declined as procedure mismatches.

---

### 2. Wire Traced Literature Retrieval & Comparator Extraction into `/ehr` (`routing.py` & `deterministic.py`)
Currently in `services/agent/routing/routing.py:237-255`, `Command(wire="summarize", slash="/ehr", ...)` lists `"search_literature"` inside `forbidden`.
1. **Unblock `search_literature` on `/ehr` (`routing.py`):**
   - Remove `"search_literature"` from `forbidden` on the `summarize` (`/ehr`) command specification.
2. **Execute Traced Literature Retrieval & Comparator Extraction in `run_command` (`deterministic.py`):**
   - During `summarize` (`/ehr`) execution, after `rag_search_sections` and `predict_readmission` complete (and only when `is_outpatient is not True` and note sections exist):
     a) Derive the `ClinicalTarget` from the retrieved note sections (`derive_target`).
     b) Invoke the live readmission literature search via `_execute_tool_calls` (or traced MCP `search_literature` execution) so that `search_literature` is recorded in `recorded` / `state["tool_calls"]` with its exact query arguments and returned articles, and appears as a first-class tool span in Langfuse.
     c) Run `extract_comparators` (including the broadened second pass when Pass 1 yields `< 2` verified rows, recording any second `search_literature` call in `tool_calls` as well).
     d) Store the verified comparator rows on `command_state` (e.g., `command_state["comparators"] = verified_rows` and `command_state["clinical_target"] = target`) so `presentation/` can render them purely as data without re-deriving phenotypes.

---

### 3. Dismantle the Static Benchmark Table & Post-Hoc Source Injection in `presentation/` (`domain.py`, `canvas.py`, `a2ui.py`)
1. **`services/agent/presentation/domain.py`:**
   - Delete `BENCHMARKS`, `CohortBenchmark`, `benchmarks_for_phenotype`, `BUCKETS`, `ADMISSION_PHENOTYPE`, `bucket_for`, `resolve_phenotype`, and `phenotype_of_text` (or reduce any symbol strictly required by `/pubmed`'s condition-lock or existing import contracts to pure, table-free helpers without hardcoded PMIDs or `hadm_id` maps).
2. **`services/agent/presentation/canvas.py`:**
   - Update `_comparative_matrix` (and `build_canvas`) to accept the verified `comparators` list directly from settled engine state rather than calling `benchmarks_for_phenotype(phenotype)`.
   - Render `ComparativeRiskMatrix` with the patient's `baseline` estimate (`predict_readmission`) and `rows` populated strictly from the verified `comparators` produced by the engine.
   - **Honest Empty-Rows State:** When a turn is eligible for `ComparativeRiskMatrix` (an inpatient `/ehr` turn with a readmission `estimate` or verified `comparators`), if `comparators` is empty (`[]`), still render the `ComparativeRiskMatrix` component with the patient's `baseline` ML estimate (`rows: []`) and set an honest `baseline_note` explaining that no condition-matched 30-day readmission rate was stated in the retrieved PubMed abstracts for this admission's primary diagnosis. Never substitute a mismatched disease or hardcoded fallback study.
3. **`services/agent/presentation/a2ui.py`:**
   - Remove `register_benchmark_sources` (or refactor citation registration so that every literature source in `sources` and `references` comes from the traced `search_literature` tool results / verified comparators already recorded in engine state, never synthesized from hardcoded constants).
   - Ensure every comparator row's `source_index` and `source_label` resolve cleanly to its corresponding entry in `sources` and `references`.

---

### 4. Close the Langfuse Trace Boundary (`turn.py` & `observability.py`)
Currently in `services/agent/turn.py:255-297`, `observability.set_trace_io` and `observability.record_turn_tools` run *before* `compose_presentation`.
1. **Reorder or Update Trace Recording in `turn.py`:**
   - Ensure the composed presentation envelope (`presentation = compose_presentation(...)`) is built **before** finalizing the Langfuse trace output/metadata.
2. **Record the Full Composed Envelope on the Trace (`observability.py` & `turn.py`):**
   - Keep `input=question` and the prose `output` intact, and record the settled presentation contract on the trace/root span metadata (specifically: rendered `canvas` component names, the full `ComparativeRiskMatrix` component payload including `baseline`, `rows`, and `baseline_note`, and the final `sources` / `references` list).
   - Verify that `state["tool_calls"]` logged via `record_turn_tools` includes the `search_literature` call(s) executed during `/ehr`.

---

## Stage Gate 2 Verification & Deliverable
1. **Hermetic Test Suite:** Update any unit tests in `tests/agent/` that previously asserted the old 9-row static `BENCHMARKS` table or `"search_literature" in summarize.forbidden` so they test the new traced `/ehr` comparator pipeline and honest empty-row state. Run `pytest tests -m "not live"` and confirm 100% pass with 0 regressions.
2. **Contract Invariant Test:** Add a hermetic test asserting that:
   - Every PMID present in `ComparativeRiskMatrix.rows` and `response["sources"]` on `/ehr` is present in the turn's traced `search_literature` tool call output.
   - `turn.py` records the composed `ComparativeRiskMatrix` and `sources` payload to the observability layer.
3. **Live End-to-End `/ehr` Run on 3 Unseen Held-Out Admissions:**
   - Pick **3 brand-new inpatient `hadm_id` admissions** that were NOT in the Stage Gate 1 eight-patient set.
   - Run the full end-to-end `/ehr` turn (`summarize`) on each of those 3 unseen admissions plus `90000090`, `90000069`, and `90000081`.
   - Print a Stage Gate 2 verification table showing for each admission:
     - `hadm_id` & derived clinical target
     - Traced tool calls recorded in `state["tool_calls"]`
     - Rendered `ComparativeRiskMatrix` (`baseline` probability + verified `rows` with PMIDs, rates, and `baseline_note`)
     - Confirmation that `sources` / `references` match the traced tool calls and that the Langfuse trace payload contains the composed matrix.
4. **STOP after reporting Stage Gate 2 results** and wait for approval before Stage Gate 3 (`evaluation/` unlock).