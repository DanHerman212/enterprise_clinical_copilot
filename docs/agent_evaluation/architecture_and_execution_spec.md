# Architecture & Execution Spec: Platform-Owned Evaluation (Langfuse Native)

**Status:** Approved for Step 0 Execution
**Date:** 2026-10-20
**Directive:** Discard uncommitted changes in `collect_http.py`. Execute Step 0 immediately, then proceed sequentially through Steps 1–7 locally. No cloud runs against live endpoints until Step 7 passes 100%.

---

## 1. Architectural Corrections to the Proposal

To prevent the failure modes of the legacy harness, the implementation must enforce three structural rules not fully specified in the draft proposal:

### A. Two Distinct Datasets (No Population Mixing)
Do not combine the Clinical Census and Suite 2 into a single dataset or experiment run.
1. `clinical-census-pilot` (1 patient, 4 items) & `clinical-census-full` (82 patients, 328 items)
2. `suite2-adversarial` (6 patients, 30 items)
Each dataset runs as its own experiment with its own `run_evaluators` gate.

### B. Multi-Turn Sequential Execution within `task(item)`
A Clinical Census conversation has 2 turns (e.g., `/risk` followed by a free-text follow-up). 
- **Do NOT flatten 2-turn conversations into 656 independent dataset items.** If flattened and run with concurrency, Turn 2 will execute out of order without Turn 1's session state.
- **Correct Pattern:** Each Census dataset item represents a **2-turn script** (82 admissions × 4 workflows = 328 dataset items). The `task(item)` function generates one `session_id`, executes Turn 1, awaits the response, executes Turn 2 on the same `session_id`, and returns both turns in the structured output.
- Set `max_concurrency=2` (not 50, and no longer 5) for live runs to prevent vertex/endpoint 429 rate-limit thrashing. Measured 2026-10-21: at 5, the rubric judge — `gemini-2.5-pro` on regional `us-east1` — exhausted its per-minute quota and five suite-2 items were lost to `429 RESOURCE_EXHAUSTED`.

### C. Explicit Schema Contracts (Solving the Manifest & Blank Output Bugs)
Every dataset item and task output must be validated via Pydantic before touching Langfuse:

- **`DatasetItemInput` Schema:**
  - `patient_id`: str
  - `workflow_type`: Literal["command_Plus_followup", "single_probe"]
  - `turns`: List[str] (1 prompt for Suite 2; 2 prompts for Census)
  - `patient_manifest`: Dict[str, Any] (Decoded via the server's native decoder during dataset seeding—**mandatory**, never fetched or rebuilt mid-run)

- **`TaskOutput` Schema (Returned by `task(item)` and written to Trace Output):**
  - `turn_1_response`: str (If Turn 1 is a slash command returning a JSON card, **serialize the card content to a human-readable string or formatted JSON** so the Langfuse `Output` column is never blank `—`)
  - `turn_2_response`: Optional[str]
  - `tool_evidence`: List[Dict[str, Any]] (Includes raw tool outputs + `synthetic` and `note` flags from `get_lab_trends`)

---

## 2. Decisions on Open Questions

1. **Step 0 (Server Capability):** Approved. Run the 10-minute throwaway dataset/experiment check against Langfuse `4.38.0` immediately.
2. **Judge Sampling:** Use **1 call per item** at `temperature=0.0` with standard exponential backoff for 429s. Drop the 2-of-3 consensus code.
3. **Deterministic Assertions:** Keep `assertions.py`. Furthermore, **short-circuit**: if a deterministic structural assertion fails (e.g., missing card schema), score `verdict=0` immediately and do not invoke the LLM judge.
4. **First Live Run Scope:** Approved. Run `clinical-census-pilot` (1 patient = 4 two-turn items) and `suite2-adversarial` (30 items) first.
5. **Release Thresholds:** Owned by `run_evaluators` in Langfuse (`pass_rate >= 0.85` for Census, `safety_failures == 0`).
6. **Rubric Ownership:** Architect-owned in `rubric.md` (v4). Include `rubric_version: "v4"` in experiment `metadata`.
7. **Historical Data:** Archive prior runs in place. Do not delete old traces until Step 8 is verified in the UI.

---

## 3. Execution Gate Sequence

Proceed sequentially. Report back after **Step 0** and after **Step 7**. Do not execute Step 8 without explicit sign-off.

- **Step 0:** Verify `run_experiment` round-trip against live Langfuse `4.38.0` server using a 1-item throwaway dataset, then delete the throwaway dataset. **(STOP and report if unsupported).**
- **Step 1:** Define Pydantic schemas for `DatasetItemInput` and `TaskOutput` and pass them to `create_dataset(input_schema=..., expected_output_schema=...)`. Write unit test verifying a missing `patient_manifest` raises a validation error.
- **Step 2:** Seed `clinical-census-pilot` (4 items), `clinical-census-full` (328 items), and `suite2-adversarial` (30 items) from local Git fixtures (bypassing bucket mounts). Verify every item contains its decoded `patient_manifest`.
- **Step 3:** Write `task(item)` (~40 lines) and `observability.set_trace_io()` ensuring both command JSON payloads and free-text strings render cleanly in the Langfuse `Output` column.
- **Step 4 & 5:** Wire `assertions.py` (deterministic) and `rubric.md` (v4 LLM Judge receiving `TaskOutput` + `DatasetItemInput.patient_manifest`).
- **Step 6:** Wire `composite_evaluator` (enforcing `fidelity`, `grounding`, `safety`, and `responsiveness >= 2`) and `run_evaluators`.
- **Step 7 (Mandatory Local Gate):** Run full end-to-end experiment locally against a mocked agent and mocked judge. Verify schema validation, manifest passage, short-circuit assertions, and parrot-rule downgrades execute in seconds. **Report results before Step 8.**
- **Step 8:** Execute live run on `clinical-census-pilot` + `suite2-adversarial`. Inspect Langfuse UI together.
- **Step 9:** Execute `clinical-census-full`.