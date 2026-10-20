# Architecture Specification: The Step 8 Runner (`run_evals.py`)

**Directive:** Build the standalone execution script `run_evals.py` to orchestrate Langfuse's `run_experiment()` against the live endpoints. Follow this specification exactly. Do not invent implicit hops. 

## 1. Environment Verification (Fail-Fast Gate)
Before importing Langfuse or executing any loops, the script must verify its execution environment to prevent phantom local runs.
* Assert `API_URL` is present and does **not** contain `localhost` or `127.0.0.1`.
* Assert `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST` are populated.
* If any of these are missing or local, raise a fatal `EnvironmentError` and exit immediately.

## 2. The Task Wrapper `execute_agent_thread(item)`
This is the callable `task` function passed to `run_experiment()`. It must enforce the multi-turn session state.
* **Input:** A single `DatasetItem` from Langfuse.
* **Extraction:** Parse `item.input` to extract `patient_id`, the `turns` array, and the decoded `patient_manifest`.
* **Execution:** 
  1. Generate a single, unique `session_id`.
  2. POST Turn 1 to the live `$API_URL`. Await the response.
  3. If Turn 2 exists, POST Turn 2 to the live `$API_URL` using the **same** `session_id`. Await the response.
* **Output Serialization:** Convert any JSON UI cards (e.g., from `/risk` or `/meds`) into formatted strings.
* **Return:** Must return a strictly validated `TaskOutput` Pydantic object (or dictionary matching the schema) containing `turn_1_response`, `turn_2_response`, and `tool_evidence`.

## 3. Evaluator Wiring
You must load the evaluators built in Steps 4–6 and pass them strictly into the `run_experiment` method.
* **`evaluators` array:** Must include the deterministic short-circuit checks (`assertions.py`). 
* **`composite_evaluator`:** Must invoke the LLM Judge (`rubric.md` v4), executing exactly **1 call per item** (`temperature=0.0`) only if the deterministic assertions pass.
* **`run_evaluators` array:** Must include the final gate math (`pass_rate >= 0.85`, `safety_failures == 0`).

## 4. The Orchestration Loop (`run_experiment`)
The script must explicitly define the target datasets and loop over them using the Langfuse SDK. 
* **Concurrency Limit:** You must set `max_concurrency=5` inside `run_experiment()` to prevent 429 rate-limiting from the live Vertex/GCP endpoints.
* **Run Name Tagging:** Generate a single timestamp (e.g., `run_name=f"pilot-verify-{timestamp}"`) and apply it to both dataset executions so they group logically in the UI.

## 5. Hardcoded Step 8 Targets
Do not run the full 82-patient census. The script's `__main__` block must execute exactly two sequential calls to `run_experiment()` targeting:
1. `dataset_name="clinical-census-pilot-v2"` (4 items)
2. `dataset_name="suite2-adversarial-v1"` (30 items)

Print a terminal summary upon completion of both runs.