# Phase 01: Codebase Audit & Multi-Turn Routing Refactor

**System:** Enterprise Clinical Copilot (`services/agent`) **Prerequisite:** Phase 00 Master Strategy Sign-Off **Execution Rule:** Zero live LLM evaluation suite runs are permitted during Phase 01. All changes must be verified locally via `pytest` unit and contract tests.

## 1. Objective

During prior single-turn optimization sprints, the agent codebase was over-fitted to pass isolated 1-turn checks. Specifically:

1. **Keyword Hijacking:** Free-text analytical questions mentioning medications or labs are intercepted by `http._route_free_text` / `routing.classify_free_text` and forced into stateless command templates (`deterministic.run_command`), dropping conversation history (`turns`) and repeating static Turn 1 captions.

2. **Tool Deprivation on the Conversational Path:** `discharge_medications` was withheld from `routing.FREETEXT_TOOLS`, and `prompts.SYSTEM_PROMPT` was instructed to refuse medication and risk questions rather than reason over prior turns or call tools when needed.

3. **Narrow Affordance Recognition:** Interactive Risk-Driver bar clicks (e.g., *"What is the oncology history for this patient?"*) and dynamic `followups.py` chips are vulnerable to mis-routing or ungrounded refusals if they do not match a single hardcoded phrase template.

4. **Numeric Rounding Drift:** Risk driver feature explanations in multi-turn follow-ups lack an explicit guard against rounding raw manifest/feature values (e.g., `18.5833` rounded to `18.58`).

**Phase 01 Goal:** Refactor the routing boundary, tool bindings, and system prompt so that **declarative slash commands** and **stateful multi-turn conversations** operate cleanly across all four UI surfaces without keyword hijacking or context loss.

## 2. Architectural Boundary Specification

| User Interaction Surface | Wire Payload Shape | Required Execution Path | Conversation History (`turns`) | UI Component Composition | 
| ----- | ----- | ----- | ----- | ----- | 
| **1. Explicit Slash Command / Starter Chip** (`/ehr`, `/risk`, `/meds`, `/labs`, `/pubmed`) | `chip` is set, OR `question` starts with `/<command>` | **Deterministic Path** (`deterministic.run_command`) | Not required (stateless baseline view) | Driven by `payload.COMMAND_COMPONENTS[command]` | 
| **2. Interactive Risk-Driver Bar Click** (e.g., *"What is the oncology history for this patient?"*) | Free text (`chip=None`), `turns` present or absent | **Stateful ReAct Path** (`graph.build_graph`) | Preserved & replayed via `replay_messages` | Driven by `payload.TOOL_COMPONENTS` for any tool called | 
| **3. Proactive Follow-Up Chip** (`followups.py` click) | Free text (`chip=None`), `turns` non-empty | **Stateful ReAct Path** (`graph.build_graph`) | Preserved & replayed via `replay_messages` | Driven by `payload.TOOL_COMPONENTS` for any tool called | 
| **4. Open-Ended Free-Form / Shorthand Query** | Free text (`chip=None`), Turn 1 or Turn 2+ | **Stateful ReAct Path** (`graph.build_graph`) | Preserved & replayed via `replay_messages` | Driven by `payload.TOOL_COMPONENTS` for any tool called | 

## 3. Tactical Engineering Tasks

### Task 1: Eliminate Free-Text Keyword Hijacking (`http.py` & `routing.py`)

* **Problem:** `http._question_or_error` calls `_route_free_text(parsed)`, which invokes `routing.classify_free_text()`. Any question containing tokens like `"medications"`, `"meds"`, `"drug"`, `"dose"`, `"labs"`, `"sodium"`, or `"creatinine"` is silently converted into `parsed["kind"] = "meds"` or `"labs"`. On Turn 2+, or on any analytical question (*"Are any high-risk or anticoagulant medications listed on discharge?"*, *"Which lab was most abnormal at discharge?"*), this drops `turns` and forces `deterministic.run_command` to emit a static summary instead of answering the question.

* **Required Refactor:**

  1. **Remove `_route_free_text` from `http.py`** (or restrict `classify_free_text` so it **never** fires when `parsed.get("turns")` is non-empty or when the query is an analytical/comparative question).

  2. **Preferred Clean Architecture:** Only explicit commands (`parsed["kind"]` set via `chip` or a leading `/slash` token parsed by `contracts._leading_command`) enter `deterministic.run_command`. All non-slash free-text questions enter the LangGraph ReAct loop (`graph.ask` $\rightarrow$ `build_graph`).

  3. Clean up obsolete keyword tables in `routing.py` (`_FREE_TEXT_ROUTES`, `_AFFORDANCE_TEMPLATES`, `_LITERATURE_CUES`) if `classify_free_text` is retired, ensuring no dead routing heuristics remain.

### Task 2: Empower the ReAct Loop to Handle Meds, Labs & Risk Follow-Ups (`routing.py`, `graph.py`, `deterministic.py`)

* **Problem:** Currently, if a user asks a free-text question about medications (Turn 1 cold start or Turn 2 follow-up), the ReAct loop does not have `discharge_medications` in `FREETEXT_TOOLS` and `SYSTEM_PROMPT` tells it to redirect to `/meds` and stop.

* **Required Refactor:**

  1. **Re-Use Replayed Evidence First:** When `turns` contains a prior `/meds`, `/labs`, `/risk`, or `/ehr` turn, `graph.replay_messages` already replays the prior tool calls and responses into the message transcript. Update `prompts.SYSTEM_PROMPT` so the model is explicitly permitted to answer analytical follow-up questions (e.g., classifying whether replayed discharge medications are anticoagulants/high-risk, or ranking which replayed lab value was most abnormal at discharge) directly from the replayed tool outputs or by calling the appropriate tool.

  2. **Admit `discharge_medications` or Structured Med Verification on Free-Text:** Add `discharge_medications` to `routing.FREETEXT_TOOLS` (or allow the ReAct loop to call `discharge_medications` when a free-text question asks about medications and no prior `/meds` turn is in `turns`). Note: `payload.TOOL_COMPONENTS` already maps `"discharge_medications": "medication_table"`, and `guardrail.py` already verifies medication doses and frequencies against the retrieved note text.

  3. **Broaden Risk-Driver & Feature Drill-Down Detection (`routing.asks_why_risk_driver`):**

     * A user clicking a Risk Card bar may send `"What in this patient's chart explains X as a readmission risk driver?"` OR a direct feature query like `"What is the oncology history for this patient?"` or `"Which top clinical drivers contributed most to this risk score?"`.

     * Update `routing.asks_why_risk_driver(question)` so that any question asking about **risk drivers, top clinical drivers, or why a factor contributes to readmission risk** binds `predict_readmission` in `_freetext_allowed`.

     * When `turns` already contains a replayed `predict_readmission` call (e.g., Turn 1 was `/risk`), ensure `replay_messages` passes the full `top_factors` list in the replayed `<tool_result>` so Turn 2 can read the exact drivers and directions without even needing a redundant tool call.

### Task 3: De-Conflict `prompts.SYSTEM_PROMPT` for Multi-Turn & Comparative Queries (`prompts.py`)

* **Problem:** `prompts.SYSTEM_PROMPT` contains rules written during single-turn isolation that directly sabotage multi-turn follow-ups:

  * *"A question that asks for a readmission RISK, or for a LIST OF MEDICATIONS, is still answered by naming the command that owns it — /risk or /meds — and by calling nothing."*

  * *"Cite a passage only from a tool call you made in THIS turn... If an answer needs note evidence, retrieve it now, in this turn, and cite that."*

* **Required Refactor in `prompts.py`:**

  1. **Analytical Medication Questions:** Clarify that while a bare request for a full readmission risk score redirects to `/risk`, **any question asking about specific medications, medication classes (e.g., anticoagulants, high-risk drugs, tapers), indications, or follow-up questions on a prior `/meds` turn MUST be answered directly** using the replayed `discharge_medications` output or by calling `discharge_medications` / `rag_search`. Never repeat a static medication count when the user asks a specific analytical question.

  2. **Comparative Lab Questions (`"most abnormal at discharge"`):** Add a clear rule under `TOOL USE` (and in `routing.LABS_FORMATTING` if `/labs` is ever called with a question): When a user asks which laboratory value was **most abnormal at discharge** (or at a specific timepoint), the **first sentence** must directly answer that comparison by examining the discharge (final timepoint) values and flags in `get_lab_trends`, explicitly naming the most abnormal lab at discharge (or stating that all labs were within their normal reference ranges at discharge). Do not list unrequested mid-stay nadirs for analytes that normalized by discharge unless asked about the full hospital course.

  3. **Exact Numeric Fidelity on Manifest & Feature Values:** Add an explicit rule under `CLINICAL SCOPE` / `IDENTIFIERS`: Never round numeric feature values from `<admission_metadata>` or tool outputs (such as `prior_inpatient_days: 18.5833` or `index_los_days`). Always state the exact decimal figure as provided in `<admission_metadata>` or the tool payload. (Only `probability_display` and `threshold_display` use pre-formatted display percentages).

  4. **Authoritative Manifest Precedence (`discharge_location`):** Reinforce that for administrative fields present in `<admission_metadata>` (specifically `discharge_location`, `admission_type`, `length_of_stay_days`), `<admission_metadata>` is authoritative for the index admission. If a retrieved note passage mentions a historical facility or prior stay (e.g., an infirmary from an earlier encounter), never report that historical facility as the index admission's discharge disposition over `<admission_metadata>`.

### Task 4: Fix Citation Indexing on Multi-Passage / Multi-Turn Retrieval (`citations.py` & `prompts.py`)

* **Problem:** On Turn 2 of `/ehr` (`"What was the discharge disposition and follow-up plan?"`), the model either cited passages out of order or `resolve_sources` applied single-section intent (`intent_sections`) when `len(cites) == 1`, overriding the actual passage index the model cited.

* **Required Refactor:**

  1. Inspect `a2ui.resolve_sources` and `citations.intent_sections`: when a follow-up question asks a composite question (e.g., disposition + follow-up plan) and the model cites `^[1]` or `^[2]` from `rag_search` or `rag_search_sections`, ensure `intent_sections` does not force-remap a valid passage citation onto the wrong passage chunk.

  2. Verify that when Turn 2 makes a fresh `rag_search` or `rag_search_sections` call, `trimmed_calls` in `http._compose_success` resolves `^[1]..^[N]` strictly against Turn 2's retrieved passage array in exact 1-based order.

## 4. Anti-Reward-Hacking Constraints

1. **No Test-String Hardcoding:** You are strictly forbidden from adding regexes or string checks for specific evaluation questions (e.g., do not match `"anticoagulant"` or `"most abnormal"` in `routing.py`). Routing must be governed by structural request properties (`chip`, leading `/slash`, `turns` presence, and general tool definitions).

2. **No Patient-ID Special Casing:** Aside from the architectural `OUTPATIENT_HADM_IDS` and `SERVED_HADM_IDS` sets, no code or prompt may branch on a specific `hadm_id` (such as `90000106` or `90000108`).

## 5. Phase 01 Verification Gate (Local Zero-Cost Tests Only)

Before Phase 01 is marked complete, the engineer must run the full local test suite (`pytest`) and add unit tests verifying:

1. **Slash Command Purity:** `/meds`, `/labs`, `/risk`, `/ehr`, `/pubmed` route to `deterministic.run_command`.

2. **Conversational Follow-Up Freedom:** Free-text questions (both Turn 1 and Turn 2+ with `turns` populated), including questions containing words like `"medications"`, `"anticoagulant"`, `"labs"`, `"sodium"`, or `"oncology history"`, route to `build_graph` (the ReAct loop) with `routed_to: None` and preserve the `turns` transcript.

3. **Replay Fidelity:** A Turn 2 follow-up after `/risk`, `/meds`, or `/labs` receives the replayed tool call and payload in `replay_messages` and `replayed_tool_calls`.

4. **No Live Eval Execution:** Do **not** deploy or trigger any Langfuse evaluation runs during Phase 01. Report the `pytest` results and git diff summary in <=40 lines and await Architect clearance for **Phase 02 (Observability & Eval Harness Remediation)**.