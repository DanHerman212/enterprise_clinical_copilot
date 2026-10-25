# Clinical Copilot Agent Surface (`services/agent/`) — Code Comprehension & Architectural Defense Guide

**Purpose:** A study and interview-prep reference mapping the 24 modules of `services/agent/`, explaining why each boundary exists, classifying modules into **Spine**, **Policy Seams**, and **Data Adapters**, and outlining how to defend and critique the architecture in senior engineering interviews.

---

## 1. The Mental Model: "Spine + Seams + Adapters"

When walking into a code comprehension or system design interview, never memorize 24 files as a flat list. Instead, organize them mentally into three roles:

1. **The Spine (Request Orchestration & Execution):** The core path a user's request travels from HTTP ingress $\rightarrow$ execution engine $\rightarrow$ HTTP response.
2. **Policy Seams (Deterministic Guardrails Around the LLM):** Pure, side-effect-free modules that enforce clinical safety, routing, citation numbering, and follow-up selection *outside* the stochastic model loop so they can be unit-tested offline in milliseconds.
3. **Data Adapters & Provenance (Translation & Telemetry):** Modules that translate data shapes between external boundaries (MCP tools, Gemini blocks, A2UI frontend components, Langfuse traces).

---

## 2. Menu-Style Breakdown of the 24 Modules

### Stage 1: Ingress (Transport, Auth & Request Validation)

| # | Module | Role Classification | What It Does (1-Line Menu Summary) | Why It Was Built This Way |
|---|---|---|---|---|
| 1 | `run.py` | **CLI Adapter** | CLI entrypoint for querying the deployed service or local stdio; not used by the browser UI. | Keeps developer/CI smoke-testing separate from the production Starlette web server. |
| 2 | `http.py` | **The Spine (Ingress + Hidden Orchestrator)** | Starlette server (`/ask`, `/ask/stream`, `/health`); parses requests, drives `_run_chain` and `_compose_success`, and emits JSON/SSE envelopes. | Built first as the server entrypoint; accumulated turn-orchestration logic (`_run_chain`, `_compose_success`) as features were added. |
| 3 | `contracts.py` | **Policy Seam (Schema)** | Validates request payloads (`question`, `hadm_id`, `turns`, `intent`, `follow_ups`) before any compute or LLM call runs. | Fail-fast boundary: prevents malformed requests or invalid parameters from burning LLM tokens or hitting MCP tools. |
| 4 | `served_admissions.py` | **Policy Seam (Access Control)** | Whitelist/registry of `hadm_id` admissions this deployment is authorized to serve. | Enforces cohort scoping at the door so arbitrary or unauthorized patient IDs are rejected immediately. |
| 5 | `observability.py` | **Provenance / Telemetry** | Configures and attaches the Langfuse callback handler to LangGraph runs as a non-blocking sink. | Designed so tracing is strictly a fire-and-forget sink: if Langfuse goes down, the clinical request still succeeds. |

---

### Stage 2: Pre-Model (Context Assembly & Routing)

| # | Module | Role Classification | What It Does (1-Line Menu Summary) | Why It Was Built This Way |
|---|---|---|---|---|
| 6 | `admission_context.py` | **Data Adapter (Pre-Flight Manifest)** | Builds the structured encounter header (age, length of stay, discharge location, prior admissions) injected before the LLM runs. | Gives the model immediate orientation on the patient's encounter without requiring a tool call just to know basic demographics. |
| 7 | `routing.py` | **Policy Seam (Traffic Cop)** | Decides slash-command (`deterministic.py`) vs. free-text (`graph.py`) and gates conditional tools (`asks_for_literature`, `wants_risk_driver`). | **Critical Safety Seam:** Prevents free-text chart questions from accidentally querying PubMed or exposing raw risk probabilities unless explicit linguistic cues or intents are present. |
| 8 | `questions.py` | **Prompt Text (User Turn)** | Templates and wording for user/command questions sent alongside the system prompt. | Separated early to keep command-specific user framing distinct from system-level instructions. |
| 9 | `prompts.py` | **Policy Seam (Tier 2 Guardrails)** | Houses the ReAct system prompt, clinical synthesis rules, abstention rules, and inline citation instructions (`^[n]`). | Centralizes LLM behavioral rules so prompt updates can be versioned and hashed cleanly. |

---

### Stage 3: Answering (Dual Execution Engines & Tool Execution)

| # | Module | Role Classification | What It Does (1-Line Menu Summary) | Why It Was Built This Way |
|---|---|---|---|---|
| 10 | `deterministic.py` | **The Spine (Fast Path Engine)** | Executes slash commands (`/ehr`, `/meds`, `/labs`, `/risk`, `/pubmed`) via fixed tool sequences and deterministic/templated formatting. | Eliminates LLM tool-selection latency and hallucination risk on the 5 core clinical workflows. |
| 11 | `graph.py` | **The Spine (ReAct Loop Engine)** | LangGraph state machine (`agent` $\rightarrow$ `tools` $\rightarrow$ `agent`) handling free-text questions and multi-turn `replayed_tool_calls`. | Isolates iterative multi-step reasoning and multi-turn transcript replay from deterministic slash commands. |
| 12 | `mcp_client.py` | **Data Adapter (Tool Bridge)** | Adapts MCP server tools (`rag_search`, `get_lab_trends`, `predict_readmission`, etc.) into LangChain/Gemini-compatible tool objects. | Decouples the agent service from the MCP transport layer so tools can be mocked cleanly in unit tests. |
| 13 | `message_text.py` | **Data Adapter (Model Output)** | Extracts normalized plain text out of Gemini's multi-block content structures. | Gemini responses can return lists of content blocks/parts rather than a single string; this normalizes the output shape. |
| 14 | `model_turn.py` | **Data Adapter (Stop Reason)** | Inspects the LLM finish reason to classify whether the turn completed normally, hit a tool call, or refused/truncated. | Separates low-level provider metadata parsing from the graph control flow. |

---

### Stage 4: Post-Model (Verification, Citations, A2UI Presentation & Follow-Ups)

| # | Module | Role Classification | What It Does (1-Line Menu Summary) | Why It Was Built This Way |
|---|---|---|---|---|
| 15 | `guardrail.py` | **Policy Seam (Factual Safety)** | Runs deterministic post-generation checks comparing the model's answer against retrieved evidence payloads. | Catches ungrounded numeric claims or leaked instructions before the user ever sees the text. |
| 16 | `citations.py` | **Policy Seam (Provenance Binding)** | Reconciles inline `^[n]` markers, renumbers them to first-appearance order (`1..K`), and resolves each marker to its exact source passage. | Guarantees that every clickable citation badge in the UI maps to a real, verified passage and sets `verified = True/False`. |
| 17 | `payload.py` | **Data Adapter (Legacy/Command Contract)** | Assembles the structured narrative/citations/components dictionary for deterministic commands. | Built in Phase 1 as the original structured output contract before the Phase 2 A2UI canvas refactor. |
| 18 | `a2ui.py` | **Data Adapter (Presentation Hub)** | Composes the final UI response envelope: A2UI visual components (`canvas`), `sources`, `verified`, and `scope`. | Acts as the single presentation assembly point connecting backend data to the frontend component schema. |
| 19 | `feature_labels.py` | **Data Adapter (Clinical Naming)** | Maps raw ML feature keys (e.g., `charlson_index`) to human-readable clinical labels across both prose and UI charts. | Ensures the risk bar chart and the written narrative never use conflicting names for the same risk driver. |
| 20 | `admission_phenotypes.py` | **Domain Taxonomy** | Classifies the clinical phenotype of an admission/query for the cohort comparator matrix. | Supports clinical contextualization by matching the patient's profile to curated reference cohorts. |
| 21 | `literature_benchmarks.py` | **Domain Data (Static Reference)** | Stores curated published readmission benchmark rates backing the comparator matrix in `a2ui.py`. | Keeps static clinical reference tables isolated from presentation rendering logic. |
| 22 | `followups.py` | **Policy Seam (Proactive UX Engine)** | Deterministic taxonomy selector (`P1`–`P11`) that generates up to 3 entity-slotted follow-up chips from verified in-memory chart state. | Replaced an untraced, stochastic LLM call with a zero-latency, 100% answerable rule engine that filters out previously asked questions. |

---

### Stage 5: Output Status & Execution Record

| # | Module | Role Classification | What It Does (1-Line Menu Summary) | Why It Was Built This Way |
|---|---|---|---|---|
| 23 | `stages.py` | **UX Adapter (SSE Progress)** | Emits human-readable interim status labels (e.g., *"Searching clinical notes..."*) over SSE while the turn is in flight. | Improves perceived latency in `/ask/stream` so clinicians see live progress steps. |
| 24 | `chain.py` | **Provenance (Audit Log)** | Stamps the immutable execution record (model ID, prompt hash, code revision, `langfuse_trace_id`, `follow_ups`) for every turn. | Provides a deterministic audit trail for clinical compliance and evaluation debugging. |

---

## 3. Architectural Evaluation: What Earns Its Keep vs. Where It Drifted

### 3.1 What Earns Its Keep (Why This Architecture Wins in Regulated AI)
* **Pure Functional Policy Seams (`routing.py`, `guardrail.py`, `citations.py`, `followups.py`):**
  In many LangGraph projects, developers stuff routing, citation formatting, and follow-up generation inside LLM nodes. That makes testing slow, expensive, and non-deterministic. By isolating those four boundaries as **pure Python functions of in-memory evidence**, your 1,326-test suite runs in ~4 minutes with zero live LLM calls.
* **Single Convergence Point (`_compose_success`):**
  Whether a turn took the deterministic slash-command path (`deterministic.py`) or the multi-turn ReAct path (`graph.py`), both paths converge into a single post-processing funnel for citations, guardrails, A2UI packaging, and follow-ups.

### 3.2 Where It Drifted ("Evolutionary Scar Tissue")
As the codebase evolved from Phase 1 $\rightarrow$ Phase 5, four seams drifted into unnecessary indirection:

1. **No Dedicated Turn Orchestrator (`http.py` Owns the Turn):**
   * *The Smell:* `http.py` handles both HTTP/SSE transport **and** high-level turn choreography (`_run_chain`, `_compose_success`). Tracing a single request requires jumping across 12 files because `http.py` is acting as a "Fat Controller."
   * *The Cleanup:* Extract `turn.py` (or `orchestrator.py`) to own the 5-stage turn lifecycle (`validate` $\rightarrow$ `route` $\rightarrow$ `execute` $\rightarrow$ `verify_and_present` $\rightarrow$ `record`), leaving `http.py` as a thin 150-line Starlette transport wrapper.
2. **Dual Presentation Adapters (`payload.py` vs. `a2ui.py`):**
   * *The Smell:* Both modules construct presentation dictionaries for the frontend. `payload.py` is an artifact of Phase 1; `a2ui.py` was added in Phase 2 for rich visual canvases.
   * *The Cleanup:* Fold `payload.py` into `a2ui.py` (or a single `presentation/` package) so there is only one presentation builder.
3. **Split Prompt & Micro-Parser Files (`questions.py` + `prompts.py`, and `message_text.py` + `model_turn.py`):**
   * *The Smell:* `questions.py` and `prompts.py` both hold prompt strings separated only by "user" vs. "system". Similarly, `message_text.py` and `model_turn.py` are tiny helpers that both parse a single Gemini response object.
   * *The Cleanup:* Consolidate prompt text into `prompts.py` and fold `message_text.py` + `model_turn.py` into a single `model_response.py` adapter.
4. **Split Provenance Modules (`chain.py` vs. `observability.py`):**
   * *The Smell:* Both exist to answer *"What code/model ran, and where is the trace?"*
   * *The Cleanup:* Unify under a single `telemetry.py` module.

---

## 4. Target Package Layout (The "If I Had 2 Hours to Refactor" Answer)

In an interview, if asked *"How would you reorganize these 24 flat files so a new engineer can grok the codebase in 5 minutes?"*, propose grouping the flat directory into **6 domain packages (~12 modules total)**:

```text
services/agent/
├── turn.py                 # NEW: Owns the end-to-end turn pipeline (extracted from http.py)
├── ingress/
│   ├── http.py             # Pure Starlette routes (/ask, /ask/stream, /health)
│   ├── contracts.py        # Request validation
│   ├── served_admissions.py# Cohort authorization
│   └── run.py              # CLI client
├── routing/
│   ├── routing.py          # Command vs. ReAct + tool gating predicates
│   └── admission_context.py# Pre-flight encounter header
├── engine/
│   ├── deterministic.py    # Slash-command fast path
│   ├── graph.py            # ReAct loop + prior-turn replay
│   ├── prompts.py          # System + question prompt templates (merged questions.py)
│   ├── mcp_client.py       # MCP tool bridge
│   └── model_turn.py       # Gemini response & finish-reason parser (merged message_text.py)
├── postprocess/
│   ├── guardrail.py        # Evidence verification checks
│   ├── citations.py        # Marker reconciliation & first-appearance renumbering
│   └── followups.py        # Deterministic P1-P11 chip engine
├── presentation/
│   ├── a2ui.py             # Unified canvas & payload composer (merged payload.py)
│   ├── feature_labels.py   # Clinical feature naming
│   ├── phenotypes.py       # Admission phenotype matching
│   └── benchmarks.py       # Literature cohort benchmarks
└── telemetry/
    ├── observability.py    # Langfuse callback & chain execution record (merged chain.py)
    └── stages.py           # SSE interim progress labels
```

---

## 5. Three Soundbites to Memorize for Interviews

1. **On Why YouHave 24 Small Modules Instead of One Giant Graph:**
   > *"In clinical AI, every failure mode—hallucinated citations, ungrounded numbers, tool leakage, or unanswerable follow-ups—happens at the boundary around the model. We intentionally carved `routing`, `citations`, `guardrail`, and `followups` into pure functional policy seams outside the LangGraph loop. That lets us run 1,300+ deterministic safety unit tests in four minutes without mocking or paying for live LLM calls."*
2. **On Recognizing Architectural Drift (Self-Awareness):**
   > *"As we iterated through five phases—moving from simple text responses to A2UI visual canvases and deterministic follow-ups—`http.py` absorbed too much turn-orchestration logic, and we kept `payload.py` alongside `a2ui.py` to avoid breaking earlier test contracts. Functionally the seams are rock-solid, but structurally my next cleanup pass would extract a `turn.py` orchestrator out of `http.py` and merge the presentation adapters."*
3. **On Deterministic vs. Probabilistic Hybrid Design:**
   > *"We don't use a ReAct loop when a deterministic pipeline suffices. Slash commands (`/ehr`, `/meds`, `/labs`, `/risk`) bypass open-ended tool selection via `deterministic.py`, and follow-up suggestions (`followups.py`) use a deterministic 11-rule clinical taxonomy over in-memory tool payloads. We reserve the LangGraph ReAct loop strictly for open-ended free-text synthesis."*