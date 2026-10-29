# Agent Service Architecture & Module Boundary Charter

## 1. Purpose

This document defines the architectural boundaries, module responsibilities, and dependency rules for `services/agent/`. Every module in the agent service has a single, well-defined scope. Cross-package imports and functional responsibilities are strictly governed by this charter and enforced in automated CI tests (`tests/agent/test_layer_separation.py`).

---

## 2. Request Execution Pipeline

Every user turn—whether invoked via a deterministic slash command (`/labs`, `/ehr`, `/meds`, `/pubmed`) or the free-text reasoning loop—flows through a single linear pipeline orchestrated by `services/agent/turn.py`:

1. **Ingress (`ingress/`):** Authenticates the caller, validates the incoming wire payload, initializes request context, and streams status stages.
2. **Routing (`routing/`):** Resolves slash commands, evaluates routing predicates, and constructs the admission context header.
3. **Engine (`engine/`):** Executes the LangGraph reasoning loop or deterministic command workflow, invokes MCP tools, and returns the raw model response alongside the recorded tool-call payloads. **The engine never mutates final prose or strips citations.**
4. **Post-Processing (`postprocess/`):** Applies deterministic clinical guardrails, enforces canonical citation numbering and bounds, resolves fallback/abstention prose when required, and generates contextual follow-up suggestions.
5. **Presentation (`presentation/`):** Transforms the settled, post-processed answer and tool payloads into UI artefacts (resolved source cards, A2UI surfaces, and clinical risk canvases). **Presentation never alters citation indices or mutates agent state.**
6. **Telemetry (`telemetry/`):** Records the final, post-processed served answer on the root Langfuse trace (`set_trace_io`), emits trace spans, and logs runtime metrics.

---

## 3. Subpackage Charter (Owns vs. Forbidden)

| Subpackage | Primary Modules | Owns (Single Responsibility) | Forbidden (Hard Red Lines) |
| :--- | :--- | :--- | :--- |
| **Root Orchestrator** | `turn.py` | End-to-end turn orchestration (`engine` → `postprocess` → `presentation` → `telemetry`); single root trace I/O settlement. | Implementing domain regexes, prompt templates, or direct MCP tool rendering. |
| **`ingress/`** | `http.py`, `contracts.py`, `run.py` | HTTP/SSE transport, authentication, wire schemas (`AgentSuccess`, `ResolvedSource`), and SSE stream reconciliation (`_prose_settled`). | Clinical domain logic, prompt templates, citation math, or guardrail evaluation. |
| **`routing/`** | `routing.py`, `admission_context.py`, `encounter.py` | Slash-command dispatch table, routing predicates, canonical tool wire-name registries (`PASSAGE_TOOLS`, `LITERATURE_TOOLS`, `TOOL_WIRE_NAMES`), and admission manifest context headers. | LLM system/formatting prompt instructions (`*_FORMATTING`), user-facing fallback prose, or response post-processing. |
| **`engine/`** | `graph.py`, `deterministic.py`, `mcp_client.py`, `prompts.py`, `model.py` | LangGraph state orchestration (`graph.py`), command execution (`deterministic.py`), MCP tool dispatch (`mcp_client.py`), LLM client setup (`model.py`), and **all** system and command-formatting prompts (`prompts.py`). | Mutating or sanitizing model answer prose; stripping citations; authoring user-facing fallback strings outside `postprocess/`; importing `presentation/`. |
| **`postprocess/`** | `guardrail.py`, `citations.py`, `answer.py`, `followups.py` | **All** prose validation and mutation: canonical citation numbering, slot counting, and range enforcement (`citations.py`); clinical safety and numerical verification (`guardrail.py`); deterministic narratives and fixed fallback/abstention sentences (`answer.py`); follow-up chip generation (`followups.py`). | Importing `presentation/`; mutating internal LangGraph state objects directly. |
| **`presentation/`** | `a2ui.py`, `canvas.py`, `domain.py` | Packaging settled prose and tool payloads into UI structures: source card resolution (`a2ui.py`), clinical risk/comparison canvases (`canvas.py`), and display formatting (`domain.py`). | Mutating answer prose or shifting citation numbers; reading hidden process-global `ContextVar` state from `routing/encounter.py`. |
| **`telemetry/`** | `observability.py`, `stages.py` | Langfuse client lifecycle, span/trace I/O recording, and SSE stage progress emission. | Importing `engine/`, `postprocess/`, or `presentation/`; defining duplicate tool-name registries. |

---

## 4. Non-Negotiable Architectural Laws

### Law 1: Single Owner for Citations (`postprocess/citations.py`)
`postprocess/citations.py` is the sole module permitted to define what constitutes a citable unit (`citable_passages`, `passage_total`, `citation_bound`), stamp positional `^[n]` markers onto tool payloads (`render_numbered`), or parse, expand, and strip `^[...]` markers from model prose (`enforce_citations`). No other module may define a citation-marker regular expression or maintain a separate slot counter.

### Law 2: Single Owner for Answer Prose Mutation & Fallbacks (`postprocess/`)
`engine/` orchestrates state and returns raw model outputs; it never regex-edits final prose. All prose sanitization, deterministic narrative assembly (`medication_narrative`), and fixed fallback/abstention sentences (`SILENT_TURN_SENTENCE`, `OUTPATIENT_SILENT_TURN_SENTENCE`, `OUTPATIENT_EHR_NOTICE`, `GUARDED_ABSTENTION`, `no_evidence_answer`) reside exclusively in `postprocess/answer.py` and `postprocess/guardrail.py`.

### Law 3: Single Owner for Prompt Instructions (`engine/prompts.py`)
All instructions that govern LLM generation—including `SYSTEM_PROMPT`, per-command formatting prompts (`LABS_FORMATTING`, `EHR_FORMATTING`, `MEDS_FORMATTING`, `LITERATURE_FORMATTING`), `FORMATTER_PREAMBLE`, `RETRIEVAL_DEMAND`, and `EHR_RECORD_QUERY`—reside exclusively in `engine/prompts.py`. `routing/` decides *what* to execute; `engine/prompts.py` defines *how* the model writes.

### Law 4: Single Owner for Tool Registries (`routing/`)
Canonical tool wire names, tool capability groupings (`PASSAGE_TOOLS`, `LITERATURE_TOOLS`), and human-readable tool stage labels are declared once in `routing/` and imported by `engine/`, `postprocess/`, and `telemetry/`. Hardcoding duplicate tool-name tuples across modules is prohibited.

### Law 5: Honest Trace Attribution (`turn.py` → `telemetry/observability.py`)
Langfuse root trace output (`observability.set_trace_io`) must record the exact final prose served to the caller. Trace output is recorded once in `turn.py` **after** `postprocess` (`guard_answer`) completes—never inside `engine/graph.py` on intermediate pre-postprocess state.

---

## 5. Dependency Direction & Allowed Import DAG

Packages must respect a strict top-down dependency graph. Lower layers may never import higher layers, and sibling leaf packages may only import across explicitly declared edges:

* **`ingress/`** → may import `turn.py`, `routing/`, `telemetry/`, `presentation/` (wire types only).
* **`turn.py`** → may import `routing/`, `engine/`, `postprocess/`, `presentation/`, `telemetry/`, and `ingress/contracts.py` (response schema validation).
* **`presentation/`** → may import `postprocess/citations.py` (read-only source resolution) and shared domain/label utilities. **May not** import `engine/`, `ingress/`, or `routing/encounter.py`.
* **`engine/`** → may import `routing/`, `postprocess/citations.py` (for tool-result numbering), `postprocess/answer.py`, and `telemetry/`. **May not** import `presentation/` or `ingress/`.
* **`postprocess/`** → may import `routing/` and `telemetry/`. **May not** import `engine/`, `presentation/`, or `ingress/`.
* **`telemetry/`** → may import `routing/` (for tool label registries). **May not** import `ingress/`, `engine/`, `postprocess/`, or `presentation/`.
* **`routing/`** → leaf foundation package. **May not** import `ingress/`, `engine/`, `postprocess/`, `presentation/`, or `telemetry/`.