# MCP Service Architecture & Module Boundary Charter

## 1. Purpose

This document defines the architectural boundaries, module responsibilities, and dependency rules for `services/mcp/`. Its goal is to keep the Model Context Protocol (MCP) service a stateless, deterministic clinical data and ML inference provider with strict separation between online serving modules and offline ingestion pipelines, enforced via automated CI tests (`tests/agent/test_layer_separation.py`).

---

## 2. Service Architecture & Execution Flow

Every online MCP tool invocation flows strictly top-down through three serving layers, completely isolated from offline data-ingestion pipelines:

1. **Transport & Registration (`server.py`, `runtime.py`, `config.py`, `contracts.py`):** Initializes the FastMCP server, authenticates incoming Cloud Run requests, loads runtime environment configuration, and registers the canonical tool export surface (`tools.__all__`).
2. **Tool Handlers (`tools/`):** Validates tool arguments and executes stateless clinical retrieval (`retrieval.py`), PubMed literature search (`literature.py`), readmission risk inference (`prediction.py`), laboratory trend assembly (`labs.py`), and admission profile lookup (`admission_profile.py`).
3. **Domain Primitives & Data Dependencies (`retrieval/`, `dependencies/`):** Provides canonical clinical note section definitions (`sections.py`), query concept extraction (`concepts.py`), BigQuery/bundle feature readers (`dependencies/features/`), Vertex AI endpoint invocation (`model_endpoint.py`), and seeded synthetic laboratory trajectories (`dependencies/labs/synthetic.py`).
4. **Offline Ingestion Pipelines (`pipelines/`):** Kubeflow/Vertex AI pipeline components for chunking, embedding, indexing, and evaluating clinical notes offline. **Online serving layers (`server.py`, `tools/`, `dependencies/`) must never import `pipelines/`.**

---

## 3. Subpackage Charter (Owns vs. Forbidden)

| Subpackage / Layer | Primary Modules | Owns (Single Responsibility) | Forbidden (Hard Red Lines) |
| :--- | :--- | :--- | :--- |
| **Root Server & Contracts** | `server.py`, `runtime.py`, `config.py`, `contracts.py` | MCP server lifecycle, tool registration from `tools.__all__`, runtime environment config (`config.py`), and tool payload TypedDict contracts (`contracts.py`). | Implementing tool business logic, duplicating hardcoded tool-name lists outside `tools/__all__`, or importing `pipelines/`. |
| **`tools/`** | `retrieval.py`, `literature.py`, `prediction.py`, `labs.py`, `admission_profile.py`, `_validation.py` | Input validation, orchestrating calls to `retrieval/` and `dependencies/`, and returning typed tool payloads with explicit data provenance. | Importing `server.py`, `pipelines/`, or `services/agent/`; generating user-facing conversational prose. |
| **`retrieval/`** | `sections.py`, `chunking.py`, `concepts.py`, `index_config.py`, `notes.py`, `embed.py`, `gate.py`, `eval_report.py` | Canonical clinical note section vocabulary (`KNOWN_HEADINGS`, `SUMMARY_SECTIONS`, `INDEX_SECTIONS`), chunking primitives, embedding helpers, and RAG index configuration (`index_config.py`). | Importing `tools/`, `server.py`, `dependencies/`, or `services/agent/`. |
| **`dependencies/`** | `features/*`, `labs/synthetic.py`, `model_endpoint.py` | Reading structured clinical features from BigQuery or local bundles, invoking the Vertex AI readmission model endpoint, and generating seeded synthetic lab trajectories. | Importing `tools/`, `server.py`, `pipelines/`, or `services/agent/`. |
| **`pipelines/`** | `rag_ingest_pipeline.py`, `recall_k.py`, `components/*` | Offline Vertex AI / Kubeflow RAG index build and recall evaluation jobs. | Being imported by any online serving module (`server.py`, `tools/`, `dependencies/`). |

---

## 4. Non-Negotiable Architectural Laws

### Law 1: Strict Isolation Between Online Serving and Offline Pipelines
Online serving modules (`server.py`, `runtime.py`, `config.py`, `contracts.py`, `tools/`, `dependencies/`) must never import from `services/mcp/pipelines/`. Offline Kubeflow/Vertex pipeline dependencies are strictly confined to `pipelines/`.

### Law 2: Zero Upward Imports Into `services/agent/`
`services/mcp/` is a foundational data and tool-execution layer. No module inside `services/mcp/` may import from `services/agent/`. Shared cross-service schemas or feature-label maps must reside in `shared/` or `services/mcp/retrieval/sections.py`.

### Law 3: Single Tool Export Registry & Distinct Config Naming
`services/mcp/tools/__init__.py` (`__all__`) is the single source of truth for exported MCP tool functions within `services/mcp/`; `server.py` registers tools by iterating `tools.__all__` rather than maintaining a duplicate list. Server runtime configuration (`services/mcp/config.py`) and offline RAG index configuration (`services/mcp/retrieval/index_config.py`) must use distinct module names.

### Law 4: Explicit Provenance on Synthetic Clinical Data
Any tool payload containing seeded synthetic clinical values (such as trajectories generated by `dependencies/labs/synthetic.py`) must include an explicit machine-readable provenance field (`"source": "synthetic"`) at the top level of the tool response dictionary, rather than relying solely on free-text notes.

---

## 5. Dependency Direction & Allowed Import DAG

* **`server.py`** → may import `runtime.py`, `config.py`, `contracts.py`, and `tools/`. **May not** import `pipelines/` or `services/agent/`.
* **`tools/`** → may import `config.py`, `contracts.py`, `retrieval/`, `dependencies/`, and `shared/`. **May not** import `server.py`, `pipelines/`, or `services/agent/`.
* **`retrieval/`** → leaf domain package; may import `shared/`. **May not** import `server.py`, `tools/`, `dependencies/`, `pipelines/`, or `services/agent/`.
* **`dependencies/`** → leaf data-access package; may import `config.py`, `contracts.py`, and `shared/`. **May not** import `server.py`, `tools/`, `retrieval/`, `pipelines/`, or `services/agent/`.
* **`pipelines/`** → offline build package; may import `retrieval/` and `config.py`. **May not** be imported by any module outside `pipelines/`.