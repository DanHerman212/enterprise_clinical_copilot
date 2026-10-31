# Phase 1 Directive: Read-Only Backend Architecture Drift Audit

## Execution Governance (STRICT READ-ONLY)
- **Zero File Modifications:** Do NOT create, edit, format, or delete any tracked or untracked file in the repository.
- **Verification Gate:** End your response by running `git status --short` and `git diff --stat` to prove the working tree is 100% untouched.

## Objective
Perform a comprehensive, file-by-file and line-by-line read-only audit of `services/agent/`, `services/mcp/`, and `evaluation/` against the three target architecture charters:
1. `docs/architecture/13-agent_module_boundaries.md` (or local equivalent charter for `services/agent/`)
2. `docs/architecture/14-mcp_service_architecture.md` (for `services/mcp/`)
3. `docs/architecture/15-evaluation_harness_architecture.md` (for `evaluation/`)

## Required Audit Deliverables

### 1. Charter Law vs. Actual Code Drift Scorecard
For every Law and Subpackage Charter row in Charters 13, 14, and 15, verify whether the actual implementation obeys both the import rules AND the behavioral responsibility boundaries. Report every violation with `file_path:line_number` and a concise explanation:
- **Charter 13 Law 1 (Single Owner for Citations — `postprocess/citations.py`):** List every module and function outside `postprocess/citations.py` that scans `tool_calls` / `replayed_tool_calls` to extract citations, counts citable passages/articles, computes citation index offsets (`start=...`), or parses/transforms citation markers.
- **Charter 13 Law 2 (Single Owner for Answer Prose Mutation & Fallbacks — `postprocess/`):** List every place in `engine/` (e.g., `deterministic.py`, `graph.py`) or `presentation/` that inspects `finish_reason`, checks for empty/truncated prose, mutates model answer text, or assigns fallback/abstention strings (`SILENT_TURN_SENTENCE`, `SAFETY_WITHHELD_SUMMARY`, etc.). Specifically inspect how mid-stream `finish_reason == "SAFETY"` or `MAX_TOKENS` truncations (where partial tokens like `"Accidental hypothermia is"` were already emitted before the stop) are currently handled across both streaming (`ingress/run.py`) and blocking paths.
- **Charter 13 Law 3, 4, 5 (Prompts, Tool Registries, Trace Attribution):** Flag any prompt strings outside `engine/prompts.py`, duplicate tool-name tuples outside `routing/`, or trace I/O writes outside `turn.py`.
- **Charter 14 Laws 1–4 (`services/mcp/`):** Check online-vs-offline (`pipelines/`) isolation, upward imports, `tools.__all__` registration, and `"source": "synthetic"` provenance on synthetic lab data.
- **Charter 15 Laws 1–4 (`evaluation/`):** Check how `evaluation/platform/task.py` counts and stamps citations compared to `postprocess/citations.py`, check `gate_verdict` / `DIMENSIONS` single-source rules, and check for any lingering dependencies on `evaluation/agent/`.

### 2. Multi-Turn Evidence & Surface Scoping Defects
Trace the exact code paths responsible for the three multi-turn anomalies observed during live clinical testing:
1. **Runaway Free-Text Tool Loops:** When a free-text query asks a literature question (e.g., `"what does pubmed say about hypothermia?"`) inside an active patient thread, trace how `routing/` and `engine/graph.py` bind tools and why the agent can execute an 18-step loop calling EHR/lab tools instead of scoping to literature retrieval.
2. **Presentation Canvas Bleed:** Trace how `presentation/a2ui.py` and `presentation/canvas.py` select which structured surfaces (lab sparkline tables, readmission risk canvas, literature drawer) to render on a turn, and why prior or incidental lab/chart tool payloads bleed onto a literature turn.
3. **Sparse vs. Sequential Citation Indexing:** Trace how `postprocess/citations.py` and `presentation/a2ui.py` number EHR note passages vs. PubMed articles on mixed or multi-turn threads (e.g., why the first PubMed badge on a turn can be numbered `18` instead of sequential display order, and where `EvidenceLedger` consolidation is needed).

### 3. Existing AST & Boundary Test Coverage Gap (`tests/agent/test_layer_separation.py`)
Inspect `tests/agent/test_layer_separation.py` and list which Charter 13, 14, and 15 rules are currently enforced by AST/import tests versus which behavioral boundaries currently lack automated structural enforcement.