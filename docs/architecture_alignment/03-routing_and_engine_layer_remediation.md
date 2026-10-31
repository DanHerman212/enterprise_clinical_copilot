# Phase 3 Directive: Routing & Engine Layer Remediation

## Execution Governance (TESTS STRICTLY READ-ONLY)
- **Test Immutability:** Do NOT modify, add, or delete any file under `tests/`. Do not use `python3` heredocs or shell redirections to touch test files.
- **Authorized Production Scope:** `services/agent/routing/`, `services/agent/engine/`, `services/agent/postprocess/`, `services/agent/ingress/`, `services/agent/presentation/a2ui.py` (for Law 4 registry import only), and `services/agent/turn.py` (to wire `finish_reason` into `postprocess`).
- **Verification Gate:** End your execution by running and reporting:
  1. `git diff --name-only tests/` (must be completely empty)
  2. `pytest tests/agent/test_layer_separation.py`
  3. `pytest tests/agent/test_architecture_remediation_contracts.py`
  4. `pytest tests -m "not live"`

## Objective
Remediate the Routing, Ingress, and Engine layers to turn the Phase 3 AST gates (`test_law2_*`, `test_law4_*`) and Behavioral Contracts (`Contract 1` and `Contract 2`) green without breaking any existing baseline tests.

## Required Production Changes

### 1. Charter 13 Law 4 — Consolidate All Duplicate Tool Registries into `routing/`
Eliminate all five duplicate tool-name tuples/sets flagged by `test_law4_no_duplicate_tool_registry_literals`:
- Define canonical tool-name constants in `services/agent/routing/` and import them at every call site:
  - `ingress/contracts.py:94-95` (`PASSAGE_TOOLS`) and `:109` (`RESULT_KEPT_TOOLS`)
  - `engine/graph.py:178-192` (`FULL_FETCH_*` tool sets)
  - `engine/mcp_client.py:64` (`EXTERNAL_TOOL`) and `:404-411` (passage-tool tuple)
  - `presentation/a2ui.py:121-123` (`rag_payload` passage-tool tuple)

### 2. Contract 2 — Intent-Scoped Tool Binding & Per-Tool Call Bounds
- **Literature Intent Scoping (`routing/` & `engine/graph.py`):** Update `_freetext_allowed` (`engine/graph.py:403-419`) using a canonical profile in `routing/` so that when `wants_literature` is `True` (without explicit lab or medication intent), the allowed toolset is scoped to `(routing.LITERATURE_TOOLS | routing.PASSAGE_TOOLS) - {"discharge_medications"}` (excluding `get_lab_trends` and `discharge_medications`).
- **Per-Tool Call Limit (`engine/mcp_client.py`):** Define `PER_TOOL_CALL_LIMIT` (≤ 2) in `engine/mcp_client.py` and enforce it per tool within a single turn alongside `MAX_TOOL_CALLS_PER_TURN` so redundant `search_literature` retries cannot loop across all 5 rounds.
- **Single Passage Counter Helper (`engine/graph.py:1276` & `engine/mcp_client.py:837`):** Consolidate the running passage offset calculation through `postprocess/citations.py` helpers.

### 3. Charter 13 Law 2 & Contract 1 — Move Fallback & Safety Cutoff Resolution to `postprocess/`
Eliminate all fallback prose selection and `finish_reason` branching from `engine/` (`test_law2_the_engine_never_selects_a_fallback_sentence`) and satisfy `Contract 1`:
- **Remove from `engine/`:**
  - Remove fallback sentence selection (`SAFETY_WITHHELD_SUMMARY`, `SILENT_TURN_SENTENCE`, `OUTPATIENT_EHR_NOTICE`) from `engine/deterministic.py:506, 611-629` and `engine/model.py:352-353`.
  - Remove `replace_final_text` blanking calls from `engine/graph.py:1122, 1144`.
  - Have `engine/` return the raw model output, tool calls, and `finish_reason` cleanly to `postprocess/`.
- **Resolve in `postprocess/` (`guardrail.py` / `answer.py`) and wire in `turn.py`:**
  - Update `guard_answer` (and its call sites in `turn.py`) to accept `finish_reason` (and encounter/outpatient context as needed).
  - When `finish_reason == "SAFETY"` occurs—even if partial tokens were already emitted mid-stream—discard the truncated fragment, replace the answer with `SAFETY_WITHHELD_SUMMARY` (when records exist) or `SILENT_TURN_SENTENCE` (when no records exist), and append `"safety_withheld"` to `guardrail_flags` so `ingress/http.py:_prose_settled` triggers a visible replacement in the browser.