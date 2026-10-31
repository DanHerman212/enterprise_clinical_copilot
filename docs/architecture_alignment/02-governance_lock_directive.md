# Phase 2 Directive: Baseline Equivalence & Governance Lock

## Execution Governance (TESTS WRITABLE, PRODUCTION CODE STRICTLY READ-ONLY)
- **Production Code Read-Only:** Do NOT modify any file under `services/`, `evaluation/`, or `docs/`.
- **Test-Only Scope:** Only files under `tests/agent/` may be created or updated in this phase.
- **TDD Contract Rule:** Every new test added in this phase encodes a target architectural requirement from `01a-audit_report.md` and Charter 13. Tests that assert the current passing baseline must pass immediately; tests that enforce the new Phase 3 and Phase 4 target architecture must fail cleanly against the un-remediated code (proving each test catches its targeted violation) without breaking pytest collection.
- **Verification Gate:** End by running:
  1. `git status --short` and `git diff --name-only services/ evaluation/` (must be empty)
  2. `pytest tests/agent/test_layer_separation.py` (showing the new AST gates catching the current drift)
  3. `pytest tests/agent/test_architecture_remediation_contracts.py` (showing the new behavioral contracts failing cleanly against the un-remediated code)

---

## Objective
Upgrade `tests/agent/test_layer_separation.py` and create `tests/agent/test_architecture_remediation_contracts.py` to mechanically lock all four remediation pillars identified in `01a-audit_report.md` before any production code is edited in Phases 3 and 4.

---

## Required Deliverables

### 1. Upgrade AST Fitness Functions (`tests/agent/test_layer_separation.py`)
Close the 4 structural blind spots identified in Section 3 of `01a-audit_report.md` and remove the dead loop at `test_layer_separation.py:294-296`:

1. **Charter 13 Law 1 (Single Citation Owner — Behavioral AST Gate):**
   - Forbid `presentation/` modules (`a2ui.py`, `canvas.py`, `domain.py`) from calling `renumber_citations`, `citation_remap`, or `len(citable_passages(...))` (catches L1-1 and L1-2 at `a2ui.py:363-364, 380`).
   - Forbid modules outside `postprocess/citations.py` (including `evaluation/`) from compiling raw inline citation marker regexes (`r"\^\[(\d+)\]"`), requiring them to use `postprocess/citations.py` helpers (catches L1-4).
2. **Charter 13 Law 2 (Single Prose Mutation & Fallback Owner — Behavioral AST Gate):**
   - Forbid `engine/` modules (`deterministic.py`, `graph.py`, `model.py`, `mcp_client.py`) from referencing or returning fallback constants (`SILENT_TURN_SENTENCE`, `OUTPATIENT_SILENT_TURN_SENTENCE`, `SAFETY_WITHHELD_SUMMARY`, `OUTPATIENT_EHR_NOTICE`) or calling `replace_final_text` to blank out answers (catches L2-1, L2-2, L2-3).
3. **Charter 13 Law 4 (Single Tool Registry — AST Tuple/Set Literal Gate):**
   - Forbid modules outside `routing/` (`ingress/contracts.py`, `engine/graph.py`, `engine/mcp_client.py`, `presentation/a2ui.py`) from defining tuple, set, or frozenset literals containing 2 or more canonical MCP tool wire names (catches all 5 duplicate registries in L4).

### 2. Behavioral Contract Tests (`tests/agent/test_architecture_remediation_contracts.py`)
Create `tests/agent/test_architecture_remediation_contracts.py` with focused unit/contract tests covering these four target invariants:

1. **Contract 1 — Mid-Stream `SAFETY` Cutoff & Visible Replacement Flag:**
   - When `engine/` returns a turn where `finish_reason == "SAFETY"` AND partial tokens were already emitted (e.g., `"Accidental hypothermia is"`), `postprocess` (`guard_answer`) discards the broken partial fragment, replaces it with `SAFETY_WITHHELD_SUMMARY` (when records exist) or `SILENT_TURN_SENTENCE` (when no records exist), and appends `"safety_withheld"` to `guardrail_flags`.
   - `ingress/http.py:_prose_settled` treats `"safety_withheld"` as a visible replacement so the browser replaces the streamed fragment.
2. **Contract 2 — Free-Text Literature Intent Tool Scoping & Per-Tool Call Cap:**
   - When a free-text query signals `wants_literature=True` without explicit lab or medication intent (e.g., `"what does pubmed say about hypothermia?"` inside an active patient encounter), `_freetext_allowed` binds `routing.LITERATURE_TOOLS | routing.PASSAGE_TOOLS` and excludes `get_lab_trends` and `discharge_medications`.
   - `mcp_client.py` enforces a per-tool call cap on `search_literature` within a single turn so a literature query cannot loop through 4+ redundant `search_literature` calls.
3. **Contract 3 — Canonical `EvidenceLedger` & Honest Trace Attribution:**
   - `postprocess/citations.py` constructs a canonical `EvidenceLedger` (or settled evidence structure) that resolves cited EHR passages and PubMed literature records once per turn without `presentation/a2ui.py` mutating answer prose or re-computing `len(citable_passages(...))` offsets.
   - On a turn combining EHR passages and PubMed articles, the final `answer` recorded on Langfuse (`set_trace_io` in `turn.py`) is byte-for-byte identical to the `answer` returned in the `a2ui` / HTTP payload (closing the L5 caveat).
4. **Contract 4 — Turn-Scoped `presentation/canvas.py` Surface Resolution:**
   - On a literature turn (`wants_literature=True` / `search_literature` payload), `compose_risk_canvas` / `a2ui.py` renders the literature view without appending an unrequested Lab Sparkline table or pulling conversation-wide `rag_payload` fallback cards from prior turns.