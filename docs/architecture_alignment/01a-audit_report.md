# Phase 1 — Read-Only Backend Architecture Drift Audit

**Date:** 2026-10-31
**Scope:** `services/agent/`, `services/mcp/`, `evaluation/`
**Charters audited against:** `docs/architecture/13-agent_module_boundaries.md`, `14-mcp_service_architecture.md`, `15-evaluation_harness_architecture.md`
**Mode:** read-only. No file under `services/`, `tests/`, `evaluation/` or `docs/` was written during the audit.

## Method

Read the three charters, then walked `services/agent/` (7 packages), `services/mcp/` (5 layers) and `evaluation/` (platform, legacy, inventory) by grep and file read. Every finding below cites a `file:line` that was opened, except where explicitly marked *not verified*.

## Verdict summary

| Charter | Laws passed | Violations |
| :--- | :--- | :--- |
| 13 — `services/agent/` | Law 3, Law 5 (with a caveat) | Law 1 ×4, Law 2 ×3, Law 4 ×1 (five duplicate registries) |
| 14 — `services/mcp/` | Laws 1–4 | none |
| 15 — `evaluation/` | Laws 1–4 | none; one marker regex outside the marker owner (no law forbids it) |

---

## 1. Charter law drift scorecard

### 1.1 Charter 13 — `services/agent/`

| # | Law | Verdict | Evidence |
| :--- | :--- | :--- | :--- |
| L1-1 | Citations single owner | **Major** | `presentation/a2ui.py:363-364` calls `renumber_citations` + `citation_remap` and **publishes the rewritten prose** as `answer`. Marker transformation and prose rewrite in the presentation layer. |
| L1-2 | No separate slot counter | **Major** | `presentation/a2ui.py:380` — `len(citable_passages(...))`, a citable-unit count outside the owner. It re-derives what `citations.citation_bound` already computes. |
| L1-3 | No separate slot counter | **Minor** | `engine/graph.py:1276` maintains the running `offset` across replayed turns (uses `citations.passage_total`, owns the counter); `engine/mcp_client.py:837` `numbered += _passage_total(shown)` is a second running counter. Charter §5 permits `engine/` → `citations.py` for tool-result numbering, so these are lawful-but-duplicated: the two places a third numbering rule could appear. |
| L1-4 | No marker regex outside the owner | **Minor** | `evaluation/platform/evaluators.py:207` `_INLINE_MARKER = re.compile(r"\^\[(\d+)\]")`; `evaluation/inventory/multiturn_chip_audit.py:93` `_CITE`. Outside Charter 13's walk, and Charter 15 has no rule against them — a gap in the law, not only in the code. |
| L1-5 | Read-only resolution permitted | **Pass** | `a2ui.py:954`, `canvas.py:88` read `citable_passages` to resolve sources; expressly allowed by Charter 13 §5. |
| L2-1 | Fallback resolution lives in `postprocess/` | **Major** | `engine/deterministic.py:611-629` inspects `finish_reason`, chooses between `SAFETY_WITHHELD_SUMMARY` and `SILENT_TURN_SENTENCE`, and writes `deterministic_answer`. The strings belong to `answer.py`; the **decision** is the engine's. |
| L2-2 | Same | **Major** | `engine/deterministic.py:506` selects and assigns `OUTPATIENT_EHR_NOTICE`; `engine/model.py:352-353` `silent_turn_sentence()` returns a fallback sentence from `engine/`. |
| L2-3 | Engine never mutates prose | **Minor** | `engine/graph.py:1122,1144` call `answer_text.replace_final_text(...)`, including `replace_final_text(state, "")`. The mutation goes through the owner's API, but the decision to blank the answer is the engine's. |
| L3 | Prompts in `engine/prompts.py` | **Pass** | Every constant (`SYSTEM_PROMPT`, `*_FORMATTING`, `FORMATTER_PREAMBLE`, `RETRIEVAL_DEMAND`, `EHR_RECORD_QUERY`) is defined there and imported by callers. `graph._context_block` assembles a block *from* those constants — composition, not authoring. |
| L4 | Tool registries in `routing/` | **Major** | Duplicate tool-name **tuples/sets** outside `routing/`: `ingress/contracts.py:94-95` (a second `PASSAGE_TOOLS`), `:109` `RESULT_KEPT_TOOLS`, `engine/graph.py:178-192` (`FULL_FETCH_*` sets), `engine/mcp_client.py:64` `EXTERNAL_TOOL` and `:404-411` (another passage-tool tuple), `presentation/a2ui.py:121-123` (`rag_payload`'s `("rag_search","rag_search_sections","discharge_medications")`). Single-name comparisons (`followups.py:176,218,262,649`, `guardrail.py:296,385,406`) are idiomatic and not counted. |
| L5 | Trace written once in `turn.py` | **Pass, with a caveat** | `set_trace_io` appears only in `turn.py:249,566` and `observability.py`. But `turn.py:249` records `guarded["answer"]` while the served payload carries the **renumbered** answer (`a2ui.py:363`) — trace output and delivered prose can differ, which is L1-1 seen from the telemetry side. |

### 1.2 Charter 14 — `services/mcp/`

| Law | Verdict | Evidence |
| :--- | :--- | :--- |
| 1 — online never imports `pipelines/` | **Pass** | No hit in `server.py`, `runtime.py`, `config.py`, `contracts.py`, `tools/`, `dependencies/`. |
| 2 — zero upward imports into `services/agent/` | **Pass** | The only textual match is a docstring in `tools/admission_profile.py:6`. |
| 3 — `tools.__all__` is the single registry | **Pass** | `tools/__init__.py:9-17` (7 tools); `server.py:91` iterates `tools.__all__`. |
| 4 — explicit synthetic provenance | **Pass** | `dependencies/labs/synthetic.py:51 SOURCE = "synthetic"` → `tools/labs.py:119` top-level `"source"`. |

### 1.3 Charter 15 — `evaluation/`

| Law | Verdict | Evidence |
| :--- | :--- | :--- |
| 1 — `passage_total` from `citations` | **Pass** | `platform/task.py:21` imports it; `:75` returns it. |
| 2 — one `gate_verdict`, one `DIMENSIONS` | **Pass** | `schemas.py:38,44,48`; `evaluators.py:485`; callers `evaluators.py:524` and `run_evals.py:323`. No duplicate `pass_rate >=` / `safety_failures ==` / `unattributed ==` anywhere else. |
| 3 — no legacy `evaluation/agent` dependency | **Pass in substance** | No import of `evaluation.agent`; the two hits are docstrings describing its removal. `sys.path.insert(REPO)` remains in `dry_run.py:27`, `sync_dev_suite.py:29`, `run.py:24`, `runner.py:31` — inserting the repo root to reach `services.*`, which the law's letter permits. |
| 4 — derived command→tool mapping | **Pass** | `platform/case_set.py:20,56` derives `_BY_WIRE` from `agent_routing.COMMANDS`. |

---

## 2. Multi-turn evidence and surface-scoping defects

### 2.1 Runaway free-text tool loops

`routing.FREETEXT_TOOLS` (`routing.py:306`) is the whole standing set — `rag_search`, `rag_search_sections`, `discharge_medications`, `get_lab_trends`, `admission_profile`. `graph._freetext_allowed` (`graph.py:403-419`) **only unions into it**:

```python
allowed = set(routing.FREETEXT_TOOLS)
if wants_literature and not declared_follow_up:
    allowed |= routing.LITERATURE_TOOLS          # adds, never narrows
```

A literature question therefore binds `search_literature` *beside* every chart tool. The only bound is `MAX_TOOL_CALLS_PER_TURN = 5` (`mcp_client.py:399`), with `RECURSION_LIMIT = 25` as a circuit breaker (`graph.py:138`). One tool round costs three supersteps, so five rounds is the "18-step" workflow a reader sees.

Live evidence from the captured trace: one free-text literature turn made **five** tool calls — `rag_search` plus four `search_literature` (`079be88f…`).

There is no stage at which a literature turn is scoped *to* literature. This is the Phase 4 remediation and it is one function.

### 2.2 Presentation canvas bleed

`compose_risk_canvas` receives both turn-scoped and conversation-wide inputs (`a2ui.py:543-565`):

- turn-scoped: `predict_payload(tool_calls)`, `labs`, `medications`
- **conversation-wide**: `rag = rag_payload(evidence_calls)` (`a2ui.py:374`), and `phenotype=turn_phenotype(tool_calls, evidence_calls)` (`:561`)

The literature swap itself is well gated (`canvas.py:65-110` `_literature_view`, `:114` `_draws_matrix`). The bleed is elsewhere: the **lab card is appended unconditionally** when a `get_lab_trends` payload is present (`canvas.py:~305-315`), and the literature branch (`~340+`) appends the comparison and records *after* it without removing it. A literature question that also called a lab tool draws a sparkline panel under a literature answer — reachable because of 2.1. Second channel: `rag` being conversation-wide lets the canvas resolve a `SourceCard` from an earlier turn's passages whenever `source` is `None`.

### 2.3 Sparse vs. sequential citation indexing

Numbering is **positional, never display-sequential**, in two places:

- `citations._renumber_citations` returns the identity map `{n: n}` — numbers are preserved by design, so the reader sees `4`, `7`, `18` and the gaps between them.
- `a2ui.py:380` continues the record numbering after **every citable unit in the conversation's evidence**, not after the units actually cited. On a turn where the conversation has shown 17 note units, the first published record is `18`.

Both are the same absence: no component owns "the display sequence of this turn's evidence". That is the Evidence Ledger, and Phase 5 is where it belongs.

### 2.4 A withheld summary swaps silently

`ingress/http.py:463` `_prose_settled` decides visible-vs-silent from `payload["guardrail_flags"]` alone. The SAFETY branch replaces announced partial prose with `SAFETY_WITHHELD_SUMMARY` and sets **no** flag (`deterministic.py:625`), so a reader who watched a sentence arrive and then lose it is told the swap was silent — the case the Dual Swap Protocol exists to make visible. This one is in the most recent change, not the older code.

---

## 3. Structural test coverage gap

`tests/agent/test_layer_separation.py` (789 lines) is a real gate: import DAG, `test_every_module_binds_the_globals_it_uses`, patch-target reachability, and one test per law. What it cannot see:

1. **Law 1 is a text scan for `\^\[`** (`:282-303`) over `services/agent/` only. L1-2 (`a2ui.py:380`, a *count* with no marker literal) passes it, and the two marker regexes in `evaluation/` are outside the walk entirely.
2. **Law 2 is `hasattr` bookkeeping** (`:305-321`): it asserts the *names* are not defined in `engine.model`/`routing`. L2-1 and L2-2 are invisible to it because the **selection** of a fallback is not a definition.
3. **Law 4 pins the canonical registries and their identity** (`:350-363`) — `PASSAGE_TOOLS`, `LITERATURE_TOOLS`, `stages.TOOL_LABELS is routing.TOOL_LABELS`. It does not forbid *additional* ad-hoc tool-name tuples, which is how five duplicates survive (L4 above).
4. **No test covers the empirical rules this phase audits**: that a turn's evidence is resolved once, that an offset is not recomputed downstream, that a canvas is scoped to its turn, or that display numbers are sequential. Those are Phase 3 fitness functions, and their absence is why every violation above has been green in CI.

Artifact worth cleaning in Phase 2: `test_layer_separation.py:294-296` builds `offenders` from `for path, _ in []` before `root` is assigned — dead code that reads as a leftover.

---

## 4. Verification gate

Captured at the end of the audit, before the tree was cleaned:

```
$ git status --short
M  .gitignore
 D docs/code_comprehension/00-master_alignment_plan.md
 D docs/code_comprehension/01-dead_code_and_hygiene.md
 D docs/code_comprehension/02-contracts_and_boundaries.md
 D docs/code_comprehension/03-orchestrator_and_model_runtime.md
 D docs/code_comprehension/04-tools_models_and_data_isolation.md
 D docs/code_comprehension/05-frontend_islands_modularization.md
 D docs/code_comprehension/05b-streaming_ux_remediation.md
 D docs/code_comprehension/06-governance_observability_and_ci.md
 D docs/code_comprehension/07-package_layout_reorganization.md
 D docs/code_comprehension/agent_surface_architecture_code_comprehension_guide.md
?? docs/architecture_alignment/

$ git diff --stat
 10 files changed, 2942 deletions(-)
```

None of that was produced by the audit: the ten `docs/code_comprehension/*` deletions were working-tree deletions that predate the inspection, `.gitignore` was staged from earlier work, and `docs/architecture_alignment/` was untracked. No path under `services/`, `tests/`, `evaluation/` or `docs/` was modified, added or deleted by the audit. The tree has since been cleaned and committed by the operator.
