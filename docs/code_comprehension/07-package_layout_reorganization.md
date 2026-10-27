# Phase 7: Agent Package Consolidation & Zero-Shim Reorganization

**Status:** `PENDING (Runs After Phase 6)`  
**Target Layers:** Layer 3 (`docs/architecture/03-orchestrator.md`) & Layer 4 (`docs/architecture/04-model_runtime.md`)  
**Reference Standard:** `docs/code_comprehension/agent_surface_architecture_code_comprehension_guide.md` (§3 & §4)  
**Repositories in Scope:** `enterprise_clinical_copilot` & `danielmherman` (`demo/fixtures.py`, `demo/test_chips.py`, `demo/test_wire_contract.py`, `demo/test_cohort_parity.py`)  

---

## 1. Executive Objective (Why This Phase Exists)

Over Phases 1–6, `services/agent/` grew to **28 flat Python files** (`turn.py`, `__init__.py`, and 26 modules). To avoid editing test imports during earlier phases, helper fragments were split into micro-files (`a2ui_envelope.py`, `questions.py`, `served_admissions.py`), private aliases were re-exported across modules (`graph._build_llm`, `http._compose_success`), and circular dependencies were worked around with function-local imports (`deterministic.py` importing `_build_llm`, `_emit`, and `_execute_tool_calls` from `graph.py` at call time).

**The Goal of Phase 7 is Code Comprehension and Zero Bloat—NOT Test-Shim Preservation:**
1. **Consolidate 28 Flat Files into 18 Cohesive Modules across 6 Domain Subpackages** (`ingress/`, `routing/`, `engine/`, `postprocess/`, `presentation/`, `telemetry/`) with `turn.py` as the sole root orchestrator spine.
2. **Zero Root Compatibility Shims:** Do **not** leave 24 stub `.py` files at the root of `services/agent/` to avoid updating imports. Update `tests/`, `evaluation/`, `Dockerfile`, and `danielmherman/demo/` to import from the canonical subpackage locations.
3. **Eliminate Circular & Function-Local Late Imports:** Break the `graph.py` $\leftrightarrow$ `deterministic.py` cycle so no module in `services/agent/` uses function-local imports to dodge a circular dependency.
4. **Lock the Architecture with AST Governance Tests:** Extend `tests/agent/test_layer_separation.py` to enforce the exact 15-module manifest, the one-way subpackage import DAG, and zero stale `.pyc` / `.DS_Store` artifacts.

---

## 2. Target 6-Subpackage Architecture (18 Canonical Modules)

Every file in `services/agent/` must map to this exact target tree. **No other `.py` files (other than `__init__.py` and `turn.py`) may exist at the root of `services/agent/`.**

```text
services/agent/
├── __init__.py
├── turn.py                        # [THE SPINE] Root turn orchestrator: validate -> route -> execute -> verify -> present -> record
├── ingress/
│   ├── __init__.py
│   ├── http.py                    # Starlette routes (/ask, /ask/stream, /health), auth header & SSE framing
│   ├── contracts.py               # Request/response wire validation + served cohort allowlist (absorbs served_admissions.py)
│   └── run.py                     # CLI developer entrypoint
├── routing/
│   ├── __init__.py
│   ├── routing.py                 # Command table (/ehr, /meds, /labs, /pubmed), ReAct tool gates & question templates (absorbs questions.py)
│   └── admission_context.py       # Pre-flight encounter header manifest (get_admission_profile)
├── engine/
│   ├── __init__.py
│   ├── deterministic.py           # Slash-command fast-path executor (/ehr, /meds, /labs, /pubmed)
│   ├── graph.py                   # LangGraph ReAct state machine + multi-turn replay
│   ├── model.py                   # Layer 4 Gemini gateway: client factory, streaming turn & response/token parser (merges model_runtime.py + model_turn.py)
│   ├── prompts.py                 # System prompt & clinical grounding behavioral rules
│   └── mcp_client.py              # MCP tool bridge (update REPO_ROOT to Path(__file__).resolve().parents[3])
├── postprocess/
│   ├── __init__.py
│   ├── guardrail.py               # Deterministic post-generation factual & numeric evidence checks
│   ├── citations.py               # Inline marker reconciliation, list deduplication & 1..K renumbering
│   └── followups.py               # Deterministic P1-P9 + literature follow-up chip selector
├── presentation/
│   ├── __init__.py
│   ├── a2ui.py                    # Presentation composer, envelope constants & citation/reference row builders (absorbs a2ui_envelope.py + note_presentation.py)
│   ├── canvas.py                  # A2UI v0.9 visual component assembly (renamed from risk_canvas.py)
│   └── domain.py                  # Clinical ML feature labels, admission phenotypes & cohort benchmarks (merges feature_labels.py + admission_phenotypes.py + literature_benchmarks.py)
└── telemetry/
    ├── __init__.py
    ├── observability.py           # Langfuse callback handler, span helpers & immutable execution record (absorbs chain.py)
    └── stages.py                  # SSE interim progress vocabulary & emit() helper
```

---

## 3. Mandatory Module Consolidations & Cleanups

### 3.1 Consolidations (14 Domain Files into 6 Subpackage Modules; 12 More Move 1:1)
1. **`ingress/contracts.py` $\leftarrow$ `contracts.py` + `served_admissions.py`:**
   * Fold `SERVED_HADM_IDS` and `is_served()` into `ingress/contracts.py`. Cohort membership is an ingress contract rule; a separate 40-line file is unnecessary indirection.
2. **`routing/routing.py` $\leftarrow$ `routing.py` + `questions.py`:**
   * Fold `CHIP_QUESTIONS`, `TOPIC_QUESTIONS`, `ADMISSION_TEMPLATE`, `own_words`, `compose_question`, and `words_of` into `routing/routing.py`. Both modules define the closed command vocabulary and its wording.
   * Delete dead `/risk` remnants from `routing.py` (`RISK_FORMATTING` and `OUTPATIENT_RISK_REFUSAL`) and `stages.py` (`QUESTION_KIND_LABELS["risk"]`).
3. **`engine/model.py` $\leftarrow$ `model_runtime.py` + `model_turn.py`:**
   * Unify Layer 4 (Model Runtime & Gateway) into a single `engine/model.py` module containing client construction (`build_llm`, `MODEL_TIMEOUT_SECONDS`), streaming turn execution (`streamed_turn`, `has_tool_call`), and response/token/finish-reason normalization (`text_of`, `finish_reason`, `token_usage`, `served_model`, `filtered_categories`, `classify`, `silent_refusal`).
4. **`presentation/a2ui.py` $\leftarrow$ `a2ui.py` + `a2ui_envelope.py` + `note_presentation.py`:**
   * Fold the 24-line `a2ui_envelope.py` constants (`CATALOG_ID`, `SURFACE_ID`, `A2UI_VERSION`, `AUDIENCE`) and the citation/reference record builders (`note_presentation.py`) into `presentation/a2ui.py`, and rename `risk_canvas.py` to `presentation/canvas.py` (`canvas.py` imports shared helpers from `presentation/domain.py` or `postprocess/citations.py`, never creating a cycle with `a2ui.py`).
5. **`presentation/domain.py` $\leftarrow$ `feature_labels.py` + `admission_phenotypes.py` + `literature_benchmarks.py`:**
   * Consolidate the three static clinical domain tables (feature display labels, phenotype classification rules, and curated literature cohort benchmarks) into `presentation/domain.py`.
6. **`telemetry/observability.py` $\leftarrow$ `observability.py` + `chain.py`:**
   * Fold `MODEL_ID`, `CODE_REVISION`, `resolve_code_revision`, `RECORD_FIELDS`, and `record_execution` from `chain.py` into `telemetry/observability.py` so trace spans and the structured execution log live in one provenance module.

### 3.2 Break the `graph.py` $\leftrightarrow$ `deterministic.py` Circular Dependency
* Currently, `deterministic.py:513` executes `from services.agent.graph import _build_llm, _emit, _execute_tool_calls` inside `run_command()` to avoid a circular import, and `graph.py:926` executes `from services.agent.deterministic import _presented_prediction` inside `tool_node()`.
* **Fix:**
  * `build_llm` lives in `engine/model.py`; `emit` lives in `telemetry/stages.py`.
  * Move shared tool-execution helpers (`_execute_tool_calls`, `_presented_prediction`, delimiter escaping) to a shared lower seam (either `engine/mcp_client.py` or have `turn.py` dispatch `routing.is_command()` directly to `deterministic.run_command` and free-text to `graph.ask` so `graph.py` never needs to import `deterministic.py`).
  * **Zero-Cycle Rule:** Neither `graph.py` nor `deterministic.py` may use function-local imports from `services.agent.*` to work around circular dependencies.

### 3.3 Path Depth, Docker Entrypoint & Bytecode Hygiene
1. **`engine/mcp_client.py`:** Update `REPO_ROOT = Path(__file__).resolve().parents[2]` to `parents[3]` so `stdio_toolbox` resolves the repository root from `services/agent/engine/mcp_client.py`.
2. **`services/agent/Dockerfile`:** Update `CMD ["python", "-m", "services.agent.http"]` to `CMD ["python", "-m", "services.agent.ingress.http"]`.
3. **Purge Stale Bytecode & `.DS_Store`:** Delete all `.DS_Store` files and stale `__pycache__` `.pyc` artifacts (`payload`, `message_text`, `server`, `predict`, `rag_search`, `feature_store`).

---

## 4. Two-Step Governance Protocol

### Step A — Read-Only Audit & Import/Patch Census
1. Inspect all 26 files in `services/agent/`, plus all import and `patch` / `patch.object` / `monkeypatch` sites across `tests/`, `evaluation/`, and `danielmherman/demo/` (`demo/fixtures.py`, `demo/test_chips.py`, `demo/test_wire_contract.py`, `demo/test_cohort_parity.py`).
2. Document in **Section 5 (Step A — Audit Findings)**:
   * The exact one-way import DAG across the 6 subpackages + `turn.py`.
   * How `graph.py` and `deterministic.py` will eliminate their function-local cross-imports.
   * The complete list of external files in `tests/`, `evaluation/`, `Dockerfile`, and `danielmherman/demo/` whose imports/patch targets will be updated in Step B.
3. **STOP.** Do NOT move or edit any files until the Architect approves Step A.

### Step B — Execution, Test Migration & AST Architecture Gate
1. Execute the consolidations and moves into the 6 subpackages (`git mv` where applicable) and delete all retired flat root modules.
2. Update all imports and `patch.object` targets across `services/agent/`, `tests/`, `evaluation/`, `Dockerfile`, `services/agent/README.md`, and `danielmherman/demo/`.
3. Add the following automated architectural guards to `tests/agent/test_layer_separation.py`:
   * **`test_agent_module_manifest_is_exact`:** Asserts that the `.py` files under `services/agent/` match the exact 15-module manifest in Section 2 (no root shims, no unapproved files).
   * **`test_subpackage_import_dag_is_acyclic`:** Walks `services/agent/` with `ast` and asserts that `postprocess/`, `presentation/`, `telemetry/`, and `routing/` never import from `ingress/` or `engine/`, and that no function-local `import services.agent...` statements exist inside functions to hide circular dependencies.
4. Run all four verification gates (`pytest tests`, `npm test`, `manage.py test`, `manage.py check --deploy`) and record the results in **Section 6**.

---

## 5. Step A — Audit Findings & Proposed Surgical Plan

**Executed 2026-10-27. Read-only: no file moved, no import edited.** Every number below was
produced by parsing the tree (`ast`), not by eye. Where the audit contradicts the plan, the
plan is wrong and the evidence is given.

### 5.1 Corrections to the plan's own arithmetic (needs a ruling)

| Claim in §1–§3 | Measured | Consequence |
| :--- | :--- | :--- |
| "26 flat Python files (`turn.py`, `__init__.py`, and 24 modules)" | **28** `.py` files: `__init__.py`, `turn.py`, **26** domain modules | two modules are unaccounted for in the plan's inventory |
| "15 Canonical Modules" (§2 title, §1 objective) | **18** modules, as listed by §2's own tree | the Step B guard "asserts the exact 15-module manifest" cannot be written against a tree that has 18 |
| "24 Domain Files → 14 Subpackage Modules" (§3.1) | **14 source files → 6 modules**; the other **12** modules move 1:1 | §3.1's heading describes a different consolidation from the one it lists |

The tree in §2 is internally consistent and I recommend adopting it verbatim; only the
*counts* are wrong. With the tree as written: 18 canonical modules + `turn.py` + 7
`__init__.py` (root + 6 subpackages) = **26 `.py` files**, down from 28. Eight domain files
are absorbed, six `__init__.py` files are created: 26 − 8 + 6 = 26. ✓

**Ruling needed:** confirm 18 as the manifest count, and correct §1/§2/§3.1 accordingly. A
guard written to "15" today would pass only by listing a tree that does not exist.

### 5.2 The exact one-way import DAG (measured, then projected onto §2)

Current module-level edges were extracted from top-level `ImportFrom`/`Import` statements
only — function-local imports are excluded here and handled in §5.3. **The current flat
layout has no module-level cycle.** Projecting those edges through the §2 mapping gives:

```text
turn                     -> engine.graph, engine.mcp_client, ingress.contracts,
                            postprocess.guardrail, presentation.a2ui
ingress.http             -> turn, ingress.contracts, engine.graph, engine.mcp_client
ingress.run              -> engine.graph, engine.mcp_client
ingress.contracts        -> routing.routing
routing.routing          -> routing.admission_context
engine.graph             -> engine.deterministic, engine.mcp_client, engine.model,
                            engine.prompts, ingress.contracts,
                            telemetry.observability, telemetry.stages
engine.deterministic     -> engine.model, engine.prompts, routing.routing,
                            ingress.contracts, presentation.domain
engine.mcp_client        -> routing.admission_context
engine.model             -> telemetry.stages
postprocess.followups    -> presentation.domain
presentation.a2ui        -> postprocess.citations, presentation.canvas,
                            presentation.domain
presentation.canvas      -> presentation.a2ui, presentation.domain            <-- §5.4
```

The proposed Step B rule — *`postprocess/`, `presentation/`, `telemetry/`, `routing/` never
import from `ingress/` or `engine/`* — **holds on this projection with zero violations.** It
is a rule about the tree the plan specifies, not about the code as written, which is why it
passes: today's worst offenders (`graph → contracts`, `deterministic → questions`) are
`engine → ingress` and `engine → routing`, and the rule as worded permits both.

### 5.3 The `graph.py` ↔ `deterministic.py` cycle — one real edge, not three

The plan cites two late imports (`deterministic.py:513`, `graph.py:926`). The audit found
**three**, of which only **one** exists to dodge the cycle:

| Site | Import | Why it is there | Verdict |
| :--- | :--- | :--- | :--- |
| `deterministic.py:546` (in `run_command`) | `from services.agent.graph import _build_llm, _emit, _execute_tool_calls` | The comment says so: "`graph` imports this module … a module-level import would be circular" | **the cycle.** Must go |
| `graph.py:978` (in nested `tool_node`) | `from services.agent.deterministic import _presented_prediction` | Not a cycle: `graph.py:67` already imports `deterministic` at module level | needless late import; delete the indirection |
| `deterministic.py:733` | `from services.agent import model_turn` | Not a cycle: "used on this exit only" | lazy convenience; becomes a normal import |

`graph.py:67` (`from … import run_command`) is a legitimate one-way edge and stays:
`graph.ask()` calls `run_command` for the command branch at `graph.py:1239`.

**The cycle break, in full.** Two of the three names `deterministic` borrows from `graph`
are re-exports whose real homes the plan already assigns:

* `_build_llm` is `model_runtime.build_llm` imported as an alias at `graph.py:54` →
  becomes `engine/model.py`; `deterministic` imports `build_llm` from there.
* `_emit` is `stages.emit` aliased at `graph.py:58` → becomes `telemetry/stages.py`;
  `deterministic` imports `emit` from there.

That leaves `_execute_tool_calls` (`graph.py:494`, async) as the only genuinely graph-owned
helper `deterministic` needs, and `_presented_prediction` (`deterministic.py:181`) as the
only thing `graph` needs from `deterministic`. **Proposal:** move both into
`engine/mcp_client.py`, which already owns the toolbox and is imported by `graph`, `http`,
`turn` and `run`; then `graph` and `deterministic` each import them at module level and the
pair is acyclic in one direction only. No 19th module is needed, so the manifest is
preserved. *(Alternative if the Architect prefers not to grow the bridge: a dedicated
`engine/tool_loop.py`, which makes the manifest 19.)*

**Consequence for tests, and the reason this is the riskiest item in Phase 7:** 71 references
to `_build_llm`, `_execute_tool_calls` and `_presented_prediction` live in ~20 test files.
The suite's idiom is `patch.object(<module alias>, "name")`, so patching the *alias* moves
with the import line — **but a patch on `graph._build_llm` only intercepts if `graph` still
binds that name.** If `graph.py` is rewritten to call `model.build_llm(...)` through the
module, every such patch silently stops intercepting and the test passes against the real
client. Step B must either keep the alias binding in `engine/graph.py` or rewrite those 71
sites to the new binding, and must prove it by an AST check rather than by the suite going
green.

### 5.4 The merge in §3.1.4 creates a cycle that the plan says it avoids

§3.1.4 folds `a2ui_envelope.py` (14 LOC: `CATALOG_ID`, `SURFACE_ID`, `A2UI_VERSION`,
`AUDIENCE`) into `presentation/a2ui.py`, and adds "`canvas.py` imports shared helpers from
`presentation/domain.py` or `postprocess/citations.py`, never creating a cycle with
`a2ui.py`". Measured: `a2ui → risk_canvas` **and** `risk_canvas → a2ui_envelope`. Merge the
envelope into `a2ui` and the target graph contains

```text
presentation.a2ui -> presentation.canvas -> presentation.a2ui
```

which is exactly the cycle the phase exists to remove. Three fixes, all preserving 18
modules:

1. **Constants into `presentation/canvas.py`.** `canvas` needs no import for them and
   `a2ui → canvas` is already one-way. Cheapest, but protocol constants then live in the
   canvas module.
2. **Constants into `presentation/__init__.py`.** Both submodules import them from the
   package, which imports neither; a third option with no new module and no odd home, at the
   cost of `__init__.py` carrying behaviour.
3. **Invert the edge:** `a2ui.py` stops importing `canvas.py`; whoever composes the full
   presentation sequence asks for both.

**Recommendation: (2).** It keeps `a2ui.py` as the composer and `canvas.py` as the renderer,
adds no module, and makes the envelope a package-level fact rather than a file — which is
what it is.

### 5.5 `engine → ingress.contracts` is pre-existing and inverts the layer

`contracts.py` (503 LOC) is imported at module level by `http`, `turn`, `graph` **and**
`deterministic`. Two of those are `engine`. §2 puts it in `ingress/`, so the target DAG has
`engine.deterministic → ingress.contracts` and `engine.graph → ingress.contracts`: the
reasoning layer depending on the request layer. The proposed Step B rule does not forbid it
(the rule constrains `postprocess/presentation/telemetry/routing` only), so the guard would
pass while the layering is inverted.

Options, in order of my preference:

1. **Accept it and write the exception down** in the rule's docstring: `contracts` is a
   shared schema, not a request handler; the alternative homes all cost more than the edge.
2. **Split the module:** wire validation stays in `ingress/contracts.py`; the served-cohort
   allowlist and the enums/constants the engine needs go to `presentation/domain.py` or a
   leaf both layers already import.
3. **Move `contracts.py` to the root** beside `turn.py` — rejected: §2 forbids root modules.

**Ruling needed.** I will not resolve this inside Step B, because either answer changes
which files move.

### 5.6 External surface that must be updated in Step B

Measured across both repositories: **69 files outside `services/agent/` reference
`services.agent`**, 190 references in total.

| Area | Files | References | Notes |
| :--- | ---: | ---: | :--- |
| `tests/` | 61 | 174 | `tests/agent/` dominates; `tests/evaluation/test_tier1_cohort_sweep.py` touches `a2ui`, `chain`, `contracts` |
| `evaluation/` | 6 | 13 | `judge.py` (`admission_phenotypes`, `literature_benchmarks`, `routing`), `build_census.py` (`routing`), `collect_http.py` + `runner.py` (`contracts`), `cohort_provenance.py` (`served_admissions`), `measure_cache_hits.py` (`prompts`) |
| `scripts/` | 1 | 2 | `scripts/agent/measure_cache_hits.py` → `engine/prompts.py` |
| `services/` (not `agent/`) | 1 | 1 | `services/agent/Dockerfile:41` `CMD ["python", "-m", "services.agent.http"]` → `services.agent.ingress.http` |
| `danielmherman/demo/` | 2 | 3 | `fixtures.py` (`a2ui`, `feature_labels`), `test_chips.py` (`questions`) |

**Corrections to the plan's file list:** §4 Step A names four sibling files.
`demo/test_wire_contract.py` and `demo/test_cohort_parity.py` **do not reference
`services.agent` at all** — verified by grep across all of `demo/`. Only `fixtures.py` and
`test_chips.py` do. Removing them from scope saves a pointless edit.

Highest-density rewrites, for ordering the work:

| Module | Target | External files |
| :--- | :--- | ---: |
| `a2ui` | `presentation/a2ui.py` | 19 |
| `mcp_client` | `engine/mcp_client.py` | 18 |
| `contracts` | `ingress/contracts.py` | 14 |
| `graph` | `engine/graph.py` | 14 |
| `routing` | `routing/routing.py` | 7 |

### 5.7 The three hygiene items, verified

| Plan item | Measured | Verdict |
| :--- | :--- | :--- |
| `mcp_client.REPO_ROOT` `parents[2]` → `parents[3]` | `mcp_client.py:40` is `parents[2]`, used at `:270` as `cwd` | correct; `services/agent/engine/` makes the repo root `parents[3]` |
| Dockerfile entrypoint | `services/agent/Dockerfile:41` | correct |
| Purge stale bytecode, ".pyc (`payload`, `message_text`, `server`, `predict`, `rag_search`, `feature_store`)" | `services/agent/__pycache__/` contains **3** of the 6 named: `message_text`, `payload`, `server` | the other three are elsewhere or already gone; purge is `find … -name __pycache__ -exec rm -rf` |
| Delete `.DS_Store` | **4** files under `services/`, `evaluation/`, `tests/` | correct |

### 5.8 Proposed Step B order (each step green before the next)

1. **Constants first, then moves.** Land §5.4's envelope decision and §5.5's ruling before
   any `git mv`, because both change the target tree.
2. **Consolidations in place, same filenames.** Fold `questions` → `routing`, `chain` →
   `observability`, `served_admissions` → `contracts`, `model_turn` → `model_runtime`,
   `a2ui_envelope` + `note_presentation` → `a2ui`, the three domain tables → one module.
   Imports outside `services/agent/` are untouched at this point, so the suite stays green
   and any breakage is attributable to the fold, not the move.
3. **Break the cycle** (§5.3) while the modules are still flat: two aliases replaced by
   direct imports, two helpers moved to the bridge, three late imports deleted. The
   function-local-import guard can be added here and must pass before any file moves.
4. **`git mv` into the six subpackages** (`git mv` throughout, so history follows), then the
   69 external files in one mechanical pass, then the Dockerfile and README.
5. **Add the two AST guards** (`test_agent_module_manifest_is_exact`,
   `test_subpackage_import_dag_is_acyclic`) and a third the plan does not list:
   `test_no_patch_target_names_a_module_that_no_longer_binds_it` — §5.3's silent-failure
   risk, which no amount of suite-passing evidence would catch.
6. **Gates:** `pytest tests`, `npm test`, `python manage.py test`, `check --deploy`; plus a
   live deploy of the agent, because the entrypoint moves and nothing else proves the
   container still starts.


---

## 6. Step B — Execution Record (Locked Until Step A Approval)

### 6.1 Executed Changes

**The layout.** 28 flat files became 26: `__init__.py`, `turn.py`, 18 canonical modules
across 6 subpackages, and 7 package `__init__.py` files. No compatibility shims exist at
the root — `test_agent_module_manifest_is_exact` pins the exact path set, so a shim left
behind names itself rather than hiding in a count.

| Subpackage | Modules | Absorbed |
| :--- | :--- | :--- |
| `telemetry/` | `observability`, `stages` | `chain` → `observability` |
| `postprocess/` | `citations`, `guardrail`, `followups` | — |
| `presentation/` | `a2ui`, `canvas`, `domain` | `note_presentation` → `a2ui`; `admission_phenotypes` + `literature_benchmarks` → `domain` |
| `routing/` | `routing`, `admission_context` | `questions` → `routing` |
| `engine/` | `deterministic`, `graph`, `model`, `prompts`, `mcp_client` | `model_turn` → `model` |
| `ingress/` | `contracts`, `http`, `run` | `served_admissions` → `contracts` |

**The cycle is gone.** `_execute_tool_calls`, its five helpers, the turn-budget constants,
`_presented_prediction` (with `_is_number` and `_presented_factor`), `RecordedToolCall` and
`validate_recorded_tool_call` — 17 definitions — moved into `engine/mcp_client.py`. Both
`graph.py` and `deterministic.py` import them at module scope; the three function-local
imports are deleted, and no function-local `services.agent` import remains anywhere in the
package.

**Three things the plan did not anticipate**, each found by a failing gate rather than by
reading:

1. **`a2ui` ↔ `canvas` was still circular after the approved constant move.** `canvas.py`
   needed `_usable_articles` from the note-presentation code, which §3.1.4 folds into
   `a2ui`. Moving the four envelope constants was necessary and not sufficient: the helper
   and its `_LITERATURE_FIELDS` tuple went to `presentation/domain.py`, the one module both
   already import.
2. **`AgentResponseError` lived in `ingress/contracts.py`** and the validator raises it, so
   moving the validator alone would have dragged an ingress import into the engine. It now
   lives in `shared/errors.py` beside `ContractError`; `contracts.py` re-exports it, so the
   wire contract's surface is unchanged and `engine` imports `ingress` nowhere.
3. **`_build_llm` was two bindings of one function.** Tests patched `graph._build_llm`
   while the command path bound its own copy from `model_runtime`, so a patch reached one
   caller and the other called the live Vertex API — 34 tests failed with
   `GoogleModelNotFoundError: 'test-model'`. Both callers now call
   `model_gateway.build_llm(...)` (aliased, because a parameter named `model` shadows the
   module) and the suite patches `services.agent.engine.model.build_llm`, the canonical
   Layer 4 gateway.

**Guards added to `tests/agent/test_layer_separation.py`:** the exact manifest, the one-way
DAG (including zero function-local `services.agent` imports), and a patch-binding check that
imports each target module and asks whether the attribute is actually bound — the only way
to catch a patch that substitutes nothing.

### 6.2 Verification Gates
| Gate | Result |
| :--- | :--- |
| `pytest tests` (`enterprise_clinical_copilot`) | *Pending* |
| `npm test` (`danielmherman`) | *Pending* |
| `python manage.py test` (`danielmherman`) | *Pending* |
| `python manage.py check --deploy` (`danielmherman`) | *Pending* |