# Phase 3: Agent Orchestrator & Model Runtime Decomposition

**Status:** `COMPLETE`  
**Target Layers:** Layer 3 (`docs/architecture/03-orchestrator.md`) & Layer 4 (`docs/architecture/04-model_runtime.md`)  
**Repositories in Scope:** `enterprise_clinical_copilot` (primary) & `danielmherman` (verify `demo/fixtures.py` import compatibility)  
**Objective:** Decompose the mixed-responsibility modules in `services/agent/` along strict Layer 3 (Orchestration) and Layer 4 (Model Runtime) boundaries—specifically extracting turn orchestration out of `http.py` into a new `turn.py`, isolating Layer 4 model runtime seams, evaluating micro-module consolidations, and modularizing `a2ui.py`—while keeping the flat package layout and 100% test compatibility.

---

## 0. Governance Protocol for This Spoke

* **Step A — The Audit Pass (Read-Only — COMPLETE):**
  * Inspect the target files in Section 1 against `docs/architecture/03-orchestrator.md` and `docs/architecture/04-model_runtime.md`.
  * Record exact file paths, line numbers, test `monkeypatch` targets, and a surgical refactoring plan in **Section 2 (Step A — Audit Findings)**.
  * **STOP.** Do NOT modify any application or test code during Step A. Await explicit Architect approval.
* **Step B — The Execution Pass (Write + Verify — COMPLETE, §3):**
  * After Architect approval of Step A, execute the approved changes, run all verification gates, and record the results in **Section 3 (Step B — Execution Record)**.

---

## 1. Step A Audit Scope & Questions to Answer

### 1.1 Track 3A — Extract Turn Orchestration (`turn.py`) from HTTP Transport (`http.py`)
* **Problem:** `services/agent/http.py` currently acts as both the Starlette HTTP/SSE transport layer (`/ask`, `/ask/stream`, `/health`, request parsing, SSE framing) and the turn orchestrator (`_run_chain`, `_compose_success`, post-model verification, and presentation assembly). Note: `turn.py` does not exist yet; the goal is to extract turn orchestration out of `http.py` into a new `services/agent/turn.py` module.
* **Audit Tasks:**
  1. Map `services/agent/http.py` by exact line ranges into (a) pure HTTP/SSE transport concerns vs. (b) turn orchestration concerns (`_run_chain`, `_compose_success`, etc.).
  2. List every symbol in `services/agent/http.py` imported or monkeypatched by `tests/` or `evaluation/`.
  3. Propose the exact boundary for creating `services/agent/turn.py` (including which re-exports or delegation shims in `http.py` must remain so existing tests and monkeypatches do not break).

### 1.2 Track 3B — Layer 4 Model Runtime Seam & Micro-Module Consolidation (`graph.py`, `model_turn.py`, `message_text.py`, `prompts.py`, `questions.py`, `chain.py`, `observability.py`)
* **Problem:**
  * Layer 4 (`04-model_runtime.md`) governs LLM client instantiation, prompt assembly, streaming token lookahead buffering, and response normalization (`model_turn.py`, `message_text.py`).
  * Several tiny two-file splits exist (`questions.py` vs. `prompts.py`, `message_text.py` vs. `model_turn.py`, and `chain.py` vs. `observability.py`).
* **Audit Tasks:**
  1. Locate every LLM client construction and model invocation across `services/agent/` (e.g., inside `graph.py`, `deterministic.py`, etc.) and document how Layer 3 (orchestration/tool policy) interacts with Layer 4 (model runtime, streaming lookahead buffer, finish-reason parsing).
  2. Audit the call sites and test imports of the three micro-module pairs:
     * `questions.py` + `prompts.py`
     * `message_text.py` + `model_turn.py`
     * `chain.py` + `observability.py`
  3. Recommend whether each pair should be consolidated (with re-exports if needed for tests) or kept separate, and whether any Layer 4 model-runtime logic inside `graph.py` or `http.py` should be extracted into a dedicated Layer 4 helper.

### 1.3 Track 3C — Presentation Composer Modularization (`services/agent/a2ui.py`)
* **Problem:** With `payload.py` deleted in Phase 2, `services/agent/a2ui.py` (~1,250+ lines) is now the sole presentation composer, but it mixes citation/reference resolution, clinical domain data shaping (risk influencers, comparative matrix, lab trends, medications), and A2UI v0.9 JSON envelope construction in one file.
* **Audit Tasks:**
  1. Map `services/agent/a2ui.py` by line range into its distinct functional responsibilities.
  2. Identify all external callers of `services/agent/a2ui.py` across `enterprise_clinical_copilot` and `danielmherman` (`demo/fixtures.py`).
  3. Propose whether and how to split `a2ui.py` into clean internal modules while keeping `a2ui.compose_presentation` (and any other externally imported functions) available at `services.agent.a2ui`.

---

## 2. Step A — Audit Findings & Proposed Surgical Plan

Read-only pass. No application or test code was modified. Line numbers are against the working tree at `b85e751` / `740cc9c` (the Phase 2 close).

### 2.0 Summary

| § | Question | Verdict |
| :--- | :--- | :--- |
| 1.1 | Can turn orchestration leave `http.py` without breaking the test suite? | **Yes, but not by re-export alone.** 55 patch sites rebind `ask`/`toolbox` *in the `http` namespace*; the names must be passed through, not re-emitted. |
| 1.2 | Should the three micro-module pairs be consolidated? | **One of three.** `message_text` merges into `model_turn`; the other two stay separate for reasons the import graph shows. One genuine Layer 4 extraction exists: the client factory and the streaming lookahead, both sitting in `graph.py`. |
| 1.3 | Can `a2ui.py` be split? | **Yes, in two moves**, with `a2ui.py` retaining re-exports for the 15 import sites and the site's fixture bridge. The canvas body (~366 lines) is the cohesive first cut. |

---

### 2.1 Track 3A — Turn orchestration out of `http.py`

#### 2.1.1 `http.py` by line range

`services/agent/http.py` is 1,076 lines.

| Range | Symbol | Concern | Proposed home |
| :--- | :--- | :--- | :--- |
| 1–48 | docstring, imports | transport | `http.py` |
| 49, 54, 61, 64 | `MAX_QUESTION_CHARS`, `ASK_TIMEOUT_SECONDS`, `STREAM_KEEPALIVE_SECONDS`, `REQUIRE_AUTH_HEADER` | transport / deploy | `http.py` |
| 74–83 | `_auth_error` | transport (IAM gate) | `http.py` |
| 85–93 | `_trace` | transport (header → trace id) | `http.py` |
| 95–135 | `_question_or_error` | transport (parse + JSON error bodies) | `http.py` |
| 137, 141 | `_SLASH_TOKEN`, `ADMISSION_NOT_FOUND` | request shape / refusal copy | `http.py` |
| 148–196 | `_unserved_admission` | **orchestration** — pre-flight served-admission policy, before the chain | `turn.py` |
| 199–228 | `_asked_questions` | request-derived context (replayed `turns` + own words) | `http.py` |
| 230–250 | `_refusal_trace_metadata` | telemetry on a refusal | `http.py` |
| 252–296 | `_unknown_command` | refusal copy, returns a response | `http.py` |
| 298–322 | `AgentAnswerUnavailable` | **orchestration** outcome type, raised by `_compose_success`, caught by both routes | `turn.py` |
| 324–337 | `_tool_error_codes` | **orchestration** telemetry over tool results | `turn.py` |
| 339–355 | `_response_fields` | **orchestration** telemetry over the model message | `turn.py` |
| 357–577 | `_compose_success` (221 lines) | **orchestration** — guardrails, presentation, payload | `turn.py` |
| 580–620 | `_StageLog` | **orchestration** — stage sink + the single execution record | `turn.py` |
| 623–667 | `_run_chain` | **orchestration** — deadline, MCP session, pre-flight hydration, graph call | `turn.py` |
| 670–690 | `health` | transport | `http.py` |
| 692–784 | `ask_route` | transport (auth, parse, status mapping) + 20 lines of record/error sequencing | `http.py` |
| 795–800, 804, 807–818 | `_sse`, `_CHAIN_DONE`, `_error_frame` | transport (SSE framing) | `http.py` |
| 821–860 | `ask_stream_route` | transport | `http.py` |
| 863–906 | `_prose_settled` | transport (settled-vs-announced verdict on the stream) | `http.py` |
| 910–1049 | `_stream_chain` | transport (queue relay, keepalive) driving **orchestration** | `http.py` |
| 1052–1067 | `app` | transport | `http.py` |
| 1069–1076 | `__main__` | process | `http.py` |

The split is clean in one respect and awkward in another. Clean: `_compose_success`, `_StageLog` and `_run_chain` are 320 contiguous lines with no Starlette types in their signatures. Awkward: both routes are themselves sequences of orchestration steps (refusal → record → compose → map to status), so `ask_route` stays a thin but not trivial function.

#### 2.1.2 The external surface of `http.py`

Everything outside the module, counted:

| Kind | Symbol | Sites |
| :--- | :--- | :--- |
| Named import | `MAX_QUESTION_CHARS` | `tests/agent/test_wire_contract_parity.py:121` |
| Named import | `_asked_questions` | `tests/agent/test_chip_dedupe.py:25` |
| Module import (`from services.agent import http as srv`) | — | 11 test modules + `tests/evaluation/test_tier1_cohort_sweep.py:280` (function-local) |
| **Patched** | `srv.ask` | **27** |
| **Patched** | `srv.toolbox` | **28** |
| Patched | `srv.REQUIRE_AUTH_HEADER` | 1 |
| Read as attribute | `srv.app` | 8 |
| Read as attribute | `srv.STREAM_KEEPALIVE_SECONDS` | 1 |
| Read as attribute | `srv.chain.CODE_REVISION` | 1 |
| Read as attribute | `srv._response_fields` | 2 (`test_error_disclosure.py:254,266`) |
| Read as attribute | `srv._prose_settled` | 1 (`test_prose_stream.py:249`) |
| Log name pinned | `logger="services.agent.http"` | 2 (`test_prose_stream.py:250`, `test_error_disclosure.py:87`) |

#### 2.1.3 The three hazards

1. **Rebinding is not re-exporting.** Every route test stubs the chain with `patch.object(srv, "ask", fake_ask)` and `patch.object(srv, "toolbox", _fake_toolbox)`. `patch.object` rebinds the name in the `http` module namespace. Today that works because `_run_chain` resolves both names as `http` globals. If `run_chain` moves to `turn.py` and resolves `turn.ask` / `turn.toolbox`, all 55 patch sites silently stop intercepting: 27 tests would run the real graph, and the failure would be a network call rather than an assertion, or — worse on a machine with credentials — a slow pass that tests nothing.
2. **Re-export does not fix hazard 1.** `from services.agent.turn import run_chain` in `http` leaves `srv.ask` as an alias nobody reads. The patch has to land on the name the callee resolves.
3. **Log names move with the code.** `caplog.at_level(..., logger="services.agent.http")` is asserted twice. Both sites cover code that stays (`_prose_settled`, and `ask_route`'s failure log), so neither breaks — but the `logger.error("agent produced a payload outside its success contract …")` at `http.py:571` moves to `services.agent.turn`, and any alerting filtered on the old name goes quiet until it is updated.

#### 2.1.4 Proposed boundary for `services/agent/turn.py`

**Moves** (in the order the file will hold them):

- `AgentAnswerUnavailable` (298–322)
- `unserved_admission` ← `_unserved_admission` (148–196), returning the response body and leaving `JSONResponse` to the caller
- `_tool_error_codes` (324–337), `_response_fields` (339–355)
- `compose_success` ← `_compose_success` (357–577)
- `StageLog` ← `_StageLog` (580–620)
- `run_chain` ← `_run_chain` (623–667)

**Stays** in `http.py`: everything in the table above marked `http.py`, plus two functions worth naming because they look like candidates and are not:

- `_asked_questions` (199–228) — it derives the clinician's words from the *request* (`turns`, `own_words`), which is request shape, and a test imports it from `http`.
- `_unknown_command` (252–296) — its decision is routing vocabulary, but its output is a response body with a message the caller sees; splitting the decision from the message would buy nothing.

**The shim.** `turn.run_chain` takes the two callables as keyword-only parameters, and `http.py` keeps `_run_chain` as the one-line seam that supplies them from its own namespace:

```python
async def _run_chain(parsed: dict, on_event=None) -> dict:
    """The seam the route tests stub through.

    `ask` and `toolbox` are passed rather than imported inside `turn`, because
    these two names IN THIS MODULE are what every route test rebinds
    (`patch.object(srv, "ask")` 27×, `patch.object(srv, "toolbox")` 28×). A
    re-export would not preserve that: the alias in `http` would no longer be
    the name `turn` resolves, and every fake would be bypassed silently.
    """
    return await turn.run_chain(
        parsed, on_event=on_event, ask=ask, toolbox=toolbox,
    )
```

with `turn.run_chain(parsed, on_event=None, *, ask, toolbox)`.

`_compose_success` and `_StageLog` are re-exported by name for `srv._response_fields` / future readers, and both routes call `turn.compose_success(...)` directly — nothing patches them.

**Consequence for `http.py`:** roughly 320 lines out, a ~12-line import block and the `_run_chain` seam in. Both routes keep their shape, so the streaming tests that count frames are untouched.

#### 2.1.5 Deliberately not doing

- No route-level abstraction (`_serve(parsed, stream=bool)` to merge the two routes). The two routes share the chain and the composer but differ in every error path — one returns status codes, the other returns terminal frames — and the merge would be a rewrite of 170 lines of tested branching for no layer gain.
- `deterministic.py`'s command branch stays where it is. It is Layer 3 orchestration that happens to live outside `http.py` already.

---

### 2.2 Track 3B — Layer 4 model runtime

#### 2.2.1 Client construction and model invocation, exhaustively

| Site | What it is | Callers |
| :--- | :--- | :--- |
| `graph.py:601–631` | `_build_llm(model)` — the only `ChatGoogleGenerativeAI` construction: `vertexai=True`, pinned model, `GEMINI_LOCATION`, temperature 0, `max_output_tokens`, `reasoning_effort`, `max_retries=3`, timeout, safety thresholds | `graph.py:914`, `deterministic.py:694` |
| `graph.py:914–916` | `_build_llm` → `bind_tools(freetext_tools)` and `bind_tools(..., tool_choice="any")` — Layer 3 policy expressed as a Layer 4 binding | `build_graph` |
| `graph.py:739–789` | `_streamed_turn(llm, messages, on_event)` — the streaming lookahead: `llm.astream`, `PROSE_LOOKAHEAD_CHUNKS` (726, = 1), holds chunks until a tool call is ruled out, drops the held prefix when the turn turns out to be routing | `graph.py:752`'s loop, reached from the answering node |
| `graph.py:951–956` | `answering_llm.ainvoke(state["messages"])` — the one non-streaming model call | answering node |
| `deterministic.py:472–509` | `_formatting_turn` — `llm.ainvoke` (`:498`) / `llm.astream` (`:501`) for the command prose pass, no tools bound |
| `deterministic.py:545` | `from services.agent.graph import _build_llm, _emit, _execute_tool_calls` — a **function-local** import, because `graph` imports `deterministic` for the command branch and a module-level import would be circular |
| `deterministic.py:694` | `_build_llm(model)  # no bind_tools: a formatting turn cannot call anything` |

**Layer 3 ↔ Layer 4 interaction, as the code actually has it.** Layer 3 decides *which tools exist and whether a turn is a routing turn*; Layer 4 decides *how a message is obtained and how it is read back*. The two meet at four places:

1. `bind_tools` (914–916) — tool policy crossing into the client.
2. `_streamed_turn` (739–789) — the lookahead is a Layer 4 concern that exists only because Layer 3 needs to know whether a turn is deliberation before announcing it. It calls `_has_tool_call` (729) and `text_of` (module `message_text`) — both response normalization.
3. `model_turn` (246 lines, stdlib only) — `finish_reason`, `token_usage`, `served_model`, `filtered_categories`, `blocked_reason`, `safety_ratings`, and the refusal/truncation sentence sets. Read by `http.py:350–351`, `deterministic.py:748–752`, `chain.py:140` (docs only).
4. `message_text.text_of` (37 lines, one function) — a content block → prose, because `str(chunk.content)` announced an envelope instead (measured on `/ehr`, 2026-10-14).

#### 2.2.2 The three micro-module pairs

**`questions.py` (141) + `prompts.py` (440) — KEEP SEPARATE.**

| | `questions` | `prompts` |
| :--- | :--- | :--- |
| Imports | stdlib only (`re`) | none (literals) |
| Importers | `contracts.py`, `http.py`, 2 tests | `graph.py`, `deterministic.py`, 4 tests |
| Content | chip/topic wording tables, `compose_question`, `own_words`, `words_of` | `SYSTEM_PROMPT` and the driver-turn rules |

Merging would make the request parser (`contracts.py`) import the 440-line system prompt, and would put the most frequently edited literal in the codebase in the same file as the closed chip vocabulary. The split is already the layer boundary; leave it.

**`message_text.py` (37) + `model_turn.py` (246) — CONSOLIDATE.**

`message_text` is one function, `text_of`, stdlib only. Its three importers are `graph.py:46`, `deterministic.py:57` and `tests/agent/test_prose_stream.py:32` — the first two already import `model_turn` or are its neighbours, and `model_turn` is exactly "read what the model sent back". A 37-line module with one 12-line function is a file boundary that pays for nothing.

Executed as: `text_of` moves into `model_turn`; `message_text.py` is deleted; `graph.py`, `deterministic.py` and the one test import from `model_turn`. No re-export file is left behind.

**`chain.py` (193) + `observability.py` (322) — KEEP SEPARATE.**

`chain` is the model pin (`MODEL_ID`), the code revision, and the single execution record (`record_execution`, `RECORD_FIELDS`); its only import is `services.mcp.config.GEMINI_MODEL`. `observability` is the Langfuse adapter (handler, span helpers, run tag, flush). **`chain` does not import `observability`** — its importers are `graph.py` and `http.py` only. Merging would put the model pin inside a tracing adapter and make the record contract depend on a telemetry SDK.

#### 2.2.3 The Layer 4 extraction: `services/agent/model_runtime.py`

There is real Layer 4 code inside `graph.py`, which is the Layer 3 state machine:

- `_build_llm` (601–631) — client construction and every generation parameter.
- `PROSE_LOOKAHEAD_CHUNKS` (726), `_has_tool_call` (729–736), `_streamed_turn` (739–789) — the streaming lookahead buffer.

Proposed: `model_runtime.py` holds `build_llm`, `streamed_turn`, `has_tool_call`, `PROSE_LOOKAHEAD_CHUNKS`; `graph.py` imports them **under their current private names**:

```python
from services.agent.model_runtime import (
    build_llm as _build_llm,
    has_tool_call as _has_tool_call,
    streamed_turn as _streamed_turn,
)
```

That detail is the whole compatibility argument: `_build_llm` is patched **11 times** as `patch.object(graph, "_build_llm", …)`. Because `graph` keeps its own module-level binding and its call site at line 914 resolves `_build_llm` as a `graph` global, every one of those 11 patches keeps working unchanged. `deterministic.py:545` imports `_build_llm` from `graph` at call time, so it sees the patch too. No test edit is needed for this move.

Not extracted, deliberately:

- The answering node's `ainvoke` (951–956) — it is a graph node; moving the call without moving the node would split a node in two.
- `deterministic.py`'s formatting turn (472–509) — extracting a second runtime seam from a Layer 3 module is a larger change than this phase should carry, and the circular import that forces the late import is worth removing on its own terms, not as a side effect of a move.

---

### 2.3 Track 3C — `a2ui.py`

`services/agent/a2ui.py` is 1,277 lines. It imports only pure domain modules — `citations`, `feature_labels`, `admission_phenotypes`, `literature_benchmarks` — and no Vertex, graph or MCP code, which is what makes it loadable from the Django fixture path.

#### 2.3.1 By responsibility

| Range | Content | Responsibility |
| :--- | :--- | :--- |
| 1–69 | docstring, imports, `CATALOG_ID`, `SURFACE_ID`, `A2UI_VERSION`, `AUDIENCE` | envelope constants |
| 70–101 | `_band`, `_usable_estimate` | risk shaping |
| 102–177 | `_usable_articles`, `_literature_view` | literature shaping |
| 178–185 | `_unavailable_text` | shared copy |
| 186–284 | `resolve_source`, `resolve_sources` | citation resolution (numbering, section intent) |
| 285–304 | `_section_label`, `narrative_surface` | narrative surface construction |
| 305–361 | `note_citations`, `_short_source` | note citations → drawer records |
| 362–412 | `literature_citations` | article records → drawer records |
| 413–446 | `_narrative` | narrative surface for either evidence kind |
| 447–541 | `_draws_matrix`, `_comparative_matrix` | comparative matrix (literature turns) |
| 543–908 | `compose_risk_canvas` (366 lines) | **the canvas itself**: components, ordering, unavailable states |
| 909–1032 | `admission_eligible`, `predict_payload`, `rag_payload`, `lab_trend_payload`, `literature_payload` | evidence selectors ("which call is this turn's evidence") |
| 1012–1068 | `turn_scope`, `turn_phenotype` | turn classification from tool calls |
| 1070–1277 | `compose_presentation` (208 lines) | the composer entry point |

Internal dependency edges, verified:

- `_literature_view` → `_usable_articles`
- `compose_risk_canvas` → `_band`, `_usable_estimate`, `_literature_view`, `resolve_sources`, `_draws_matrix`, `_comparative_matrix`, `display_probability`, `humanize`
- `_narrative` → `narrative_surface` → `note_citations` / `literature_citations` (→ `_section_label`, `_short_source`)
- `compose_presentation` → `_usable_articles`, `_usable_estimate`, `resolve_sources`, `_narrative` (only at 1122)
- The matrix helpers (447–541) use no in-file helper; they take dicts.

#### 2.3.2 External callers

In `enterprise_clinical_copilot`, by imported name (count = distinct import statements):

| Name | Count | Where |
| :--- | :--- | :--- |
| `compose_presentation` | 15 | `http.py:30`, plus 14 test imports |
| `compose_risk_canvas` | 3 | `test_a2ui.py`, `test_a2ui_literature.py`, `tests/evaluation/test_tier1_cohort_sweep.py:28` |
| `predict_payload` | 2 | `test_a2ui.py`, `test_tier1_cohort_sweep.py` |
| `AUDIENCE`, `CATALOG_ID`, `SURFACE_ID`, `_narrative`, `_usable_articles`, `literature_citations`, `literature_payload`, `rag_payload`, `resolve_source` | 1 each | `test_a2ui.py`, `test_a2ui_literature.py` |
| module attribute access (`from services.agent import a2ui`) | 4 | `test_citation_intent.py:19`, `test_affordance_and_meds_surface.py:79,108`, `test_eval_contract.py:345` — uses `rag_payload`, `resolve_sources`, `citation_remap`, `compose_presentation` |

Inside the agent, three modules reach `a2ui` as a module attribute, so the split has to keep those attributes resolvable on `a2ui`:

- `guardrail.py:245` → `a2ui._usable_articles`
- `guardrail.py:1046` → `a2ui.compose_presentation`
- `graph.py:184` → `a2ui.rag_payload`
- `evaluation/inventory/multiturn_chip_audit.py:139` → `a2ui.compose_presentation`

In `danielmherman`, the entire surface is one function plus one label helper, imported lazily with the sibling repo pushed onto `sys.path` (`demo/fixtures.py:41–51`):

```python
from services.agent.a2ui import compose_presentation
from services.agent.feature_labels import label_for
```

(`demo/test_chips.py:35` additionally imports `services.agent.questions.CHIP_QUESTIONS`.) Fixture mode is the only site code that imports the agent at all, and it is rejected in production.

#### 2.3.3 Proposed split — two moves, both reversible

**Move 1 — `services/agent/risk_canvas.py`** (the cohesive first cut, ~490 lines out):

- `_band`, `_usable_estimate`, `_usable_articles`, `_literature_view`, `_unavailable_text`
- `_draws_matrix`, `_comparative_matrix`
- `compose_risk_canvas`

It imports `resolve_sources` from `a2ui` **or** (better) from wherever move 2 puts it; to avoid a cycle in a single-step execution, move 1 should import `resolve_sources` from `a2ui`, which keeps the dependency one-directional (`risk_canvas → a2ui` is impossible if `a2ui` re-exports `compose_risk_canvas`) — so the correct order is move 2 first, or both in one commit with `resolve_sources` living in the citation module. See §2.5 for the ordered plan.

**Move 2 — `services/agent/note_presentation.py`** (~270 lines out):

- `_section_label`, `narrative_surface`, `note_citations`, `_short_source`, `literature_citations`, `_narrative`, `resolve_source`, `resolve_sources`, `_unavailable_text`

**What `a2ui.py` keeps:** the envelope constants, the evidence selectors (909–1068), `turn_scope`, `turn_phenotype`, `compose_presentation`, and **re-exports** of every name the census above shows being imported elsewhere:

```python
from services.agent.note_presentation import (  # noqa: F401
    _narrative, literature_citations, note_citations, resolve_source, resolve_sources,
)
from services.agent.risk_canvas import (  # noqa: F401
    _usable_articles, compose_risk_canvas,
)
```

This is the whole compatibility contract: all 29 name-imports across 12 files, the four module-attribute call sites and the Django fixture bridge keep working untouched. What changes is where the code lives, not what `services.agent.a2ui` offers.

#### 2.3.4 Not splitting

- The evidence selectors and the composer stay together. The selectors answer "which call is this turn's evidence" — a question about the transcript — and `compose_presentation` is their only caller besides tests.
- `compose_presentation` itself is not decomposed. Its 208 lines are sequential composition over already-prepared inputs; cutting it would produce two functions that only ever run in one order.

---

### 2.4 Doc-vs-code mismatches recorded during the audit

| Spoke says | Actual |
| :--- | :--- |
| `docs/remediation/03-orchestrator_and_model_runtime.md` | `docs/code_comprehension/03-orchestrator_and_model_runtime.md` |
| `a2ui.py` "~1,250+ lines" | 1,277 |
| `http.py` performs "post-model verification" | verification is `guardrail.guard_answer`, called at `http.py:417`; `http.py` only maps its flags and decides `verified` |
| Track 3B names `chain.py` vs `observability.py` as a candidate pair | they do not import each other; the pin and the tracing adapter are orthogonal |

---

### 2.5 Proposed Step B order

Each step is independently verifiable; the suite should be run after every one.

1. `turn.py` extraction (§2.1.4) + `_run_chain` seam. Expect: no test edits.
2. `message_text` → `model_turn` (§2.2.2) + the one test import. Expect: 1 test edit.
3. `model_runtime.py` extraction with `graph` private aliases (§2.2.3). Expect: no test edits.
4. `note_presentation.py` extraction + `a2ui` re-exports (§2.3.3, move 2).
5. `risk_canvas.py` extraction + `a2ui` re-exports (§2.3.3, move 1).
6. Update the layer documents and this Spoke's §3.

Order note: 4 before 5, because `risk_canvas` calls `resolve_sources`, and after step 4 that name lives in `note_presentation` — so step 5 imports it from there rather than reaching back into `a2ui`.

---

### 2.6 Decisions for the Architect

| # | Decision | Recommendation |
| :--- | :--- | :--- |
| 1 | `turn.py` seam: inject `ask`/`toolbox` through `http._run_chain` (§2.1.4), or leave them imported in `turn` and rewrite 55 patch sites | **Inject.** It preserves 100% test compatibility and makes the seam explicit; rewriting 55 patch sites is churn with no design gain. |
| 2 | `message_text.py`: merge into `model_turn` and update one test import, or keep the 37-line module | **Merge.** One function, one home, one test line. |
| 3 | `model_runtime.py`: extract `_build_llm` + the streaming lookahead from `graph.py`, or leave Layer 4 in `graph.py` this phase | **Extract**, with `graph` keeping private aliases so the 11 `_build_llm` patches are untouched. |
| 4 | `a2ui.py` split depth: two modules as proposed, canvas-only, or leave whole | **Two modules.** The canvas (366 lines) and the citation/reference block (270 lines) are each cohesive and each have exactly one internal caller group. |
| 5 | New module names: `risk_canvas.py` + `note_presentation.py` vs alternatives | **`risk_canvas.py` + `note_presentation.py`.** Flat, consistent with the existing package, and `note_presentation.py` avoids colliding with `citations.py`, which is the renumbering module. |

---

## 3. Step B — Execution Record

Architect approval received for §2.6 in full, with the §2.5 order enforced. All six
steps executed; the suite run after each. No test file changed except the one import
that Decision #2 named, and no route behaviour changed.

### 3.1 Executed Changes

| Step | File | Change | Tests edited |
| :--- | :--- | :--- | :--- |
| 1 | `services/agent/turn.py` | **New** (505 lines). `run_chain`, `compose_success`, `StageLog`, `unserved_admission`, `refusal_trace_metadata`, `_tool_error_codes`, `_response_fields`, `AgentAnswerUnavailable`, `ADMISSION_NOT_FOUND`. | 0 |
| 1 | `services/agent/http.py` | 1,076 → 642 lines: the orchestration blocks removed, `_run_chain` reduced to the seam, the moved names imported under their old private spellings so `http._compose_success` / `http._StageLog` / `http._response_fields` still resolve. | 0 |
| 2 | `services/agent/model_turn.py` | `text_of` merged in (246 → 281 lines), with the history that made it a function rather than an inline copy. | — |
| 2 | `services/agent/message_text.py` | **Deleted** (37 lines). | — |
| 2 | `graph.py`, `deterministic.py`, `tests/agent/test_prose_stream.py` | Repointed at `model_turn.text_of`; three prose references updated. | 1 |
| 3 | `services/agent/model_runtime.py` | **New** (171 lines). `build_llm`, `streamed_turn`, `has_tool_call`, `PROSE_LOOKAHEAD_CHUNKS`, `MODEL_TIMEOUT_SECONDS`. | 0 |
| 3 | `services/agent/stages.py` | `emit` moved in from `graph._emit` (it validates a stage event against `TOOL_LABELS`, which is this module's invariant). | 0 |
| 3 | `services/agent/graph.py` | 1,529 → 1,371 lines: Layer 4 removed, private aliases kept (`_build_llm`, `_has_tool_call`, `_streamed_turn`, `MODEL_TIMEOUT_SECONDS`), `emit as _emit`. | 0 |
| 4 | `services/agent/a2ui_envelope.py` | **New** (24 lines). `CATALOG_ID`, `SURFACE_ID`, `A2UI_VERSION`, `AUDIENCE`, so neither consumer imports `a2ui` back. | 0 |
| 4 | `services/agent/note_presentation.py` | **New** (312 lines). Citation and drawer records: `resolve_source(s)`, `note_citations`, `literature_citations`, `narrative_surface`, `_narrative`, `_section_label`, `_short_source`, `_unavailable_text`, `_usable_articles`, `_LITERATURE_FIELDS`. | 0 |
| 5 | `services/agent/risk_canvas.py` | **New** (573 lines). `compose_risk_canvas`, `_band`, `_usable_estimate`, `_literature_view`, `_draws_matrix`, `_comparative_matrix`, the two literature notes and `_BASELINE_SOURCE`. | 0 |
| 5 | `services/agent/a2ui.py` | 1,277 → 451 lines: the composer, the evidence selectors, and re-exports of every name the 29 import sites and 4 module-attribute call sites use. | 0 |

### 3.2 Where the audit was wrong, and what was done

1. **`ASK_TIMEOUT_SECONDS` was a fourth monkeypatch target (§2.1.2 said three).**
   `tests/agent/test_model_failures.py:175` overrides it through a generic helper
   (`_post(model, ASK_TIMEOUT_SECONDS=1.0)` → `patch.object(srv, name, value)`), so the
   census regex — which read literal `patch.object(srv, "…")` calls — could not see it.
   The test failed loudly on the first run (504 became 200). **Resolution:** the deadline
   stays in `http` and is passed into `turn.run_chain(..., deadline=…)`. That is also
   better layering: the route's 504 message quotes the number, so the route owns it.
   Lesson recorded: a patch helper with `**kwargs` hides targets from a literal census;
   the run-time proof is the suite, not the grep.

2. **`_usable_articles` belongs with the notes, not the canvas.** §2.3.1 put it in the
   canvas block by line position, but `_narrative` (notes) reads it and the canvas only
   consumes it. Left where it was, step 5 would have created a `note_presentation →
   risk_canvas → note_presentation` cycle. It moved with the notes, and
   `_LITERATURE_FIELDS` — which it uses — moved with it.

3. **Two companion moves the extraction needed, not listed in §2.5.**
   `streamed_turn` calls `graph._emit`, so `emit` moved to `stages.py` (where the stage
   vocabulary and label table already live) and `graph` re-imports it as `_emit`;
   `build_llm` reads `MODEL_TIMEOUT_SECONDS`, so that constant moved with it and `graph`
   re-binds it still. Both keep `deterministic.py:545`'s late `from services.agent.graph
   import _build_llm, _emit, …` working unchanged.

### 3.3 Every patch site still intercepts

The point of the seam design, verified rather than argued — the suite cannot pass
otherwise for these four:

| Patch | Sites | Now resolves as |
| :--- | :--- | :--- |
| `patch.object(srv, "ask", fake)` | 27 | `http._run_chain` passes `ask` into `turn.run_chain` |
| `patch.object(srv, "toolbox", fake)` | 28 | same |
| `patch.object(srv, "ASK_TIMEOUT_SECONDS", 1.0)` | 1 | passed as `deadline` |
| `patch.object(graph, "_build_llm", fake)` | 11 | `graph` keeps its own binding; `deterministic` reads it from `graph` |

### 3.4 Behaviour changes

1. **Two log names moved.** `agent produced a payload outside its success contract`
   now logs under `services.agent.turn`; `no progress label for tool …` and
   `progress callback failed; continuing` now log under `services.agent.stages`. Any
   alert filtered on `services.agent.http` for the first, or `services.agent.graph` for
   the other two, goes quiet until it is updated. No test pins either name.
2. Nothing else. No route, contract, prompt, tool policy or payload changed.

### 3.5 Verification Gates

| Gate | Result |
| :--- | :--- |
| `pytest tests` (`enterprise_clinical_copilot`) | **1349 passed, 1 skipped** (208s) |
| `python manage.py test` (`danielmherman`, `ENVIRONMENT=development`) | **243 tests OK** (the fixture bridge imports the refactored `a2ui.compose_presentation` and `feature_labels.label_for`) |
| `python manage.py check --deploy` (`ENVIRONMENT=production`) | **System check identified no issues (2 silenced)** |

Suite arithmetic: unchanged at 1,349 / 1 skipped. Nothing was deleted, added or renamed
at the test level except `test_prose_stream.py:32`'s import.

### 3.6 Deliberately not done

- **The layer documents needed no edit.** `docs/architecture/03-orchestrator.md` and
  `04-model_runtime.md` describe responsibilities and name no modules, so the split
  neither contradicts nor dates them. Verified by grep for every module name this phase
  touched.
- **`deterministic.py`'s formatting turn is still a Layer 4 call site inside a Layer 3
  module**, and still imports `_build_llm` at call time to dodge the `graph` ↔
  `deterministic` cycle. Removing that cycle is its own change: it means teaching the
  command branch to receive its model rather than build one.
- **`a2ui.py` still carries re-exported names.** They are the module's public
  surface for the tests, `guardrail` and the Django fixture bridge; the cost is one
  `# noqa: F401` comment per line against a second call path everywhere.