# Langfuse v4 Migration Plan

**Status:** IN PROGRESS · 2026-08-23 · Sprint C (observability upgrade)
**Decision source:** `projects/agent-harness/docs/session_2026-08-20.md` (#4 —
"proper integration, not a shim"; the official LangGraph callback needs the
modern stack).
**Target:** migrate the agent's observability to **Langfuse v4 (OTel-native)** +
**langgraph 1.2.11** + **langchain 1.x**, wire the official LangGraph callback,
upgrade the self-hosted Langfuse server, and re-validate the eval gate.

---

## 0. DESIGN DECISION 2026-08-23 — GROUND-UP, NOT A HYBRID

The user committed to the **ground-up LangChain-native rewrite** over a hybrid
bridge. Rationale: our agent calls `genai.Client` and `MCPToolbox` directly
(outside LangChain), so the official `CallbackHandler` sees nothing; a hybrid
would mean maintaining two observability mechanisms forever. The ground-up
approach makes observability a **side effect of the architecture**.

### Spike findings (scratch venv, validated 2026-08-23)
- `langfuse==4.14.4` + `langgraph==1.2.11` + `langchain==1.3.16` +
  `langchain-core==1.6.0` + `langchain-google-genai==4.3.5` coexist cleanly.
  `CallbackHandler.last_trace_id` exists and is populated after
  `graph.invoke(config={"callbacks": [handler]})`.
- **Model integration: use `langchain-google-genai.ChatGoogleGenerativeAI`
  (NOT `langchain-google-vertexai.ChatVertexAI` — deprecated in 3.2.0, removed
  in 4.0.0).** It supports the Vertex backend (`project`, `location`,
  `vertexai=True`, ADC), `bind_tools`, `ainvoke`, `max_output_tokens`, and
  `thinking_budget`.
- **`langchain-mcp-adapters==0.3.2` is NOT viable**: it declares
  `mcp<2.0.0,>=1.24.0` and imports `mcp.server.fastmcp` (removed in 2.0), so it
  is incompatible with our `mcp==2.0.0`. The adapter surface we need (wrap MCP
  tools as LangChain `BaseTool`s, fire `on_tool_start`/`on_tool_end`) is small
  and we already own `MCPToolbox` — wrap it ourselves instead of the adapter.
- Result: the ground-up rewrite keeps `mcp==2.0.0` + our MCP server unchanged,
  adds the LangChain stack, and replaces the hand-rolled `@observe` tracing.

### Target architecture
```
graph.ainvoke(..., config={"callbacks": [handler]})   ← official CallbackHandler
└── agent node (LangGraph)                            ← native span / edge
│     └── ChatGoogleGenerativeAI (langchain-google-genai)
│             (Vertex backend, ADC)                   ← generation span (native,
│             full prompt + response + usage)              no manual serialize)
└── tools node (LangGraph)
      └── MCP tools wrapped as BaseTool (our own)     ← tool spans (native)
handler.last_trace_id → state["langfuse_trace_id"]    ← eval loop contract kept
```
No `@observe`, no `langfuse_context`, no `_serialize_contents` in graph.py.

---

## 1. Why

The current pins are mutually incompatible for the official LangGraph
integration:

- `langfuse==2.60.10` + `langgraph==1.2.10` cannot use the official LangGraph
  callback (`CallbackHandler`) — the graph-view spike (#4) came back needing the
  modern stack.
- The current code therefore hand-rolls observability with `@observe`
  decorators + `langfuse_context` in `agent/graph.py` (`agent.ask` trace,
  `gemini.generate` generation span, `mcp.tool` spans) — it works, but it is
  not the official integration, so the native graph view (nodes/edges per
  trace) and the full OTel pipeline are unavailable.

Migrating to Langfuse v4 + the official callback gives us:
- Native **LangGraph trace** (each node as a span, edges visible in the UI).
- OTel-native ingestion (the v4 stack is the supported forward path).
- One source of truth for tracing instead of hand-maintained decorators.
- Keeps the existing guarantees: self-hosted only (no PHI/PII to Langfuse
  Cloud), scores attached via `collect.py` / `judge.py`, JSONL archive in GCP
  stays the durable evidence.

---

## 2. Current state (baseline)

**Code:** `projects/agent-harness/agent/graph.py`
- `LANGFUSE_ENABLED` gate (all three env vars present → enable).
- `observe()` decorator (no-op when disabled).
- `_generate` → generation span; `_call_tool` → `tool.{name}` span;
  `ask` → `agent.ask` trace + trace id published on state for the eval loop.
- `collect.py` records `langfuse_trace_id`; `judge.py` attaches 6 per-dimension
  scores to the matching trace.

**Pins:** `projects/agent-harness/agent/requirements.txt`
```
mcp==2.0.0
langgraph==1.2.10
google-genai==1.75.0
google-auth==2.50.0
uvicorn==0.47.0
langfuse==2.60.10
```

**Runtime:** self-hosted Langfuse on Cloud Run
(`observability.danielmherman.com` → Langfuse). Agent + mcp-server on Cloud Run.

**Working today:** traces + generation/tool spans + scores in the self-hosted
UI; eval loop (collect → judge) attaches scores; 1,854 scores across the
2026-08-19 run + pilot.

---

## 3. Scope

### In
1. Upgrade agent runtime deps to the compatible set:
   `langfuse` v4 (OTel-native), `langgraph 1.2.11`, `langchain 1.3.16`,
   `langchain-google-genai 4.3.5` (NOT the deprecated langchain-google-vertexai),
   `langchain-core 1.6.0` (keep the image narrow — no BigQuery/GCS clients
   beyond what the model integration pulls).
2. **Rewrite `graph.py` on the LangChain stack** (ground-up):
   - model: `genai.Client` → **`ChatGoogleGenerativeAI`**
     (langchain-google-genai, Vertex backend + ADC), so the model call is a
     native generation span with full prompt + usage;
   - tools: wrap our existing `MCPToolbox.call()` in thin **`BaseTool`
     subclasses** (NOT `langchain-mcp-adapters` — it is incompatible with
     `mcp==2.0.0`), so tool calls fire `on_tool_start`/`on_tool_end` natively;
   - graph: keep `StateGraph` (already LangChain-native).
3. Wire the **official `CallbackHandler`** via
   `graph.ainvoke(..., config={"callbacks": [handler]})`; publish
   `handler.last_trace_id` → `state["langfuse_trace_id"]` so
   `collect.py`/`judge.py` keep working. Preserve:
   - the `LANGFUSE_ENABLED` gate / no-op-when-disabled behavior,
   - the full prompt in trace inputs, the tool span inputs/outputs, and the
     final answer + tool_calls on the trace,
   - the thinking-budget / `MAX_TOKENS`-with-no-text handling.
4. Upgrade the self-hosted Langfuse server (Cloud Run) to the v4/OTel image +
   migrate its data/schema as the upgrade requires.
5. Verify the **graph view** live (nodes/edges per trace).
6. Re-validate the **eval gate**: run the golden eval, confirm scores attach to
   the right traces in the upgraded UI.

### Out
- No change to MCP server, predict/rag tools, or the site (mcp stays 2.0).
- No new observability features beyond what the migration requires (Playground
  / Prompt Management stay pinned until after synthetic cohort).
- No PHI/PII to Langfuse Cloud (self-hosted stays self-hosted).

---

## 4. Approach

1. **Pin study + spike (DONE 2026-08-23).** Validated in a scratch venv:
   `langfuse 4.14.4` + `langgraph 1.2.11` + `langchain 1.3.16` +
   `langchain-google-vertexai 3.2.4` coexist; `CallbackHandler.last_trace_id`
   populated after `invoke(config={"callbacks": [handler]})`. Confirmed
   `langchain-mcp-adapters` is NOT viable with `mcp==2.0.0` (declares
   `mcp<2.0.0`, imports removed `mcp.server.fastmcp`) → wrap tools ourselves.
2. **Branch + rewrite.** On `langfuse-v4`: upgrade pins; rewrite `graph.py` to
   `ChatVertexAI` + `BaseTool`-wrapped MCP tools + official `CallbackHandler`;
   delete `@observe`/`langfuse_context`/`_serialize_contents`.
3. **Wire the trace id** for `collect.py`/`judge.py` (from
   `handler.last_trace_id` after each invoke).
4. **Upgrade self-hosted server** on Cloud Run (image + any schema/data
   migration per the v4 upgrade notes).
5. **Verify live:** deploy via CI/CD (push → `deploy-on-push` trigger — do NOT
   `gcloud builds submit` manually), then:
   - run the risk/meds/summarize chips → confirm native traces + graph view,
   - run the golden eval → confirm scores attach,
   - confirm trace inputs still contain the full prompt.
6. **Rollback plan:** keep the old image tag available; if the migration
   regresses tracing or the eval gate, redeploy the previous agent image +
   Langfuse server image and revert pins. The JSONL archive is untouched and
   remains the source of truth regardless.

---

## 5. Risks / mitigations

| Risk | Mitigation |
|---|---|
| ChatVertexAI changes model behavior (thinking budget, MAX_TOKENS) | Preserve the config explicitly; re-run the golden eval and compare gate scores |
| BaseTool wrapper diverges from MCPToolbox semantics | Keep the wrapper a thin call-through to `MCPToolbox.call` (errors stay structured dicts) |
| Agent image bloats with langchain deps | Keep runtime deps narrow; only add what ChatVertexAI + callback truly import |
| Score attachment breaks (trace id lost) | Preserve the trace-id publish contract; verify in eval re-validation |
| Self-hosted upgrade data migration | Test the server upgrade on a scratch instance / snapshot first |
| Regresses live tracing | Rollback to old images; JSONL archive is unaffected |

---

## 6. Exit criteria

- [ ] `agent/requirements.txt` on the ground-up set (langfuse v4, langgraph
      1.2.11, langchain 1.3.16, langchain-google-vertexai 3.2.4); image builds.
- [ ] `graph.py` drives Gemini via `ChatVertexAI` and tools via `BaseTool`
      wrappers; `LANGFUSE_ENABLED` gate + no-op behavior preserved.
- [ ] Official `CallbackHandler` wired via graph config; `@observe`/
      `langfuse_context`/`_serialize_contents` deleted.
- [ ] `state["langfuse_trace_id"]` still published; `collect.py`/`judge.py`
      attach scores in the upgraded UI.
- [ ] Native graph view renders live for a chip turn (nodes/edges).
- [ ] Golden eval re-validated (scores land on the right traces; gate scores
      unchanged within tolerance).
- [ ] Deployed via CI/CD push trigger; rollback noted.
- [ ] Session doc updated; this plan checked off in `go_live_plan.md`.

---

## 7. Reference

- Decision: `projects/agent-harness/docs/session_2026-08-20.md` (#4)
- Ground-up decision + spike: `docs/session_2026-08-23.md`
- Current tracing: `projects/agent-harness/agent/graph.py`
- MCP toolbox: `projects/agent-harness/agent/mcp_client.py`
- Eval loop: `projects/agent-harness/eval/collect.py`, `eval/judge.py`
- Prior wrap-up: `projects/agent-harness/docs/phase6_wrapup_2026-08-19.md`
- Repo memory: `/memories/repo/eval-observability.md`
