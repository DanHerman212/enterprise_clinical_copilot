# Langfuse v4 Migration Plan

**Status:** planned · 2026-08-22 · Sprint C (observability upgrade)
**Decision source:** `projects/agent-harness/docs/session_2026-08-20.md` (#4 —
"proper integration, not a shim"; the official LangGraph callback needs the
modern stack).
**Target:** migrate the agent's observability to **Langfuse v4 (OTel-native)** +
**langgraph 1.2.11** + **langchain 1.x**, wire the official LangGraph callback,
upgrade the self-hosted Langfuse server, and re-validate the eval gate.

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
   `langfuse` v4 (OTel-native), `langgraph 1.2.11`, `langchain 1.x`
   (add only what the official callback imports; keep the image narrow).
2. Replace the hand-rolled `@observe`/`langfuse_context` tracing in
   `graph.py` with the **official LangGraph `CallbackHandler`** (or the
   documented v4 wiring), while preserving:
   - the `LANGFUSE_ENABLED` gate / no-op-when-disabled behavior,
   - the trace id on `state["langfuse_trace_id"]` for `collect.py`/`judge.py`,
   - the full prompt in trace inputs (system prompt + messages), the tool
     span inputs/outputs, and the final answer + tool_calls on the trace.
3. Upgrade the self-hosted Langfuse server (Cloud Run) to the v4/OTel image +
   migrate its data/schema as the upgrade requires.
4. Verify the **graph view** live (nodes/edges per trace).
5. Re-validate the **eval gate**: run the golden eval, confirm scores attach to
   the right traces in the upgraded UI.

### Out
- No change to MCP server, predict/rag tools, or the site.
- No new observability features beyond what the migration requires (Playground
  / Prompt Management stay pinned until after synthetic cohort).
- No PHI/PII to Langfuse Cloud (self-hosted stays self-hosted).

---

## 4. Approach

1. **Pin study (start of sprint, ~15 min).** Resolve the exact compatible set:
   `langfuse` v4 + `langgraph 1.2.11` + `langchain 1.x` + the callback import
   path. Confirm whether `langchain` is required as a direct dep or only a
   transitive pin (keep the agent image narrow — the runtime should not drag in
   Vertex/BigQuery/GCS clients).
2. **Branch + spike.** On a branch (e.g. `langfuse-v4`), upgrade pins, swap the
   tracing to the official callback, run the agent locally with a local/self-
   hosted Langfuse to confirm traces render with the native graph view.
3. **Wire the trace id** for `collect.py`/`judge.py` (callback must expose the
   trace id so score attachment keeps working).
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
| Version incompatibility not fully resolvable | Pin study first; stay on the branch until the spike runs clean locally |
| Agent image bloats with langchain deps | Keep runtime deps narrow; only add what the callback truly imports |
| Score attachment breaks (trace id lost) | Preserve the trace-id publish contract; verify in eval re-validation |
| Self-hosted upgrade data migration | Test the server upgrade on a scratch instance / snapshot first |
| Regresses live tracing | Rollback to old images; JSONL archive is unaffected |

---

## 6. Exit criteria

- [ ] `agent/requirements.txt` on `langfuse` v4 + `langgraph 1.2.11` +
      `langchain 1.x`; image still builds.
- [ ] `graph.py` uses the official LangGraph callback; `LANGFUSE_ENABLED`
      gate + no-op behavior preserved.
- [ ] `state["langfuse_trace_id"]` still published; `collect.py`/`judge.py`
      attach scores in the upgraded UI.
- [ ] Native graph view renders live for a chip turn.
- [ ] Golden eval re-validated (scores land on the right traces).
- [ ] Deployed via CI/CD push trigger; rollback noted.
- [ ] Session doc updated; this plan checked off in `go_live_plan.md`.

---

## 7. Reference

- Decision: `projects/agent-harness/docs/session_2026-08-20.md` (#4)
- Current tracing: `projects/agent-harness/agent/graph.py`
- Eval loop: `projects/agent-harness/eval/collect.py`, `eval/judge.py`
- Prior wrap-up: `projects/agent-harness/docs/phase6_wrapup_2026-08-19.md`
- Repo memory: `/memories/repo/eval-observability.md`
