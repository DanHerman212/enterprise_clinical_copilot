# Langfuse v4 Migration Plan — proper LangGraph integration

_Decided 2026-08-20. Owner: Dan. Companion to `session_2026-08-20.md` and
`go_live_plan.md` (Phase 6 observability)._

> **STATUS: DEFERRED (2026-08-20).** Code migration (Phases A + B) is **done and
> verified** against a local Langfuse v4 instance (graph view + scores). The
> **production cutover (Phase C) is deferred to a planned effort**: v4 requires
> ClickHouse + S3/blob storage + a Postgres→ClickHouse data migration, and the
> official path is v2 → v3 → v4. This work is parked on the `langfuse-v4-migration`
> branch; `main` stays on the v2 stack for the demo. See `session_2026-08-20.md`.

## Why

The graph-view spike (#4) needs the **official LangGraph integration**, which only
works on the modern Langfuse stack. The current pins are mutually incompatible for
the official callback path:

- `langfuse 2.60.10`'s LangChain callback only supports `langchain 0.3.x`
- `langgraph 1.2.10` requires `langchain-core 1.x` (which `langchain 0.3.x` forbids)

So the official callback cannot be installed on the current pins. Decision: **migrate
to the modern, OTel-native stack instead of a custom decorator shim.** This also
aligns tracing with the OpenTelemetry standard used across the AI-engineering
ecosystem (Arize, Langfuse v3+, etc.).

## Target stack

| Piece | From | To |
|---|---|---|
| `langfuse` (Python SDK) | 2.60.10 (v2 API) | 4.x (OTel-native, v4 API) |
| `langfuse` (self-hosted server) | v2.95.11 on Cloud Run | v4.x |
| `langgraph` | 1.2.10 | 1.2.11 |
| `langchain` / `langchain-core` | — / 1.6.0 | langchain 1.x (now compatible) |
| Graph tracing | custom `@observe` on functions | official LangChain callback on the graph |

## Phases

- **A. Dependencies.** Upgrade pins in `agent/requirements.txt` + harness
  `requirements.txt`; install into the harness venv; verify a coherent resolve
  (langfuse 4.x + langgraph 1.2.11 + langchain 1.x + langchain-core 1.x + OTel).
- **B. SDK migration (v2 → v4).** Migrate `agent/graph.py`, `eval/collect.py`,
  `eval/judge.py` to the v4 API:
  - `@observe` import/root-span semantics (v2 `langfuse.decorators` → v4 `langfuse`)
  - init/config via `Langfuse(...)`/`get_client()`; no `langfuse_context.configure`
  - trace attributes via `propagate_attributes()` (v4 replaces `update_current_trace`)
  - observation updates → v4 equivalents
  - score attachment in `judge.py` → v4 Scores API
  - wire the **official LangGraph callback** via `config={"callbacks": [...]}` on
    `graph.ainvoke` (this is the piece that renders the agent↔tools graph).
- **C. Server upgrade.** Bump self-hosted Langfuse on Cloud Run v2.95.11 → v4.x
  (image + DB migrations), verify `observability.danielmherman.com`. Rollback plan:
  keep the v2 image tag + snapshot DB before migrating.
- **D. Rebuild & verify.** Rebuild the agent image; deploy endpoints (predict + RAG);
  pilot collect; confirm a trace renders as the agent↔tools graph with scores attached.
- **E. Re-validate.** The pinned eval stack changed → re-run at least a regression
  sample of the golden eval; full 300-trace re-run later.

## Risks / caveats

- Eval gate (95%, reproducible) ran on old pins → re-validation is part of the deal.
- v4 OTel smart span filtering can drop intermediate spans and disconnect trace
  trees — allowlist instrumentation scopes if the graph renders disconnected.
- Server upgrade has DB-migration risk → snapshot + rollback plan.
- v4 exports only LLM-relevant spans by default (less billing noise) — verify the
  eval's trace completeness is unaffected.

## Phase B verification findings (2026-08-20, live against self-hosted Langfuse)

> **UPDATE — verification PASSES against a real v4 server.** The three "issues"
> below were all **artifacts of testing against the incompatible v2 production
> server** (which 404'd the v4 SDK's export), not code bugs. Verified on a local
> Langfuse v4 (docker, `LANGFUSE_MIGRATION_V4_WRITE_MODE=dual`) with a stub
> toolbox (no endpoints):
>
> - Trace lands with **trace id = the OTel trace id** → `get_current_trace_id()`
>   is the correct id for `create_score`; scores attach (verified: faithfulness 3,
>   safety 1, verdict 1).
> - **Full graph structure renders**: `agent.ask` (AGENT root) → `LangGraph`
>   (CHAIN) → `agent`/`tools`/`route` node spans (official callback), plus
>   `gemini.generate` (GENERATION) and `tool.*` (TOOL) observations.
> - **Nesting is correct** (`parentObservationId` set; earlier "flat parent=None"
>   was a wrong field name in the inspection script).
> - No export errors against the v4 server.
>
> **Prerequisite discovered:** the v4 SDK cannot export to the v2 server (404).
> The self-hosted server must be v4 (Phase C). Local verification was done in
> `dual` write mode so the classic `/api/public/traces` API works alongside the
> new events tables.

## Remaining work

- **Phase C — upgrade the self-hosted server to v4** (`langfuse-web`/`langfuse-worker`
  image `:2` → `:4`, DB snapshot + rollback plan). Must use `dual` write mode for
  the classic traces API + current eval tooling to keep working.
- **Phase D — rebuild the agent, deploy endpoints, pilot collect, verify live.**
- **Phase E — re-validate the eval gate (regression sample).**

## Status

- [x] A — dependencies
- [x] B — SDK migration + **verified against local Langfuse v4 (graph view + scores)**
- [ ] C — server upgrade (**on the critical path**)
- [ ] D — rebuild + live verify graph view
- [ ] E — re-validate eval gate
