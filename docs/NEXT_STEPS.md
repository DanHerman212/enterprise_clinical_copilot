# Next Steps — Architecture Sequencing Strategy

_Date: 2026-07-24_

The MLOps model is deployed to a live Vertex endpoint. This document captures the
strategy for building out the remaining pieces of the architecture (MCP tool, RAG,
agent, UI) in the right order.

## Guiding principle: thin vertical slice first

Since RAG / agent / UI are all new territory, don't build each layer to completion
before the next. Build a **walking skeleton** — the narrowest possible end-to-end
path through all layers — then thicken it. This de-risks the unfamiliar parts early
and avoids over-investing in a layer that needs rework once it's wired to the others.

The README dependency graph is the key: `ML → Agent`, `RAG → Agent`, `Agent → UI`.
**The Agent is the integration hub.** Everything else is a tool it calls or a face it
wears. So the agent is what we stand up first, with the smallest number of tools.

## Recommended order

> **⚠️ SUPERSEDED (2026-07-28).** The A/B/C ordering below was the original reasoning and
> is kept for context. The website moved to **Step 0 (first)** because Phase 1 is already
> complete and publishable, and the site is fully independent. See
> **[Roadmap (updated 2026-07-28)](#roadmap-updated-2026-07-28--website-first)** for the
> current plan.

### Phase A — Prediction tool + minimal agent (next)

1. Wrap the live endpoint as an MCP server exposing one tool:
   `predict_readmission(hadm_id)` → prob + threshold + top_factors
   (reuse the `smoke_test.py` glue).
2. Stand up a minimal orchestration agent that can call that one tool.
3. Prove the loop: "readmission risk for hadm 20924467?" → agent calls tool →
   structured answer.

Highest-leverage next step: closes the loop from the strongest area (ML) into the
weakest (agents) across the smallest surface.

### Phase B — RAG as the second tool

Biggest new build, so it comes after the agent skeleton exists and can already hold a
tool. Corpus (EHR notes) → Vector Search index → `rag_search` tool → register as the
agent's second tool. Now the agent does predict + retrieve. Don't build RAG in
isolation — it plugs into the existing agent.

### Phase C — UI / website + A2UI

Last, because it's the presentation layer over a working agent. Wrapping a
half-working agent in a UI just means debugging two things at once. Once the agent
gives good answers with both tools, put Django + A2UI in front for dynamic UIs.

## Direct answers to the guiding questions

- **MCP for the endpoint first?** Yes — good first move. The tool logic is trivial
  (call the endpoint), so all the complexity is MCP plumbing, which is exactly the
  skill worth learning, on a low-risk surface. Alternative is an in-process tool
  function (faster, no protocol), but MCP buys a clean, reusable seam and the learning.
- **RAG next?** Yes, second — but as the agent's *second tool*, not a standalone system.
- **When website / A2UI?** Last. It depends on the agent being good. Building it
  earlier entangles UI bugs with agent bugs.

## Open decisions — RESOLVED (2026-07-24)

1. **Agent framework** — **LangGraph on Cloud Run** (Stack 2, portable/self-run).
2. **LLM provider** — **Gemini Flash** via Vertex (cheap, managed, stays in-GCP).
3. **RAG corpus** — **RESOLVED.** `physionet-data.mimiciv_note.discharge` (+ `radiology`)
   in BigQuery, keyed by `hadm_id`. Unblocks Step 3. See [rag_requirements.md](rag_requirements.md).
4. **UI base** — **RESOLVED.** A page on the existing Django site:
   `danielmherman.com/projects/clinical_copilot/demo`.
5. **MCP hosting** — build one transport-agnostic server: **stdio for dev**,
   **streamable HTTP** on Cloud Run for prod.
6. **Rendering layer** — **A2UI is committed** (design decision made). The demo page
   renders agent output via A2UI; the agent's output contract is designed to be
   A2UI-renderable from the start rather than retrofitted.
   Requirements and constraints researched 2026-07-29 — see
   [a2ui_requirements.md](a2ui_requirements.md). Key outcomes: A2UI does **not** require
   WebSockets (they are a *proposed*, unimplemented transport), so the Redis/Postgres
   decision below is unaffected; the real constraint is that all maintained renderers
   are client-side JS, so we use the **Lit (Web Components)** renderer inside Django and
   explicitly avoid the documented CopilotKit/Next.js path.
7. **Region** — **`us-east1`** everywhere (website + mlops aligned; avoids cross-region
   latency/egress).
8. **Redis / Memorystore** — **SKIP.** The demo is auth-gated with few users, so Cloud SQL
   (Postgres) covers quota, sessions, and LangGraph checkpointing — all shared across
   Cloud Run instances and durable. Saves ~$43/mo. Revisit only for a live multi-client
   WebSocket dashboard (the one thing Postgres genuinely can't do).
9. **Demo access** — public website, but the LLM-backed demo sits behind **Django auth**
   with per-user quota in Postgres (issued to selected employers / partners). Gives a
   natural audit trail and caps LLM cost/abuse.

Full Phase 2 design captured in
[projects/agent-harness/docs/architecture.md](../projects/agent-harness/docs/architecture.md).

---

## Roadmap (updated 2026-07-28 — website-first)

**Step 0 — Website launch (do first).** Deploy the clean-sheet Django site to Cloud Run
per the [GCP deployment guide](../../danielmherman/docs/GCP_DEPLOYMENT_GUIDE.md).
Rationale: it is fully independent (zero dependencies), de-risks unfamiliar infra in
isolation, and — critically — Phase 1 (MLOps) is **already complete and publishable**,
so the site turns finished work into visible output and lets each phase be published as
it lands. The Django BFF that will front the agent also lives here.

> **Guardrails:** ship the skeleton + portfolio only — **do not build the demo page yet**
> (its requirements don't exist until the agent does). Timebox it, then return to Step 1.

Then:

1. **Full training pass, then deploy the endpoint** \u2014 the model must go through a
   **complete pipeline run** before deployment; the existing
   `readmission-final-20260723172647` is not the deployment candidate. Run the full
   pipeline, then `deploy_cpr.py` (image is cached; deploy path already validated).
2. **MCP server + `predict_readmission`** \u2014 local, stdio, BigQuery feature path.
   **Exit = Tier 1 acceptance.**
3. **LangGraph agent** \u2014 calls the tool via MCP, still fully local.
4. **Deploy both to Cloud Run** \u2014 agent **private**, MCP over HTTP. Check A2UI maturity
   and its integration story here, before it lands on the critical path.
5. **Django BFF + A2UI demo page** \u2014 auth-gated with per-user quota. The agent emits a
   **single fixed component** (the risk card: probability, threshold, top factors).
   \u27f5 *Walking skeleton complete: BigQuery \u2192 model \u2192 MCP \u2192 agent \u2192 live website.*
   **First tool call + UI done \u2014 this is the milestone gate.**
6. **RAG as the second tool** \u2014 add the citation/passage component; the agent now
   genuinely chooses between output shapes.
7. **Finalize orchestration + demo UI** \u2014 user journeys, the demo script, and the Tier 2
   eval (faithfulness/groundedness rubric + golden set).
8. **Publish**, then the deployment-optimization cleanup pass.

### Risk mitigation at Step 5 (agreed)

Learning A2UI *and* debugging the agent at the same time is the main risk. Mitigation:
lock the agent's output to **one fixed A2UI component** first and prove the render path
end-to-end with a known-good payload. Only Step 6 introduces real dynamism — onto a pipe
already trusted. This is why A2UI's multi-shape capability is deliberately deferred to
Step 6 even though A2UI itself is committed from Step 5.

### Status of gaps (updated 2026-07-24)

**Step 1 (MCP/agent) — CLEARED to start:**
- **Repo layout** — ✓ `projects/agent-harness/mcp/` (server + tools),
  `projects/agent-harness/agent/` (LangGraph runtime).
- **Dev-first sequence** — ✓ build + test locally (MCP over stdio, tool against local
  CPR container or BigQuery) before Cloud Run; redeploy the Vertex endpoint only for the
  live integration test.
- **Tier 1 acceptance** — ✓ base bar: *agent, given `hadm_id=20924467`, calls
  `predict_readmission` via MCP → prob `0.1314` + decision + top_factors, schema-valid.*
  Additions: (a) invalid `hadm_id` returns a graceful tool error (no crash); (b) assert
  the call routed through MCP (not a bypass); (c) response schema contract test
  (types + `top_factors` shape).
- **Feature source** — start on **BigQuery** for dev/CI; the **online Feature Store**
  path is the showcase, turned on only when actively demoing (bills continuously; the
  `create-feature-store` pipeline component already does most of the wiring).

**Website / UI:**
- **Domain** — ✓ page on the existing site: `danielmherman.com/projects/clinical_copilot/demo`.
- **UI ↔ agent auth** — ✓ **thin BFF in Django**: browser → Django view →
  (service-to-service ID token) → **private** agent Cloud Run. Agent off the public
  internet, no CORS, auth + rate-limiting centralized in Django.
- **A2UI** — ✓ committed as the rendering layer. Single fixed component at Step 5;
  multiple shapes from Step 6.
- **Deployment guide** — reviewed and corrected (region `us-east1`, `min-instances 0`,
  migrate-before-deploy in CI, clean-sheet §17 with no data migration, collectstatic
  failures no longer suppressed, superuser password via Secret Manager).

**RAG:**
- **Corpus** — ✓ RESOLVED: `physionet-data.mimiciv_note.discharge` (+ `radiology`),
  keyed by `hadm_id`. Unblocked.

**Demo / publish:**
- **Golden set** — ✓ exists: ~1,000 unique holdout patient IDs reserved for the demo;
  Tier 2 eval draws from this.
- **Publish plan** — `danielmherman.com/projects/`, social syndication, per-section Vimeo
  videos, and a **reproducible GitHub repo** for others. A deployment-optimization
  **cleanup pass is the explicit final step**.

**New action items (research / enablement):**
- **Enable Gemini on Vertex** — turn on Vertex AI + confirm Gemini Flash model access and
  the correct **region** (Gemini may require a different region than the mlops `us-east1`,
  e.g. `us-central1` / a global endpoint). Needed before Step 1 agent wiring.
- **PhysioNet DUA + LLM compliance** — research the credentialed-use terms on sending
  MIMIC text to LLMs. Staying **in-GCP via Vertex** (no prompt retention/training under
  enterprise terms) is the defensible path and reinforces the Gemini-via-Vertex choice.
  The reproducible repo must **not** publish any MIMIC data — document access steps instead.

**Cross-cutting:** secrets/auth (Secret Manager), CI/CD for the two new Cloud Run
services, PHI framing (MIMIC is de-identified — state it explicitly).

### Cleared to start Step 1
All three Step-1 gates are answered. Remaining pre-req: enable Gemini/Vertex (action
item above).
