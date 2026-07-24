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

Full Phase 2 design captured in
[projects/agent-harness/docs/architecture.md](../projects/agent-harness/docs/architecture.md).

---

## Finalized 5-Step Roadmap (2026-07-24)

1. **Agent Runtime + MCP** — MCP server with `predict_readmission`, LangGraph agent on
   Cloud Run + Gemini Flash, pluggable feature source, single-turn.
   **Exit = Tier 1 integration test passes.**
2. **Website + UI** — deploy clean-sheet Django to Cloud Run (per the GCP guide), wire a
   UI to the agent runtime.
3. **RAG tool** — build the Vector Search index + `rag_search`, add as the agent's
   second tool.
4. **Demo + evaluation** — define user journeys and the Tier 2 eval
   (faithfulness/groundedness rubric + golden set of `hadm_id`s).
5. **Publish results.**

### Status of gaps (updated 2026-07-24)

**Step 1 — CLEARED to start:**
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

**Step 2 — website/UI:**
- **Domain** — ✓ page on the existing site: `danielmherman.com/projects/clinical_copilot/demo`.
- **UI ↔ agent auth** — RECOMMEND **thin BFF in Django**: browser → Django view →
  (service-to-service ID token) → **private** agent Cloud Run. Agent off the public
  internet, no CORS, auth + rate-limiting centralized in Django.
- **A2UI** — experiment on the demo page; plain Django templates as the first-cut fallback.

**Step 3 — RAG:**
- **Corpus** — ✓ RESOLVED: `physionet-data.mimiciv_note.discharge` (+ `radiology`),
  keyed by `hadm_id`. Step 3 unblocked.

**Step 4–5:**
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
