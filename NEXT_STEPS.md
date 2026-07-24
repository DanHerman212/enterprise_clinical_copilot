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

## Open decisions to resolve before writing the detailed plan

1. **Agent framework** — Vertex Agent Engine + ADK (stays in the GCP/Vertex world) vs
   LangGraph / other? Shapes everything downstream.
2. **RAG corpus** — what unstructured data is actually available? (MIMIC-IV discharge
   summaries / notes?) Gates RAG scope; the single biggest unknown.
3. **UI base** — extend the existing `danielmherman` Django site, or a fresh app under
   `agent-harness`? And how much A2UI maturity are we counting on?
4. **MCP hosting** — local/dev MCP server first, or Cloud Run from the start?
