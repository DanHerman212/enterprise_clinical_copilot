# Agent Harness

The orchestration layer: an agent that reasons over the risk model's output, exposed to
a clinician through a web interface. The model answers *what* the risk is; this project
answers *why*, allowing a clinician to interrogate the prediction.

## Architecture

```mermaid
flowchart LR
  U["Clinician"] --> DJ["Django BFF<br/>auth + quota"]
  DJ --> LG["LangGraph Agent<br/>Cloud Run · private"]
  LG <--> GM["Gemini Flash<br/>Vertex"]
  LG --> MCP["MCP Server<br/>stdio dev · HTTP prod"]
  MCP --> T1["predict_readmission"]
  MCP --> T2["rag_search"]
  T1 --> EP["Vertex Endpoint"]
  T1 --> FS["Feature source<br/>BigQuery"]
  T2 --> VS["Vector Search"]
  LG --> A2["A2UI components"]
```

## Stack

| Concern | Decision | Why |
|---|---|---|
| Agent framework | **LangGraph** | Portable and model-agnostic; not coupled to one cloud's agent product |
| Runtime | **Cloud Run** | Serverless, scales to zero, we own the container |
| LLM | **Gemini Flash** via Vertex | Near-free per token, managed, and keeps clinical text inside GCP |
| Tool exposure | **MCP server** | Reusable by this agent, by desktop clients, by future agents |
| Transport | **stdio** (dev) + **streamable HTTP** (prod) | One implementation; develop locally, deploy remote |
| Feature source | **BigQuery** | Low-latency for demos, near-free for dev and CI |
| Rendering | **A2UI** | Agent output is designed to be renderable as components, not just prose |
| UI ↔ agent auth | **Thin BFF in Django** | Agent stays private; auth and quota centralized in one place |
| Shared state | **Cloud SQL (Postgres)** — no Redis | Postgres covers quota, sessions, and checkpoints durably across instances |

## Design notes

**Tools are stateless.** `predict_readmission(hadm_id)` hides all feature plumbing
behind a single identifier: resolve features from the configured source, call the Vertex
endpoint, return probability + decision + parent-aggregated TreeSHAP factors. It is a
pure function of `hadm_id`, which makes it trivially testable and cacheable.

**The MCP server is the reusable seam.** Putting tools behind MCP rather than wiring
them directly into the agent means the same server works from a local stdio client
during development and over authenticated HTTP in production, with no change to tool
logic. Adding retrieval later is adding a tool, not rearchitecting.

**Orchestration is deliberately thin at first.** With one tool, the agent's routing is
nearly trivial — and that is intentional. The value of the first build is the plumbing
(MCP + runtime + deploy path), which makes adding subsequent tools inexpensive.

**Cloud Run is stateless and scales to zero**, so anything crossing requests — per-user
quota, sessions, agent checkpoints — lives in Postgres. Redis is deliberately absent;
it would only be warranted for WebSocket broadcast to multiple simultaneous viewers,
which is out of scope.

## Evaluation

Two tiers, kept separate because conflating them is how agent projects lose credibility.

**Tier 1 — integration (deterministic, CI-able).** Does the agent call the right tool
with the right arguments? Does it route through MCP rather than bypassing it? Does the
response schema validate? Does `hadm_id=20924467` return the known-good `0.1314`? Does
an invalid ID fail gracefully?

**Tier 2 — output quality (semantic).** Is the narrative *faithful* to the tool outputs
— no invented SHAP factors — and *grounded* in retrieved passages? Measured against a
300-trace golden set (100 held-out patients × 3 prompts: risk, medications, summarize)
with an LLM-as-judge against a versioned rubric. Latest run (`gemini-2.5-flash`):

| Metric | Result |
|---|---|
| Pass rate | **95%** (285 / 300) |
| Agent errors | 0 |
| Faithfulness | 96% |
| Groundedness | 98.7% |
| Citation accuracy | 99.7% |
| Clinical correctness | 99% |
| Safety | 99% — 3 medication-safety failures remain (contradictory instructions, an invented dosage, an incorrect frequency) |

The critical principle: this is **not** re-evaluating the model. Model correctness is
the MLOps AUCPR, already established. Agent evaluation measures only whether the
narrative is supported by what the tools actually returned.

## Deeper reading

- [architecture.md](docs/architecture.md) — full component design and rationale
- [BUILD_GUIDE.md](docs/BUILD_GUIDE.md) — step-by-step build plan (MCP → agent → A2UI demo)
- [a2ui_spike_findings.md](docs/a2ui_spike_findings.md) — verified A2UI rendering recipe and the four failure modes behind it
- [rag_requirements.md](../../docs/rag_requirements.md) — contract for the planned retrieval tool
- [NEXT_STEPS.md](../../docs/NEXT_STEPS.md) — build sequencing
