# Architecture deep dive — one request, end to end

For tonight's discussion with another engineer. The diagram is
`docs/diagrams/request-flow.mmd`; the numbered steps below match its `autonumber`
so you can talk to the picture rather than describe it.

To render it as an image for slides:
`npm install -g @mermaid-js/mermaid-cli` then `bash scripts/render_diagrams.sh`
(writes `assets/diagrams/request-flow.png`). In the editor, right-click the
`.mmd` → **Preview Diagram**.

Already rendered and committed alongside this doc:
`assets/diagrams/request-flow.png` (1904×1879, for slides) and
`assets/diagrams/request-flow.svg` (vector — use this one if the deck scales it,
since it stays sharp). Both were produced through hosted renderers because
`mmdc` is not installed here; if you re-render, the repo script is the intended
path, and the hosted fallback is mermaid.ink or kroki.io with the source
base64-encoded.

A separate component-level view already exists as a Mermaid diagram inside
`docs/architecture/00-reference-architecture.md` — that one is the twelve-layer
map; this one is the request lifecycle.

---

## The cast, and why each piece is separate

| Component | Runtime | Exposure | Scales to zero |
|---|---|---|---|
| **Site** (Django) | Cloud Run `danielmherman` | Public | Yes |
| **Agent** (FastAPI + LangGraph) | Cloud Run `agent`, 1 Gi | **IAM-private** | Yes |
| **MCP server** (tools) | Cloud Run `mcp-server`, 512 Mi | **IAM-private** | Yes |
| **Model** | Gemini, pinned in code | — | — |
| **Predictor** | Vertex endpoint `readmission-endpoint` | IAM | Endpoint (billable while deployed) |
| **Retrieval** | Vertex index behind `readmission-rag-index` | IAM | Index (billable while deployed) |
| **Data** | BigQuery: `hybrid_features`, `hybrid_notes`, `demo_cohort` | IAM | — |
| **Conversation store** | Cloud SQL | Private | — |
| **Observability** | Cloud Logging + self-hosted Langfuse v4 | UI is public, behind its own login | Web yes, data plane on a VM |

The two private services are the ones worth emphasising: **neither the agent nor
the tool server is reachable from the internet.** The site holds an identity token
for the agent, and the agent holds one for the tool server. There is no path from
a browser to the model except through both hops.

---

## The request, step by step

**1 — Browser → site.** `POST /demo/a2ui/ask/` with the session cookie and CSRF
token. The body is either a chip (`{"chip": "risk", "hadm_id": ...}`) or free text
(`{"question": "...", "hadm_id": ...}`), plus `conversation_id` when the turn
continues one.

**2 — The site decides what this turn is.** In order:

- authenticate, then claim a credit (`10` per user per day, refunded when the
  failure was ours)
- resolve the conversation: pin the patient to it, check the 24-hour TTL, check
  the turn ceiling (`6`)
- if it is a follow-up, assemble `turns` from the stored rows — **the most recent
  6**, answers clamped to 8,000 characters, tool results carried only where they
  cannot be re-derived

This is the point to make clearly: **the site, not the agent, owns conversation
state.** The agent receives everything it needs in the request.

**3 — Site → agent.** `POST /ask` (or `/ask/stream` with
`Accept: text/event-stream`) with `Authorization: Bearer <ID token>` whose
audience is the agent's own URL. If the token is missing or the audience is wrong,
the request is rejected by Google's front end before any of our code runs — a 401
that looks nothing like an application error, deliberately.

**4 — The agent validates and starts.** The request contract is a **closed field
set**: `question`, `hadm_id`, `chip`, `turns`. Anything else is refused **by
name**, including conversation-shaped fields, with a message that says conversation
state belongs to the caller. The question is capped at 2,000 characters. Then a
LangGraph run starts with the question and replayed turns in state.

**5 — Agent → MCP server.** Tool calls go over the HTTP MCP transport, with the
agent's own ID token for the MCP audience. Why a separate service rather than
functions in the agent process: the tool server is the only component with data
access, it has its own identity and its own audit surface, and it can be revised,
scaled and reasoned about independently of the reasoning loop.

**6 — Inside the tools.** Three tools, all typed:

- `predict_readmission(hadm_id)` → the Vertex endpoint returns probability,
  threshold, decision, top factors, model version
- `rag_search(hadm_id, query, top_k)` → vector search over the note index
- `rag_search_sections(hadm_id)` → the deterministic section path

Every call carries the admission it is about, and **`readmission.demo_cohort` is
the authorisation boundary**: an admission outside the served cohort is rejected
rather than looked up. That is the control that makes a cross-patient question
return "unknown patient" instead of another patient's notes.

**7 — The model call.** Gemini, pinned in code rather than read from the
environment, bounded at 60s. The timeouts nest by construction and are **enforced
at startup**, not documented: model 60s < tool 100s < agent 110s. An operation that
could outlast its caller would fail invisibly, so the service refuses to start with
a bad combination.

**8 — The loop.** If the model asks for more evidence, the agent calls the tool
again and re-asks. The loop is step-capped, which is the cheap insurance against a
runaway bill.

**9 — Deterministic guardrails, after the model.** This is code, not a prompt:

- **numbers must be grounded** in tool output. A probability the model composed
  itself is removed and the fact recorded by name (`risk_number_unsupported:0.14`)
- **citations must be in range** for the turn (`citation_out_of_range:^1`)
- a follow-up is judged against **both** this turn's tool calls and the replayed
  turns' — because the earlier turns are in the transcript, so they are evidence
  for what the answer may say. Citations stay scoped to the current turn, because
  their numbering addresses this turn's presentation.

The served answer is the **guarded** one. What the model wrote and what the user
read are different objects, and the difference is recorded.

**10 — The presentation is composed here, not in the front end.** Citation
renumbering, source resolution and the A2UI canvas are evidence semantics: they
belong to the layer that ran the guardrails and saw the tool results. The site
passes them through; the browser renders them and interprets nothing.

**11 — The response contract is validated** before it leaves, so a payload outside
it becomes a stable error rather than a surprise in the front end.

**12 — Observability, as a sink.** Two things are emitted and **neither can fail
the answer**:

- one JSON execution record to Cloud Logging: revision, model, served model,
  outcome, finish reason, tokens, question length, duration, per-stage timings,
  tool names, tool errors, guardrail flags by name, and the trace id
- Langfuse spans for the graph, the route, the tools and the model calls

The record deliberately carries **no question or answer text** — length and shape
instead. A log store is not where clinical prose lives, and the same reasoning
governs retention.

**13 — Agent → site.** `200` with: `answer`, `guardrail_flags`, `tool_calls`,
`a2ui`, `sources`, `model`, `code_revision`, `mcp_transport`,
**`langfuse_trace_id`**.

**14 — The site stores the turn.** The question row and the answer row are written
**in one transaction** — half a turn cannot be replayed — carrying the model, the
code revision, citation identities (not passage text), tool calls with payloads
only where they cannot be re-derived, guardrail flags, and the trace id.

**15 — Back to the browser**, as JSON or as SSE frames
(`planning → tool → result → verify → answer`), which is what makes a slow answer
feel like work in progress rather than a hang.

**Out of band — the evaluation harness.** It calls the same `/ask` endpoint over
267 cases and attaches judge scores to each trace by id. That is the layer that
turns "it seems good" into a number and a browsable failure.

---

## Two design decisions worth dwelling on

**State lives with the caller, not the agent.** The agent is stateless: it holds
nothing between requests, which is why it can scale to zero, be replaced
mid-conversation, and be load-balanced without session affinity. Everything a
follow-up needs arrives in the request. The cost of that decision is the replay
window — the caller must choose what to send back, and it sends the last 6 turns.

**Observability may never break the answer.** Tracing is a sink. Unconfigured
means silently off; a broken SDK, a dead span or an unreachable UI degrades to
"no trace" and never to "no answer". The same rule is in the deploy: the Cloud
Build step looks the Langfuse host up and swallows the failure, so a renamed UI
cannot block an agent release.

---

## Failure taxonomy — what a user sees, and why

| Failure | What happens |
|---|---|
| Model or tool timeout | `502` with a stable code and a correlation id; the credit is refunded; nothing internal is disclosed |
| Tool returns an error | The tool error travels as a typed result; if the answer cannot be grounded the agent says so rather than inventing |
| Unsupported field | `400`, refused **by name**, with the reason (a conversation field is refused as a product decision) |
| Cold start | Both services at `min-instances 0`; the first request pays ~8s. It is a cost decision, and it is visible |
| Retrieval misconfigured | The eval preflight catches it in ~30s: six smoke questions, and a retrieval-failure rate above threshold aborts the run |

---

## Numbers to have ready

| | |
|---|---|
| Model | `gemini-3.1-flash-lite`, pinned in code |
| Timeouts | model 60s < tool 100s < agent 110s, enforced at startup |
| Agent / MCP | 1 Gi, max 3 instances / 512 Mi, max 3; **both IAM-private** |
| Conversation | 24-hour TTL, swept by a daily job; ceiling 6 turns |
| Credits | 10 per user per day, refunds capped at 3/day for failures that cost model spend |
| Replay window | last 6 turns, answers clamped to 8,000 chars |
| Predictor | AUCPR 0.328 vs HOSPITAL 0.251 baseline; threshold 0.11; Brier 0.112 |
| Evaluation (today) | 267 clinical cases, 94.0% pass, 6 safety failures; 20 adversarial probes, 19 pass |

---

## Questions to expect

**"Why not call the model straight from the browser?"** → API keys would have to
live in the client, and the client would become the place where guardrails,
contract validation, evidence and the audit record are supposed to be. The site
exists so nothing sensitive reaches the browser.

**"Why MCP instead of importing the tools?"** → separate identity, separate blast
radius, and one place that holds data credentials. It also lets the eval harness
and the demo use the same tools without duplicating access.

**"How do you stop one patient's notes appearing in another's answer?"** → the
cohort table is the authorisation boundary; every tool call carries its admission;
and there is an adversarial probe that asks for a second patient by id, which is
refused.

**"What stops hallucination?"** → three things, none of them the prompt: retrieval
returns cited passages; the judge scores groundedness against the evidence the
agent had; and deterministic guardrails delete ungrounded numbers and out-of-range
citations after the model runs.

**"How do you know a change helped?"** → the eval harness, the frozen rubric, the
judge validated against human labels, and a gate at 95% with zero safety failures.
It does not run in CI yet — that is a known gap, and today's run is manual.

**"What's the weakest part?"** → say it plainly: medication questions. 12 of 16
failures today were the agent reporting no medication list while the retrieved
passage contained the regimen, and the dedicated medication section is only
returned for 38 of 89 such questions. That is diagnosed, quantified, and the next
fix.

**"Why self-hosted Langfuse?"** → traces contain prompts and retrieved clinical
text; the data plane runs in our project on our VM, with the UI behind its own
login. The trade is that we operate ClickHouse, Redis, MinIO and a worker, and
that cost is real.
