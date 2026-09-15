# Layer 3 — Orchestrator (the chain)

Status: audited 2026-09-13. Gap 1 closed 2026-09-15 by decision B (stream
progress, not prose) and verified against the live stack. Gaps 2–7 open.

---

## 1. What the orchestrator is

The orchestrator is the component that decides **what happens between a question
and an answer**: which prompt the model receives, whether it must consult
evidence before answering, what it does with what the tools return, and when to
stop. Everything else in the architecture is a capability it calls. This is the
only layer that owns control flow.

```
  question ─▶ [ prompt template ] ─▶ [ model turn ] ─┬─▶ answer ─▶ [ guardrails ] ─▶ response
                                        ▲            │
                                        │            └─▶ tool call ─▶ [ tool ] ─┐
                                        └───────────────────────────────────────┘
                                          loop until the model stops calling tools
```

Terms used in this document:

| Term | Meaning |
|---|---|
| Chain | The prompt template, the control flow, and the tool wiring, considered as one thing. Google calls it "the chain"; other vocabularies say agent, graph, or pipeline. |
| Agent | A chain that may call tools several times before answering, rather than following a fixed sequence. |
| Superstep | One pass of a graph execution: a model turn, or the tool node that runs what the model asked for. Bounding supersteps is how you bound cost. |
| Prompt-as-code | Instructions, tool contracts and guardrails: they change behaviour, so they belong in version control with review and tests. |
| Prompt-as-data | Retrieved passages, few-shot examples, the user's question: they arrive at runtime and need validation rather than review. |
| Stateless | The process holds nothing between requests. Any instance can serve any request, and a restart loses nothing, because there is nothing to lose. |
| Streaming | Sending the answer as it is generated rather than after it is complete. The distinction matters for perceived latency, not for correctness. |
| Session state | The conversation so far. If a product needs follow-up questions, this must live somewhere that outlives the process. |
| Guardrail | A check that runs after the model answers — deterministic where possible — that can correct or refuse the answer. |
| Model pin | The exact model revision in use, recorded so an answer can be reproduced later. |

**Layer 3 is: the prompt, the loop, and the wiring that decides what the model
sees and does.** It owns no user interface, no tool implementation, and no
storage — but it is the layer most likely to be asked about in an interview,
because it is where an AI system's behaviour actually lives.

---

## 2. What Google requires of this layer

From *Single-agent AI system using ADK and Cloud Run* and the architecture
document in the requirements file (`docs/google-cloud-ai-architecture-requirements.html`),
grouped here by what the orchestrator owns rather than by where they appear.

**The API it exposes**

1. **The front end never calls the model directly.** It talks to the agent over
   an API. That API must support streaming, be stateless, and keep conversation
   state outside the process. *(required)*
2. **The agent is stateless.** Session state lives in an external store
   (Cloud SQL, Firestore, Memorystore, or a platform session service) so any
   instance can serve any request and a restart loses nothing. *(required)*
3. **The agent runs on an agent runtime** — Cloud Run, a managed agent runtime,
   or Kubernetes — and is built with an agent framework. *(required)*
4. **The model comes from a managed runtime.** Serving the model is a
   dependency, not something this application hosts. *(required)*

**The chain as an artifact**

5. **The prompted model component is the smallest deployable unit.** Prompt
   template, chain definition, tool wiring and model pin are versioned together
   as one artifact with its own revision history. *(required)*
6. **Prompt parts are classified** into prompt-as-code (template, guardrails,
   instructions — version control, review, tests) and prompt-as-data (examples,
   retrieved context, the user's question — validation and drift detection).
   *(required)*
7. **Every execution logs its inputs, its outputs, the intermediate state of
   each step, and the chain configuration used.** *(required)*
8. **Chains are evaluated end to end**, not component by component. *(required
   — the evaluation work itself belongs to the evaluation layer; the
   orchestrator's obligation is to be evaluable.)*

**The loop**

9. **A step cap on any agent loop** so a pathological turn cannot buy unbounded
   spend. *(required)*
10. **Start with a single agent** and improve prompt and tools before adding
    agents; multi-agent only when one agent measurably fails. *(required)*
11. **Every loop pattern has an explicit exit condition** — iteration cap,
    token budget, or quality threshold. *(required)*
12. **Define the requirements before choosing a pattern**: task shape, latency
    tolerance, cost budget, and how much human judgement is needed. *(required)*

**Owned by other layers, listed so the boundary is explicit**

- Tools exposed over MCP with typed schemas, and small tool definitions —
  belongs to the tools layer (layer 5).
- Cross-session memory in an external store — belongs to the memory layer
  (layer 8), and is only required if the product needs personalisation.
- Structured logging and distributed tracing — belongs to the observability
  layer (layer 10). The orchestrator's obligation is to emit the data;
  collecting and querying it is that layer's job.

---

## 3. How this application implements the layer

Repository `enterprise_clinical_copilot`, service `services/agent/`. The
front end calls Django; Django calls the agent; the agent calls the tools.

### 3.1 The chain

`services/agent/graph.py` builds it explicitly rather than through a
prebuilt ReAct helper, so the control flow is readable:

| Element | Where | What it does |
|---|---|---|
| Prompt injection | `graph.py` 310–311 | One `SystemMessage` carrying the prompt, then one `HumanMessage` carrying the question. Nothing else goes in. |
| Model turn | `graph.py` 242 (`agent_node`) | `llm.ainvoke(state["messages"])`, returning the model's message. |
| Tool turn | `graph.py` 259 (`tool_node`) | Runs whatever the model asked for, one call at a time. |
| Loop decision | `graph.py` 265 (`route`) | If the model's last message contains tool calls, go to the tool node; otherwise end. |
| Edges | `graph.py` 271–275 | `START → agent`, `agent → tools` conditionally, `tools → agent`. |
| Tool wrapper | `graph.py` 172 (`_MCPTool`), 209 (`_tools`) | One LangChain tool per MCP tool, advertising the real parameter names and types from the MCP `input_schema`. |
| Model | `services/mcp/config.py` 83, 89; `graph.py` 192 (`_build_llm`) | `gemini-2.5-flash` by default, 2048 output tokens, temperature 0, Vertex backend, 3 retries. Two things to be precise about. First, `gemini-2.5-flash` is an **alias**, not a dated revision: what it resolves to is Google's decision, so it is not a pin in the sense this document uses the word. Second, the value is read from the environment (`os.environ.get("GEMINI_MODEL", ...)`), so the effective model is deployment configuration; the agent's `cloudbuild.yaml` happens not to set it, so today the code default applies. |
| Thinking budget | `config.py` 85–89 | None set explicitly. The 2048-token output budget covers thinking *and* the answer, and the comment records the failure that produces: spend it all on thinking and the call returns 200 with empty text. `final_text` exists to catch that case. There is no separate knob for how much the model may think, which is a real latency and cost lever left at its default. |
| Loop bound | `graph.py` 101 (`RECURSION_LIMIT = 10`) | LangGraph supersteps. Agent → tools → agent is three, so this allows about four tool rounds before the graph raises. |
| Per-turn tool bound | `graph.py` 100 (`MAX_TOOL_CALLS_PER_TURN = 5`) | Calls beyond the budget are refused with a structured error rather than executed. |
| Per-request rebuild | `graph.py` 229; `http.py` 174 (`_run_chain`) | `build_graph()` runs on every request, which rebuilds the model client and re-wraps the tools, and `toolbox()` opens a fresh MCP session and lists the tools each time. This is what makes the service stateless. It is also a fixed cost paid before the first model turn; it has now been measured (section 7). |

The shape is deliberate for this product: one question is one or two tool calls,
so a graph that loops until the model stops asking is sufficient and cheap. A
fixed pipeline would be cheaper still, but the model needs to choose between a
risk score and a notes search, and occasionally needs both.

### 3.2 The API it exposes

`services/agent/http.py` is a Starlette app with three routes (`http.py` 414–419):

| Route | Method | Behaviour |
|---|---|---|
| `/health` | GET | Status, model name, MCP transport (`http.py` 186). No topology disclosure. |
| `/ask` | POST | Takes `{"question": "..."}`, returns one JSON object. `http.py` 204. |
| `/ask/stream` | POST | The same chain, with progress stages streamed ahead of the answer. `http.py` 288. Added 2026-09-15; see 3.7 and 6.1. |

`/ask` is unchanged by the streaming work: it takes the same input, returns the
same object, and fails with the same codes. The stream is a second route rather
than a second behaviour of the first, so a caller that does not ask for it
cannot be affected by it.

The error contract is part of the API, and the caller depends on it:

| Status | `error` | When | Where |
|---|---|---|---|
| 401 | `unauthenticated` | No `Authorization` header on Cloud Run | `http.py` 67, 73 |
| 400 | `invalid_json` | Body is not JSON | `http.py` 93 |
| 4xx | `question_too_long` and the shape errors | From `parse_agent_request` | `contracts.py` 53–75 |
| 504 | `timeout` | The chain exceeded its wall-clock bound | `http.py` 216, 220 |
| 502 | `agent_failed` | Any infrastructure failure; carries a 12-character `correlation_id` and no exception text | `http.py` 236–240 |
| 502 | `answer_unavailable` | The final model turn was empty, or the payload failed its own contract | `http.py` 246–249 |

Everything above the first byte is the same for both routes. The streamed route
returns those same statuses while nothing has been written; once a frame has
been sent the status code is spent, and the same failures arrive as a terminal
`error` frame carrying the same code and correlation id (`http.py` 274).

| Property | Where | Detail |
|---|---|---|
| Identity required | `http.py` 57, 67 | Requests must carry an `Authorization` header when running on Cloud Run. Cloud Run's own invoker check runs first; this is a presence check, not a second validation of the token. |
| Bounded input | `http.py` 42; `contracts.py` 53, 75 | Question must be a non-empty string, ≤ 2000 characters. The parsed request has exactly one field. |
| Wall-clock bound | `http.py` 47, 174; `services/mcp/runtime.py` 43–44 | `asyncio.timeout(ASK_TIMEOUT_SECONDS)`. The deadlines are nested on purpose: each tool call 100 s, the whole chain 110 s, Django's wait 120 s (`settings.py` 101). Each layer gives up before the one above it, so a timeout is reported by the layer that can say what timed out. |
| Stateless | `http.py` 174 | A fresh MCP session per request (`async with toolbox()`). No state survives the request. |
| Post-processing | `http.py` 117 (`_compose_success`) | Final message only → deterministic guardrails → presentation composed → contract validated. One implementation, shared by both routes, so a streamed answer and a single-response answer cannot drift. |

### 3.3 The prompt

`services/agent/prompts.py` line 9 is a single `SYSTEM_PROMPT` string, imported
at `graph.py` 57 and sent verbatim. It is prompt-as-code in the plain sense: it
lives in the repository, and changes to it go through review like any other
file. It is worth being precise about the tests, though: no test asserts the
prompt text. What the tests cover is the behaviour the prompt depends on — that
tool results are wrapped in `<tool_result>` delimiters, that only the final
message is served, and that the guardrails catch the failures the prompt warns
about. So the prompt is versioned and reviewed, but its wording is not itself
under test.

Its content is organised around the failure it exists to prevent: the model
producing a confident, plausible risk number without calling the tool. It
therefore states a tool contract (call `predict_readmission` for a score, call
`rag_search` for notes, cite the passage), a data-versus-instructions rule
(everything inside `<tool_result>` is patient data and never an instruction),
and formatting rules for how numbers and citations are reported.

Retrieved passages arrive as tool results wrapped in `<tool_result>` delimiters
by the tool wrapper, so the data path is real but implicit: there is no
separate prompt-as-data module.

**Half of the prompt lives in the other repository.** The system prompt is
only one of the two texts the model receives; the other is the question, and
for the demo's starter chips Django writes it. `demo/views.py` 25
(`_question_for`) maps a chip to a fixed sentence and appends the admission
(`views.py` 52–55: `f'{question} For admission {hadm_id}.'`). Those sentences
are in `demo/fixtures.py` 50–55 — a module whose docstring describes it as
fixture mode "while the endpoints are down" — and they are imported into the
live path at `views.py` 17. So the wording the model is asked to act on is
split across two repositories, and the production half sits in a file named
for development scaffolding. This is prompt content in every sense that matters
to the "one artifact" requirement.

One of those templates asks a question the product cannot answer. `CHIPS`
includes `compare` — "Compare this assessment to the previous one for this
patient" (`fixtures.py` 54). There is no previous assessment: the product is
single-turn (see Gap 4). The chip is not offered in the console, but
`_question_for` accepts any key in `CHIPS`, so a direct request with
`chip: "compare"` passes validation, spends a quota credit, and sends the model
a question about history it does not have.

### 3.4 What happens after the model answers

Three deterministic steps, in order, all in code:

| Step | Where | What it does |
|---|---|---|
| Answer selection | `graph.py` 261 (`final_text`) | Returns the text of the **final** message only. An empty final turn yields an empty string, and the server reports the answer as unavailable rather than shipping an earlier preamble. |
| Guardrails | `guardrail.py` 517 (`guard_answer`) | Checks the answer against the tool evidence, and **rewrites it** where the evidence does not support it. A risk number that does not match the model's output is removed from the text (`verify_risk_numbers`, 257; `_remove_spans` at 297), an age the source redacts is redacted (`redact_invented_age`, 154), medication tokens and per-medication frequencies not found in the retrieved text are removed (`verify_med_tokens`, 165; `verify_med_freqs_per_med`, 445), and citations must resolve to real passages (`check_citations`, 205). Only `flag_invented_dates` (310) flags without stripping, and its docstring says why. Every guard also appends a flag, so the response carries both the corrected text and the list of what was corrected. |
| Presentation | `a2ui.py` 328 (`compose_presentation`), 154 (`resolve_sources`) | Builds the canvas payload and resolves citation numbers to sources. This is the agent's job by decision of layer 1: the numbering the user sees is evidence semantics, so it is composed where the evidence is. |

The consequence for this layer is worth stating: the response is **not** a
string, and the text in it is **not necessarily what the model wrote**. It is a
validated object containing the guarded answer, the guardrail flags, the
trimmed tool calls, the resolved sources, and the canvas. Any change to how the
answer reaches the browser has to keep that contract intact, and any design
that shows the model's text before the guardrails have run is showing a
different answer from the one the system stands behind.

### 3.5 The caller

Django is the only public caller, and it is not quite a pass-through: it
composes part of the prompt (3.3) and it inspects the chain's output.

| Property | Where |
|---|---|
| One blocking POST per question, `Authorization: Bearer <identity token>` | `demo/agent_client.py` 110, 140 |
| Upstream deadline 120 s | `demo/agent_client.py` 144; `danielmherman/settings.py` 101 |
| At most 4 concurrent agent calls per instance | `demo/agent_client.py` 44, 122; `settings.py` 112 |
| The view that calls it, and returns the payload plus the remaining quota | `demo/views.py` 109, 150 |
| Reads the agent's `tool_calls` and turns a 200 with any tool error into a 502 plus a quota refund | `demo/views.py` 93 (`_tools_errored`), 168 |
| Served by uvicorn (ASGI), two workers | `Dockerfile` 42–45 |
| What the browser shows while waiting | a pending turn with `text: '…'`, `meta: 'working'` — `static/js/demo_flow.js` 603 |

`_tools_errored` matters for this layer because it is a second consumer of the
chain's internal structure: it depends on tools reporting failure as
`{"error": ...}` inside `tool_calls[].response`. Any change to how the agent
surfaces tool failure has to keep that shape or change Django with it.

So the user experience today is: send a question, watch a placeholder for
several seconds, receive the whole answer and the canvas at once.

### 3.6 Tests

25 test modules under `tests/agent/` (26 files including `conftest.py`). The
ones that speak to this layer directly: `test_graph_rewrite_smoke.py` (the
graph is constructed and the tools call through, without contacting Vertex),
`test_answer_integrity.py` (only the final message is served; tool results are
wrapped and delimited), `test_spend_caps.py` (both caps hold — it asserts the
per-turn refusal at lines 46–53 and the recursion bound at 60), `test_guardrail.py`
(the post-hoc checks), `test_agent_contract.py` (the response shape Django
depends on), and `test_progress_stream.py` (added 2026-09-15, section 3.7).

### 3.7 The progress stream (added 2026-09-15)

The chain takes about ten seconds warm and longer cold, and until now the caller
had nothing to show for any of it. `/ask/stream` streams *stages*, not prose: the
answer still arrives whole, and section 6.1 explains why that is not a
compromise but the only honest option.

| Element | Where | What it does |
|---|---|---|
| The vocabulary | `services/agent/stages.py` 51 (`STAGES`) | The closed set of stage values: `planning`, `reviewing`, `tool`, `verify`, `answer`, `error`. A client can switch on them exhaustively, and a test asserts the set has not grown by accident. |
| The labels | `stages.py` 70 (`TOOL_LABELS`) | Display text per MCP tool, kept next to the prompt rather than in the browser: the browser cannot know what tools exist, and must not be where wording about the agent's internal steps is invented. A test walks the live tool list and fails if a tool has no label. |
| Where a stage is emitted | `graph.py` 104 (`_emit`), called at 152 and 252 | Immediately before the tool call and immediately before the model call — derived from an execution about to happen, never from a script of what should happen. |
| A refused call | `graph.py` 130 (`_execute_tool_calls`) | Announces nothing. The per-turn budget refusal returns a synthetic error without touching the tool, and announcing it would describe work that never started. |
| The relay | `http.py` 318 (`_stream_chain`) | Drains a queue of stages, emits SSE frames, then exactly one terminal frame. A keepalive every 15 s (`http.py` 54), because a tool call may legitimately take 100 s and a connection silent for that long is closed by an idle timeout before the answer arrives. |
| The terminal frame | `http.py` 262 (`_sse`), 274 (`_error_frame`) | The `answer` frame carries exactly the object `/ask` returns, validated by the same `_compose_success`; the `error` frame carries the same code, message and correlation id `/ask` would have put in its body. |

Three rules hold this together, and they are the reason the display can be
trusted:

1. **A stage is emitted where the work happens, not before it.** There is no
   timer and no scripted sequence, so a stage cannot outrun the work it
   describes.
2. **A stage describes an action being taken, never a result.** "Searching the
   discharge notes" is true whether the search returns passages or fails.
   Whether the answer is good is the terminal frame's business.
3. **A call that is refused is not announced.** Announcing it would tell the
   user about work that never started, which is the specific dishonesty this
   design exists to avoid.

Across the proxy, Django relays the frames as they arrive (`demo/views.py` 148,
`_stream_frames`) and the browser puts the stage label where "working" used to
be (`static/js/demo_flow.js` 644). The answer body stays a placeholder until the
guarded answer arrives, so no path exists by which a stage can be mistaken for
the answer.

---

## 4. Current state against the requirement

| Google requires | What exists | Met? |
|---|---|---|
| Front end never calls the model directly | Browser → Django → agent; the agent is IAM-private | **Yes** |
| The agent API supports streaming | `/ask/stream` streams progress stages while the chain runs, then the answer as the final frame; `/ask` is untouched. Closed 2026-09-15. | **Partly** — the API streams, and a caller sees the work as it happens instead of a blank wait. It is not a streamed *answer*: the guardrails rewrite the text after the model finishes, so the corrected answer is delivered whole (6.1). |
| Stateless API | Fresh MCP session per request, fresh graph per call, no state between requests | **Yes** |
| Conversation state externalised | There is no conversation: the wire contract carries one question and nothing else | **Not applicable yet** — needs a decision, not code |
| Agent framework on an agent runtime | LangGraph/LangChain on Cloud Run, deployed through a build pipeline | **Yes** |
| Model from a managed runtime | `gemini-2.5-flash` on Vertex through `langchain-google-genai`, ADC | **Yes** |
| Prompt, chain, tools and model pin versioned as one artifact | The system prompt, the graph and the tool wiring live in this repository. The model is an alias, not a revision, and is overridable by an environment variable at deploy time. The question templates live in the *other* repository. Nothing records which combination served a given answer | **No** — the pieces are in version control, but not together, not pinned, and not recorded |
| Prompt-as-code classified and tested | The prompt is in version control and reviewed, and the behaviours it depends on are tested — but no test asserts its text | **Partly** |
| Prompt-as-data classified (examples, retrieved context, drift) | Retrieved passages flow as tool results, but there is no separation of data-side prompt parts and no drift detection | **Partly** |
| Every execution logs inputs, outputs, intermediate steps and chain config | One line per request: trace id, tool count, guardrail flag count. No question, no answer, no per-step state, no prompt or chain revision | **No** |
| Step cap on the agent loop | Recursion limit 10 supersteps, 5 tool calls per turn, wall-clock timeout | **Yes** |
| Single agent before multi-agent | One agent. No multi-agent structure to justify or remove | **Yes** |
| Explicit exit condition on every loop | The loop exits when the model stops calling tools, and three independent caps bound it | **Yes** |
| Requirements defined before pattern chosen | This document set; the layers are defined before they are built | **Yes** |

---

## 5. Gaps

**Gap 1 — No streaming, and the requirement is explicit.** *Closed 2026-09-15 —
see 6.1 for the decision and 7 for the evidence. The analysis below is the reason
the decision went the way it did and is kept as it was written.* The front end
does not support streaming and neither does the API behind it (`http.py` 204
returns a single `JSONResponse`). The user watches `working` until the whole
chain finishes: a model turn, one or two tool calls, a second model turn,
guardrails, and composition. This is the same gap the client layer recorded and
deferred, and it lands here because the API that would have to stream is this
one.

There is a complication worth understanding before choosing an option: the
response is not a string, and its text is not what the model wrote. The canvas
and the resolved sources are composed from the tool results (`a2ui.py` 328),
and the guardrails (`guardrail.py` 517) remove unsupported numbers,
medications and ages from the text before it leaves. Streamed tokens would
therefore not be the final answer — they would be a draft that is corrected on
the way out, and for a clinical answer the difference between the draft and the
corrected text is exactly the part that matters.

**Gap 2 — The chain has no single identity, and none at runtime.** Google
wants prompt, chain, tool wiring and model pin versioned together. Here the
four parts are in version control but not as one thing: the system prompt,
graph and tool wiring are in this repository; the question templates are in
the website repository (`demo/fixtures.py` 50–55); the model is an alias
(`gemini-2.5-flash`) rather than a dated revision, and it is read from the
environment (`config.py` 83), so it can change at deploy time with no commit
anywhere. And nothing *records* what ran: there is no chain or prompt revision
in the logs, the model name appears only in the response payload and
`/health`, and the one log line per request (`http.py` 177) carries a trace
id, a tool count and a flag count. The requirement asks for inputs, outputs,
the intermediate state of each step, and the chain configuration; none of
those four is currently emitted. The practical consequence is that a bad
answer today cannot be tied back to the exact prompt, template and model
revision that produced it — which is also the first thing an evaluation loop
and an incident review need.

**Gap 3 — Prompt-as-data is not separated, and the prompt itself is untested.**
All prompt content is prompt-as-code. Retrieved context reaches the model only
as tool results, and the user's question is validated for type and length
(`contracts.py` 53–75). What does not exist: any few-shot examples, any explicit
classification of which parts of the prompt are data, and any drift detection on
what arrives at runtime. Google's requirement is a classification, and this
application has effectively classified everything as code — which is defensible
for a fixed clinical prompt, but it should be a recorded decision rather than an
accident. There is a second half to this: prompt-as-code is supposed to mean
*tests*, and no test asserts the prompt's text. The instructions the model is
actually given are the least-covered artifact in this layer.

**Gap 4 — Conversation state: absent, and the requirement assumes it exists.**
The requirement says session state must live in an external store so any
instance can serve any request. This application satisfies the "stateless"
half completely — and the state half is moot because the product is single-turn:
`parse_agent_request` accepts one question (`contracts.py` 53, 75) and the view
never sends history. If a follow-up question is ever wanted, this becomes real
work: a session identifier on the wire, a store, and a retention policy. Worth
deciding deliberately, because "our agent is stateless" is only half the
sentence, and an interviewer will ask about the other half. The `compare` chip
(3.3) is the concrete instance: a question about a previous assessment that
the system accepts and cannot answer.

**Gap 5 — Live verification of this layer is limited by the tool endpoints.**
The model is live and the MCP protocol works end to end; the prediction and
retrieval endpoints behind the tools are not running, so a production request
today completes the chain with tool errors and Django returns 502 rather than an
answer (observed and recorded in the edge layer document). The chain, the caps,
the guardrails and the contract can all be exercised this way; a trustworthy
clinical answer cannot. Full verification of this layer therefore waits on the
model-runtime and tools layers.

**Gap 6 — Housekeeping from the observability teardown.** The agent's Cloud
Run environment still references two secrets for the observability stack that
was torn down; the code path is disabled, so this is dead configuration rather
than a fault. The code still describes that stack as live: `graph.py` 13–30
explains how every `/ask` becomes a native Langfuse trace, and the module
imports and wires a handler that is now always the no-op. Both are removed the
next time the agent is touched, and the observability layer decides what
replaces them.

**Gap 7 — Latency: one of its two inputs is now measured, the other is not.**
Production shows 3 to 43 seconds per question (edge layer document, section
7.5), and the streaming decision in 6.1 is really a decision about what to do
with that time. The per-request rebuild (graph, model client, MCP session and
tool listing — 3.1) has now been measured and is what dominates the wait before
the first stage: about 22 s on a cold instance against about 1 s warm (section
7), which makes it the largest single lever on perceived latency in this layer.
The model's thinking still has no explicit budget and no measurement, and
remains the untested one.

---

## 6. Design decisions

Gaps 2, 3 and 4 are questions of intent more than effort, Gap 6 is
mechanical, and Gap 7 is a measurement that should precede the rest. Gap 1 is
the substantial one.

### 6.1 Streaming (Gap 1)

| Option | What it is | Cost | Trade-off |
|---|---|---|---|
| **A. Stream the answer tokens.** | Agent `/ask` becomes a streamed response; Django proxies it without buffering; the browser renders text as it arrives. | Days. Touches the agent API, the Django proxy, the client, and the tests for all three. | Best perceived latency. But it breaks the current contract: the guardrails run *after* the text is complete and rewrite it, so streamed text can be amended or refused afterwards — the user would see a number that is then removed. It also splits a validated object into a stream plus a trailing payload. |
| **B. Stream progress, not prose.** | Keep the single final payload, and stream *events* about the chain: "reading the risk model", "searching the discharge notes", then the answer and canvas as today. | Hours to a day. One SSE path plus client handling; the response contract is unchanged. Django already runs under uvicorn (ASGI), so no server change is needed to hold an open response. | The user sees what is happening, which is where most of the perceived slowness comes from, and no unguarded text ever reaches the screen. It is not what Google says a production front end needs, and it does not reduce time-to-first-token for the answer itself. |
| **C. Do not stream; cut the latency.** | Record it as a deliberate deferral. Measure the per-request rebuild and the thinking budget (Gap 7), shorten the chain where the evidence allows, keep instances warm, and keep the placeholder honest about what is happening. | Hours. Configuration and prompt work. | Cheapest and keeps the answer contract exactly as verified. Does not meet the requirement as written, so it needs to be a stated decision with a reason, not an omission. |

A fourth possibility, worth naming because it is what the requirement actually
describes: **stream the answer but only after the guardrails have seen it** —
buffer the model's final turn, guard it, then stream the corrected text and
deliver the canvas when it is ready. That gives real streaming without ever
showing unguarded clinical text, at the cost of losing the earliest tokens
(which is where most of the latency gain lives for a short answer).

Recommendation: **B now, with the design for streaming written down**, unless a
requirement appears for genuine time-to-first-token on the answer text. The
answers here are short — a few sentences plus a canvas — so the honest win is
telling the user which step is running, not spilling tokens.

Decision (2026-09-15): **B**, implemented and verified against the live stack.
The browser asks for `Accept: text/event-stream`; the agent relays stages while
the chain runs and the answer as the final frame. A caller that does not send
that header gets exactly what it got before.

Two consequences are recorded rather than glossed over:

- **The requirement is met in part, and the wording matters.** A production
  front end that must show the answer as it is written is not served by this.
  The reason is the complication above: the guardrails rewrite the text after
  the model finishes, so streamed tokens would be a draft the user watches being
  corrected, and for a clinical answer the corrected part is the part that
  matters. If genuine time-to-first-token on the answer ever becomes a
  requirement, the design to build is the fourth option — buffer the final turn,
  guard it, then stream the corrected text — and it should be built knowing it
  gives up the earliest tokens, which is where most of the gain lives.
- **A failure after the stream opens arrives as an event, not a status code.**
  Before the first frame the response is still an ordinary request, so a caller
  gets a real 502/504 with the same body as before. After the first frame the
  status is spent, and the same failure arrives as a terminal `error` frame
  carrying the same code and correlation id. This is a real change to how a
  caller must handle failure on this route, and it is the price of streaming at
  all.

### 6.2 Chain identity and execution records (Gap 2)

Three separable pieces. First, make the artifact one thing: move the question
templates out of `demo/fixtures.py` and into the agent alongside the system
prompt (Django would send the chip name and admission id; the agent would own
the wording), and pin the model to a dated revision in code rather than an
environment default. Second, give it an identity: a chain revision constant,
bumped whenever prompt, templates, tool wiring or model change. Third, record
it: one structured log line per execution carrying the revision, the model,
the question length, the tool names called, the guardrail flags and the
timings. That satisfies the intent of the requirement without building an
observability platform, which belongs to the observability layer; the
alternative is to wait for that layer and choose the storage once. Recorded as
open, because the third piece decides where this data finally lives — but the
first two do not depend on it.

### 6.3 Conversation state (Gap 4)

Options: leave the product single-turn and record that finding — in which case
the `compare` chip should be removed, since it advertises a capability the
system does not have; or add a session identifier with history in Cloud SQL
(already in the architecture, so no new dependency) when follow-up questions
are wanted. Recorded as open, because it is a product decision with an
architectural consequence, and the demo does not need it yet.

### 6.4 Prompt-as-data (Gap 3)

Options: record the current state deliberately ("everything is prompt-as-code
because the prompt is fixed and reviewed"), or split the prompt into a
versioned template plus a runtime data section with its own validation. The
first is honest and costs nothing; the second matters only when examples or
retrieved context start changing the prompt's wording.

Decision: _pending_.

---

## 7. What changed

**Gap 1 closed 2026-09-15: progress streaming, option B.** The answer is still
delivered whole; what streams is which step of the chain is running.

*The agent.* `services/agent/stages.py` is new — the display labels and the
closed set of stage values (`stages.py` 51). A stage is emitted where the work
happens and nowhere else: `graph._emit` (`graph.py` 104) is called immediately
before each tool call (`graph.py` 152) and immediately before each model call
(`graph.py` 252). `ask()` takes an optional `on_event` callback (`graph.py` 279);
with no listener the graph behaves exactly as it did before, which is what keeps
`/ask` unchanged. `POST /ask/stream` (`http.py` 288) runs the same chain, relays
the stages, then emits one terminal frame carrying the same object `/ask`
returns — composed by the same `_compose_success` (`http.py` 117), so the two
routes cannot drift into serving different answers. A keepalive comment frame
goes out every 15 s (`http.py` 54) because a tool call may legitimately take 100 s
and a silent connection that long is closed by an idle timeout.

*Django.* `demo/agent_client.py` `ask_stream` (`agent_client.py` 199) reads the
agent's stream and guarantees a terminal frame, so no caller can be left waiting
for an answer that is not coming. `demo/views.py` relays the frames as
Server-Sent Events, claiming the quota before dispatch and refunding exactly as
the blocking path does (`views.py` 148). The first frame is pulled before the
response commits to being a stream (`views.py` 319), so a failure before any
frame is still an ordinary 502 with the same body the blocking path returns.

*The browser.* `static/js/demo_flow.js` reads the stream (`demo_flow.js` 615) and
puts the stage label where "working" was (`demo_flow.js` 644). The answer body
stays a dotted placeholder until the guarded answer arrives, so there is no path
by which progress text can be mistaken for the answer. The live label gets its
own class (`turn-meta-live`) and a larger size than the small print it shares a
slot with — 0.82rem against 0.7rem, still under the 0.88rem answer body — because
it is read at a glance by someone who is waiting rather than skimmed afterwards.
Changing the stylesheet and the module meant bumping their cache-bust versions
(`demo_splitpane.css?v=8`, `demo_a2ui.js?v=13`, and the `demo_flow.js` import at
`?v=19`), which a test asserts exactly so a stale asset cannot reach a browser.

**Two bugs the live test found, both invisible to the offline tests.**

1. **Django buffers a synchronous streaming iterator.** The first version relayed
   frames with a sync generator. Under ASGI, `StreamingHttpResponse.__aiter__`
   consumes a sync iterator with `sync_to_async(list)`, which materializes the
   whole stream and sends it when the response ends. Measured on the live path:
   every frame arrived at the same millisecond (9.963 s), which is
   indistinguishable from not streaming at all. Django warns about exactly this
   in the server log, naming the fix. The relay is now an async generator, which
   is consumed part by part; the same measurement then showed frames at 1.1 s,
   2.1 s, 3.3 s, 4.8 s, 6.7 s and 9.3 s.

2. **The relay waited out a keepalive after the chain had finished.** The first
   loop only noticed a finished chain between keepalives, so every answer was
   delayed by up to 15 s — a progress stream that made the common case slower
   than not streaming at all. A sentinel queued when the task completes
   (`http.py` 271) ends the loop the moment the work does. The offline suite went
   from 17.7 s to 2.7 s when this was fixed, which is the size of the stall.

A third problem was caught by the test suite rather than by the live run: the
async relay runs on the event loop, where Django refuses synchronous database
access, so the quota calls hop to a worker thread (`views.py` 133). Without that
the streamed path would have raised `SynchronousOnlyOperation`.

**Live verification.** Both tool endpoints were deployed for this
(`readmission-endpoint`, `readmission-rag-index`). In the browser at
`/demo/a2ui/`, a risk question showed *Reading the question* at 1.1 s, *Reading
the risk model* at 2.4 s, *Reviewing the evidence* at 3.3 s, *Searching the
discharge notes* at 4.3 s, *Reviewing the evidence* at 5.4 s, then the answer at
9.1 s — a real probability of 0.250531 above the 0.11 threshold, five attributed
factors, and a citation resolving to a real note section. No console errors.
`/ask` still returns its full eight-field contract live (HTTP 200).

**Latency now has a number, and it is not the model.** A cold streamed call took
about 22 s to reach its first stage; a warm one reached it in about 1.1 s. The
22 s is the per-request rebuild the audit flagged as unmeasured — a fresh graph,
a fresh MCP session and a fresh tool listing on every request, plus container and
MCP cold starts with `min-instances 0`. A warm blocking call measured 11.3 s end
to end. So the wait before anything is shown is dominated by per-request setup,
not by the model or the tools, which is the opposite of where the intuition
points and the first thing worth attacking.

**One setting outside this layer changed.** The load balancer's backend service
had no `timeout_sec`, so it took the 30 s default — shorter than the 110 s chain
deadline and the 120 s Django wait, and short enough to truncate a streamed
answer with no error at all. It is now 300 s, so the load balancer is the last
layer to give up rather than the first. That is an edge-layer setting
(`danielmherman/infra/edge/edge.tf`), recorded here because this layer is where
the symptom would have shown up.

**Tests.** Agent: 13 new in `tests/agent/test_progress_stream.py`, including a
regression test for the keepalive stall; 253 pass, with the same 10 pre-existing
errors from live-model tests that need cloud credentials this machine does not
have. Django: 10 new in `demo/tests.py`, including one that asserts the response
is an async iterator, because that is the only offline proof that Django will not
buffer it; 94 pass.

---

## 8. Interview questions this layer answers

**Why a graph at all, rather than a prompt and an API call?**
Because the model has to consult evidence before it can answer. The question
"why was this patient flagged?" needs the risk score, and "what do the notes
say about follow-up?" needs retrieval. Rather than guess the order, the graph
lets the model ask for what it needs, and the loop ends when it stops asking.
The cost of that freedom is bounded three ways: ten supersteps, five tool calls
per turn, and a wall-clock timeout derived from the caller's deadline.

**The requirement says the API must stream. Does yours?**
Partly, and I would rather say which part than answer yes. The API streams
*progress*: while the chain runs, the caller gets events naming the step that is
running, and the answer arrives as the final event. It does not stream the answer
as the model writes it, and that is deliberate — the guardrails run after the
model finishes and rewrite the text, removing a risk number the evidence does not
support, a medication that is not in the retrieved notes, an age the source
redacted. Streaming tokens would put a draft on the screen and correct it
afterwards, and for a clinical answer the corrected part is exactly the part that
matters. So the honest position is: streaming is met for progress, deferred for
prose, with the design for real answer streaming written down — buffer the final
turn, guard it, then stream the corrected text, accepting that the earliest
tokens are what you give up.

**What did streaming actually cost you?**
Two bugs only the live stack could show, and both are worth knowing because
neither is visible in a unit test. First: Django's ASGI handler consumes a
*synchronous* streaming iterator with `sync_to_async(list)`, which materializes
the whole stream and sends it at the end — every frame arrived at the same
millisecond, which looks exactly like no streaming at all. The response has to be
an async iterator. Second: the relay loop originally only noticed a finished
chain between keepalives, so every answer waited out a keepalive interval after
the work was done; a sentinel queued when the chain finishes fixed it, and the
offline suite got fifteen seconds faster, which is the size of the stall. The
general lesson is that "it streams" is a claim about the whole path — agent,
proxy, and the server that serves it — and any hop can silently buffer.

**Does streaming make it faster?**
No, and that is the interesting part. It does not reduce time-to-first-token for
the answer at all; it removes the blank wait. The measurement is why I would not
claim otherwise: a warm streamed call reaches its first stage in about a second,
and a cold one takes about twenty-two, because the per-request rebuild — a fresh
graph, a fresh MCP session and a fresh tool listing on every request — dominates
everything before the model even runs. So the next latency win is that setup,
not the streaming.

**What stops the model from inventing a risk number?**
Four things, and the important one is not the prompt. The prompt states the tool
contract; the guardrails check the answer against what the tool actually
returned, so a risk number that does not appear in the model output is removed
from the text and flagged rather than shipped; the tool wrapper declares real
parameter types so the model cannot call a tool with invented arguments; and
the answer is selected from the final message only, so an empty final turn is
reported as unavailable instead of falling back to an earlier preamble.

**Where does the prompt live, and how do you know what version produced a given answer?**
In two places, which is the honest answer. The system prompt is in
`services/agent/prompts.py` alongside the chain; the question templates the
chips use are in the website repository. The model is an alias read from the
environment, so it is not pinned. And nothing at runtime records which
combination served a request. None of that is a design decision — it is the
first thing I would fix in this layer, because the same data is what an
evaluation loop and an incident review both need.

**Is the agent stateless?**
Yes, and completely: a fresh MCP session and a fresh graph per request, and the
wire contract carries a single question with no history. The corollary is that
there is no conversation today. If follow-ups are wanted, the session has to go
somewhere external — Cloud SQL is already in the architecture — and the wire
contract needs a session identifier. Saying "stateless" without that sentence is
only half an answer.

**Does the user see what the model wrote?**
Not necessarily, and that is deliberate. The guardrails run on the model's
final text and remove anything the evidence does not support — a risk number
that is not the model's output, a medication or dose not in the retrieved
notes, an age the source redacted. The user sees the corrected text plus a
list of what was corrected. It is also the reason token streaming is not a
free win here: the first tokens on screen might be the ones that get removed.

**Why is the canvas composed in the agent instead of the browser?**
Because citation numbering and source resolution are statements about evidence.
The agent is where the tool results are visible and where the guardrails ran, so
it is the only layer that can number a citation and mean something by it. The
browser renders what it is given; it does not decide what a number refers to.

**Could you cut cost or latency by adding a second agent, or a router?**
Not defensibly. The task is one question with at most two tool calls, and the
measurement that would justify multi-agent — one agent failing at tool selection
— has not happened. Google's own guidance is to improve prompt and tools first,
and the interesting failure here was never tool selection: it was the model
answering confidently without consulting the tool, which is addressed by the
prompt contract and the guardrails.
