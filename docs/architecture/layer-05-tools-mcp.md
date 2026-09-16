# Layer 5 — Tools & grounding

Status: audited 2026-09-16, and independently reviewed the same day. Six gaps recorded,
none closed. Sections 5 and 6 are paired one to one: each gap has exactly one decision, in
the same order.

---

## 1. What the layer is

Every fact this system states about a patient comes through a tool. This layer is the
boundary those tools sit behind: the three functions that read the patient's notes and
score an admission, the protocol that carries the call, and the validation on both sides
of it.

It is not the chain — layer 3 decides when to call a tool and what to do with the result
— and it is not the data itself, which layer 7 owns. It is where the model is told what it
may ask for, and where the answers are checked before the model sees them.

```
  ┌─────────────────────────────────────────────┐
  │  LAYER 3 — the chain                        │   owns the decision to call a
  │  _execute_tool_calls, graph.py 113          │   tool, the per-turn budget, and
  └───────────────────┬─────────────────────────┘   what the answer may claim
                      │  IN:  a tool name and the arguments the model produced
                      v
  ┌─────────────────────────────────────────────┐
  │  LAYER 5 — the tool boundary                │   owns what a tool accepts, what
  │  the client, mcp_client.py 115              │   it may return, and the transport
  │  the server, services/mcp/server.py 26      │   between them: one server object,
  │  three tools, two transports                │   two ways to reach it
  └───────────────────┬─────────────────────────┘
                      │  IN:  arguments validated · the admission filter applied in the query
                      v
  ┌─────────────────────────────────────────────┐
  │  THE GROUND TRUTH — not operated here       │   owns the rows, the vectors and
  │  Vector Search index · BigQuery notes ·     │   the score; this layer treats
  │  prediction endpoint · feature source       │   everything it says as untrusted
  └───────────────────┬─────────────────────────┘
                      │  BACK:  chunk ids and distances · note text · a score · a failure
                      v
                      OUT: one validated payload, or a structured error — never an exception
```

What crosses each boundary, and what does not:

| Boundary | Crosses | Does not cross | Who decides |
|---|---|---|---|
| Chain → tool call | A tool name the server advertises, with arguments matching the declared input schema | A tool the server does not advertise; an argument of the wrong type; a call beyond the per-turn budget | The server's tool list decides what can be asked; layer 3's budget decides how often |
| Tool → chain | One JSON payload with a declared shape, or `{"error": code, "message": …}` | Exceptions, stack traces, and anything the contract rejects | The tool function builds it; `contracts.py` decides whether it may leave |
| Ground truth → tool | Chunk ids, distances, note text, a score | Another patient's note: every resolved row is re-checked against the requested admission, and a mismatch is refused rather than served | The index filter decides what is retrieved; `_fetch_texts` decides what is believed |
| Transport | The same server object over stdio locally and authenticated streamable HTTP on Cloud Run | Session state — the deployed transport is stateless, so a follow-up request may land on any instance | The environment selects the transport; the graph never knows which it got |

Terms used in this document:

| Term | Meaning |
|---|---|
| MCP | Model Context Protocol — the wire protocol between an agent and a tool server. Tools declare a name, a description and an input schema, and the client advertises them to the model as callable functions. |
| Tool | One callable capability with a typed signature. Three here: `predict_readmission`, `rag_search`, `rag_search_sections`. |
| Tool contract | The declared shape of what a tool may return, enforced in code before the payload leaves the function — distinct from the schema the protocol advertises. |
| Structured error | A failure returned as data — `{"error": code, "message": …}` — so the model can report it. An exception instead would collapse the graph and lose the reason. |
| Filter | A server-side restriction passed into the vector query, so retrieval is scoped to one admission before ranking rather than discarded after it. |
| Passage | One citable unit of note text, carrying its chunk id, its section and its text. A citation points at a passage. |
| Grounding | Making an answer traceable to a source. Here: a clinical claim cites a passage, and the passages come from the patient's own notes. |
| Focused toolset | Few tools with one job each, rather than one tool with a mode parameter that changes what it does. |

**Layer 5 is: the only way the model reaches a fact it did not already have.**

---

## 2. What Google requires

| # | Requirement | Level |
|---|---|---|
| A4 | Tools exposed through MCP (or custom function tools when no MCP server exists), each with typed input and output schemas. | MUST |
| A5 | Tool definitions kept small: primitive types, < 5 parameters, enums instead of free text. Unbundle monolithic tool servers into focused toolsets. | MUST |
| G1 | Treat all inputs as untrusted — user, retrieved documents, tool results. Filter and validate external content before it enters a prompt (indirect injection). | MUST |
| B3 | Serving uses the same embedding model and parameters as ingestion. | MUST |
| F2 | Structured logging in the agent: which tools were called, inputs/outputs, latency per step. Distributed tracing (Cloud Trace) across frontend → agent → tools → model. | MUST |
| G4 | Least-privilege service accounts per service; service-to-service auth with workload identity, not shared keys. | MUST |

---

## 3. How this application implements it

One server object serves both transports (`server.py` 26–40 defines it, 92–105 chooses),
so the local path and the deployed path cannot drift apart. Three tools are registered on
it (`server.py` 38–40). The server also carries an `instructions` string describing which
tool suits which question; it is not what reaches the model — the client reads a tool's
name, description and input schema only — so the rules that steer tool choice are the ones
in the prompt (`prompts.py` 39–50).

The client (`services/agent/mcp_client.py`) turns each advertised tool into a LangChain
tool for the model to call, with the tool's `input_schema` cleaned of the keys Gemini
rejects (`_clean_schema`, 43–49; `graph.py` 212–229 does the wiring). That cleaning matters
because pydantic emits `title`, `additionalProperties` and `default`, and an unknown key is
a 400 at generate time rather than at declaration time. What comes back is normalised from
whatever the protocol returned (`_payload`, 66), and a transport failure becomes a
structured error (`call`, 115) so the model can report it rather than crash.

Input validation has one definition applied by every tool entry point — `valid_hadm_id`
(`tools/_validation.py` 11), which rejects booleans because `bool` is an `int` subclass —
plus the ranges each tool owns, such as `top_k` between 1 and 20 (`retrieval.py` 203–204).
Above that sits the SDK's own coercion of arguments to the declared types, which happens
before the tool function is entered.

Output validation runs before any payload leaves a tool. `validate_prediction_result` and
`validate_retrieval_result` (`services/mcp/contracts.py` 76 and 110) check fields, types and
internal consistency — a `returned` count that disagrees with the passages is an error —
against the `TypedDict`s that name each shape (`services/mcp/contracts.py` 7–44). A payload
that fails is replaced with `invalid_tool_response` and logged server-side, never returned as
a malformed success.

Retrieval stays inside one patient by two independent means. The admission id is passed
into the index query as a filter (`retrieval.py` 230), so it constrains ranking rather
than being applied afterwards; and every note the index returns is re-checked at the
BigQuery layer (`_fetch_texts`, 119), where a row belonging to another admission raises
`IsolationViolation` (145) and the call returns a structured error instead of serving the
passage. An id the server cannot parse, and a note that is missing text, are both errors
rather than silently dropped passages — a dropped passage looks like a retrieval gap and
is hard to debug.

Section retrieval does not depend on the index at all. `_search_sections` (418) re-parses
the note and re-chunks it with the same deterministic chunker that built the index, returns
one passage per section in a fixed order, and marks those passages `retrieval:
deterministic` rather than giving them an embedding score they do not have. The section
vocabulary is single-sourced from that chunker (`retrieval.py` 45 and 67), so a build and a
serving path cannot disagree about which sections exist.

Tool results reach the prompt wrapped in a literal delimiter (`graph.py` 150), and the
prompt states that everything inside it is data about the patient and never an instruction
to the model (`prompts.py` 30–36). Two tests keep the wrapper and the prompt in step — that
they name the same literal, and that the tools named in the prompt are exactly the tools the
server registers (`test_prompt_contract.py` 68, 79).

The deployed path is Cloud Run. `services/mcp/Dockerfile` and
`services/mcp/cloudbuild.yaml` build and deploy `mcp-server`; the agent resolves `MCP_URL`
from the live service at deploy time rather than holding a copy (`services/agent/cloudbuild.yaml`
2–4, 46); every request except `/health` must carry an `Authorization` header (`server.py`
63–77), with the token itself verified by Cloud Run IAM and minted for the service's
audience and cached for 45 minutes (`mcp_client.py` 192). One tool
call is bounded by the tool leg of the timeout chain — 100 seconds, inside the question's
110 and above the model's 60 — with the HTTP client's own timeout of 110 seconds set just
above the per-call read timeout, so the tool deadline fires first and fails with a
structured error (`mcp_client.py` 220). The image copies the package wholesale, which also
ships `pipelines/` and the `__pycache__` directories of deleted modules; that is image
hygiene rather than a requirement, and is noted here so it is not rediscovered as a surprise.

---

## 4. Current state against the requirement

| Requirement | State | Why |
|---|---|---|
| A4 — typed schemas both ways | **Partly** | The input schemas are derived from the signatures and keep their types all the way to the model. The output contract is enforced inside the tool and used nowhere else — not declared to any client, not checked by the consumer (gap 1). |
| A5 — small definitions, enums, focused toolsets | **Partly** | One parameter, three and one, all primitive and all below the limit, on one focused server with one job per tool; the two retrieval tools are separate rather than merged behind a mode flag. No parameter is an enum, and the one bounded parameter's range is in prose and in code but not in the schema (gap 2). |
| G1 — tool results treated as untrusted | **Partly** | Provenance is enforced twice — the admission constrains the vector query, and every resolved row is re-checked — and a mismatch refuses to serve the text. The content is not validated: it reaches the prompt unbounded and unscreened (gap 3). The failure paths return internal detail, including another patient's identifiers on the isolation path (gap 4). |
| B3 — same embedding model and parameters both sides | **Partly** | Both sides use `gemini-embedding-001` at 768 dimensions with the asymmetric task types today. The same facts are defined in four places, one of them an environment read on the serving path (gap 5). |
| F2 — which tools, inputs and outputs, latency, distributed tracing | **Partly** | Tool names and a per-step latency are on every execution record, and the stream emits tool and result events as they happen. Inputs and payloads are absent by decision — note text is patient data, the same reason layer 4's record carries no question or answer text (F6). The tool's own error code does not reach the record either, so a refused call and a failed one look alike there. No trace id crosses to the MCP server; that clause belongs to layer 10's plane. |
| G4 — least privilege per service, service-to-service auth | **Partly** | The agent authenticates to the MCP service with a per-audience identity token, and the server refuses a request that carries no `Authorization` header — while the token itself is verified by Cloud Run IAM rather than in-process. Whether the accounts are least-privilege, and who may invoke the service, cannot be evidenced from the repository: no invoker binding or role for `mcp-server` appears in it. That half is deferred to layer 11 rather than claimed here. |

---

## 5. Gaps

Each entry states the defect and the evidence for it. The decision that closes it is the
entry of the same number in section 6.

**1 — The tool contract is enforced inside the tool and known nowhere else.**
All three tools advertise `{"type": "object", "additionalProperties": true}` — an object
with no declared properties, dumped from the running server. The validators behind it are
real, tested and enforced (`services/mcp/contracts.py` 76 and 110, called by every tool
before it returns), but that is the only place the shape exists. Nothing downstream checks
it: the agent's own guard requires only that the response is a dict
(`services/agent/contracts.py` 215–225). The cause is the return annotation — the SDK
derives an output schema from the return type, and every tool is annotated
`-> dict[str, Any]` (`prediction.py` 100, `retrieval.py` 332 and 465). The contracts are
also incomplete for what the tools emit: `granularity` (`retrieval.py` 325) and `note`
(`retrieval.py` 435 and 460) are returned and declared nowhere. Measured over stdio against
the installed SDK (mcp 2.0.0): annotating a tool with a union of two `TypedDict`s advertises
a schema whose only property is `result`, the structured payload then arrives wrapped as
`{"result": …}` — which is what `_payload` returns (`mcp_client.py` 66) — and a key the
schema does not declare is dropped from it.

**2 — The one bounded parameter does not declare its bound.**
`top_k` is advertised as `{"type": "integer", "default": 5}` and nothing more. Its range
lives in prose ("1-20, default 5", `retrieval.py` 342) and in a check that returns
`bad_request` when it is exceeded (`retrieval.py` 203–204), so the constraint rejects a
call rather than preventing it. No parameter in any tool is an enum — the only free text is
`query`, a search phrase that cannot be enumerated — so that half of A5 is unexercised
rather than violated.

**3 — Tool results enter the prompt without content validation.**
Nothing bounds the size of what arrives: a `granularity: note` fallback passage is a whole
discharge note (`retrieval.py` 325) and up to `top_k` of them arrive at once, so the text
entering the prompt has no ceiling. Nothing handles the characters either. `json.dumps`
escapes quotes and newlines but not `<` or `>`, measured, so a passage whose text contains
the closing tag produces a second closing tag in what the model receives (`graph.py`
147–155) — and the prompt's rule covers "EVERYTHING inside those tags" (`prompts.py`
30–36), which leaves text outside the wrapper outside the rule. No test feeds one through:
the tests that exist check that the prompt and the wrapper name the same literal
(`test_prompt_contract.py` 68). Nothing screens the text at all, which is the part of G1
that lands here — what may enter a prompt from the outside — as against the injection
*policy* that layer 11 owns.

**4 — Error messages carry internals into the prompt and back to the caller.**
`call` returns `f"{type(exc).__name__}: {exc}"` when the transport fails
(`mcp_client.py` 129). Measured over MCP: a wrongly typed argument comes back as
`tool_call_failed` carrying pydantic's message and a documentation URL, so a bad argument
is also indistinguishable from a broken connection. The isolation refusal returns the
exception text, which names the foreign note ids (`retrieval.py` 283, message built at
145–148) — and a note id is `{subject_id}-DS-{note}`, so that is another patient's
identifier, asserted by a test (`tests/agent/test_rag_search.py` 147). `missing_text`
embeds the table name (`retrieval.py` 305–306), `unparsed_datapoint` echoes the raw
datapoint id (294–298), and `incomplete_features` lists internal column names
(`prediction.py` 65–67). Every one of those *is* the tool result, so each enters the
prompt, and the success payload carries each tool call's response to the caller verbatim
(`http.py` 354). The record cannot compensate: it keeps the stage, the tool and the timing
and drops the payload (`http.py` 226–232), so a refused call and a failed one are the same
row there.

**5 — The embedding space is defined in four places, and the serving copy is settable.**
`retrieval/embed.py` 17–20 holds literals, used by the ingest component
(`pipelines/components/embed_chunks.py` 185–189) and the build script. `services/mcp/config.py` 99–100
holds environment reads, used by the serving path (`retrieval.py` 209 and 212).
`rag_config.yaml` 36–39 holds a third copy, in a file that describes itself as the place
those settings live "so the corpus switch is ONE value". `rag_ingest_pipeline.py` 55
carries a fourth as a default. `RESTRICT_NAMESPACE` is defined twice as well
(`embed.py` 21, `services/mcp/config.py` 101): a drift in that one makes the serving filter match
nothing and retrieval return zero without an error. The failure the requirement is about is
not an exception — it is plausible neighbours from a different space. The contrast is in
the same file: the section vocabulary *is* single-sourced (`retrieval.py` 45), with a
comment saying build and serving "can never drift".

**6 — An unknown admission is indistinguishable from an admission with no notes.**
`predict_readmission` returns `unknown_patient` when the admission is not in the feature
source (`prediction.py` 44–48). Both retrieval tools return success with `returned: 0` and
a note saying nothing was found (`retrieval.py` 435 and 460) — including when the admission
id does not exist at all. So the model reports "no notes found" for a patient that does not
exist, and a typo in an identifier reads as an empty record. This is not a requirement row;
it was found auditing one tool's contract against another's.

**Recorded, not owned here:** the index, the note tables and the pipeline that builds them
belong to layer 7; the distributed-tracing clause of F2 belongs to layer 10's plane, which
is also where the record's shape is owned; and injection and jailbreak as a policy is layer
11's, which is why gap 3 is about what may enter a prompt rather than about a classifier.

---

## 6. Design decisions

### 6.1 — for gap 1: complete the contract, declare it, and check it where it is used

Three steps, in this order, because probing it showed it is not the one-line change it first
appears to be. First the contracts gain the keys the tools already emit (`note`,
`granularity`), because a declared schema that omits them drops them from the structured
payload — measured. Then each tool's return is annotated with its contract union, which is
what makes the SDK advertise a real schema. Then, because that annotation makes the SDK wrap
the payload as `{"result": …}`, the client unwraps it in `_payload` and validates what it
received against the same contract — the consumer is where an untrusted result has to be
checked, and today it is where nothing is checked. The three land together: the second
breaks the first if it ships alone. Two tests: every advertised tool declares properties,
and a payload the contract rejects does not reach the model.

### 6.2 — for gap 2: put the bound in the signature

Declare `top_k`'s range in the signature so it reaches the advertised schema, and keep the
existing check as the enforcement for a caller that ignores the schema. The bound does not
change; what changes is that the model can see it instead of discovering it by failing.
A test asserts the advertised schema carries it.

### 6.3 — for gap 3: bound and validate what may enter the prompt

Three things at one boundary, because they are one decision: a ceiling on the text one
passage may contribute, handling for the delimiter — escaped rather than stripped, so a
citation and the passage it cites stay in step — and a check for the characters that should
never appear in note text. The prompt rule stays as it is; what changes is that it stops
being the only thing between a note and the instructions. Screening for *injection* remains
layer 11's; this is the gate that decides what reaches a prompt at all. Tests: a passage
carrying the closing tag yields exactly one, and an oversized passage is refused rather
than silently truncated.

### 6.4 — for gap 4: a code and a sentence out, the detail to the log

Every failure path returns a stable code with a message a caller can act on, and the detail
goes to the log — which is what the routes do and what these paths do not. The isolation
refusal is the one that matters most: it may say that a note did not belong to the requested
admission, and must not say which note or whose. Two existing tests pin the current shape
and have to change with it (`tests/agent/test_rag_search.py` 147,
`tests/agent/test_mcp_contract_boundary.py` 79); a test that pins a leak is not a reason to
keep one. The record should also carry the tool's error code, so a refused call can be
counted rather than only seen.

### 6.5 — for gap 5: one definition, and the serving path stops reading the environment

`retrieval/embed.py` already holds the parameters ingestion uses; it becomes the only
definition, and the serving path, the pipeline configuration and the pipeline default all
resolve from it. `EMBEDDING_MODEL` and `EMBEDDING_DIM` stop being deploy-time inputs, on the
same argument that pinned the model and the region in layer 4: a value that can change
silently and produce plausible wrong answers is not a setting. `RESTRICT_NAMESPACE` is
included, because a drift in that one is silent zero recall rather than a wrong neighbour.
A test asserts every consumer resolves the same object, so drift means editing one place.

### 6.6 — for gap 6: one meaning for "this admission has nothing"

The retrieval tools answer an unknown admission the way the prediction tool already does —
an `unknown_patient` error rather than an empty success — so that an empty result means the
record is empty and nothing else. A test each: an admission that does not exist, and an
admission that exists with no discharge note.

---

## 7. What changed

Nothing yet. This layer is audited and its gaps are recorded; this section is filled in as
they are closed, one at a time.

---

## 8. Interview questions this layer answers

**How does the model reach anything it did not already know?**
Three tools over MCP, all on one server: one scores an admission and returns the
attributions behind the score, one retrieves passages for an open-ended question, one
returns a passage per discharge-note section for a summary. The model cannot invent a
patient fact because it has no other source — the tools are the only path to the notes,
the features and the model score.

**How do you stop one patient's notes reaching another patient's answer?**
Two independent mechanisms, and neither trusts the other. The admission id goes into the
vector query as a filter, so it constrains ranking rather than being applied to results
afterwards; then every note the index hands back is re-checked against the requested
admission at the database layer, and a mismatch raises rather than being dropped. That
second check exists because an upstream defect — a missing filter, a stale index, a
truncated id — would otherwise flow another patient's text straight into a citation.

**What stops a retrieved note from giving the model instructions?**
An instruction, today: the prompt says everything inside the result wrapper is data and
never a directive, and that imperative text in a note is clinical content to report rather
than obey. That is the weaker half of the defence, and gap 3 is the reason: the wrapper's
closing tag can be written by the note itself, so text can land outside the region the rule
covers. What the layer does have is structural enforcement of *provenance* — where the text
came from and who it belongs to — which is a different problem from what the text says.

**What happens if the index returns a note that belongs to another patient?**
The call refuses to serve the text. The admission id constrained the vector query, so an id
from another admission means an upstream defect — a missing filter, a stale index, a
truncated id — and the second check catches it, logs it, and returns an error rather than a
passage. What it does *not* do is keep quiet about which notes were wrong: the message
carries the foreign note ids, and an id embeds the other patient's subject id, so the
refusal leaks an identifier while correctly withholding the text. That is gap 4, and it is
the sharpest instance of it.

**A tool fails mid-question. What does the model see?**
A structured error rather than an exception, because the model is instructed to report a
failure instead of inventing a number, and a raised exception would collapse the graph and
lose the reason. Every failure path returns `{"error": code, "message": …}`: an unknown
tool, a bad argument, a missing note, an isolation violation. What it should not see is the
exception text itself, which is gap 4.

**Why declare an output schema when the output is validated anyway?**
Because they answer different questions. The validator answers "may this payload leave the
server", and it runs at the moment the answer is produced. The schema answers "what shape
is this", and it is read before the call, by anything that is not this process — another
agent, a test, or a reviewer. Today the second answer is an empty object, which means a
client can see the input contract but not the output one.

**Why three tools rather than one that takes a mode?**
Because a mode parameter is a hidden branch: the model has to be told in prose when to use
which value, nothing stops a value that does not exist, and the schema describes all of
them at once. Three tools with one job each put that choice in the names and descriptions,
which is what the model actually selects on — and it is what makes the summarization rule
expressible as a tool choice rather than as a caveat.
