# Layer 8 — Memory and session state

Status: audited 2026-09-18, written in the four-part form used from layer 7 onward, and
rewritten the same day after an independent review found that its central factual claim was
wrong: the layer is not empty. The browser keeps a thread, one episode per patient, in page
memory, and it draws continuity that the answers do not have. Five gaps are recorded and none
is closed. They are product decisions rather than defects, and the owner has stated the
intention to build a multi-turn experience in which a physician asks free-form follow-up
questions; the interaction model and the memory policy are therefore prerequisites rather than
options. The remediation recorded for the entries that depend on it is a strategy session,
whose agenda is `layer-08-memory-and-ux-strategy.md`, beside this document.

---

## 1. Overview

### 1.1 The role of the layer

The memory and session state layer is what makes a conversation a conversation rather than a
sequence of unrelated questions. It holds the transcript of what has been asked and answered,
the identity that lets a later request be related to an earlier one, and, at a larger scope,
what the system remembers about a user or a patient across sessions.

The layer exists because the model is not the memory. A language model receives a context
window and returns a completion; it holds nothing between calls. Every property that is
usually attributed to conversational memory — that the assistant remembers the earlier
question, that it does not repeat itself, that "and what about the potassium" resolves to a
sensible antecedent — is produced by the calling system placing the earlier material into the
next context window. When that material is absent, the model does not notice: it answers the
new question as though it were the first, which is a correct answer to a poorly framed
question and is therefore worse than an error.

The layer therefore carries two separable concerns that are easy to conflate. The first is
state: where the transcript lives between requests, who may read it, how long it is kept, and
what happens to it when the session ends or the account is deleted. The second is the
interaction model: what a user is entitled to expect from a second turn, and how a system
tells the difference between a follow-up question about the patient under discussion and a
fresh question about a different one.

What makes this layer's state of affairs unusual is that the two have come apart. A
conversation is displayed: the console page keeps one episode per patient in the browser, adds
each question and answer to it, and renders the result as a thread. The conversation is not
transmitted: every request carries a single question and an admission, so the agent answers
each turn as though it were the first. Two turns are therefore related to one another on the
screen and nowhere else, and the gap between the display and the answer is the subject of this
layer.

In a clinical setting that gap carries a safety weight it would not carry in a general
assistant. A thread implies continuity, and a reader who sees three turns stacked in one
episode reasonably expects the third to have been answered with knowledge of the first two. If
the interface offers that impression while the answer is composed in isolation, the interface
is making a claim the system does not support. The same risk appears from the other direction
once continuity is added: a follow-up answered about a patient other than the one under
discussion produces a plausible answer about the wrong person. The design of this layer is
therefore as much about constraining continuity as about providing it.

```
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  ONE TURN                                                                │
  │                                                                          │
  │  client                    keeps one episode per patient in page memory: │
  │                            the turns, their sources and the last query;  │
  │                            writes nothing to storage, and sends none of  │
  │                            it                                            │
  │  site                      decides what to ask and sends the intent plus │
  │                            the admission; keeps an authentication session│
  │                            and a daily counter of requests               │
  │  agent                     one question to completion, with the model    │
  │                            calls and tool calls of that turn bounded     │
  │  tools                     stateless between calls, and every call       │
  │                            carries the admission it is about             │
  └──────────────────────────────────────────────────────────────────────────┘
                      │  the answer leaves with its citations and the passages behind them
                      v
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  WHAT SURVIVES THE REQUEST                                               │
  │                                                                          │
  │  the thread                in the browser's memory, per patient; a reload│
  │                            ends it, and the next request does not carry  │
  │                            it                                            │
  │  authentication session    which user this is; the cookie dies with the  │
  │                            browser, the server-side row outlives it      │
  │  daily counter             one credit per request, refunded on a failure │
  │                            that was not the user's fault; live path only │
  │  the conversation          in the browser only, never on the wire        │
  └──────────────────────────────────────────────────────────────────────────┘
                      │  and what does not exist
                      v
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  WHAT DOES NOT EXIST                                                     │
  │                                                                          │
  │  session identity          the wire carries no turn-to-turn relation     │
  │  conversation store        no server-side record of a question or answer │
  │  retention rule            nothing states what would be kept, for how    │
  │                            long, or who could read it                    │
  │  follow-up signal          the composer exists; the request cannot say   │
  │                            that a turn continues an earlier one          │
  └──────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Components

**The thread in the browser.** The console page keeps a map of episodes keyed by admission,
and each episode holds the turns for that patient, the assessments drawn from each answer, the
sources behind them and the query that produced them
(`danielmherman/static/js/demo_flow.js` 150–168). Sending a question pushes the user's turn and
a pending agent turn into the current episode before the request is made, and the reply
replaces the pending turn, so the page shows a thread that grows turn by turn
(`danielmherman/static/js/demo_flow.js` 675–736). One request is in flight at a time by
design, so the thread is ordered by construction (`danielmherman/static/js/demo_flow.js`
676–678). This is conversation state, and it is real: it is what the interface renders, it is
what a physician sees when they scroll, and it lives only for the life of the document. The
page writes nothing to browser storage and reads nothing from it but the CSRF cookie, so a
reload ends the rendered thread (`danielmherman/static/js/demo_flow.js` 150–168). What the page
does carry is the name of the conversation its thread belongs to: an answered turn returns an
identifier, the episode keeps it, and the next question sends it back so that it is answered as
a follow-up rather than as a first question (`danielmherman/static/js/demo_flow.js` 685–686,
797). The rendered thread is still the page's own — it is never re-fetched — and the identifier
is what makes the site's copy of the conversation the one the answer is built from. The
interface states the boundary while a conversation is open, in one line under the thread
header: that the thread lives in this page and that reloading starts a new conversation
(`danielmherman/demo/templates/demo/a2ui_console.html` 151–154,
`danielmherman/static/js/demo_flow.js` 336–348, `danielmherman/static/css/demo_splitpane.css`
226–231).Decided 2026-09-18, in the strategy session's record: **a session is a page load**,
and the interface says so rather than letting a reload be discovered from a follow-up that was
answered as a first question.

**The composer.** The console page offers a follow-up field: a pinned input with its send
button sits below the thread, and the thread renders in chapters with an "earlier messages"
control (`danielmherman/demo/templates/demo/a2ui_console.html` 132–145). Asking a second
question of the same patient works, and as of 2026-09-18 the request says that it *is* a second
question: the site names the conversation the follow-up belongs to, and the site — not the
browser — supplies the earlier turns the agent is asked to consider
(`danielmherman/demo/agent_client.py` 111–115, `danielmherman/demo/views.py` 158–173). The
interaction model this layer had to settle was about meaning rather than about affordance, and
section 4 records how it was settled: the field was already there, and the question was what
the answer does with the turn above it.

**The agent's in-invocation state.** The chain keeps a transcript while it is answering, and
this is the only transcript the agent has. It is a list of messages annotated so that each
graph node appends to it, together with the record of the tool calls made during the turn
(`services/agent/graph.py` 64–69). It is created when a question arrives and discarded when the
question is answered, because the entry point runs one question to completion and returns the
final state to its caller (`services/agent/graph.py` 347–384).

**In-process state that survives between requests.** The services are written to be stateless,
and they hold several caches with lifetimes inside one instance. The tool client caches an
identity token for forty-five minutes (`services/agent/mcp_client.py` 241, 249), and the
handles to the prediction endpoint, the feature manifest, the retrieval clients and the concept
tagger are memoised for the life of the process (`services/mcp/dependencies/model_endpoint.py`
5–16, `services/mcp/dependencies/features/manifest.py` 26–66, `services/mcp/tools/retrieval.py`
91–114, `services/mcp/tools/prediction.py` 37). The tool client also opens a session per
request and closes it with the request, while the server keeps none between them
(`services/agent/http.py` 272–275, `services/mcp/server.py` 97–102). None of this holds a
conversation, and all of it is state with a lifetime that a multi-turn design will interact
with.

**The tool boundary.** The tool server is stateless by default and says so: a streamable
session kept in process memory would be lost when the next request landed on a different
instance, and the tools hold no per-session state, so there is nothing to lose
(`services/mcp/server.py` 97–102). A flag exists to keep sessions in memory for a
single-instance deployment, and it is not the deployed configuration
(`services/agent/Dockerfile` 34). Every tool call carries the admission it is about, so the
tool layer has no need of the notion of a session.

**The request contract, and the second boundary behind it.** What the agent accepts is a closed
set of fields: a typed question or a named starter chip, an admission identifier, the earlier
turns of a conversation, and nothing else (`services/agent/contracts.py` 8–17, 78–95). The
closure is deliberate, and the reasoning is recorded in the file: an earlier version ignored
unknown fields, so a caller that sent a history received a confident answer that had silently
dropped it, and refusing is what keeps single-turn a decision rather than an accident (layer 3,
gap 4). That protection now covers the site's proxy in front of it as well, which refuses an
unknown field by name and accepts four of them, one of which is the conversation identifier its
own store keys on (`danielmherman/demo/views.py` 33–71). The two closed sets differ from each
other deliberately rather than by omission: the site's boundary carries the conversation
identifier, and the agent's carries the turns a caller wants considered, because the agent must
hold nothing and the caller must hold everything (A1, A3). A browser that posted a history is
refused at the boundary it can reach, and a caller that names a conversation still cannot push
its contents.

**The only per-user counter.** Each account has a daily allowance of agent calls, held in the
site's database, and it is claimed before the work is done and refunded when a failure was not
the user's fault (`danielmherman/demo/models.py` 55–92, 94–125). It is the only counter a
request leaves behind. It is not the only durable per-user record: an authentication session
row and, when a login fails repeatedly, a lockout record are written by the site as well
(`danielmherman/danielmherman/settings.py` 137–152). The counter is also charged on the live
path only: in fixture mode the answer is composed locally and the allowance is reported rather
than spent (`danielmherman/demo/views.py` 262–270).

**Concurrency, which is per instance.** The site allows a fixed number of agent calls to be in
flight at once, enforced by a semaphore in the process, with a timeout on each call
(`danielmherman/demo/agent_client.py` 45, `danielmherman/danielmherman/settings.py` 101–112).
That bound is per instance and is what a multi-turn interface, with several conversations open
at once, will exhaust before it exhausts anything else.

**What the answer carries.** Each resolved source is a citation number, a section, the passage
text and the query that retrieved it (`services/agent/contracts.py` 25–29). The passage text
therefore travels in the answer payload, which matters for this layer: a store of "the
citations that were shown" holds note text unless the stored shape deliberately drops that
field. Each tool call in the same payload carries its name, its arguments, the result, and
whether the result can be obtained again (`services/agent/contracts.py` 12–17, 32–47): the
first three are what a faithful replay needs, and the fourth is what lets a storing caller keep
the result only when it cannot be produced again.

**The absent components.** There is still no cross-session memory, and that is a decision
rather than an omission: the requirement's own condition is that the product needs
personalisation across sessions, and the product has not asked for it (A7, section 2). What no
longer belongs in this list is the rest of it. A conversation store exists and is where a
conversation lives (`danielmherman/demo/models.py` 196–316), it has a retention rule and a job
that enforces it (`danielmherman/demo/management/commands/purge_expired_conversations.py`
25–66), an identity for a conversation travels on the wire
(`danielmherman/demo/views.py` 158–173), and the interface sends the earlier turns of a
conversation back to the agent that answers from them
(`danielmherman/demo/conversations.py` 133–162). Section 3 records each of those and section 4
records the state of the entries that describe them.

**Three holdings that already exist without a retention rule.** The evaluation harness archives
runs as a JSONL trace file it calls the durable archive, and a record carries the question, the
answer and the tool calls, including passage text (`evaluation/agent/collect.py` 1–8). The
retrieval layer caches notes outside the repository, in a file with its own manifest
(`services/mcp/retrieval/notes.py` 1–34). The demonstration fixtures committed to both
repositories contain captured answers with whole passages. None of the three has a stated
lifetime, a stated reader or a deletion rule, and each holds clinical text. They are the
precedent any conversation policy has to reconcile with, and the reason the policy question is
not hypothetical.

### 1.3 Integration with the stack

Upward, the layer is used by the client, which already keeps the thread, and by the
orchestrator, which assembles the context window. Downward, it would be written to a store of
its own: memory is not a view of the data layer, and the distinction matters here, because a
conversation contains text a user typed together with clinical passages the system retrieved,
and that mixture has its own retention question.

The boundaries with the adjacent layers are stated once here rather than tabulated. The
orchestrator owns what is placed into the context window and in what order, and it has handed
four questions to this layer — whether the prompt template and the replayed material must be
separated, how a replayed passage escapes the tool-result boundary that currently marks
retrieved text as data, whether history is replayed as structure or as text, and how the code
revision that produced an answer travels with the stored turn (layer 3, section 6). Until this
layer answers them, the orchestrator cannot add continuity without putting retrieved note text
into the prompt outside the boundary it established. The client layer owns what the interface
offers, which already includes a composer and a thread; the interaction model that makes a
second turn meaningful is this layer's. The data layer owns the clinical corpus, written
offline and read-only during a request; a conversation store is a different kind of store,
written on the request path, and it must not become a second place where clinical text
accumulates without a retention rule. The observability plane owns what is logged about a
request and for how long, which overlaps at one point: a transcript kept for the product and a
transcript kept for debugging are two holdings, and the policy has to name which is which.

**Terms used throughout this document.**

| Term | Definition |
|---|---|
| Turn | One question and the answer produced for it, including the tool calls and citations that produced the answer. One request is one turn. |
| Episode | The browser's per-patient container for the turns of one conversation, held in page memory for the life of the document. |
| Thread | What the page renders from the episode: the turns in order, with the sources behind each answer. |
| Transcript | The ordered list of turns in a conversation as the model would receive it. In this system it exists only inside one invocation. |
| Session | An identity that relates two or more turns to one another, with a lifetime. No such identity exists on the wire. |
| Session state | The transcript and any other material held between turns, wherever it is held. Today that is the browser's episode. |
| Cross-session memory | Material that outlives a session and is reused in a later one, such as a standing preference. Google marks this optional. |
| Retention policy | The rule stating what is kept, for how long, who may read it, and what happens to it when the account or the demonstration ends. |

---

## 2. What Google requires

| # | Requirement | Level |
|---|---|---|
| A1 | The front end never calls the model directly and reaches the agent over an API. A production front end needs streaming support, a stateless API, and externalised conversation state. | MUST |
| A3 | The agent is stateless and session state lives in an external store, so that any instance can serve any request and a restart does not lose context. | MUST |
| A7 | Long-term, cross-session memory lives in an external store, and exists only if the product needs personalisation across sessions. | SHOULD |

A3 is met in its statelessness clause and unmet in its store clause, and the distinction is
worth stating precisely because the two halves point in opposite directions here. The agent
holds nothing between requests, which is what the first clause asks for and is verified: the
services are deployed without session affinity, the tool server is stateless by default, and no
transcript is returned to a caller. The second clause asks that session state live in an
external store; session state exists, in the browser, and a store in the browser is the
opposite of external. The requirement is therefore partly met, and the gap table records the
part that is not.

A1 is also partly met. The interface never calls the model, it reaches the agent over an API,
and streaming is implemented: one route returns the completed answer, and a second streams the
stages of a chain as it runs and terminates with the same payload
(`services/agent/http.py` 307, 410, 439–480). The clause about externalised conversation state
is unmet for the same reason as A3's.

A7 is not applicable rather than unmet: its own condition is that the product needs
personalisation across sessions, and it does not. The decision is recorded, and Google's level
is SHOULD, so nothing in the requirement set is left unresolved by declining it.

Three further requirements in the same group belong to other layers and are not audited here.
A2, which requires an agent framework and an agent runtime, is audited in the orchestrator
layer. A4 and A5, which require typed tools with few parameters, are audited in the tools
layer. A6, which requires the model to be served from a managed runtime, is audited in the
model runtime layer.

One requirement outside this group governs the retention question that gap 3 records, and it is
named here so that the policy work is not written without it: F6, which requires that no
personal or confidential data reach logs, and G6, which requires data minimisation. Both are
audited in other layers; the decision about what a conversation may keep is where they apply to
this one.

---

## 3. Implementation in this repository

The console page holds the conversation and renders it. One episode per patient is kept in page
memory, holding the turns, the assessments derived from each answer, the sources behind them
and the last query issued for that patient
(`danielmherman/static/js/demo_flow.js` 150–168). A question is posted with its admission, the
user's turn and a pending agent turn are appended immediately, and the answer replaces the
pending turn (`danielmherman/static/js/demo_flow.js` 675–736). One request is in flight at a
time, which is what keeps the thread ordered (`danielmherman/static/js/demo_flow.js` 676–678).
Nothing is written to browser storage, so a reload ends the rendered thread. What the page keeps
beyond the life of one request is the identifier of the conversation its thread belongs to
(`danielmherman/static/js/demo_flow.js` 163, 685–686, 797), which it sends back with the next
question; the turns themselves are never posted by the browser, and a request that tries to is
refused (`danielmherman/demo/views.py` 33–71).

Each request is composed as a first question unless it names a conversation. The site turns a
payload into an intent — a starter chip, a clicked patient, or typed free text — and sends that
intent with the admission, leaving the wording of the question to the agent
(`danielmherman/demo/views.py` 175–231, `danielmherman/demo/agent_client.py` 111–115). The
agent validates a closed field set, runs one question to completion through the chain, and
returns the answer with its sources, its tool calls, the model and the code revision that
produced it (`services/agent/contracts.py` 8–17, 78–95, 360–413,
`services/agent/graph.py` 347–384). The transcript the chain builds while doing this is
returned to the caller and not stored, and the answer payload carries the passages behind each
citation rather than a reference the caller would have to resolve
(`services/agent/contracts.py` 25–29).

The demonstration has a second mode that answers without the agent at all. In fixture mode the
site composes the answer from a captured payload, which is what local work and the offline
demonstration use; the allowance is reported rather than charged, and the question wording is
the site's own (`danielmherman/demo/views.py` 409–420, `danielmherman/demo/fixtures.py` 46–52).
The mode is refused in production (`danielmherman/danielmherman/settings.py` 117–122). Decided
2026-09-18 and implemented the same day: this path stays explicitly single-turn, and the
interface labels it as such, so the offline demonstration never implies the continuity the live
agent has. Fixture mode opens no conversation and writes no turn, and the page carries a note
saying that answers come one question at a time (`danielmherman/demo/views.py` 236–258,
`danielmherman/demo/templates/demo/a2ui_console.html` 88–98), with two tests holding both
halves (`danielmherman/demo/test_conversation_memory.py` 262–281).

What lives between requests is the browser's episode, an authentication session, a daily
counter and a few caches inside each service. The session's cookie expires with the browser
(`danielmherman/danielmherman/settings.py` 137), and its server-side row does not: the session
engine is the database backend, the age is Django's default of two weeks, and no job sweeps
expired rows. The counter is claimed before the work and refunded after a failure that was not
the user's fault, with the rollover performed lazily on the first request of a new day
(`danielmherman/demo/models.py` 55–92, 94–125). The caches are the tool client's identity token
and the memoised service handles described in section 1.2.

The agent accepts the earlier turns of a conversation, which is what makes the store usable. The
request contract gained one field and kept its closure: `turns` is a list of at most six earlier
turns, each a question and an answer and nothing else, each bounded in length, and any other
field inside a turn is refused rather than carried (`services/agent/contracts.py` 88–95,
105–113, 182–252). A turn is never decoded or reformatted: what the clinician asked and what
they were told are replayed as they were. The chain turns each replayed turn into the pair of
messages a conversation would have produced — the question as a human message, the answer as the
assistant message it was — and seeds them between the system prompt and the question now being
asked, so the template stays a template and the replayed material stays data
(`services/agent/graph.py` 386–411). Both routes carry the field through unchanged
(`services/agent/http.py` 90–133, 293–319, 477).

The rules about continuity and spend are enforced in the ask path, where a caller can feel them.
Opening a conversation is what spends a credit: a request that names no conversation and carries
an admission opens one pinned to that admission and claims a credit, while a follow-up inside a
conversation claims nothing because it is already paid for (`danielmherman/demo/views.py`
89–142, 432–447). The rules that protect the boundary are checked before the agent is called: a
conversation belonging to another account is reported exactly as one that does not exist, a
conversation past its window is refused even when the sweep has not reached it, a follow-up
whose admission differs from the conversation's is refused, and a conversation that has spent
its turns is refused with the way forward named rather than only the refusal
(`danielmherman/demo/views.py` 89–142). A turn is counted only when an answer exists, on both
the blocking and the streaming path, which is what makes the ceiling a limit on answers rather
than on attempts; a failed opening turn refunds its credit and leaves the conversation in place,
so the retry is the first successful turn and costs exactly one credit
(`danielmherman/demo/views.py` 144–156, 276–289, 341–352, 484–492). Nine tests cover the rules
through the endpoint with the agent mocked
(`danielmherman/demo/test_conversation_lifecycle.py` 1–182).

A replayed turn carries its tool calls as well as its dialogue, so a replay is structurally the
same thing to the model as a live turn. A turn may hold up to five calls, each with a name, its
arguments, and optionally the payload the call returned
(`services/agent/contracts.py` 109–111, 116–179). The chain rebuilds the pair a live turn
produces — the assistant turn that asked for the call, then the result — and the result goes
through the same renderer a live one does, so the `<tool_result>` wrapper and the guard that
neutralises the wrapper's own delimiter inside a payload apply identically
(`services/agent/graph.py` 421–478, 102–142). Replaying a passage outside the wrapper would put
clinical text where the prompt says instructions live, which is the boundary the wrapper exists
to draw.

A payload is carried only when it cannot be recomputed. A prediction score is stored, because
recomputing it would need the endpoint version that produced it and could still return a
different number; a retrieval result is not stored but resolved again by calling the tool, which
is deterministic here because the corpus is written offline and never mutated by a request.
Re-resolving is the price of keeping passage text out of the conversation store, and it is paid
only for the calls that stored nothing (`services/agent/graph.py` 465–477). Ten tests cover this
half of the replay, including the delimiter guard applied to a stored payload and the refusal of
an oversized passage on replay exactly as on a live call
(`tests/agent/test_replay_tool_calls.py` 1–168).

The interface was the last piece of this, and it was implemented on 2026-09-18. The site stores
each answered turn and sends the earlier ones back. An answer is written as the pair it was —
the question the agent composed and the answer it produced, with the citation identities, the
tool calls, and the model and code revision behind them
(`danielmherman/demo/conversations.py` 51–88) — and the next question goes out with up to six of
those turns, in the shape the agent replays and bounded by the same caps
(`danielmherman/demo/conversations.py` 133–162, `danielmherman/demo/views.py` 158–173). The
browser supplies only the identifier of the conversation it is continuing: it is given that
identifier with each answer, sends it with the next question, and forgets it when the site
refuses to continue the conversation, which is what makes a refusal's advice actionable by
asking again (`danielmherman/static/js/demo_flow.js` 39–41, 685–686, 749–751, 797,
`danielmherman/demo/views.py` 484–492).

What is stored is checked against the policy rather than assumed to follow it. A citation keeps
its identity and drops its passage, so no discharge-note text enters this database, and a tool
result is kept only when it cannot be obtained again
(`danielmherman/demo/conversations.py` 90–131). That distinction is not the site's to guess:
the agent states it on every call it emits, because only the layer that ran the tool knows
(`services/agent/contracts.py` 32–47, `services/agent/http.py` 193–219), and the
classification is held exhaustive over the tools the server registers by a test, so a new tool
fails the suite rather than defaulting into a privacy decision
(`tests/agent/test_agent_contract.py` 369–383). The answer also states the code revision that
produced it (`services/agent/http.py` 228–236), which is what lets a stored turn be explained
after a deploy. Fourteen tests cover the store and the replay, through the endpoint with the agent
mocked and against the replay builder directly
(`danielmherman/demo/test_conversation_memory.py` 1–281), and the site's 135 tests pass.

One ordering decision was recorded rather than left implicit, and it was resolved in the
direction it predicted. The site's closed set accepts the field name it needs and refuses the
field it must not accept: `conversation_id` is approved at the site's boundary and `turns` is
refused there, because the browser names a conversation and the store supplies its contents
(`danielmherman/demo/views.py` 33–71). A caller that could push its own turns could put words in
the copilot's mouth and have the answer stored as though they had been said. The two closed sets
therefore differ on purpose, and the difference is the requirement: the site may hold the
conversation, and the agent may not.

The reply to the boundary question is the one line the interface now carries: a conversation
lives in the page, and a reload starts a new one. It is the honest form of the decision rather
than a gap left silent — the thread was already the page's, and what was missing was saying so
while the user could still see it. Resuming the rendered thread instead was considered and
deferred, and the reason is technical rather than a matter of preference: a stored turn holds
the question, the answer, the citation identities and the tool calls, but not the A2UI canvas
the turn drew or the passages behind its citations, because those are composed by the agent
from live tool evidence and Django is deliberately a pass-through. A resumed thread would
therefore be a transcript wearing the app's clothes, and the alternative — storing the composed
presentation per turn — would put note text and prediction scores in this database, which is
the thing the retention policy refuses.

The conversation store was built on 2026-09-18 and is where a conversation lives. Two models hold it: a conversation belongs to one account and
one patient, carrying the window it was opened under and the number of turns spent
(`danielmherman/demo/models.py` 196–267), and a turn carries the question, the answer, the
model and code revision that produced it, the citation identities and the tool calls
(`danielmherman/demo/models.py` 269–316). The patient is pinned: the model refuses a save that
would repoint an existing conversation at another admission, which is a policy enforced in
code rather than a convention (`danielmherman/demo/models.py` 237–253). The retention window
is fixed at creation rather than sliding, and both it and the turn ceiling are settings
(`danielmherman/danielmherman/settings.py` 109–114). What removes an expired conversation is a
job, not a reader: `purge_expired_conversations` deletes in batches and cascades to the turns,
with a dry run for inspection (`danielmherman/demo/management/commands/purge_expired_conversations.py`
25–66). Administrators can read a conversation and cannot edit one
(`danielmherman/demo/admin.py` 37–58). Nine tests hold the policy
(`danielmherman/demo/tests.py` 234–360), and the platform schedule that runs the sweep is a
deployment step rather than repository content: the command is safe to run on any schedule,
and Cloud SQL has no native row expiry, so it needs `pg_cron` in the instance or a scheduler
that invokes it.

The deployment runs more than one instance of every service. Each permits up to three, with a
minimum of zero so that an idle demonstration costs nothing
(`services/agent/cloudbuild.yaml` 54, `services/mcp/cloudbuild.yaml` 45–46). The site bounds
agent calls per instance with a semaphore and times each one out, so a burst queues rather than
multiplies (`danielmherman/demo/agent_client.py` 45,
`danielmherman/danielmherman/settings.py` 101–112). Nothing in the design depends on a single
instance, and nothing in it keeps a conversation in one.

---

## 4. Gaps, remediation and record of change

Against the requirements in section 2: A1 and A3 are met, each in the clause about
externalised conversation state and in the clauses about the agent and the API. The state is
external to the agent, it lives in a store the site authorises, and the interface sends the agent
what it needs to answer a follow-up while holding nothing itself. A7 is not applicable and is
recorded as a decision rather than left unresolved. Five entries were opened; all five are now
closed, four by implementation and one by a decision, and the two pieces of work that remain are
named where they belong: the schedule that runs the sweep at deployment, and the stated
lifetimes for the trace archive and the note cache, which are harness-side actions owned by
other layers.

| # | Gap | Remediation | State |
|---|---|---|---|
| 1 | A thread is displayed and not transmitted. The browser keeps one episode per patient and renders a conversation, while every request is composed as a first question, so the interface implies continuity that the answer does not have (A1, A3). | Settle the interaction model in the strategy session: whether the page's thread is to be answered from, and if so, by whom — the caller sending the turns it wants considered, or a session the agent resolves. Record the answer, then implement it on the path the demonstration uses. | Open. Decision required; the decision is the remediation. Decided 2026-09-18 in the strategy session: **session-scoped continuity, with cross-session memory declined on the record.** The work is now implementation rather than decision. **Closed 2026-09-18.** The agent accepts earlier turns and replays their tool results, the store holds a conversation and the answers it produced, and the interface continues the thread it renders: the browser keeps the identifier the site returns and sends it back, and the site — not the browser — supplies the turns the agent is asked to consider (`danielmherman/demo/views.py` 158–173, 432–447, 484–492, `danielmherman/static/js/demo_flow.js` 685–686, 797). Fourteen tests cover the store and the replay (`danielmherman/demo/test_conversation_memory.py` 1–281), and the flow was exercised end to end in a browser. What the entry did not settle, and the session-boundary decision does, is what a reload means: **a session is a page load, and the interface says so while a conversation is open** (`danielmherman/demo/templates/demo/a2ui_console.html` 151–154). |
| 2 | No session identity exists on the wire, and the site's proxy is an open boundary: it reads three keys and drops every other key in silence, which is the failure that closing the agent's contract was meant to prevent, at the boundary a browser can reach (`danielmherman/demo/views.py` 41–96). | Close the site's boundary as the agent's is closed: refuse an unknown field rather than ignore it, and state in the layer 3 record that the guarantee now holds at both boundaries. If the interaction model introduces a session field, it is added to both closed sets in the same change. | **Half closed 2026-09-18.** The site's proxy now refuses an unknown field rather than dropping it, using the same two-wordings refusal as the agent — a conversation field is refused as a product decision made visible, any other unknown field is refused by naming what the endpoint accepts — and the check runs before the fixture branch, so neither mode can answer a request whose extra fields it would have dropped. Three tests in the site's suite hold it (`danielmherman/demo/tests.py`, `RequestBoundaryTests`). **Closed 2026-09-18.** The session identity is on the wire and is named at the site's boundary: `conversation_id` is one of the four fields the proxy accepts, and the browser is given it with each answer and sends it back with the next question (`danielmherman/demo/views.py` 33–71, 484–492). The agent's closed set does not accept it and must not: the requirement that the agent hold nothing is exactly why the caller sends the turns instead (A1, A3). The two sets therefore differ on purpose, and section 3 records the difference as the requirement rather than as an inconsistency. |
| 3 | No retention policy exists for a conversation, and three holdings of clinical text already exist without one: the evaluation harness's trace archive, the retrieval layer's note cache, and the captured fixture payloads committed to both repositories (`evaluation/agent/collect.py` 1–8). The answer payload carries passage text per citation, so a store of "the citations that were shown" holds note text unless the stored shape drops that field (`services/agent/contracts.py` 23–27). | Fix the policy before the store, and reconcile it with the three holdings rather than writing it around them: what is kept, for how long, who may read it, what an account deletion or the end of the demonstration does to it, and what the logs hold as distinct from what the product holds. Name the stored shape explicitly, including which fields of a citation are kept. | Open. Policy decision required; it precedes the store because the store is built to it. Decided 2026-09-18: **twenty-four hours, swept by a job rather than by lazy expiry** — a row that is never read is never swept by a reader, and PostgreSQL has no native row expiry — **a stored citation keeps its identity and not its passage text, and each of the three existing holdings is given its own stated lifetime.** **Decided 2026-09-18: the store keeps tool names, their arguments and the payloads that cannot be re-derived — the prediction scores — and retrieval payloads are re-derived on demand, so no clinical note text enters the session store. The evaluation trace archive gets a fixed longer lifetime for auditing, the note cache is treated as ephemeral and regenerable with a very short lifetime, and the committed fixtures carry no runtime lifetime at all: they are a repository decision, kept synthetic or de-identified.** **Implemented 2026-09-18: the store enforces the window and the ceiling from settings, the sweep is a management command with a dry run, nine tests hold it, and the site's 98 tests pass. Still outstanding: the schedule that runs the sweep at deployment, and the stated lifetimes for the trace archive and the note cache, which are harness-side actions.** |
| 4 | The orchestrator layer handed four decisions to this layer and none is taken: whether the prompt template and replayed material must be separated, how a replayed passage is kept outside the tool-result boundary that marks retrieved text as data, whether history is replayed as structure or as text, and how the code revision that produced an answer travels with the stored turn (layer 3, section 6). | Take them in the strategy session, since they constrain what the context window may contain and therefore what a stored turn must carry. The injection boundary is the one with a security consequence: replaying retrieved text into the prompt outside its tool-result marker reopens the channel that boundary exists to close. | Open. Four decisions, one of which is a security boundary rather than a product preference. Partly decided 2026-09-18: **history is replayed as structure, the prompt template is separated from what is replayed into it, and the code revision is stored with each turn.** The injection boundary is accepted as a requirement and its stored shape is carried forward: a tool result is a message role plus a wrapper with a delimiter guard (`services/agent/graph.py` 98–102, 208–215), so replay has to reconstruct that structure. **Decided 2026-09-18: the store keeps what is needed to rebuild the structure, which is the tool name and its arguments plus the payloads that cannot be re-derived, and retrieved passages are re-derived; the wrapper and its delimiter guard are applied on replay exactly as they are applied to a live tool result. Implemented the same day for the dialogue: earlier turns are accepted under a closed, bounded turn shape and replayed as messages, with six tests proving order, fidelity, the cap and the separation of the template from the replayed material. **Implemented in full on the harness side the same day: the tool calls of a replayed turn are re-inserted as the assistant call and its result, rendered through the same wrapper and delimiter guard as a live result, with stored payloads used where they exist and the rest re-resolved from the deterministic corpus. **Closed 2026-09-18**: the site stores each answered turn with its tool calls and sends them back in the agent's replay shape, and the agent states on every call it emits whether the result can be obtained again, so the store keeps only what it cannot re-derive (`danielmherman/demo/conversations.py` 51–162, `services/agent/contracts.py` 32–47). The code revision is populated: the agent returns it with the answer and the store keeps it, which is what makes a stored turn explainable after a deploy (`services/agent/http.py` 228–236).** |
| 5 | Continuity is undefined where it has a clinical consequence. Each request carries its own admission, and isolation is enforced per request; with a thread that continues, a session must not be able to move between patients implicitly, and the allowance must say what a follow-up costs. The daily allowance bounds the total spend but nothing bounds the shape of a conversation, and failed turns are refunded individually (`danielmherman/demo/views.py` 195–215, 279–310). | Settle in the strategy session: the admission is pinned when a conversation opens, a change of patient is an explicit act, every turn re-asserts the restriction the tools enforce, a turn spends a credit as a request does, and a conversation carries a ceiling on turns as well as the per-turn step cap that exists. Implement with a test that a conversation cannot be answered about an admission other than the one it was opened for. | Open. Three decisions — identity, isolation, cost — with one implementation. Decided 2026-09-18: **the admission is pinned when a conversation opens and re-asserted on every turn, and only the first turn spends a credit.** The credit decision decouples spend from the allowance, so the turn ceiling is the bound on what one credit buys, and **Decided 2026-09-18: the ceiling is a hard limit on user turns, provisionally six, and reaching it offers a button that starts a new conversation and spends the next credit rather than truncating the thread or blocking the user without a path forward. A conversation whose first turn fails keeps its state and is refunded, so the retry is the first successful turn and costs exactly one credit. **Implemented 2026-09-18: opening a conversation spends the credit, a follow-up does not, the ceiling refuses with the way forward named, a turn is counted only when an answer exists, and a failed opening turn refunds and keeps its conversation. Nine tests hold it through the endpoint.** |

**Record of change.** This document was written on 2026-09-18 and corrected the same day after
an independent review. The correction was to its central claim. The first version asserted that
the layer held no conversation state, that the browser held nothing, and that the interface
offered no field for a follow-up; all three were wrong. The browser keeps one episode per
patient, the console page has had a follow-up composer and a rendered thread throughout, and
the accurate statement of the defect is narrower and more useful: the thread is displayed and
not transmitted, so the interface claims a continuity that the answers do not have. Two earlier
layer documents had already recorded this, the client layer as state living in the browser's
memory and the orchestrator layer as a thread that is page memory and not saved state; this
document is the first to treat it as this layer's subject rather than the next layer's problem.
The review also corrected four secondary claims and added the material that was missing: the
session cookie's lifetime is not the session row's, the daily counter is charged on the live
path only, the services hold caches with lifetimes and permit up to three instances each, and
the site's request boundary is open where the agent's is closed.

The strategy session was held on 2026-09-18 and every decision it raises is settled, in
section 4 and section 5 of `layer-08-memory-and-ux-strategy.md`. The first piece of work
the session ordered was implemented the same day: the site's request boundary was open,
reading three keys and dropping every other key, and it now refuses an unknown field with
the same two-wording refusal the agent uses, checked before the fixture branch so that
neither mode answers a request whose extra fields it discarded. Three tests were added and
the site's 89 tests pass. That closes the boundary half of entry 2. The second item, the conversation store, was
built the same day: two models behind one authorisation boundary, the patient pinned in
code, the retention window and the turn ceiling taken from settings, a sweep that
deletes by batch under a dry run, a read-only administrative view, and nine tests. The third item, the contract change, followed: the agent accepts `turns` under a
closed and bounded shape, the chain replays each turn as its own messages between the
prompt and the current question, both routes carry the field, and eleven tests were
added or updated to hold it. The site still refuses the field, deliberately, until the
interface forwards it — the two closed sets are not yet the same, and the document says
so. The fourth item completed the replay on the harness side: a replayed turn's tool
calls are rebuilt as the assistant call and its result, through the same wrapper and
delimiter guard a live result passes, with stored payloads used where they exist and
retrieval re-resolved from the corpus where they do not. Ten tests hold it. The fifth item put the money and continuity rules into the ask
path: a credit is claimed when a conversation opens and not when it continues, the
ceiling refuses with a way forward, a turn counts only once an answer exists, and a
failed opening turn keeps both its credit and its place. Nine more tests cover the
rules through the endpoint. The sixth item wired the interface to the store and closed it: an answer is written as the pair it was, the next question goes out with
the earlier turns in the agent's replay shape, the browser carries the conversation
identifier and is told what it means, and the offline path is labelled as the
single-turn thing it is. Two pieces of the answer had to be added for this to be
faithful rather than plausible — each emitted tool call now carries its arguments and
a statement of whether its result can be obtained again, and the answer carries the
code revision behind it — because a replayed call without its arguments misstates
what was asked, and a stored turn with no revision cannot be explained later
(`services/agent/http.py` 193–236). The tool classification is held exhaustive over
the tools the server registers: a new tool fails the suite rather than defaulting
into a privacy decision (`tests/agent/test_agent_contract.py` 369–383).

The seventh item answered the question the sixth one exposed: what a session is. The
rendered thread was always the page's, while the store's copy outlives it, so a reload
produced a new conversation and a second credit for a first turn. Decided 2026-09-18:
**a session is a page load** — the thread and the conversation end together — and the
interface says so in one line while a conversation is open, so the boundary is
understood rather than discovered. Resuming the rendered thread from the store was
considered and deferred rather than rejected, and the reason is that the store holds no
rendered turn: the canvas and the passages behind citations are composed by the agent
from live evidence, so a resumed thread would be a transcript without its artifacts, and
the only way to store the artifacts is to put note text and scores in the site's
database, which the retention policy refuses (`docs/architecture/layer-08-memory-and-ux-strategy.md`,
the boundary of a session).

Every entry in the table above is closed. The
entries that were closed by implementation were held to the layer's own rules rather
than to a successful-looking answer: no clinical note text enters the session store, a
replayed turn carries its structure, and a boundary that is open is said to be open. The gaps were
decisions that had not been taken rather than defects that could be fixed, which is why the
remediation for three of them was the strategy session rather than a change. The one property worth recording as a
positive finding is that the demonstration cannot currently answer about the wrong patient by
carrying an identifier forward, because there is nothing on the wire to carry; that property
follows from the single-turn contract rather than from a constraint that would survive the
addition of continuity, which is why entry 5 is a design requirement and not a note.

The strategy session is set out in `layer-08-memory-and-ux-strategy.md`: the decisions it must
settle, the options for each, the evidence available, and the places the outcome has to be
written down so that the next reader finds the policy rather than the code.
