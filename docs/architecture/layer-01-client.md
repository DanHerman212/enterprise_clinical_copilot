# Layer 1 — Client

Status: audited and refactored 2026-09-11. Gap 2 closed; streaming (Gap 1) deferred to layer 3 by decision.

---

## 1. What a web application is

A web application is a set of programs that talk to each other over a network.
For this application there are three of them.

```
  user's machine                Google Cloud                     Google Cloud
 ┌────────────────┐   HTTPS    ┌────────────────────┐   HTTPS   ┌────────────────┐
 │  Browser       │ ─────────▶ │  Django web server │ ────────▶ │  Agent service │
 │  HTML, CSS, JS │ ◀───────── │  (public)          │ ◀──────── │  (private)     │
 └────────────────┘    JSON    └────────────────────┘    JSON   └────────────────┘
```

**The browser** runs on the user's machine. It receives HTML, CSS and
JavaScript from the web server and executes them. Everything in the browser is
under the user's control: they can read it, modify it, or replay it. Nothing
the browser holds can be trusted, and nothing secret can be given to it.

**The web server** is the program the browser talks to. In this application it
is Django, running on Cloud Run, reachable from the public internet. It has
four jobs: serve the page, identify the user (login and session), check the
request is well-formed and allowed, and call the private services on the
user's behalf. Because the browser can only ever reach the web server, the web
server is where all the enforcement happens.

**The agent service** does the actual work — runs the model, calls tools,
composes the answer. It is private: Cloud Run will refuse any request that does
not carry a Google identity token from an authorised service account. The
browser has no such token, so it cannot reach the agent even if it knew the
URL.

Terms used in this document:

| Term | Meaning |
|---|---|
| Session cookie | A cookie the web server sets at login. It identifies which logged-in user is making the request. |
| CSRF token | A second token that proves the request was issued by our own page, not by another website the user happens to have open. Required on every POST. |
| Content Security Policy (CSP) | A response header that tells the browser which origins it may load scripts from and which origins it may connect to. The browser enforces it. |
| Static assets | The JavaScript, CSS and image files. Served as-is; versioned with a query string (`?v=17`) so browsers pick up new versions. |
| Backend for frontend | A web server whose only purpose is to serve one user interface and mediate between it and internal services. That is Django's role here. |

**Layer 1 is the browser plus the web server that serves it.** Its job is to
render results and collect input — nothing more. All intelligence lives behind
it.

---

## 2. What Google requires of this layer

From *Choose your agentic AI architecture components* (Google Cloud
Architecture Center, reviewed 2026-04-21) and *Single-agent AI system using
ADK and Cloud Run* (reviewed 2025-12-09):

1. The frontend does not call the model. It talks to the agent through an API.
2. The frontend holds no secrets.
3. A production frontend supports streaming — the user sees output as it is
   produced rather than waiting for the whole answer.
4. The API between frontend and agent is stateless: each request is complete
   on its own.
5. Conversation state is kept in an external store, not in the page and not in
   the web server's memory.

Everything else about the frontend — which framework, how it looks — Google
leaves open.

---

## 3. How this application implements the layer

Repository `danielmherman` (Django). File paths are relative to that repository.

### 3.1 The page

`demo/templates/demo/a2ui_console.html` is the single demo page. Django fills
in three things before sending it: the list of demo patients (from the
database), the user's remaining daily quota (line 81), and the URL the browser
should post questions to (line 22, `data-ask-url`). It loads one JavaScript
module, `static/js/demo_a2ui.js` (line 190).

### 3.2 The browser code

Two files, about 1,100 lines together.

`static/js/demo_flow.js` (713 lines) owns the patient list, the conversation
thread, the starter chips, and the ask flow. The only network call in the
browser is `fetch(state.askUrl, ...)` at line 620, inside `async function
post(body, userText)` (line 590). It posts JSON to Django with the CSRF token
in the `X-CSRFToken` header (line 622) and waits for one JSON reply.

`static/js/demo_a2ui.js` (167 lines) draws the right-hand "context canvas" by
feeding the agent's response into a vendored renderer (the A2UI library,
imported at lines 16–20).

### 3.3 The web server

`demo/urls.py` defines two routes for the demo: `a2ui/` renders the page,
`a2ui/ask/` accepts a question.

`demo/views.py`:

- `a2ui_console` (line 81) renders the page. Requires login.
- `a2ui_ask` (line 109) handles a question. Requires login and POST. It parses
  the body, turns it into a question (`_question_for`, line 25), claims one
  quota credit, calls the agent, refunds the credit on failure, and returns the
  agent's JSON with the remaining quota added (line 176). On agent failure it
  logs the detail server-side (line 154) and returns a generic message to the
  browser.

`demo/agent_client.py`:

- `_id_token` (line 47) mints a short-lived Google identity token for the
  agent's URL. In production this comes from the Cloud Run metadata server.
- `ask` (line 98) posts `{"question": ...}` to the agent's `/ask` endpoint
  with that token (`requests.post`, line 121), with a 120-second timeout
  (line 125). A semaphore limits concurrent agent calls to four per instance
  (line 44).

`demo/agent_contract.py` — `validate_agent_response` (line 26) checks the
shape of the agent's reply before Django forwards it.

### 3.4 Configuration that enforces the boundary

`danielmherman/settings.py`:

- Content Security Policy (line 285). `connect-src` is `'self'` (line 297):
  the browser may only make network calls back to the Django origin.
- The Django secret key is read from Secret Manager in production (lines
  56–70). The agent URL is an environment variable (line 98); it is a location,
  not a secret.
- Fixture mode (`DEMO_FIXTURE_MODE`, line 117) serves captured responses for
  local development. Production refuses to start with it enabled (line 122).

### 3.5 The agent's side of the contract

Repository `enterprise_clinical_copilot`. `services/agent/http.py` exposes
`POST /ask` (`ask_route`, line 69) and returns a single `JSONResponse`. There
is no streaming endpoint.

---

## 4. Current state against the requirement

| Google requires | What the code does | Met? |
|---|---|---|
| Frontend does not call the model; talks to the agent via an API | Browser's only call is to Django's `a2ui/ask/`. CSP forbids any other origin. Agent is private and needs an identity token the browser never has. Django proxies one request per question. | Yes |
| Frontend holds no secrets | Browser receives patient rows, quota count, and a relative URL. Secret key lives in Secret Manager. Error detail stays in server logs. | Yes |
| Stateless API | `a2ui_ask` is one request in, one response out. Django keeps no conversation state. | Yes |
| Streaming | Django blocks on a single `requests.post` for up to 120 s. The browser shows "…" until the entire answer arrives. The agent has no streaming endpoint. | **No** — deferred to layer 3 by decision (section 6). |
| Conversation state in an external store | State lives in the browser's memory (`state.episodes`, `demo_flow.js` line 128). A page reload loses it. | **No** — but this belongs to the memory layer (layer 8), not to the client. Recorded there; not counted against layer 1. |

One further observation that is not on Google's list but bears directly on
"the client only renders":

**The browser interpreted the agent's output (fixed 2026-09-11).** Before the
refactor, `demo_flow.js` held a hand-copied list of clinical note section
headings (`SECTION_ALIASES`) duplicating a list in the agent repository, and
an `extractSection` function that used it to cut a passage out of a full note.
`demo_a2ui.js` `envelopeForCite` decided, when a user clicked a citation
number, *which passage that number referred to*, using heuristics about the
question's intent to correct for the model sometimes mis-numbering its
citations. That was interpretation. After the refactor the agent sends a
resolved `sources` list and the browser looks a clicked footnote up by number
(`envelopeForCite`, `demo_a2ui.js` line 117; the lookup is line 119).

---

## 5. Gaps

**Gap 1 — No streaming.**
The user waits, blind, for the whole pipeline (two possible cold starts, tool
calls, generation). Django holds a worker thread for the duration. Fixing this
needs two halves: the agent must emit a stream, and Django and the browser
must pass it through and render it progressively.

**Gap 2 — The browser interprets citations. (Closed 2026-09-11; see section 7.)**
Roughly 200 lines of JavaScript, plus a duplicated vocabulary across two
repositories, existed because the agent's response did not tell the browser
which passage each citation pointed to. The agent already knew: it renumbers
the citations and holds the passages. It now includes a resolved `sources`
list in its response; the browser displays `sources[n]` and the
interpretation code is deleted.

---

## 6. Design decisions

Both gaps require a change to the agent's response, so both cross the
repository boundary.

| Option | Scope | Trade-off |
|---|---|---|
| **A. Close Gap 2 now; defer Gap 1 to the orchestrator layer (layer 3).** | Agent: add resolved `sources` to the response. Browser: delete interpretation code, render `sources[n]`. | Small, self-contained, testable. Streaming is designed once, when the agent's request handling is on the table. |
| B. Close both now. | As A, plus a streaming endpoint on the agent and pass-through in Django and the browser. | Larger. Decides how the agent streams before the agent layer has been studied. |
| C. Audit only; refactor after layer 3. | None now. | Keeps momentum through the layers; leaves this layer open. |

Recommendation: **A**.

Decision (2026-09-11): **A.** Close Gap 2 now. Streaming is deferred to the
orchestrator layer (layer 3), where the agent's request handling is studied.

---

## 7. What changed

The agent now resolves every citation to its passage and sends the result;
the browser only renders it. Both suites were green before and after:
`pytest tests/agent` (239 passed) and `python manage.py test demo` (34 OK).

**Agent repository `enterprise_clinical_copilot`**

- `services/agent/a2ui.py` — new `resolve_sources` (line 154) returns one
  `{cite, section, text, query}` per citation in the answer, using the
  resolution logic the canvas already had. `compose_presentation` (line 328)
  calls it and adds `sources` to the presentation (line 357). The canvas's
  Source card is now defined as `sources[0]`, so the two cannot disagree.
- `services/agent/citations.py` — `cited_numbers` (line 184) exposed as a
  public helper so `a2ui.py` does not import a private name.
- `services/agent/contracts.py` — `sources: list[ResolvedSource]` added to
  the response type (line 34) and validated (lines 109–118).
- `services/agent/http.py` — `/ask` forwards `sources` (line 160).
  `citation_map` and `intent_sections` are no longer sent; they remain
  internal inputs to the resolution.
- `tests/agent/test_a2ui.py` — `TestResolvedSources` (line 329) pins the
  shape, the ordering, and agreement with the canvas.

**Web repository `danielmherman`**

- `static/js/demo_flow.js` — deleted `SECTION_ALIASES`, `SECTION_HEADERS`,
  `extractSection` and `escapeRegex`. Each turn now stores `sources` from the
  response (`agentTurnFromResponse`, line 567) instead of `intentSections`
  and `citationMap`. 855 → 713 lines.
- `static/js/demo_a2ui.js` — `envelopeForCite` (line 117) is a lookup,
  `turn.sources.find(s => s.cite === n)` (line 119); the heuristics and
  `unavailableText` are gone. 240 → 167 lines.
- `demo/agent_contract.py` — validates `sources` (lines 47–53).
- `demo/views.py`, `demo/fixtures.py`, `demo/tests.py` — comments and
  assertions updated for the new field; no behaviour change in Django's proxy.
- `demo/templates/demo/a2ui_console.html` — script version bumped to `?v=12`
  (line 190) so browsers fetch the new module.

**Verified in the browser** (local, fixture mode, admission 90000009 — Linda
Marchetti): the medications answer renders four footnotes; clicking [1], [2]
and [3] swaps the Source card to discharge medications, brief hospital course
and discharge diagnosis respectively, each taken directly from `sources[n]`.
The fixture path composes its response with the agent's own
`compose_presentation`, so this exercised the changed code.

**Not changed:** Django's request/response proxy, the quota, the login, the
Content Security Policy, the agent's tools or prompts. Nothing has been
committed or deployed yet.

---

## 8. Interview questions this layer answers

**Why can't the browser call the agent directly?**
Three independent reasons: the Content Security Policy only allows connections
back to the Django origin; the agent is a private Cloud Run service that
requires a Google identity token; and the browser is never given one. Django
is the single public surface, so login, quota, and input checks happen in one
place.

**What is Django's job in this architecture?**
It is a backend for frontend. It serves the page, authenticates the user,
validates the request, mints an identity token, forwards one request to the
agent, checks the response shape, and returns it. It does not compose the
answer — the agent owns that.

**Why does Google want streaming?**
Agent requests are slow — seconds to tens of seconds. Without streaming the
interface is blind and the server holds a thread per request. Streaming gives
the user progress and bounds server resources.

**Where should conversation state live, and where does it live today?**
Google says in an external store, so any server instance can handle any
request and a reload loses nothing. Today it lives in the browser's memory.
That is addressed in the memory layer.
