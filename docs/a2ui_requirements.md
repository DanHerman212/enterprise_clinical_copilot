# A2UI — Requirements (Draft)

_Date: 2026-07-29 · Status: research findings + decisions, not yet a design_

Requirements for the A2UI rendering layer, captured from a review of the official
specification and docs. This is the **WHAT/WHY** and the constraints we must design
around. See [NEXT_STEPS.md](NEXT_STEPS.md) for sequencing — A2UI lands at Step 5.

Sources: [a2ui.org](https://a2ui.org/), the
[transports](https://a2ui.org/concepts/transports/),
[A2UI over MCP](https://a2ui.org/guides/a2ui_over_mcp/), and
[renderers](https://a2ui.org/reference/renderers/) references.

## What A2UI actually is

A **declarative JSON format** that lets an agent describe a UI without emitting code.
The client holds a **catalog** of pre-approved components; the agent may only request
components from that catalog. The payload is data, never executable code — which is why
this is defensible in a clinical context.

The analogy the spec uses: A2UI is HTML, the agent is the server, the renderer is the
browser.

## Headline finding: WebSockets are not required

The earlier concern — that committing to A2UI would force a bidirectional connection
and drag in a `websocket` key, sticky sessions, and eventually Redis — **is resolved.
It does not.**

A2UI is explicitly transport-agnostic: "use any method that can send JSON." The
official transport status table:

| Transport | Status |
|---|---|
| A2A Protocol | Stable |
| AG-UI | Stable |
| REST API | Planned |
| **WebSockets** | **Proposed only — not implemented** |
| **SSE** | **Proposed only — not implemented** |

WebSockets are not merely optional; they are not even built. Under the MCP transport,
**user interactions come back as ordinary tool calls** (`a2ui_action`), not as messages
on a held-open socket. The entire loop is request/response.

**Consequence:** the absent `'websocket'` key in `asgi.py` stays absent. Redis stays
out. The Postgres-only decision holds.

## The real risk is the frontend, not the transport

Every maintained web renderer is a **client-side JavaScript package** — React, Lit
(Web Components), or Angular. There is no server-rendered or Django-template-native
renderer, and none is planned.

Worse, the documented "use A2UI with any agent framework" path assumes
**CopilotKit + Next.js + React**, and the only documented LangGraph integration runs
through `CopilotKitMiddleware`. Following that path literally would mean standing up a
second frontend service and abandoning Django as the UI host.

That is the thing that would actually bite us.

## Requirements

### R1 — Render inside Django; do not adopt Next.js or CopilotKit

The demo page is served by the existing Django app at
`/projects/clinical_copilot/demo`. Introducing a Next.js frontend would mean a second
Cloud Run service, a second deploy path, a second auth surface, and would defeat the
thin-BFF design.

**Use the Lit renderer.** Lit compiles to standard **Web Components**, which mount into
an ordinary Django template as a custom element with a `<script>` tag. It is the only
maintained renderer that does not impose a frontend framework on the host page. React
and Angular both assume they own the page.

### R2 — No JS build step; load Lit natively

_Revised 2026-07-29 — this requirement was originally written defensively and was wrong._

Lit does **not** require Vite, Webpack, or any bundler. Running without a build step is
an explicit Lit design goal. Three viable paths, in order of preference:

1. **Pre-built bundle from a CDN** — Lit has shipped official pre-built bundles since
   v2.2.0, loadable directly via `<script type="module">` with no bare-module
   resolution.
2. **ESM CDN** (`esm.sh`, `unpkg`, `skypack`) — resolves dependencies on the fly for
   standard `import` statements.
3. **`npm i lit` + a native HTML import map** — serve `node_modules` as static files;
   the browser resolves bare imports itself.

**Requirement:** no Node in the Django build or runtime image, and no bundler in the
repo. Pin the CDN URL to an exact version — an unpinned CDN import is a supply-chain
risk and a silent-breakage risk. If we later vendor the file into `static/`, WhiteNoise
serves it and the dependency is fully self-hosted, which is the stronger end state for a
clinical demo.

A bundler would only become worth it for TypeScript authoring or production
minification — neither is needed for a single risk card.

### R3 — MCP tools stay pure; the agent composes A2UI

A2UI over MCP supports returning UI directly from a tool as an `EmbeddedResource` with
MIME type `application/a2ui+json`. **We should not do this for `predict_readmission`.**

Our MCP server is deliberately reusable — the same `predict_readmission` must work from
Claude Desktop, from CI, and from the agent. A tool that returns UI payloads is coupled
to one presentation layer and is no longer a clean prediction interface.

Therefore: **tools return structured JSON; the LangGraph agent translates that into
A2UI.** The `a2ui-agent-sdk` Python package (schema manager + validator) is the
supported way to do this without CopilotKit.

This preserves the Tier 1 acceptance criterion — the tool returns `0.1314` and a schema
that a non-UI client can consume.

### R4 — Pin the spec version and isolate it behind an adapter

The project is in **early public preview** and says plainly: "Expect changes." Four
versions are live — v0.8 legacy, v0.9 stable, **v0.9.1 current**, v1.0 release
candidate. v1.0 introduces breaking renames (`theme` → `surfaceProperties`) and adds
client-to-server RPC.

Target **v0.9.1**, pin it explicitly, and confine payload construction to a single
module so a version bump is one file, not a rewrite. Re-evaluate at Step 4 — if v1.0
has shipped by then, start there instead and avoid a migration.

### R5 — Define a small clinical catalog

The catalog is the security boundary and the thing the agent is prompted against. Start
with `includeBasicCatalog` plus a minimal set:

- **RiskCard** — probability, threshold, decision
- **FactorList** — parent-aggregated SHAP contributions with direction
- **CitationList** — retrieved note passages with `hadm_id` provenance (Step 6)

Component descriptions are injected into the agent's prompt, so they are prompt surface
and must be written as carefully as tool descriptions.

### R6 — Stage adoption exactly as planned

Step 5 emits a **single fixed component** (RiskCard) from a known-good payload. This
proves the render path end-to-end without the agent choosing shapes. Step 6, once
`rag_search` exists, is where the agent selects among component types — which is the
point of A2UI and also where it can go wrong.

### R7 — Control LLM visibility of payloads

MCP resource annotations govern whether the model can read A2UI JSON on later turns:

| `audience` | Effect |
|---|---|
| _(empty)_ | Visible to both user and LLM |
| `["user"]` | Rendered to the user, hidden from LLM context |
| `["assistant"]` | Available for reasoning, not rendered |

Default to `["user"]` for rendered payloads. This cuts token cost and — more
importantly — stops the model from re-reading its own UI output and treating it as
evidence, which is a faithfulness risk given the Tier 2 criteria.

### R8 — Degrade gracefully

Always accompany an A2UI payload with plain `TextContent`. If the renderer fails, the
version mismatches, or the bundle does not load, the clinician still gets the
assessment as text. A demo that renders nothing is worse than a demo that renders
plainly.

## Out of scope

- Multi-client or collaborative surfaces (the only case that would need WebSockets and
  a Redis channel layer)
- Mobile renderers (Flutter, SwiftUI, Compose)
- A2A protocol transport — relevant only for multi-agent meshes, which we do not have
- Building a custom renderer

## Open questions — verify at Step 4

1. ~~Does Lit require a bundler?~~ ~~Does the A2UI Lit renderer load from an ESM CDN?~~
   **Both answered: yes, it works with no build step.** Verified 2026-07-30 by
   [a2ui_cdn_spike.html](../projects/agent-harness/spikes/a2ui_cdn_spike.html) — nine
   checks pass, including Lit dedupe, custom-element registration, and data binding.
   The exact working import map and the v0.8-vs-v0.9 payload trap are recorded in
   [BUILD_GUIDE.md §16](../projects/agent-harness/docs/BUILD_GUIDE.md#16-a2ui-rendering-layer).
2. Does `a2ui-agent-sdk` work cleanly with Gemini structured output via LangGraph, given
   that first-class LangGraph support is still on the roadmap?
3. Will v1.0 ship before Step 5? If so, target it directly.
4. Does the Lit bundle coexist with the existing CKEditor 5 assets without conflict?
5. Is progressive/incremental rendering worth it for a single card, or is one payload at
   the end simpler and sufficient for the demo?

## Decisions recorded

| Decision | Choice |
|---|---|
| Transport | Plain HTTP request/response; SSE only if streaming is wanted later. **Not WebSockets.** |
| Renderer | **Lit (Web Components)** mounted in a Django template |
| Frontend framework | **None** — no Next.js, no CopilotKit, no second service |
| Build tooling | **None** — verified: pinned ESM CDN / import map, no Node in the image |
| A2UI generation | In the **LangGraph agent** via `a2ui-agent-sdk`; MCP tools return plain JSON |
| Spec version | **spec v0.9** via npm `@a2ui/lit@0.10.2` `/v0_9` subpath (package version ≠ spec version), behind an adapter module |
| Fallback | Always emit text alongside the payload |
