# A2UI CDN spike — findings

**Date:** 2026-07-30
**Verdict:** **PASSED** — 9/9 checks, visually confirmed
**Artifact:** [a2ui_cdn_spike.html](../spikes/a2ui_cdn_spike.html)

## The question

Does the **A2UI Lit renderer** (`@a2ui/lit`, distinct from Lit core and carrying its own
dependency tree) load and render from an ESM CDN with **no Node and no bundler**?

This was the last open question in [a2ui_requirements.md](../../../docs/a2ui_requirements.md)
and the single largest risk in the build plan. Requirement **R2** — "no JS build step" —
depended on it. A negative result would have forced one of three worse options: vendor a
pre-built bundle, introduce Node + Vite into the Django image, or hand-write a component
against the A2UI schema.

## The answer

**Yes.** Nine stages pass: both packages resolve, Lit dedupes to a single instance,
`<a2ui-surface>` registers, `MessageProcessor` constructs, the v0.9 payload is accepted,
the DOM contains bound data, and Markdown renders as a real `<h2>`.

Rendered output, confirmed by screenshot:

> **Readmission risk — synthetic patient**
> Probability: 0.1314
> Decision at threshold 0.12: FLAGGED
> · prior inpatient days (increases risk)

R2 holds. No Node in the image. The Django template can mount A2UI directly.

## Verified package coordinates

The prior research doc recommended Lit but never recorded the **actual npm package
names** — which was the very first thing the spike needed. Recording them here so that
gap does not recur.

| Package | Version | Notes |
|---|---|---|
| `@a2ui/lit` | **0.10.2** | Renderer + `basicCatalog`; import the `/v0_9` subpath |
| `@a2ui/web_core` | **0.10.5** | `MessageProcessor`; import the `/v0_9` subpath |
| `@a2ui/markdown-it` | **0.1.0** | Independently versioned. **Not optional** — see below |
| `lit` | 3.2.1 | Peer, must be externalized |
| `zod` | **3.25.76** | Peer, must be ≥ 3.25 |
| `@lit/context` | 1.1.6 | Transitive; needed directly to provide the Markdown renderer |

Upstream repo is **`a2ui-project/a2ui`** (not `google/A2UI`), Apache-2.0, ~15.9k stars.

> **Package version ≠ spec version.** Package `0.10.x` implements **spec v0.9** via the
> `/v0_9` subpath. Our notes previously said "pin v0.9.1", which would send you hunting
> for a package that does not exist. Spec track: v0.8 legacy → v0.9 stable → v0.9.1
> current → v1.0 release candidate.

## The working import map

Copy verbatim. Every entry is load-bearing; each was added in response to an actual
failure.

```html
<script type="importmap">
{
  "imports": {
    "lit": "https://esm.sh/lit@3.2.1",
    "lit/": "https://esm.sh/lit@3.2.1/",
    "zod": "https://esm.sh/zod@3.25.76",
    "zod/": "https://esm.sh/zod@3.25.76/",
    "@lit/context": "https://esm.sh/@lit/context@1.1.6?external=lit",
    "@a2ui/markdown-it": "https://esm.sh/@a2ui/markdown-it@0.1.0",
    "@a2ui/web_core/v0_9": "https://esm.sh/@a2ui/web_core@0.10.5/v0_9?external=lit,zod",
    "@a2ui/lit/v0_9": "https://esm.sh/@a2ui/lit@0.10.2/v0_9?external=lit,zod"
  }
}
</script>
```

## Four failures, and what each teaches

The spike failed four times before passing. Each failure is a bug that would otherwise
have surfaced at §16 with the entire MCP + agent stack already built and a much larger
surface to blame.

### 1. `Failed to resolve module specifier "zod/v3"`

Two causes stacked. A bare-name import map entry does **not** cover subpaths — a
trailing-slash `"zod/"` entry is required. And zod must be **≥ 3.25**, the release that
introduced the `zod/v3` subpath. Verified directly: `esm.sh/zod@3.25.76/v3` → 200,
`esm.sh/zod@3.24.1/v3` → **404**. Same reasoning applies to `"lit/"`, since the renderer
imports `lit/decorators.js` and `lit/static-html.js`.

### 2. `net::ERR_CONNECTION_CLOSED` on a cold esm.sh artifact

Transient, not a code defect — esm.sh builds artifacts on first request and the first hit
died mid-build. Warmed with three curls (all 200). **But the page did not boot when it
happened**, which is the real lesson. See "Vendor before production" below.

### 3. v0.8 payload shape sent to a v0.9 renderer — *the expensive one*

Symptom: console warning `Component implementation not found for type: [object Object]`
and a silently empty box. **No exception thrown.** You can stare at a blank card with a
clean-looking console for a long time.

In **v0.9**, `component` is a **string** and properties sit **inline**:

```js
// v0.9 — correct
{ id: 'root',  component: 'Card',   child: 'card-body' }
{ id: 'title', component: 'Text',   text: 'Readmission risk', variant: 'h2' }
{ id: 'prob',  component: 'Text',   text: { path: '/probability' } }

// v0.8 — silently rejected
{ id: 'title', component: { Text: { text: { literalString: 'Readmission risk' } } } }
```

Three further v0.9 differences, all of which bit:

- `updateDataModel` is a plain `path`/`value` upsert. The v0.8 `contents` adjacency list
  of typed `valueString`/`valueNumber` entries is gone
- `createSurface` takes **no `root` property** — the component whose `id` is literally
  `"root"` *is* the tree root
- `Text` uses `variant` (`h1`–`h5`, `caption`), not v0.8's `usageHint`

This matters disproportionately because **most A2UI examples findable online are v0.8**,
and the two versions fail against each other without erroring.

### 4. False-negative assertion via `shadowRoot.textContent`

The card *had* rendered, but the test reported empty. Each A2UI component is its own
custom element with its **own** shadow root, so text does not live in the surface's
shadow root. Any DOM assertion needs recursive piercing:

```js
function deepText(node) {
  let out = '';
  if (node.shadowRoot) out += deepText(node.shadowRoot);
  for (const child of node.childNodes) {
    out += child.nodeType === Node.TEXT_NODE ? child.textContent : deepText(child);
  }
  return out;
}
```

Worth remembering for §13 acceptance tests — a naive selector will report failure
against working UI.

## Markdown is mandatory, not cosmetic

A2UI `Text` properties are **Markdown**, and `variant: 'h2'` is implemented by literally
prepending `## `. With no Markdown renderer wired up, the user sees raw `## Heading` on
screen.

The renderer is injected through **Lit context**, not a global setter — so a
`ContextProvider` must sit **above** `<a2ui-surface>` in the DOM:

```js
const { basicCatalog, Context } = await import('@a2ui/lit/v0_9');
const { ContextProvider } = await import('@lit/context');
const { renderMarkdown } = await import('@a2ui/markdown-it');

new ContextProvider(host, { context: Context.markdown, initialValue: renderMarkdown });
```

## Duplicate-Lit risk: investigated and cleared

Two copies of Lit is the classic silent CDN failure for web components — two
`CustomElementRegistry` attempts, two `ReactiveElement` base classes, components never
upgrade, no error. Mitigated by `?external=lit,zod` (encodes as path segment
`X-ZWxpdCx6b2Q`); the compiled `.mjs` then imports **bare** `"lit"`, resolved once by the
import map.

`@lit/context@1.1.6` was the one suspicious transitive — unpinned `^1.1.6` and not
externalized. Inspected the module body directly: **5,353 bytes, zero import specifiers,
zero `LitElement`/`ReactiveElement` references.** Standalone and duck-typed. No second
Lit copy. Stage 3 of the spike asserts identity (`litA === litB`) so a regression here
fails loudly rather than silently.

## Rendering pattern

```js
processor.onSurfaceCreated(s => {
  const el = document.createElement('a2ui-surface');
  el.surface = s;                 // PROPERTY, not attribute
  host.replaceChildren(el);
});
processor.processMessages(messages);
```

Message order: `createSurface` → `updateComponents` → `updateDataModel`.

## Vendor before production

The spike uses esm.sh, which is correct for local iteration. **Vendor the pinned files
into `static/` before the demo goes live.**

Two reasons, one of them observed firsthand: esm.sh returned `ERR_CONNECTION_CLOSED`
mid-spike and the page failed to boot. That is a third-party outage sitting directly in
the render path of a demo that may be shown to an employer. Vendoring also removes the
supply-chain exposure of executing CDN-served JavaScript, and WhiteNoise already serves
`static/` with hashing.

## Side finding: A2UI has an official MCP guide

[a2ui.org/guides/a2ui_over_mcp](https://a2ui.org/guides/a2ui_over_mcp/) — an MCP server
can return A2UI **directly** as an `EmbeddedResource` with MIME type
`application/a2ui+json`. It also documents the `a2ui_action` round-trip for user
interaction and the `audience: ["user"]` annotation.

This is a sanctioned alternative to our §6 decision (MCP tool returns plain JSON, agent
composes A2UI). **Recommendation: keep the current decision.** A JSON-pure
`predict_readmission` stays usable from Claude Desktop, from CI, and from any future
non-A2UI client; binding the tool's output to a UI protocol would couple the model
service to the presentation layer. But the alternative is now documented rather than
merely unconsidered — useful to be able to discuss.

## Where this landed

| Change | File |
|---|---|
| §16 rewritten against verified behaviour | [BUILD_GUIDE.md](BUILD_GUIDE.md) |
| Open question 1 closed; package coordinates recorded | [a2ui_requirements.md](../../../docs/a2ui_requirements.md) |
| Runnable spike | [a2ui_cdn_spike.html](../spikes/a2ui_cdn_spike.html) |

Re-run any time — from the repo root:

```bash
python3 -m http.server 8777
# then open http://localhost:8777/projects/agent-harness/spikes/a2ui_cdn_spike.html
```

Machine-readable result is left on `window.__SPIKE_RESULT__` for later automation.

## Resume point

Spike work is **closed**. Remaining plan work:

- [ ] **§17 expansion** — Django template integration, `/projects/clinical_copilot/demo`
      route, login gate, cohort search + advanced `hadm_id` field, BFF post, surface
      mount, synthetic-name disclaimer, model-version footer. Ordinary Django wiring now
      that the render path is proven; no unknowns blocking it
- [ ] **§3 — deploy the Vertex endpoint** (`deploy_cpr.py`), verify with
      `smoke_test.py 20924467`, expect ≈ **0.1314**. This is the natural first build step
- [ ] **§4 — teardown script**, written immediately after §3 so the endpoint is never
      left billing

Carry-forward pitfalls already noted in the guide: parameterize the expected probability
in Tier 1 tests from `tests/fixtures/expected.json` rather than hardcoding `0.1314` (it
changes at §20); SSE views need **async** generators under ASGI; ID token audience is the
Cloud Run **service URL**.
