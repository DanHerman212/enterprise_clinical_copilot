# Phase 5: Frontend Islands & Shadow-DOM Boundary Modularization

**Status:** `COMPLETE`  
**Target Layer:** Layer 1 (`docs/architecture/01-client.md`)  
**Repositories in Scope:** `danielmherman`  
**Objective:** Decompose the monolithic `static/js/copilot_canvas.js` (~1,040 lines) into focused, single-responsibility scripts and replace manual recursive Shadow-DOM piercing (`inlineMarker` / `revealMarker`) with an explicit boundary contract on `a2ui-cited-narrative`.

---

## 0. Governance Protocol for This Spoke

* **Step A — The Audit Pass (Read-Only):**
  * Inspect `static/js/copilot_canvas.js`, `static/js/copilot_composer.js`, `static/js/copilot_app.js`, `static/js/copilot_shell.js`, `static/js/a2ui_clinical_components.js`, and `demo/templates/demo/a2ui_console.html` against `docs/architecture/01-client.md`.
  * Record exact line ranges, shared state couplings, test assertions in `demo/tests.py`, and a surgical modularization plan in **Section 2 (Step A — Audit Findings)**.
  * **STOP.** Do NOT modify any application or test code during Step A. Await explicit Architect approval.
* **Step B — The Execution Pass (Write + Verify):**
  * After Architect approval of Step A, execute the approved changes, rebuild the clinical bundle if `a2ui_clinical_components.js` is touched, run all verification gates, and record the results in **Section 3 (Step B — Execution Record)**.

---

## 1. Step A Audit Scope & Questions to Answer

### 1.1 Decompose `static/js/copilot_canvas.js` (~1,040 Lines)
* **Architectural Standard (`01-client.md`):** Each vanilla JS island on the console page owns a narrow, well-defined slice of UI state. Currently, `static/js/copilot_canvas.js` combines at least five distinct responsibilities in one IIFE:
  1. **Streaming Cadence & SSE Frame Consumer:** `PACING`, `drawn`, `drain`, `appendProse`, `consume`.
  2. **TSV Table Export:** `structuredRows`, `percent`, and blob download wiring.
  3. **Turn DOM Lifecycle & Chrome:** `createTurn`, `fillTurn`, `presentFacts`, `presentReferences`, `presentFollowUps`, `postFeedback`, rating/copy buttons.
  4. **A2UI Scene Loader & Surface Mounter:** `scene`, `proseMarkdown`, `mountSurface`.
  5. **Sidebar Thread List, Context Bar & Session Persistence:** `loadThreads`, `renderThreads`, `threadRow`, `openRowMenu`, `confirmDelete`, `threadAction`, `openConversation`, `restoreConversation`, `syncToPatient`, `newConversation`, `updateContext`, `rememberConversation`.
* **Audit Tasks:**
  1. Map `static/js/copilot_canvas.js` by exact line range and trace the shared closure variables (`conversationId`, `loadedHadm`, `threadEntries`, `declaredIntent`, `turnOrdinal`, etc.) that cross between the feed/turn lifecycle and the sidebar thread manager.
  2. Inspect `demo/tests.py` (and any other frontend tests in `demo/test_*.py`) to identify every test that reads `static/js/copilot_canvas.js` as text or asserts on script tags in `a2ui_console.html`.
  3. Propose a clean split (either extracting pure/self-contained modules such as TSV export, SSE/pacing, or the sidebar thread manager into companion scripts loaded by `a2ui_console.html`, or ES module imports) that minimizes shared-state coupling and keeps all template/source tests green.

### 1.2 Replace Recursive Shadow-DOM Piercing (`inlineMarker` / `revealMarker`)
* **Architectural Standard (`01-client.md`):** The host page and the Lit/A2UI Web Components (`a2ui_clinical_components.js`) communicate via props and CustomEvents/public methods—never by the host page recursively walking `shadowRoot` trees or mutating inline styles on internal shadow nodes.
* **Problem:** In `static/js/copilot_canvas.js`, `inlineMarker(node, index)` recursively walks `node.querySelectorAll('*')` and pierces every `.shadowRoot` looking for `button.cite[data-cite]`, and `revealMarker(index)` directly mutates `marker.style.outline` and `marker.style.outlineOffset` from outside the component.
* **Audit Tasks:**
  1. Inspect `inlineMarker` and `revealMarker` in `static/js/copilot_canvas.js` and `CitedNarrativeElement` (`a2ui-cited-narrative`) in `static/js/a2ui_clinical_components.js`.
  2. Propose a clean public method or event contract on `CitedNarrativeElement` (e.g., `element.revealCitation(index)` owning its own scroll and CSS class/highlight state inside its shadow stylesheet) so `copilot_canvas.js` never walks arbitrary `.shadowRoot` trees or writes inline styles into shadow DOM nodes.

---

## 2. Step A — Audit Findings & Proposed Surgical Plan

Read-only pass. No application or test code modified. Line numbers against `danielmherman` at `554925b`.

### 2.0 Summary

| § | Question | Verdict |
| :--- | :--- | :--- |
| 1.1 | Is `copilot_canvas.js` still ~1,040 lines? | **No — 1,989.** The Spoke's figure is from before the streaming work; the file has roughly doubled. |
| 1.1 | Are the five responsibilities really one closure? | **Yes, plus a sixth (boot and submit wiring), and the largest single block is one function.** `createTurn` is 672 of the 1,989 lines. |
| 1.1 | Do the tests block a split? | **Yes, and only because of how they read source.** `test_ux_restoration.py` reads `copilot_canvas.js` as text and asserts marker strings and even its indentation. That is one `setUp` line away from being fixable. |
| 1.2 | How deep does the Shadow-DOM piercing go? | **Two levels, both walking.** The A2UI surface renders components into *its* shadow root, and the narrative renders markers into *its* own — so `inlineMarker` recurses through arbitrary hosts to reach a button the component already owns. |
| 1.2 | Does the component already offer a contract to build on? | **Yes.** `citation-popover` exposes `open`/`toggle`/`close`, and the narrative's own stylesheet already draws the highlight the canvas writes inline. |

---

### 2.1 Track 1.1 — decomposing `copilot_canvas.js`

#### 2.1.1 By line range

| Range | Responsibility |
| :--- | :--- |
| 1–37 | file docstring (the design argument for the pacing and the stream) |
| 38–70 | IIFE open; 18 DOM `getElementById` references; 4 `data-*` config attributes |
| 72–108 | closure state: `declaredIntent`, `turnOrdinal`, `conversationId`, `loadedHadm`, `threadEntries`, `CONVERSATION_KEY` |
| 109–137 | session persistence: `rememberConversation`, `storedConversation` |
| 133–172 | utilities: `conversationUrl`, `csrfToken`, `el`, `scrollIntoView` |
| 173–253 | composer bridge: `askedFor`, `offeredCommands`, `commandForIntent` — re-reads `#cp-command-data`, which `copilot_composer.js:215-229` also parses |
| 228 | `PACING` (reading cadence constants) |
| 254–375 | turn chrome: `postFeedback`, `markChosen`, `iconButton`, `ratingButton`, `flash`, `copyButton`, `copyText` |
| **376–477** | **TSV export: `percent`, `structuredRows`** (102 lines, pure) |
| 478–488 | `askFollowUp` (module scope, by test) |
| **489–1160** | **`createTurn` (672 lines)** — the turn DOM, pacing (`drawn` 841, `drain` 872, `appendProse` 912), references drawer (`inlineMarker` 975, `revealMarker` 992, `referenceRow` 1018), `presentFacts` 1115, and the returned turn API 1146–1159 |
| 1162–1346 | A2UI scene and stream consumer: `turns` 1162, `scene` 1168, `proseMarkdown` 1199, `mountSurface` 1210, `fillTurn` 1234, `consume` 1264 |
| 1347–1417 | context bar: `updateContext`, `clearContext`, `patientName`, `setPatient`, `entryForHadm` |
| **1400–1753** | **sidebar thread manager** (354 lines): `loadThreads` 1400, `renderThreads` 1418, `threadRow` 1450, row menu 1509–1588, `confirmDelete` 1589, `threadAction` 1611, `setPinned` 1634, `deleteThread` 1638, `markActiveThread` 1649, `setSearchOpen` 1660, `clearFeed` 1674, `restoreConversation` 1682, `openConversation` 1700, `syncToPatient` 1731 |
| 1754–1989 | `newConversation` 1754, `send` 1769 (193 lines), submit/search/boot wiring 1962–1989 |

#### 2.1.2 The shared state, and who touches it

| Variable | Declared | Feed / turn side | Sidebar side |
| :--- | :--- | :--- | :--- |
| `conversationId` | 105 | via `rememberConversation` after a turn | read in `openConversation`, `send`, `restoreConversation` |
| `loadedHadm` | 106 | `send` sets it | `restoreConversation`, `openConversation`, `syncToPatient`, `newConversation` |
| `threadEntries` | 117 | — | `loadThreads`, `renderThreads`, `entryForHadm` |
| `turnOrdinal` | 91 | `createTurn` mints ids for ratings | — |
| `declaredIntent` | 80 | `send` consumes it | — |
| `renderer`, `rendererPromise`, `inFlight` | 68–70 | `scene`, `consume`, `send` | — |
| `feed` (DOM) | 43 | `createTurn`, `turns`, `clearFeed` | `clearFeed`, `restoreConversation` |

**Three functions are the whole seam.** `restoreConversation` (1682) calls `clearFeed`, then `createTurn` + `fillTurn` per stored turn, then `setPatient` and `updateContext`. `syncToPatient` (1731) and `newConversation` (1754) do the same in reverse to empty the view. `send` (1769) calls `loadThreads` after storing a turn, so the sidebar's list is refreshed by the feed.

So the coupling is real but narrow: **one direction drives the feed (restore/clear), the other triggers a sidebar refresh (a turn was stored).** That is an interface, not a tangle — and it is the reason the split below is two files and not five.

#### 2.1.3 What the tests pin

| Assertion | Where | Effect on a split |
| :--- | :--- | :--- |
| `self.canvas = (STATIC / 'copilot_canvas.js').read_text()` | `test_ux_restoration.py:111` | Every assertion below reads **this one file**. No split survives it unchanged. |
| `'cp-turn-toolbar'`, `'fa-copy'`, `'fa-download'`, `'fa-thumbs-up'`, `'fa-thumbs-down'`, `'cp-turn-actions'` | `:115-117` | chrome stays somewhere the reader can see |
| `'data-feedback-url'`, `'keepalive: true'` | `:120-121` | `postFeedback` location |
| `'cp-stages'`, `'function stageHistory'`, `'stages.push(label)'` | `:125-126` | `stageHistory` is inside `createTurn` (733) |
| `self.canvas.count("addEventListener('a2ui-follow-up'") == 1` | `:140` | exactly one listener, module scope |
| `'\n  function askFollowUp('` | `:143` | **two-space module-scope indentation**, asserted as a literal — the test's own comment records that a nested definition broke the click path once |
| `'declaredIntent'`, `'askFollowUp(detail.question, [], detail.intent)'` | `:141-142` | the intent bridge |
| `'payload.verified === true'`, `'Verified against the chart'`, `'Reconciled with medical record'` | `:147-150` | the badge copy |
| `.replace(/^[ \t]*#{1,6}[ \t]+/gm` | `:153` | the markdown heading rule in `proseMarkdown` |
| `assertContains(response, 'copilot_composer.js' / 'copilot_canvas.js' / 'copilot_shell.js' / 'copilot_app.js')` | `tests.py:858, 888, 907, 908` | a new script needs its own assertion, or the template could drop it and nothing fails |
| four retired script names absent; `"'SourceCard'" not in a2ui_clinical_components.js` | `tests.py:1035-1040` | unrelated to this split, unaffected by it |

The indentation assertion is the one to respect rather than work around: it exists because a nested `askFollowUp` silently threw on every click. Any new file that takes module-scope functions must use the same two-space convention, and its own test must assert it there.

#### 2.1.4 Proposed split — two files, in two stages

**Stage 1 — break the test coupling, then extract the one pure block.**

1. `demo/test_ux_restoration.py:111` reads the console's source **set** rather than one file: `self.canvas = ''.join(p.read_text() for p in sorted(STATIC.glob('copilot_*.js')))`. Every existing assertion keeps working, and it stops mattering which file a marker lives in — which is the property the split needs and which the test should have had anyway. One assertion must move with its subject: `count("addEventListener('a2ui-follow-up'") == 1` stays meaningful across the set.
2. Extract `percent` + `structuredRows` (376–477) into **`static/js/copilot_tsv.js`** — 102 lines, pure, takes the surface object and returns text. It is the only block in the file with no DOM, no closure state and no async.

**Stage 2 — extract the sidebar behind an explicit contract.**

3. Move 1400–1753 into **`static/js/copilot_threads.js`**, keeping `conversationId`, `loadedHadm`, `threadEntries`, `CONVERSATION_KEY` and the sidebar DOM with it. The feed keeps the turn lifecycle and the context bar.

   The contract, in both directions and nothing else:

   | Direction | Mechanism | Today's call |
   | :--- | :--- | :--- |
   | threads → feed | `CopilotFeed.restore(turns)` / `CopilotFeed.clear()` | `restoreConversation` → `createTurn`+`fillTurn`; `clearFeed` |
   | threads → feed | `CopilotFeed.patient(name, hadmId)`, `CopilotFeed.context(expiresAt)` | `setPatient`, `updateContext` |
   | feed → threads | `document.dispatchEvent(new CustomEvent('cp-turn-stored'))`, `'cp-thread-changed'` | `send` → `loadThreads`; pin/delete → `loadThreads` |

   `newConversation` and `syncToPatient` stay with the threads module: both are thread selection, and both drive the feed through the two calls above.

**Resulting sizes.** Stage 1: 1,989 → ~1,890. Stage 2: → ~1,530. With §2.2's reveal contract: ~1,495.

#### 2.1.5 What should not be split, and why the file stays large

`createTurn` is 672 lines and holds the turn's DOM, its prose buffer, its pacing ticker, its stages, its drawer and its badge. Extracting any part of it means re-passing six or seven pieces of per-turn state through a new boundary on every frame — a longer contract than the function it replaces. **The honest fix for its size is decomposition inside the function, not another file**, and that cannot be verified here: this repository has no DOM harness, and every frontend test in it reads source text. Recorded as a decision rather than assumed away.

---

### 2.2 Track 1.2 — the reveal contract

#### 2.2.1 What the code does today

| Site | Lines | Behaviour |
| :--- | :--- | :--- |
| `inlineMarker(node, index)` | 975–990 | `node.querySelectorAll('button.cite[data-cite]')`, then `querySelectorAll('*')` and **recurse into every `.shadowRoot` it finds** until a match or exhaustion |
| `revealMarker(index)` | 992–1006 | finds the marker, scrolls it, then writes `marker.style.outline` and `outlineOffset` inline, and clears them after 1,400 ms |
| call site | 1058 | the drawer row's toggle calls it when a row opens |
| comment at 971–974 | — | states the reason: "no selector crosses a shadow boundary, and the marker the reader clicked and the row they opened have to be the same record" |

**How deep it actually is.** `mountSurface` (1210) appends `<a2ui-surface>` to the turn's canvas host. The surface renders through Lit (`node_modules/@a2ui/lit/src/v0_9/surface/a2ui-surface.js:156-171`), which renders into **its own shadow root** — so `a2ui-cited-narrative` is not a light-DOM descendant of the turn root. The narrative then builds `button.cite[data-cite]` into *its* shadow root (`_marker`, 765-812). Two boundaries, and the recursion walks all of them because it cannot know which.

#### 2.2.2 What the component already owns

`CitedNarrativeElement` (`a2ui_clinical_components.js:637-835`) already:

- creates the markers (`_marker` 765), including `aria-haspopup`/`aria-expanded`;
- owns a delegated click listener on its own shadow root (`_onMarkerClick` 818-828), wired once in `updated()`;
- owns the popover, whose `open`/`toggle`/`close` (473/498/513) are the precedent for a public method API;
- **already styles the exact highlight**: `NARRATIVE_STYLES` has `.cite:focus-visible { outline: 2px solid #1f2937; outline-offset: 1px }` (375). The canvas writes the same colour inline, at 2px offset instead of 1px — a duplicate that already disagrees.

So the reveal belongs where the marker does. Nothing in the contract has to be invented except the entry point.

#### 2.2.3 Proposed contract

**On `CitedNarrativeElement`:**

```js
/** Highlight the citation marker for `index` and bring it into view. */
revealCitation(index) {
  // `data-cite` holds the marker's digits as written by `_marker`, so the
  // selector needs no escaping — it is a number, not reader text.
  const marker = this.renderRoot.querySelector(
    `button.cite[data-cite="${String(index)}"]`);
  if (!marker) return false;                 // inert while streaming, or absent
  marker.scrollIntoView({ block: 'center',
                          behavior: matchMedia('(prefers-reduced-motion: reduce)').matches
                            ? 'auto' : 'smooth' });
  marker.classList.add('is-target');
  clearTimeout(this._revealTimer);
  this._revealTimer = setTimeout(() => marker.classList.remove('is-target'), 1400);
  return true;
}
```

plus, in `NARRATIVE_STYLES`:

```css
.cite.is-target { outline: 2px solid #1f2937; outline-offset: 1px; }
```

The 1,400 ms and the dismissal move inside the component with the class, so the timer is cancelled per element rather than racing a detached node.

**On the canvas side**, the element reference is obtained once, at mount, without a query whose depth depends on the renderer's internals:

- `createTurn` listens on its own `root` for `a2ui-cited-narrative-ready` (`{ bubbles: true, composed: true, detail: { element } }`), dispatched by the component on first update;
- `revealMarker(index)` becomes `narrative?.revealCitation(index)`;
- `inlineMarker` is deleted.

`composed: true` is what lets the announcement cross the surface's shadow boundary; nothing walks downward, and the canvas never names the renderer's element structure. Both no-op cases stay no-ops: a prose-only turn never announces, and a streaming turn's markers are inert `<span class="cite inert">` (777-787), so the lookup inside the component finds no button.

#### 2.2.4 Consequences to check in Step B

- `a2ui_clinical_components.js` is **bundled** (`a2ui_clinical.js` → `vite.clinical.config.js`, output `static/js/bundled/`, gitignored), so the change requires `npm run build:clinical`, and the template's `data-renderer-url="…?v=14"` (`a2ui_console.html:263`) must bump or a browser serves the old component.
- The canvas's own `?v=35` (`a2ui_console.html:321`) must bump for the same reason.
- No Python test asserts `inlineMarker`/`revealMarker` today, so nothing fails when they go. New source assertions are proposed instead, in the shape this repository already uses for browser behaviour it cannot run: the canvas contains no `shadowRoot`; the component defines `revealCitation`; `NARRATIVE_STYLES` contains `.cite.is-target`; the announcement is dispatched with `composed: true`.

---

### 2.3 Doc-vs-code mismatches recorded during the audit

| Spoke says | Actual |
| :--- | :--- |
| `copilot_canvas.js` is "~1,040 lines" (title and §1.1) | 1,989 |
| five responsibilities (§1.1) | six: the boot/submit/search wiring at 1962–1989 is a seventh block and belongs to none of the five |
| `PACING`, `drawn`, `drain`, `appendProse`, `consume` are one responsibility | they are, but `drawn`/`drain`/`appendProse` are *inside* `createTurn`, not at module scope — a split has to carry the whole function or none of it |
| `a2ui-cited-narrative` is reachable as a light-DOM element | it is not: the surface renders into its own shadow root (`@a2ui/lit` `a2ui-surface.js:156-171`) |

---

### 2.4 Proposed Step B order

1. Test decoupling (`test_ux_restoration.py` reads the source set) — no behaviour change, and it makes the rest reviewable.
2. `copilot_tsv.js` extraction + its template script tag + its `assertContains` line in `tests.py`.
3. `copilot_threads.js` extraction behind the contract in §2.1.4.
4. The §2.2 reveal contract: component method + style + announcement, canvas deletion of both functions, bundle rebuild, both `?v=` bumps.
5. Source assertions for each new boundary; §3 record; tracker update.

Steps 1–3 touch only `danielmherman`. Step 4 touches the component bundle, so it is the one that needs the build.

---

### 2.5 Decisions for the Architect

| # | Decision | Recommendation |
| :--- | :--- | :--- |
| 1 | Split depth: the two-stage split above (~1,495 lines), or canvas-only, or decompose `createTurn` internally | **Two stages.** It removes the two cohesive, testable blocks; `createTurn`'s size is a separate problem that needs a DOM harness to do safely. |
| 2 | Change `test_ux_restoration.py` to read the console's source set | **Yes, first.** Without it no split can be reviewed on design; with it the tests get stronger, because they stop depending on which file a marker lives in. |
| 3 | New scripts as classic `<script>` with one frozen namespace each (`window.CopilotTSV`, `window.CopilotThreads`) vs ES modules | **Classic + one namespace per file.** The page loads four classic scripts today and the only module on it is the A2UI bundle; making these modules changes load ordering, the `fetch`-free boot path and every `assertContains` in `tests.py` at once. Cost: the codebase currently has no `window.X =` globals at all, and this introduces two, deliberately bounded. |
| 4 | Reveal contract: announcement event + `revealCitation` (recommended), vs one deliberate shadow hop from the canvas to the surface | **The event.** A single hop is less code but still encodes the renderer's internal structure in the page's own script — which is the thing the layer document forbids. |
| 5 | Whether the 1,400 ms highlight duration should move into the component | **Yes, with the class.** A timer that outlives its node is the kind of leak the current code cannot prevent, because the node is not the component's. |


## 3. Step B — Execution Record

All five §2.5 decisions approved as proposed; steps 1–5 executed in the §2.4 order.
Every change is in `danielmherman`; `enterprise_clinical_copilot` is untouched.

### 3.1 Executed Changes

| Step | File | Change |
| :--- | :--- | :--- |
| 1 | `demo/test_ux_restoration.py` | `setUp` reads the console's source **set** (`copilot_*.js`) instead of one file. Name kept (`self.canvas`) so all ten existing assertions are unchanged. |
| 2 | `static/js/copilot_tsv.js` | **New** (121 lines): `percent`, `structuredRows`, `window.CopilotTSV` frozen with two members. |
| 2 | `static/js/copilot_canvas.js` | 102 lines removed; both call sites read `window.CopilotTSV.structuredRows`. |
| 3 | `static/js/copilot_threads.js` | **New** (507 lines): the sidebar, row menu, thread actions, restore/open/sync/empty, `conversationId`/`loadedHadm`/`threadEntries`, session persistence, and the sidebar's own wiring and boot. `window.CopilotThreads` frozen with seven members. |
| 3 | `static/js/copilot_canvas.js` | ~370 lines removed; `window.CopilotFeed` frozen with twelve members; the sidebar wiring and the boot moved out; five seam calls rewritten. |
| 4 | `static/js/a2ui_clinical_components.js` | `revealCitation(index)`, `.cite.is-target` in `NARRATIVE_STYLES`, `_revealTimer`, and the `a2ui-cited-narrative-ready` announcement (`bubbles` + `composed`). |
| 4 | `static/js/copilot_canvas.js` | `inlineMarker` **deleted**; `revealMarker` reduced to one call into the stored element reference, obtained from the announcement. |
| 4 | `static/js/bundled/a2ui_clinical.js` | Rebuilt with `npm run build:clinical` (803 kB, gitignored — the build is the artefact). |
| 4 | `demo/templates/demo/a2ui_console.html` | Two script tags added before the canvas; `copilot_canvas.js?v=35 → 36`, renderer `?v=14 → 15`. |
| 5 | `demo/test_frontend_boundaries.py` | **New** (10 tests): both namespace directions, lazy handle, the frozen TSV surface, no `shadowRoot`, no inline marker styles, the component owning the reveal and its timer, the announcement crossing both boundaries, load order, and module-scope indentation. |
| 5 | `demo/tests.py` | Two `assertContains` lines for the new script tags. |

**Sizes:** `copilot_canvas.js` 1,989 → **1,464** (−525, −26%). With the two new files the
console's script total went 2,586 → 2,092.

### 3.2 What the run found that the audit did not

1. **A multi-line `{# … #}` comment is not a Django comment.** The template test
   (`test_the_console_says_so`) caught it precisely as designed: braces rendered as
   page text. Rewritten as two single-line comments.
2. **Backticks inside `NARRATIVE_STYLES`.** The stylesheet is a `css` template literal,
   so a comment quoting `` `revealCitation` `` terminated the string. Caught by
   `node --check` before anything ran; the comment now names the method without
   quoting it.
3. **`thisTurnId()` read `conversationId` directly** — a page-level call into thread
   state that neither the audit's seam list nor the call census found, because it is
   inside `createTurn` rather than in the sidebar. It now reads
   `window.CopilotThreads.conversationId()` at click time, which is also more correct:
   a rating is given after the turn, by which point the thread may have moved on.
4. **`clearFeed` was defined in the moved block and called only from it.** All three
   call sites became `canvas().clear()` + the local `closeRowMenu()`, so the definition
   was deleted rather than moved.

### 3.3 Verification, and its limit

| Gate | Result |
| :--- | :--- |
| `pytest tests` (`enterprise_clinical_copilot`) | **1377 passed, 1 skipped** — unaffected, as expected |
| `python manage.py test` (`danielmherman`, `ENVIRONMENT=development`) | **256 tests OK** (was 246; +10 boundary tests) |
| `python manage.py check --deploy` (`ENVIRONMENT=production`) | **System check identified no issues (2 silenced)** |

**This is the first phase whose deliverable is browser code that no gate executes.**
Three things were verified, and one was not:

- `node --check` on all four scripts (the components file checked as a module).
- The bundle rebuilt, and the rebuilt file contains `revealCitation` and the
  `a2ui-cited-narrative-ready` dispatch.
- The seams are asserted from source in both directions, which is what catches a
  rename, a member missing from a namespace, or a script tag in the wrong order.

**Not verified: that the page behaves.** There is no DOM harness in this repository
and no browser in the loop, so a logic error inside a moved function would pass all
three gates. The mitigations were mechanical: blocks were sliced verbatim rather than
rewritten, the seam is twelve named members in one frozen object, and every crossing
call is enumerated in a test. The honest next step for this file's remaining size
(`createTurn` is still 672 lines) is a DOM harness — without one, further splitting is
unreviewable.

### 3.4 Behaviour changes

1. **The highlight is a class, not inline styles, and its offset is now 1px.** It
   matches `.cite:focus-visible` in the same stylesheet; the old inline value was 2px,
   i.e. two definitions that had already disagreed.
2. **A drawer row opened on a streaming turn no longer scrolls to a marker.** Inert
   markers are spans, so there is nothing to reveal — the old code returned early for
   the same reason. No change in outcome, but the path is now explicit.
3. Nothing else: same requests, same rendering, same DOM ids, same copy.

### 3.5 Deliberately not done

- **`createTurn` was not decomposed**, per Decision #1. It is 672 lines and its parts
  share eight pieces of per-turn state; splitting it without a DOM harness would be
  unverifiable.
- **No ES modules.** The page loads classic scripts, and the only module on it is the
  A2UI bundle.
- **The `?v=` bumps were manual**, as they were before. A build-time hash would be
  better and is out of scope here.
