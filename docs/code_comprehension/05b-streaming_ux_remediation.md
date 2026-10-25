# Phase 5B: Universal Canvas Streaming & Reveal UX Remediation

**Status:** `COMPLETE`  
**Target Layers:** Layer 1 (`docs/architecture/01-client.md`), Layer 3 (`03-orchestrator.md`), & Layer 4 (`04-model_runtime.md`)  
**Repositories in Scope:** `danielmherman` (primary) & `enterprise_clinical_copilot` (verify SSE `answer_delta` emission across all slash commands and free-text turns)  
**Objective:** Remediate the turn-streaming and canvas-reveal UX across **all user inputs** (all 5 slash commands `/ehr`, `/meds`, `/labs`, `/risk`, `/pubmed`, follow-up chips, and free-text questions). Replace the broken 45-char/sec plain-text trickle + abrupt DOM-swap hack with a unified, OpenEvidence-style progressive reveal ("water filling a glass") across the entire canvas, backed by an executable JS/DOM verification gate.

---

## 0. Governance Protocol for This Spoke

* **Step A — The Audit Pass (Read-Only — CURRENT STEP):**
  * Trace the end-to-end streaming and canvas-mounting lifecycle across **all turn modalities** from `enterprise_clinical_copilot` (`deterministic.py`, `graph.py`, `model_runtime.py`, `turn.py`, `http.py`) through `danielmherman` (`views.py`, `copilot_canvas.js`, `a2ui_clinical_components.js`, `demo_shell.css`).
  * Record the exact root causes of the "2 sentences trickle, then everything dumps at once" defect and the light-DOM-to-Shadow-DOM swap across all turn types, and propose a unified architectural fix in **Section 2 (Step A — Audit Findings)**.
  * **STOP.** Do NOT modify any application or test code during Step A. Await explicit Architect approval.
* **Step B — The Execution Pass (Write + Verify — LOCKED UNTIL STEP A APPROVED):**
  * After Architect approval of Step A, execute the approved changes, verify them with an executable JS/DOM test (in addition to the 3 standard gates), and record the results in **Section 3 (Step B — Execution Record)**.

---

## 1. Step A Audit Scope & Questions to Answer

### 1.1 Audit Every Input Modality Across Backend & Frontend
* **Observed Symptom:** Across streamed turns, the first 1–2 sentences trickle out slowly in unformatted plain text, and then as soon as the `answer` SSE frame arrives, the remaining ~85% of the stream is discarded and the entire finished A2UI surface (`CitedNarrative`, `RiskInfluencers`, `MedicationGrid`, `LabTrendSparklines`, `ComparativeRiskMatrix`, `LiteratureCard`) slams onto the screen in a single abrupt frame.
* **Audit Tasks:**
  1. **Map All Turn Modalities (Backend → Frontend):**
     * Trace what SSE frames (`stage`, `answer_delta`, `answer`) are emitted for each input type:
       * Free-text ReAct turns (`graph.py` + `model_runtime.py` `streamed_turn`)
       * Prose-generating slash commands (`/ehr`, `/pubmed`, driver turns in `deterministic.py` `_formatting_turn`)
       * Surface-first / structured slash commands (`/meds`, `/labs`, `/risk` in `deterministic.py` and `risk_canvas.py`)
       * Follow-up chip clicks (`askFollowUp`)
  2. **Frontend Pacing vs. Cutover (`danielmherman/static/js/copilot_canvas.js`):**
     * Trace `PACING` (`charsPerSecond: 45`, `catchUpQueue: 350`, `tickMs: 40`), `releasePrefix`, `drawn()`, `drain()`, `appendProse()`, `consume()`, and `fillTurn()`.
     * Document how `drain()` builds up an undrawn backlog in `pending` during streaming, and how `fillTurn()` immediately calls `block.answer('')` -> `stopPacing()`, wiping `.cp-answer-text` and dumping `<a2ui-surface>` into `.cp-canvas` in one frame.
  3. **Double-Render / DOM-Swap & Unused `streaming` Prop (`copilot_canvas.js` vs. `a2ui_clinical_components.js`):**
     * Inspect `CitedNarrativeElement` (`a2ui_clinical_components.js`), which already declares `streaming: z.boolean().default(false)` and renders inert `.cite.inert` badges during streaming, yet is never mounted or updated during `answer_delta` frames.
     * Inspect how non-prose A2UI cards (`RiskInfluencers`, `MedicationGrid`, `LabTrendSparklines`, `ComparativeRiskMatrix`, `LiteratureCard`) mount inside `.cp-canvas` and whether they snap in abruptly without a staged reveal/fade transition.

### 1.2 Design the Universal "Water Filling a Glass" Canvas Architecture (OpenEvidence Standard)
* **UX Standard:** For **every input**, the canvas must fill smoothly and progressively from top to bottom—like water filling a glass:
  * **No 2-Sentence Choke + Abrupt Dump:** Streaming prose must never trickle out 10% of the answer and then snap the remaining 90% onto the page in a single frame when the `answer` event arrives. Either the visual stream flows at a natural, responsive cadence that tracks generation and smoothly finishes pouring (`await block.settle()`) before settling interactive citations, OR streams directly through the final formatted component without a jarring teardown.
  * **No Raw-Text-to-Formatted-Component Jump:** Eliminate the jarring visual swap between unformatted light-DOM plain text in `.cp-answer-text` and Shadow-DOM formatted Markdown in `<a2ui-cited-narrative>`. Headings, paragraphs, lists, and inert citation pills should render in their final typography from the start, with newly revealed blocks/clauses animating in with a soft CSS opacity/settle transition (`opacity: 0 -> 1`, subtle `translateY` settle).
  * **Graceful Card & Surface Entrance:** When visual A2UI cards (`RiskInfluencers`, `MedicationGrid`, `LabTrendSparklines`, `ComparativeRiskMatrix`, `LiteratureCard`, plus the verification badge, references drawer, and follow-up chips) mount, they should enter with a cohesive, choreographed CSS fade/settle rather than popping in with a hard layout jolt.
* **Audit Tasks:**
  1. Evaluate and propose the cleanest unified architecture across `copilot_canvas.js`, `a2ui_clinical_components.js`, and `demo_shell.css` so **every** turn type (free-text, `/ehr`, `/meds`, `/labs`, `/risk`, `/pubmed`, and follow-up chips) streams and reveals smoothly through the canvas.
  2. Detail the exact state machine for `createTurn`, `consume`, and `fillTurn` (including how `pending` prose drains/settles before final citation activation, and how `prefers-reduced-motion` is respected).

### 1.3 Executable JS / DOM Verification Harness
* **Audit Tasks:**
  1. Inspect `danielmherman/package.json` and available Node/DOM tooling (`node --test`, Vite, JSDOM, etc.).
  2. Propose an executable JS test suite (runnable via `npm test` and invoked from Django tests or CI) that exercises the `createTurn` streaming and settle lifecycle across simulated multi-chunk SSE streams and surface mounts, proving that:
     * Prose streams progressively to completion without dropping the `pending` tail on `answer` arrival,
     * Markdown/citations render without raw `**` / `##` flashes or layout-swap regressions, and
     * Both prose-heavy turns and card-heavy turns settle cleanly.

---

## 2. Step A — Audit Findings & Proposed Surgical Plan

Every reference below is to the tree at Phase 5's commit (`danielmherman` `948e355`,
`enterprise_clinical_copilot` `951fe62`). **No application or test code was modified to
produce this section.**

### 2.1 What each input actually sends

The seven inputs named in §1.1 are four streaming paths, one single-frame path, and one
path with no prose at all. They do not behave alike, and two of them cannot be fixed on
the client.

| Input | `answer_delta` frames | Emitted at | Draft the reader sees |
| :--- | :--- | :--- | :--- |
| Free text | one per model chunk | `graph.py:794` → `model_runtime.py:121` | paced, tail dropped |
| `/ehr` | one per model chunk | `deterministic.py:722` (`prose_is_model_output`) | paced, tail dropped |
| `/labs` | one per model chunk | same | paced, tail dropped |
| `/pubmed` | one per model chunk | same | paced, tail dropped |
| Follow-up chip | one per model chunk | `askFollowUp` (`copilot_canvas.js:340`) → `requestSubmit` → the free-text path | identical to free text |
| `/meds` | **exactly one, carrying the whole narrative** | `deterministic.py:770-771` | **whole answer in one paint** |
| `/risk` | none — `prose=False` (`routing.py:441`) | — | nothing, then a cold surface |

Three facts follow, and each of them changes the plan:

1. **A follow-up chip is not a modality.** `askFollowUp` writes the question into the
   composer and calls `requestSubmit`, so a chip is a free-text turn with different
   wording. There is no separate code path to fix.
2. **`/meds` already announces its answer, but the announcement is the entire answer**
   (`medication_narrative(...)`, one frame). It therefore hits `drain()`'s
   `releasePrefix` burst (`copilot_canvas.js:750-756`), which paints *all* of `pending`
   in the first tick. `/meds` is not slow-and-then-truncated; it is instantaneous-and-
   then-replaced. A cadence fix that does not address `releasePrefix` would leave
   `/meds` exactly as it is.
3. **`/risk` has nothing to pour.** Its answer *is* its surface. It cannot be made to
   stream, and pretending otherwise would mean inventing prose the agent deliberately
   does not write.

### 2.2 Root causes

**RC-1 — the tail is discarded at the terminal frame.** `fillTurn` (`copilot_canvas.js:1088`)
takes the `drawsProse` branch and calls `block.answer('')` (1102), whose body is
`say('')` (626) → `stopPacing()` (638) → `pending = ''` (727-735). Every character that
arrived but had not been painted is dropped at the exact moment the answer becomes
complete. This is the reported symptom and it is a deliberate line of code, not a race.

**RC-2 — the cadence cannot converge.** `PACING` (192-197) is 45 characters/second at a
40 ms tick, so `budget += 1.8` per tick and `floor(budget)` releases 1 character, 2 on
every fifth tick — 45 cps exactly. Catch-up only engages above `catchUpQueue = 350`
and only triples to 135 cps. A 900-character answer that the model writes in 5 s is
therefore drawn to about 225 characters; the last two seconds drain at 135 cps and the
backlog at the terminal frame is still ~400 characters. **Any answer longer than
`45 × stream_duration` loses its tail by construction** — 225 characters for a 5 s
stream, against a 20 s draw time. The pacing was designed so the reader *would not
finish* the draft before the answer landed; the swap then assumed the reader was
already at the end.

**RC-3 — the draft has no block structure.** `draw()` (686-688) appends a bare text
node, and `.cp-answer-text` (`demo_shell.css:426`) sets no `white-space`, so every `\n\n`
collapses to a space. The draft is one unbroken run of words — headings survive only as
the stripped text that `drawn()` (713-722) leaves behind. Even a draft the reader had
time to finish is not the document they then see.

**RC-4 — the surface is inserted in one synchronous frame, above the prose.**
`mountSurface` (1064) appends `<a2ui-surface>` inside `onSurfaceCreated` (1073-1077),
synchronously within `processMessages`. `.cp-canvas` is `display:none` while `:empty`
(350, 424) and becomes a full-height block on that append, sitting *above* the prose
element (`answerSection.append(canvas, prose)`, 445), which was emptied one statement
earlier. `.cp-answer` is the scroll container (341-345), so its `scrollTop` is preserved
while the content beneath it is replaced wholesale. No transition, no measurement, no
frame between the two states.

**RC-5 — the draft and the final prose are two typographies.** `.cp-answer-text` is
`line-height: 1.65` (426); the narrative is `line-height: 1.6` (`NARRATIVE_STYLES`,
`a2ui_clinical_components.js:317`). Paragraph margins happen to agree (427 vs 353), and
headings exist only after settle (343-347). A 3 % line-height difference compounds over a
long answer into a visible reflow at mount.

**RC-6 — the `streaming` prop is dead.** `CitedNarrativeApi` declares `streaming`
(`a2ui_clinical_components.js:638`) and the component renders an inert dashed pill
(798-805) and a "markers are inert" note (709-714) from it. The value is hardcoded
`False` at `note_presentation.py:181` and passed through at `risk_canvas.py:277`; the
surface is only ever mounted from the settled `answer` frame. No caller can set it true.
Three agent tests (`test_a2ui.py:410`, `test_a2ui_literature.py:576`) and two site
fixtures (`demo/tests.py:69`, `demo/test_conversation_memory.py:263`) pin the `False`, so
this is a contract key with no live behaviour — a decision, not a bug to patch.

**RC-7 — the chrome pops in.** `presentFacts` (969) is called before the surface mounts
(1093) and unhides `scopeEl`, `routedEl`, `evidence` (817/831), the references drawer
(933) and the follow-up rows (952) by flipping `hidden`. Every one of those is an
instant layout insertion under a turn whose answer has not been drawn yet.

**RC-8 — reduced motion has no reveal path.** `appendProse` (777-784) paints everything
immediately and never starts the ticker. That is correct policy, but the surface still
mounts abruptly, so a reader who asked for no animation gets the harshest version of the
swap.

### 2.3 Where this Spoke's premises do not match the code

Recorded so the approved plan is built on the code rather than on the description.

1. **"Dribbles out 1–2 sentences"** — the opening paint is not a sentence count. It is
   `releasePrefix` drawing the entire current backlog in one burst (750-756). For the
   chunked commands that burst is one chunk, because `PROSE_LOOKAHEAD_CHUNKS = 1`
   (`model_runtime.py:108`); for `/meds` it is the whole answer.
2. **"`fillTurn()` calls `block.answer('')` → `stopPacing()`"** — correct, but only on
   the `drawsProse` branch (1101). `/risk` takes the other branch and writes
   `payload.answer`, which is the empty string by design (`deterministic.py:690`).
3. **"`CitedNarrativeElement` … renders inert badges during streaming"** — it *can*; it
   never does (RC-6). §2.6 asks for a ruling rather than a patch.
4. **"Raw-text-to-formatted-Markdown swap"** — the markdown filters in `drawn()`
   (713-722) mean the draft is not raw Markdown. It is Markdown with `*`, `^[n]` and
   `##` removed. The swap is *unstyled run of words* → *styled document*.
5. **"The six components mount"** — there is one mount, not six. All of them are
   children of a single `<a2ui-surface>` composed by `compose_risk_canvas`
   (`risk_canvas.py:210`+) and created in one `processMessages` call. There is one
   insertion to choreograph, and then a per-child stagger inside a shadow root the page
   cannot select into.

### 2.4 Proposed surgical plan

Six steps. Steps 1–4 are the defect; 5 is the choreography; 6 is the gate the Spoke
requires.

**Step 1 — extract the reveal state machine (`static/js/copilot_reveal.js`, new).**
Everything about *when* characters appear, and nothing about the DOM. It takes a
`paint(fragment)` callback and an injected clock (`now`, `schedule`), and exposes
`push`, `flush`, `cancel`, `text`, `pending`, `settledFraction`. `copilot_canvas.js`
keeps the DOM half and loses `PACING`, `drain`, `stopPacing`, `budget`, `releasePrefix`
and `ticker`. This is the change that makes the gate in §2.5 possible at all: a state
machine with an injected clock is testable; a closure inside `createTurn` is not.

**Step 2 — a cadence that converges.**
`PACING = { minCps: 45, maxCps: 900, lagMs: 900, flushMs: 320, tickMs: 16 }` and
`cps = clamp(pending.length / lagMs * 1000, minCps, maxCps)`. The draft then tracks the
inflow with a bounded lag instead of falling arbitrarily behind, which is the "water in
a glass" property: the level follows the pour. Two supporting rules:

- `releasePrefix`'s burst is capped: `burst = min(pending.length, FIRST_PAINT_CAP)` with
  `FIRST_PAINT_CAP = 120`. The intent it encodes — do not pay the agent's lookahead hold
  twice — survives; the `/meds` whole-answer dump does not.
- On the `answer` frame, `flush()` raises the rate to `pending.length / flushMs` and
  resolves when drained. A 400-character tail pours in ~320 ms instead of being deleted.

**Step 3 — the draft renders in final form.**
`draw()` becomes `renderDraft()`, partitioning the accumulated text on `\n\n`: complete
blocks rendered through the renderer's own markdown provider (`proseMarkdown`, 1053),
the incomplete tail as a filtered text node. `send()` (1245) warms `scene()` (1022) at
submit so the provider exists before the first block completes; a block that completes
before it does is drawn as text and re-rendered when the provider arrives.
`demo_shell.css:426-427` is restated to the narrative's metrics — `line-height: 1.6`,
`h1`–`h4` at `0.95rem/600/1.45` with `1.25rem 0 0.25rem`, first-child flush — plus an
inert-pill rule matching `.cite.inert`. With the metrics equal, the mount is the same
glyphs in the same places.

**Step 4 — settle is a cross-fade, not a teardown.**
`fillTurn` (1088-1116) is reordered: `presentFacts` → `await block.flushProse()` →
`mountSurface` with `cp-entering` (opacity 0) → two `requestAnimationFrame`s →
`block.releaseDraft()` → remove `cp-entering`. The draft is removed only after the
surface has been laid out and painted, so no frame is blank and no layout is displaced.
Restored turns (`copilot_threads.js:372-374`) take the same path with a zero-length
flush. The `cp:swap` path loses `block.answer('')` entirely.

**Step 5 — choreographed entrance inside the bundle.**
`BLOCK_SPACING` (`a2ui_clinical_components.js:83-86`) becomes `BLOCK_SETTLE`: the same
margin rule, plus `:host { animation: a2ui-settle 260ms ease-out both }`, `:host(:nth-child(n))`
delays at 60 ms intervals for children 2–6, and a `prefers-reduced-motion` off switch.
`:host(:nth-child())` is the only selector available — the children live in the
surface's shadow root, so no rule in `demo_shell.css` and no `querySelectorAll` from the
page can reach them. `demo_shell.css` gets the container fade on
`.cp-canvas > a2ui-surface`, which *is* light DOM. NB: the stylesheet is a template
literal; no backticks in the comment, per the two prior parse failures at 343 and in
Phase 5.

**Step 6 — the executable gate.** See §2.5.

### 2.5 The executable JS gate

Node is `v22.22.3`; `node --test` needs no dependency. `node_modules` (37 packages) has
no DOM.

**Layer 1 — `tests/js/copilot_reveal.test.mjs`.** The state machine on a virtual clock:
- 200 pushes of 5 characters at 20 ms: `paint` is called with monotonically appended
  text and the concatenation equals the input.
- **the regression**: push 900 characters, advance 5 s, call `flush()` — assert the
  painted length reaches 900 and `pending` is empty before the caller is permitted to
  swap. This is RC-1, executed.
- **the `/meds` regression**: one 400-character push — assert the first paint is capped,
  not the whole string. This is RC-2/`releasePrefix`, executed.
- lag stays under `lagMs × 1.5` for the whole stream.
- `cancel()` paints nothing further; `flush()` on an empty queue resolves immediately.
- reduced motion: one paint, no scheduled tick.

**Layer 2 — `tests/js/copilot_draft.test.mjs`.** The block partitioner: no output ever
contains `**`, `## ` or `^[`; the concatenated blocks equal the filtered whole; a partial
block never becomes a block; a marker split across two pushes is never painted raw.

**Layer 3 — the component.** `CitedNarrativeElement` under a DOM, if one is available
(§2.6, decision 2).

**Wiring.** `package.json` gains `"test": "node --test tests/js/"`.
`demo/test_frontend_stream.py` shells out to it under `skipUnless(shutil.which('npm') and
node_modules.is_dir())`, so `manage.py test` stays one command and the JS gate is part of
the standard three.

### 2.6 Decisions required before Step B

1. **The `streaming` prop: delete or wire.**
   *Delete* removes the prop, the inert branch (`a2ui_clinical_components.js:798-805`),
   the note (709-714), `.cite.inert`, and the `"streaming": False` key at
   `note_presentation.py:181` / `risk_canvas.py:277`, with the three agent tests and two
   site fixtures updated. *Wire* means the surface is mounted before the envelope exists
   — a new `prose_open` frame from the agent, plus an update-in-place path in
   `mountSurface` — which is the higher-fidelity answer and a wire-contract change.
   **Recommend delete**: Step 3 makes the light-DOM draft carry the streaming
   typography, and a second implementation of "prose while streaming" inside the
   component would be the same fact in two places.
2. **A DOM for Layer 3.** `jsdom` as a dev-only dependency and a real component test, or
   a DOM-free gate plus the existing source assertions? **Recommend jsdom** — without it
   no test executes a line of the bundle, which is the limit Phase 5 had to declare.
3. **Whether `/meds` changes on the agent side.** It already emits its whole narrative as
   one delta (`deterministic.py:770-771`); Step 2's capped first paint paces it on the
   client. No agent change is proposed. Confirming that reading is enough.

### 2.7 What these gates prove, and what they do not

- The reveal state machine and the block partitioner are exercised end to end.
- CSS is not executed by anything. Step 3's typography match is asserted by parsing both
  stylesheets and comparing the numbers — that proves the two definitions agree, not that
  the browser lays them out identically.
- With jsdom, component *state* is exercised, not its rendering fidelity.
- **No gate can prove the swap is invisible.** That is a judgement made in a browser,
  and it should be made before Step B is signed off rather than after.

---

## 3. Step B — Execution Record

Executed 2026-10-25 against `danielmherman` and `enterprise_clinical_copilot`. Steps 1–6 of
§2.4, with the three rulings in §2.6 applied as given.

### 3.1 Executed Changes

**`danielmherman`**

| File | Change |
| :--- | :--- |
| `static/js/copilot_reveal.js` | **new, 264 lines.** The cadence as a state machine: `PACING { minCps 45, maxCps 900, lagMs 900, flushMs 320, tickMs 16, firstPaintCap 120 }`, `create(paint, { now, schedule, cancel, reducedMotion })` → `{ push, flush, drainNow, cancel, pending, released, settledFraction }`. `window.CopilotReveal`, frozen. |
| `static/js/copilot_draft.js` | **new, 105 lines.** `partition(raw)` → `{ blocks, tail }` on the last `\n\n`; `segments(raw)` → `[{text}` / `{cite}]` with the unclosed-marker hold. `marker` is exported so the bundle's copy can be compared to it. |
| `static/js/copilot_canvas.js` | 1,464 → **1,564**. `PACING`, `drain`, `stopPacing`, `budget`, `releasePrefix`, `ticker`, `drawn` and the `pending` queue removed; the reveal instance, `renderDraft`, `renderBlock`, `plain`, `renderTail`, `markerPill`, `decorateInert`, `resetDraft`, `flushProse`, `settle`, `SETTLE_MS`, `nextFrame` and `wait` added. `mountSurface` returns the element it created and marks it `cp-entering`. `fillTurn` rewritten (see 3.3). |
| `static/css/demo_shell.css` | `.cp-answer-text` restated to the narrative's metrics (1.65 → **1.6** leading, `h1`–`h4` at `0.95rem/600/1.45`, `1.25rem 0 0.25rem`, first-child flush, paragraph margin `0 0 0.75rem`). Added `.cp-draft-block` (`cp-settle 200ms`), `.cp-draft-tail` (`pre-wrap`), `.cp-cite` (dashed inert pill), `@keyframes cp-settle`, `.cp-canvas > a2ui-surface` + `.cp-entering`, `.cp-answer.is-settling` grid stacking, `.cp-answer-text.cp-leaving`, and a reduced-motion off switch. `animation: cp-settle` on `.cp-scope`, `.cp-evidence`, `.cp-references`, `.cp-follow-ups`. |
| `static/js/a2ui_clinical_components.js` | `BLOCK_SPACING` → **`BLOCK_SETTLE`** on all five cards: the same margin rule plus `:host` animation, `:nth-child(2..6)` delays at 60 ms, `@keyframes a2ui-settle`, reduced-motion off switch. `streaming` deleted: schema key, `render()` branch, `_decorate` local, `_marker` parameter and inert branch, `.cite.inert`, `.streaming-note`. |
| `demo/templates/demo/a2ui_console.html` | `copilot_reveal.js?v=1` and `copilot_draft.js?v=1` before the canvas; canvas `?v=36 → 37`; renderer `?v=15 → 16`. |
| `package.json` / `package-lock.json` | `jsdom` as a dev dependency; `"test": "node --test tests/js/*.test.mjs"`. |
| `tests/js/copilot_reveal.test.mjs` | **new, 7 tests.** The state machine on a virtual clock. |
| `tests/js/copilot_draft.test.mjs` | **new, 11 tests.** The block partitioner and the marker segments. |
| `tests/js/cited_narrative.test.mjs` | **new, 6 tests.** The component under `jsdom`. |
| `demo/test_frontend_stream.py` | **new, 15 tests.** Typography parity, settle ordering, island wiring, and `npm test` run as part of `manage.py test`. |
| `demo/tests.py`, `demo/test_conversation_memory.py` | the `'streaming': False` key removed from the two fixtures whose envelopes are parsed against the component schema. |

**`enterprise_clinical_copilot`**

| File | Change |
| :--- | :--- |
| `services/agent/note_presentation.py` | `narrative_surface` returns `{text, citations}`; the reason the key is gone is in the docstring. |
| `services/agent/risk_canvas.py` | the `CitedNarrative` component drops the key. |
| `tests/agent/test_a2ui.py`, `test_a2ui_literature.py` | the `streaming is False` assertions become closed-set assertions on the payload. |

### 3.2 What the run found that the audit did not

1. **The pour deadline was not a deadline.** Giving the flush a rate of `backlog / flushMs`
   is exponential decay — each tick takes a fraction of what is *left* — and `floor(budget)`
   then rounds the residue to zero for a long time. Measured on 500 characters: **2.0 s for
   a pour of 320 ms**, and the last characters never arrived at all. Replaced with a quota
   fixed when the pour is armed (`ceil(queue / 20)` per tick), which drains in exactly 20
   ticks. The probe that found it is the reason `test_the_pour_has_a_deadline` exists.
2. **The draft needs the renderer, and the renderer was loaded on the first answer only.**
   `scene()` was called from `mountSurface` and nowhere else, so a block finishing before
   the import resolved would have rendered as words. `renderBlock` awaits `scene()` itself
   and falls back to plain text, and `send()` warms nothing else — the import now begins at
   the first block instead of at the answer, which is still far later than the first paint.
3. **A `vm` realm hands back arrays the test runner will not compare.** The draft tests
   failed with "Values have same structure but are not reference-equal" on every
   `deepEqual`, because the arrays were minted in the sandbox. The module is pure and hangs
   itself on `window`, so it now runs in the host realm; the reveal tests keep the sandbox
   and take the host's `Promise` instead.
4. **`node --test tests/js/` fails on Node 22** — it resolves the directory as a module.
   The script uses the `*.test.mjs` glob, which the shell expands.
5. **`jsdom` has neither `matchMedia` nor `scrollIntoView`**, and the component calls both.
   Stubbed in the harness, where they are environment rather than behaviour.
6. **The component does not render prose without a markdown provider.** `_syncProse` sets
   `_source` in `updated()` and asks for nothing; the second update is the provider's
   promise resolving. The harness supplies a one-paragraph renderer, which is also what
   makes the marker decoration test real — the markers have to survive rendering to be
   decorated.
7. **The closed schema made the deletion atomic across four files.** `CitedNarrativeApi`
   is `.strict()`, so a payload still carrying `streaming` fails validation and the surface
   does not draw. The agent, both site fixtures and the schema had to change together.

### 3.3 The two handovers, as built

**The pour.** `fillTurn` no longer calls `block.answer('')`. It calls `presentFacts`, then
`await block.flushProse()` — which resolves when the last character is on the screen — and
only then mounts the surface. A restored turn has nothing queued and the flush resolves
without doing anything.

**The cross-fade.** `mountSurface` appends the surface with `cp-entering` (opacity 0) and
`settle` puts `.cp-answer.is-settling` on the answer section, which stacks both children in
one grid cell. Two frames later the surface begins to fade in and the draft begins to fade
out **in the same frame**; the draft is emptied when the fade ends, at which point the
pixels have stopped changing. The draft therefore never occupies its own row under the
answer — the failure the ruling named — and the answer is never blank, because the draft is
still at full opacity for the whole of the surface's fade-in.

Which is why `releaseDraft` became `resetDraft`: the draft is not released at the start of
the transition but at its end, and the transient state in between is two overlapping copies
at the same coordinates with the same metrics. `test_the_surface_is_laid_out_before_it_is_faded_in`
pins that order from the source.

### 3.4 Verification Gates

| Gate | Result |
| :--- | :--- |
| `pytest tests` (`enterprise_clinical_copilot`) | **1377 passed, 1 skipped** (268 s) |
| `npm test` (`danielmherman`) | **24 passed, 0 failed** (2.9 s) — 7 reveal, 11 draft, 6 component |
| `manage.py test` (`danielmherman`) | **271 OK** (57 s) — was 256; +15 from `test_frontend_stream.py` |
| `manage.py check --deploy` (`danielmherman`) | **No issues identified (2 silenced)** |
| `npm run build:clinical` | `static/js/bundled/a2ui_clinical.js` **804.03 kB** (gzip 192.51 kB); bundle contains `a2ui-settle`, contains neither `streaming-note` nor `cite inert` |

### 3.5 Behaviour changes a reader will notice

- The draft grows in the answer's own typography, with headings, paragraph breaks and inert
  citation pills, instead of one collapsed run of text with the markers removed.
- The draft finishes pouring before the surface replaces it. On a long answer that is up to
  `flushMs` of extra reading; it is time the reader was going to spend reading the same
  characters anyway.
- `/meds`, which announces its whole narrative in one frame, is paced instead of pasted.
- The cards on a canvas settle in with a 60 ms stagger.
- Under `prefers-reduced-motion` there is no pacing, no animation, and the surface replaces
  the draft in a single frame.

### 3.6 Deliberately Not Done

- **No `prose_open` frame and no early surface mount.** The ruling on decision 1 was delete,
  so the streaming typography lives in the light-DOM draft and `CitedNarrative` renders only
  settled prose. A living surface that updates in place remains the higher-fidelity answer
  and remains unbuilt.
- **No agent change for `/meds`.** The single delta is paced on the client, which is what
  the third ruling confirmed. Nothing was moved earlier in the pipeline, so time to first
  token is unchanged.
- **`/risk` still arrives as a single reveal.** It has no prose to pour; it gets the card
  stagger and nothing else.
- **`prose_settled` is untouched.** A guardrail rewrite still replaces the text and still
  says so in the Evidence-Check pill.
- **`demo_shell.css` has no cascade layer work.** The `is-settling` grid is the thinnest
  thing that stacks two children; it is not a general layout mechanism.

### 3.7 What These Gates Prove, and What They Do Not

- **Proved:** the reveal state machine on a virtual clock, including the two regressions
  this phase exists for (the tail is painted before a swap is permitted; a single large
  frame is not painted whole). The block partitioner and the marker segments. The
  component's schema, render, marker decoration, reveal and ready event, in a DOM.
- **Proved by comparison, not by rendering:** that the draft's stylesheet and the
  narrative's declare the same font size, leading, heading metrics and paragraph margin.
  That is not the same claim as "the browser paints them at the same y", and it is the
  strongest claim a test can make about CSS in a repository with no browser.
- **Not proved:** that the cross-fade is invisible. `jsdom` computes no layout and applies
  no stylesheet, so the grid stacking, the 180 ms transition and the `cp-entering` handoff
  are asserted as source and as declarations, never as pixels. **This is a browser
  judgement and should be made in a browser before this phase is signed off.**
- **Conditional:** the JS gate runs from `manage.py test` through `npm`. On a machine
  without `node` or `node_modules` it raises `SkipTest`, so the site gate reports 270 tests
  plus one skip rather than failing. That is deliberate — a checkout with no dependencies
  cannot compile the bundle either — but it means the gate is only as present as the
  toolchain.

---