# UX Design Plan — Clinical Readmission Copilot

**Phase:** post-integration UX design (after the demo integration is wired)
**Source framework:** the 3-step process (Split-Pane · Trust & Explainability · Progressive Disclosure)
**Status:** plan for the next phase — not yet implemented
**Design decisions:** Q1–Q6 locked 2026-08-11 (see §6); personas + journeys locked 2026-08-11 (see §7)

---

## 0. Memory design (CoALA-aligned) — FINALIZED 2026-08-11

**Framework:** CoALA (Cognitive Architectures for Language Agents) defines four
memory modules — **working** (current state), **episodic** (past experiences),
**semantic** (facts), **procedural** (how-to / code). Agents both *read* (retrieve)
and *write* (learn) long-term memory.

**Where the project sits today:**

| CoALA memory | Current system | Read/Write |
|---|---|---|
| Working | LangGraph `AgentState.messages` + `tool_calls` (per-turn) | read+write, not persisted |
| Semantic | BigQuery feature table + Vector Search notes index | **read-only** |
| Episodic | ❌ none | — |
| Procedural | `graph.py`, MCP tools, prompts | read-only (by design) |

**Decision (user-approved):** add **episodic memory = the conversation/assessment
thread**, stored **browser-side** (client-held), matching the split-pane plan and
the stateless Cloud Run topology.

**Concrete design (Option A — episodic = chat thread):**

1. **What is stored (an episode):** per patient, an ordered list of turns, each
   `{user_question, agent_answer, tool_calls_used: [names], cited_passage_ids}`.
   Optionally structured assessment records (risk, drivers, passages) for
   "compare to earlier" — **Option B, stretch goal, deferred.**
2. **Where it lives:** browser-held JS state (an array of episodes in the demo
   console). No server persistence. Survives page reloads only within the tab
   session; resets on refresh. **Backend stays stateless.**
3. **Working-memory injection rule (the important part):** on each new question,
   the *last N turns* (e.g., 5) of the current patient's episode are prepended to
   the prompt as prior context. This is **retrieval into working memory**. Keep N
   small — CoALA + the R7 rule warn against dumping everything in.
4. **Write rule (learning):** after each turn completes, append the new turn to
   the episode. Only *episodic* write — never write to semantic/procedural.
5. **Retrieval policy:** by default, inject the most recent turns for the *same
   patient*. (Cross-patient memory is deliberately NOT shared — patient isolation
   extends to memory, matching the R1 privacy posture.)
6. **Guardrail:** the agent must not cite a passage it did not retrieve this turn;
   memory carries citation *IDs* for the UI, not as grounds for new claims.

**Why this shape:** it is the smallest change that makes the split-pane chat real
(conversation persists across turns for a patient), it names what the UX plan
already required, and it keeps the demo's stateless backend + cloud-only constraint.
CoALA is used as a *design vocabulary*, not a library — there is no CoALA SDK.

---

## 1. Summary of my understanding

The demo will be judged on the **user experience first**, not on the underlying ML.
Doctors (and demo viewers) are time-constrained and need to **trust the output
instantly**. The framework is three moves:

1. **Split-Pane Architecture** — separate the *conversation* (agent thread) from the
   *data artifacts* (structured widgets). Chat is for orchestration and narration;
   structured results (prediction, SHAP, RAG passages) render as persistent widgets
   in a dedicated context canvas — never dumped into a chat bubble.
2. **Trust & Explainability** — surface the MLOps we already have:
   - a **risk gauge** (color-coded)
   - **top SHAP features** as a simple bar chart
   - **citation highlighting** — superscripts in the agent's prose that, when
     clicked, highlight the exact source sentence in the retrieved note (in the canvas)
3. **Progressive Disclosure** — don't drown the doctor in 49 features:
   - **zero-state prompts** (starter chips on patient select)
   - **expandable details** ("Why is risk high?" → short answer + a "View full
     feature breakdown" button that expands the data pane)

**Key insight:** the copilot isn't a chatbot that answers — it's an **orchestrator
that brings the right data visualization to the clinician exactly when asked.**

---

## 2. Where we are today (what already exists)

| Capability | Current state | Relevance to UX plan |
|---|---|---|
| Demo console (Django) | Patient list (left) + single output region (right) | Already has the *spatial split* seed — no chat thread yet |
| A2UI renderer | Vendored, working, v0.9, `basicCatalog` (Card/Column/Text, Markdown) | Canvas widgets will be A2UI components |
| Risk card | Single A2UI card: probability, threshold, decision, top factors (text) | The embryo of the "risk gauge + features" widget |
| Text fallback (R8) | Every card has `fallback_text` | Keep this — it protects the demo |
| Agent (LangGraph) | One-shot `ask(question)` per request; returns `{a2ui, answer}` | Needs to become a *thread* with tool-call history for split-pane |
| `predict_readmission` | probability, threshold, decision, top_factors (SHAP, logit) | Feeds the gauge + feature bar |
| `rag_search` | passages with section + text + score | Feeds citations; **ids include `{note_id}_{section}_{ordinal}`** |
| MCP tools | registered in one server | Split-pane can render per-tool outputs |

### Key architectural constraints that shape the design

- **Tools return JSON; the agent composes UI** (`a2ui.py` is the single adapter).
  The split-pane canvas must therefore be driven by *structured tool outputs the
  agent relays*, not by the agent emitting raw UI per widget.
- **R7: rendered payloads are `audience: ["user"]`** — the model never re-reads its
  own UI output. A persistent canvas must be fed by the *tool responses*, which the
  agent already records in `tool_calls` state.
- **R8: never render without a text fallback.** Every new widget keeps a text path.
- **Citations need a source mapping.** `rag_search` returns passage text + section,
  but *sentence-level* highlighting requires either (a) the index to return char
  offsets, or (b) client-side matching of a quoted span against the passage text.
  This is the biggest open technical question (see §6 Q4).

---

## 3. The UX plan, step by step

### Step 1 — Split-Pane Architecture

**Goal:** conversation on the left, persistent context canvas on the right.

**Layout:**
```
┌─────────────────────────────┬──────────────────────────────┐
│  Agent Thread (left)        │  Context Canvas (right)      │
│                             │                              │
│  [starter chips]            │  ┌────────────────────────┐  │
│  User: "why is risk high?"  │  │ RISK GAUGE  (widget 1) │  │
│  Agent: "…^[1] …"           │  │ 28.4%  ▓▓▓▓░  (red)   │  │
│    ├─ cite 1 →              │  └────────────────────────┘  │
│  User: "meds?"              │  ┌────────────────────────┐  │
│  Agent: "…^[2]"             │  │ SHAP FEATURES (widget2)│  │
│                             │  │ bar chart              │  │
│                             │  └────────────────────────┘  │
│                             │  ┌────────────────────────┐  │
│                             │  │ SOURCE NOTE (widget 3) │  │
│                             │  │ …highlighted sentence… │  │
│                             │  └────────────────────────┘  │
└─────────────────────────────┴──────────────────────────────┘
```

**Behaviors:**
- The **left pane is a real chat thread**: user messages + agent turns, with
  `tool_calls` shown as a subtle "used predict/rag_search" indicator (provenance).
- The **right pane holds persistent widgets** keyed by patient + tool type:
  - a prediction widget (updates if a new prediction is run)
  - a RAG widget (appends passages as new retrievals happen)
- Widgets **persist across turns** for the same patient — they don't vanish when a
  new message is sent. Switching patient clears/resets the canvas.
- Structured outputs never go into chat bubbles; the agent's prose references them.

**Implementation note:** this is the largest change — the demo backend currently
serves a single `{a2ui, answer}` per request with no session/thread state. Split-pane
needs either (a) a session-scoped thread store on the Django side, or (b) the browser
holding accumulated state and the server returning tool results + prose per turn. The
browser-accumulation path is simpler and matches the stateless Cloud Run topology.

### Step 2 — Trust & Explainability

**Goal:** make the MLOps visible and legible.

**Widgets (all in the Context Canvas, all with text fallback):**

1. **Risk Display — big number + progress bar** (user decision 2026-08-11)
   - Big probability number, color-coded by risk band (bands derived from the
     *operating threshold*, not hardcoded: green < threshold, amber near, red above).
   - A progress bar with a **threshold marker** so the decision point is visible at
     a glance.
   - Text fallback keeps the exact number (R8: don't round into a band).

2. **SHAP Feature Bars**
   - The top ±5 parent features as a horizontal bar chart (contribution sign →
     direction; magnitude → bar length).
   - Reuses `top_factors` already returned by `predict_readmission`.
   - Decision (2026-08-11): **spike an A2UI bar-chart component** so the
     "agent composes UI" architecture stays intact (see §6 Q3).

3. **Citation Highlighting**
   - Agent prose emits superscripts `^[1]` `^[2]` tied to a numbered source list.
   - Clicking a citation **highlights the cited section** in the source note,
     rendered as widget 3 (decision 2026-08-11: whole-section anchoring from the
     passage id `{note_id}_{section}_{ordinal}`, not sentence offsets).
   - Sources are the `rag_search` passages; the numbered list maps citation → passage.

**Prompt-side contract change (agent/prompts.py):** the agent must be told to
(a) cite passages with `^[n]`, (b) never invent a citation, and (c) only cite what
`rag_search` returned. This is a guardrail, same spirit as the existing "never
invent a risk factor" rule.

### Step 3 — Progressive Disclosure

**Goal:** the doctor sees the right density at the right moment.

1. **Zero-state prompts (starter chips)** — confirmed 2026-08-11
   - On patient select, the thread shows three fixed starter chips:
     - "Run 30-day readmission risk"
     - "Summarize recent discharge notes"
     - "What medications were they discharged on?"
   - A fourth chip, **"Compare to prior assessment"**, is surfaced only once a
     prior assessment exists for this patient (episodic memory).
   - Chips are patient-specific (from the cohort summary) and just send a
     pre-composed question (like the current picker already does — the pattern exists).

2. **Expandable details**
   - Default: gauge + top-3 factors + cited answer.
   - "View full feature breakdown" expands the SHAP widget to all significant
     parent groups.
   - RAG passages collapse to section + snippet; expand to full text.

3. **Density control**
   - One patient assessment = one glance: risk decision + drivers + evidence.
   - Everything else is behind a click. The canvas is organized by *what was asked*,
     not by all 49 features at once.

---

## 4. Suggested build order (for the next phase)

| # | Work | Depends on |
|---|---|---|
| 1 | Thread state in the demo (accumulate turns client-side) | — |
| 2 | Canvas widget framework: predict widget (gauge + bars) | 1 |
| 3 | RAG widget: passage list + source note | 1, and rag_search live |
| 4 | Citation contract (`^[n]`) in prompts + prose renderer | 3 |
| 5 | Sentence highlight | 4 (needs §6 Q4 answer) |
| 6 | Zero-state chips + expandable details | 1 |
| 7 | Polished split-pane layout + responsive | 2-6 |

Each step keeps the R8 text fallback so the demo never renders nothing.

---

## 5. What this plan deliberately keeps / does not change

- **Tool contract unchanged:** `predict_readmission` and `rag_search` stay plain
  JSON. All UI composition stays in `a2ui.py` on the agent side.
- **R8 text fallback** on every surface.
- **The one-shot backend can stay stateless** — the thread lives in the browser.
- **Credentialing boundary:** note-text rendering stays behind one clean component,
  as the guide already requires.

---

## 6. Open questions — ANSWERED 2026-08-11

All six are resolved. Each decision is recorded where it applies above.

**Q1 — Thread scope → browser-only episodic memory.** The chat thread per patient
lives client-side (an episode array in the demo console); backend stays stateless.
Injection rule: prepend the last ~5 turns of the same patient as prior context;
write on completion; cross-patient memory deliberately not shared (R1 privacy).
See §0.

**Q2 — Risk display → big number + progress bar.** Not a needle gauge. Big exact
probability, color-coded by band derived from the operating threshold, progress bar
with a threshold marker. R8 text fallback keeps the raw number.

**Q3 — SHAP chart → spike an A2UI bar-chart component.** Keeps the "agent composes
UI" architecture (tools return JSON, a2ui.py adapts); no separate front-end chart
stack. Top ±5 parent features, sign → direction, magnitude → bar length.

**Q4 — Citation granularity → whole cited section.** Clicking `^[n]` highlights the
cited section in the note, anchored by the passage id `{note_id}_{section}_{ordinal}`.
No backend change. Sentence-level offsets deferred unless probed.

**Q5 — Demo audience → clinical readability first, trace behind a toggle.** Default
view prioritizes speed and readability (risk + drivers + cited answer); a
"show trace" toggle reveals tool calls, SHAP details, feature source, model version
for technical peers.

**Q6 — Starter chips → 3 fixed + dynamic compare.** Fixed: run 30-day readmission
risk · summarize recent discharge notes · what medications were they discharged on.
The fourth, "Compare to prior assessment," surfaces only once a prior assessment
exists for the patient (episodic memory). Chips are patient-specific.

---

## 7. Personas & user journeys — LOCKED 2026-08-11

Three personas. Two are clinical, one is the evaluator. The demo is designed
around the primary persona, then made to satisfy the other two.

**Decision (user):** primary persona = **Dr. Ortiz (the doctor)** — the demo
leads with "is this patient at risk, and why?" as a quick, trustworthy read.
**Decision (user):** the patient list **shows risk affordances (dots)** — a small
color-coded dot per patient so the list can be scanned for who needs attention.

### Persona 1 — Dr. Lena Ortiz, Hospitalist (PRIMARY)
- **Goal:** during rounds, quickly know "is this patient at risk of bouncing back,
  and why?" — and whether to trust the number.
- **Context:** time-pressured, scanning many patients, zero patience for a wall of
  features. Will distrust a number with no grounding.
- **Emotional driver:** fear of a bad discharge call. Confidence with *less*
  cognitive load.

**Journey — morning rounds (cast: Leonard Castellano, 0.19 high risk):**
1. **Orient** — patient list shows an amber dot on Castellano. One glance tells
   her who needs attention; no opening action needed.
2. **Enter** — tap the patient. Thread opens empty with starter chips. Tap
   "Run 30-day readmission risk."
3. **Assess** — within a second: big number 19.5% on a progress bar, threshold
   marker visible, band amber, with a two-line driver summary and the cited
   answer prose.
4. **Probe** — tap "View full feature breakdown" → SHAP bars expand.
5. **Verify** — click a citation `^[1]` → the discharge-notes section highlights
   in the source panel. Trust established.
6. **Decide** — confident: flag for discharge review. Next patient. Under 2 min.

**Derived requirements:** patient list carries a risk affordance (dot) → needs a
cheap per-patient risk signal; assessment = one glance (number + bar + threshold
+ drivers, no scrolling); progressive disclosure (probe → SHAP, verify → cited
section); every claim in prose clickable to a section (Q4).

### Persona 2 — Maya Chen, RN, Discharge Coordinator (SECONDARY)
- **Goal:** build a safe discharge plan — meds, instructions, follow-up — and see
  how the picture changed across the stay.
- **Context:** operational, detail-oriented, asks follow-ups, signs off on the plan.
- **Emotional driver:** accountability — every answer she gives the patient traces
  to evidence.

**Journey — discharge planning (same patient, later that day):**
1. **Continue** — the thread already holds this morning's assessment (episodic
  memory). Starter chips now include "Compare to prior assessment."
2. **Meds** — tap "What medications were they discharged on?" → RAG returns the
  discharge-medications passage, cited.
3. **Verify** — click the citation → the `discharge_medications` section
  highlights in the note; she reads the context directly.
4. **Compare** — tap "Compare to prior assessment" → the canvas shows this
  assessment vs. this morning's side by side (risk + drivers + what changed).
5. **Act** — flags the plan for a follow-up; the thread now holds both
  assessments.

**Derived requirements:** episodic memory actually works (thread persists across
patient visits; the Compare chip appears only after a first assessment);
medications retrieval + citation to section; a side-by-side comparison widget
(risk + drivers across two assessments).

### Persona 3 — Alex Rivera, Technical Evaluator (DEMO AUDIENCE)
- **Goal:** verify the engineering is real — R1 isolation, no hallucination,
  citations that actually map to note text.
- **Context:** portfolio reviewer / technical peer. Skips polish, pokes for fakery.
- **Emotional driver:** skepticism. The trace toggle is his trust mechanism.

**Journey — portfolio review:**
1. **Assess** — run the risk chip; big number + drivers appear fast.
2. **Probe grounding** — ask a question with no answer (e.g., "does the note
  mention a discharge destination?") → agent says **no supporting passage found**
  (empty-is-real-answer), no fabrication.
3. **Toggle trace** — flip "show trace" → tool calls appear (`predict_readmission`,
  `rag_search`) with feature source, model version, and passage scores.
4. **Test isolation** — open a second patient, ask about the first patient's data
  → R1 restrict holds; nothing leaks.
5. **Judge** — citations map to real note sections; numbers match the model
  version shown. Trust → the engineering is real.

**Derived requirements:** trace toggle reveals `tool_calls`, `feature_source`,
`model_version`, passage scores; empty-is-real-answer behavior is demonstrable;
R1 isolation is *visible* (the demo can run a cross-patient probe).

### Cross-cutting requirements (from all three)
| Requirement | Serves | Build order ref |
|---|---|---|
| Patient list risk dots | Dr. Ortiz | needs cheap per-patient risk signal (precompute for cohort) |
| Big number + progress bar + threshold marker | Dr. Ortiz | Step 2 (predict widget) |
| SHAP bars behind "View full feature breakdown" | Dr. Ortiz, Alex | Step 2 (A2UI bar chart spike, Q3) |
| Citation → section highlight | All | Step 3/4 (Q4) |
| Starter chips (3 fixed + dynamic Compare) | Dr. Ortiz, Maya | Step 6 (Q6) |
| Episodic thread + Compare widget | Maya | Step 1 + §0 memory |
| Trace toggle | Alex | Step 2 (provenance) |
| Empty-is-real-answer, no fabrication | Alex | already in rag_search + prompts |
| R1 isolation visible | Alex | already true server-side; surface it |

---

## 8. Next steps (after this plan)

1. Build order (see §4) — the integration demo lands the current card + citations
   first; the full split-pane lands in the redesign phase.
2. Wire the persona journeys to concrete screens/wireframes in the demo phase.
3. Revisit: any reference screenshots or the interactive sandbox "density
   experiment" if you still want it (it would refine Q3/Q5 choices, now decided).
