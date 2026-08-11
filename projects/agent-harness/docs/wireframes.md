# Wireframes — Clinical Readmission Copilot demo

**Date:** 2026-08-11 · **Status:** layout-level (boxes + hierarchy, not pixels)
**Maps to:** `ux_design_plan.md` §7 personas/journeys, §6 decisions Q1–Q6
**Built on:** real fixtures in `data/demo_fixtures/`

These are **structure, not design**. No color/typography decisions yet — that's
the polish pass after the skeleton runs. Every region notes what feeds it (tool
output or fixture) so the skeleton build has a data contract per box.

---

## Legend

Risk bands are derived from the operating threshold (0.12), not hardcoded:

| Band | Range | Dot marker |
|---|---|---|
| Low | probability < 0.12 | `●` green |
| Borderline | 0.12 – 0.20 | `◐` amber |
| High | > 0.20 | `▲` red |
| Unscored | no precomputed value | `○` gray |

Demo patients with known values (from fixtures):
- Erica Abernathy `●` 0.045 · Danielle Achebe `●` 0.051 · Gloria Kowalski `◐` 0.131 · Leonard Castellano `◐` 0.195

> Engineering note: dots on the list require a **precomputed risk signal for the
> whole cohort** (predict is ~seconds/patient, so this is a build-time cache, not
> a per-open call). Flagged in `ux_design_plan.md` §7 cross-cutting table.

---

## Screen 1 — Patient list (pre-select state)

**Serves:** Dr. Ortiz step 1 (Orient) · **Feeds:** `demo_cohort.json` + cached risk

```
┌──────────────────────────────────────────────────────────────────────┐
│  Readmission Copilot                                    [env: demo ▾] │
├──────────────────────────────────────────────────────────────────────┤
│  🔍 Search patients…                    Showing 32 of 32 · test split │
├──────────────────────────────────────────────────────────────────────┤
│  ● Erica Abernathy    32F   urgent · 2 prior · 3d · home              │
│  ● Danielle Achebe    31F   urgent · 3 proc · 3d · home               │
│  ◐ Gloria Kowalski    66F   observation · 3d · rehab   ← borderline   │
│  ◐ Leonard Castellano 72M   urgent · 5d · skilled care ← attention    │
│  ○ Monique Vasquez    36F   urgent · 1 proc · 2d · home               │
│  … (rows scroll)                                                     │
├──────────────────────────────────────────────────────────────────────┤
│  ○ risk unscored · ● low · ◐ borderline · ▲ high                     │
└──────────────────────────────────────────────────────────────────────┘
```

**Behaviors / notes**
- Row = dot + name + age/sex + the cohort summary line (already in
  `demo_cohort.json`).
- Dots let the doctor **scan** before opening anything (Orient).
- Selecting a row → Screen 2 for that patient.
- The "borderline / attention" hint is a design idea to validate — a subtle text
  tag next to the dot for borderline and high cases. Could be noise if too many
  rows have it; revisit in the polish pass.

---

## Screen 2 — Split-pane assessment (thread + canvas)

**Serves:** Dr. Ortiz steps 2–5, Maya steps 1–4 · **Feeds:** `rag_search` +
`predict_readmission` outputs, episodic thread state (browser-held)

```
┌──────────────────────────────┬───────────────────────────────────────┐
│ THREAD                       │ CONTEXT CANVAS                        │
│ L. Castellano · 72M · demo   │                                       │
│ ┌──────────────────────────┐ │                                       │
│ │ ○ Run 30-day risk        │ │                                       │
│ │ ○ Summarize notes        │ │                                       │
│ │ ○ Discharge meds?        │ │                                       │
│ │ ○ Compare to prior  ←episodic, appears after 1st assessment       │
│ └──────────────────────────┘ │                                       │
│                              │                                       │
│ You: Run 30-day risk         │   RISK                                │
│                              │   ┌─────────────────────────────┐    │
│ Agent: 19.5% — above the     │   │ 19.5%                       │    │
│ 0.12 threshold. Risk driven  │   │ ██████████░░░░░░░░  [0.12|] │    │
│ by prior inpatient days^[1]  │   │ low        borderline  high  │    │
│ and age.^[2]  (used: predict)│   └─────────────────────────────┘    │
│                              │                                      │
│ You: discharge meds?         │   DRIVERS  (top ±5)                   │
│                              │   ┌─────────────────────────────┐    │
│ Agent: Discharged on …^[3]   │   │ prior_inpatient_days  +0.11  │    │
│  (used: rag_search)          │   │ age                   −0.09  │    │
│                              │   │ rdw_max               −0.06  │    │
│                              │   │ [view full breakdown ▾]       │    │
│                              │   └─────────────────────────────┘    │
│                              │   SOURCE (cited section)              │
│                              │   ┌─────────────────────────────┐    │
│                              │   │ ╔═ discharge_medications ══╗│    │
│                              │   │ ║ …highlighted passage…   ║│    │
│                              │   │ ╚═════════════════════════╝│    │
│                              │   └─────────────────────────────┘    │
│ [message input…]             │   [show trace ▤]                      │
└──────────────────────────────┴───────────────────────────────────────┘
```

**Behaviors / notes**
- **Left = real thread.** User turns + agent turns; each agent turn shows a
  subtle `(used: predict / rag_search)` provenance line. The episodic memory
  design: last ~5 turns injected into the next prompt, thread held browser-side.
- **Right = persistent canvas**, keyed by patient + tool type. Widgets persist
  across turns for the same patient; switching patient clears/resets.
- **RISK widget** (Q2): big number + progress bar + threshold marker `|`, band
  labels underneath. Exact number kept (R8 text fallback).
- **DRIVERS widget** (Q3): top ±5 SHAP parent features, sign → direction,
  magnitude → bar length. Collapsed by default → "view full breakdown" expands.
- **SOURCE widget** (Q4): the passage for the most recent citation; clicking a
  `^[n]` in prose swaps/highlights the cited section here.
- **Starter chips** (Q6): 3 fixed + dynamic "Compare to prior" (only after a
  first assessment exists).
- **Show trace toggle** → Screen 3 (overlays/replaces the canvas).

**State variants (same screen, different canvas states)**
- **A — zero-state:** canvas empty; only chips shown; "compare" absent.
- **B — after risk:** RISK + DRIVERS + cited answer.
- **C — citation clicked:** SOURCE widget shows the highlighted section.
- **D — breakdown expanded:** DRIVERS grows to all significant parent groups.
- **E — compare-to-prior:** canvas shows this assessment vs. the earlier one
  side by side (RISK + DRIVERS columns + "what changed" line).

---

## Screen 3 — Trace view (toggle on)

**Serves:** Alex Rivera steps 2–5 · **Feeds:** `tool_calls` from the thread
state + the raw tool payloads

```
┌──────────────────────────────┬───────────────────────────────────────┐
│ THREAD (unchanged)           │ TRACE                                 │
│                              │ ┌───────────────────────────────────┐ │
│ …prior turns…                │ │ tool calls (this thread)          │ │
│                              │ │                                   │ │
│ You: does the note mention   │ │ predict_readmission(20724182)     │ │
│ a discharge destination?     │ │   probability 0.194512            │ │
│                              │ │   threshold   0.12                │ │
│ Agent: no supporting         │ │   model  readmission-final-…      │ │
│ passage was found for that.  │ │   source BigQuery                 │ │
│  (used: rag_search)          │ │                                   │ │
│                              │ │ rag_search("medications", top_k=5)│ │
│                              │ │   returned 1                      │ │
│                              │ │   discharge_medications 0.2479    │ │
│                              │ │                                   │ │
│                              │ │ [cross-patient isolation is       │ │
│                              │ │  enforced server-side (R1)]       │ │
│                              │ └───────────────────────────────────┘ │
│                              │ [show trace ▤]  (toggle off)          │
└──────────────────────────────┴───────────────────────────────────────┘
```

**Behaviors / notes**
- The trace is the **technical-evaluator surface**: real tool calls, arguments,
  returned counts/scores, model version, feature source.
- **Empty-is-real-answer** is demonstrated here (the "no supporting passage"
  turn above) — a core no-hallucination proof for Alex.
- A static note that **R1 isolation is enforced server-side** turns our privacy
  posture into a visible, checkable claim.

---

## Interaction & content rules (apply to all screens)

1. **R8 — never render nothing.** Every widget has a text fallback (the exact
   number, the section name, the tool error message).
2. **Structured outputs never go in chat bubbles** — prose references them via
   `^[n]`; the widgets hold the data.
3. **Citations:** superscripts map to the numbered source list; click → the
   cited section highlights in SOURCE (Q4). Agent never cites a passage it did
   not retrieve (prompt contract, already in `agent/prompts.py`).
4. **Progressive disclosure:** glance = number + bar + top drivers; everything
   else behind a click (breakdown, trace, full note text).
5. **Patient isolation extends to memory:** the Compare chip and thread are
   per-patient; cross-patient context is never shared.

## Mapping to journeys & fixtures

| Journey step | Screen / state | Fixture that drives it |
|---|---|---|
| Ortiz 1 Orient | Screen 1 dots | demo_cohort.json + cached risk |
| Ortiz 2–5 Assess/Probe/Verify | Screen 2 B–D | predict_20724182.json, rag_sepsis…json |
| Ortiz 6 Decide | Screen 2 (post-state) | — |
| Maya 1 Continue | Screen 2 A (chips incl. Compare) | episodic thread state |
| Maya 2–3 Meds/Verify | Screen 2 B/C | rag_medications_20724182.json |
| Maya 4 Compare | Screen 2 E | two predict fixtures side by side |
| Alex 2–5 Grounding/Trace/Isolation | Screen 3 | tool_calls + predict/rag payloads |
