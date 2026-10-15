# Phase 4: Clinical Intelligence, Streaming Architecture & Advanced UX

## Executive Summary
Phase 4 elevates the Enterprise Clinical Copilot from a stateful chart reader into a real-time, point-of-care clinical decision support workspace that rivals OpenEvidence and Gemini[cite: 1]. It closes the Phase 3 hygiene items, implements end-to-end token and component streaming with clinical safety swap protocols and client-side cadence control, introduces OpenEvidence-style multi-source risk synthesis tables, follow-up navigation, and reference-level feedback traces (feeding Phase 6 Evals), unifies the EHR and Risk views, and expands the MCP toolset with authoritative pharmacology (`/drug`) and DUA-compliant synthetic longitudinal lab trends (`/labs`)[cite: 1].

---

## Execution Roadmap (Milestones)
To maintain velocity and verify core protocol changes before adding new tools, Phase 4 is executed in three sequential milestones:
* **Milestone 4A (Core Experience — In Progress):**
  * **4A-0 (Completed & Committed):** Phase 3 Carry-Over Hygiene (§0).
  * **4A-1 (Completed & Committed — `1c84d67`, `8ba5dbf`):** Real-Time Lookahead Streaming & Safety Swap Protocol (§1).
  * **4A-1b (Implemented, uncommitted):** Client-Side Streaming Cadence & Pacing Queue (§1.4), the auto-expanding composer from §6, and §1.5 below — free text now retrieves, so its answers carry citations.
  * **4A-2 (Next Up):** OpenEvidence-Style Follow-Ups, Citation Drawers (§2), and the remaining table-export affordance in §6.
* **Milestone 4B (Risk & Literature Synthesis):** Unified `/ehr` + `/risk` View, Interactive Driver Bars, Multi-Source Condition Risk Matrix, and Reference-Level Feedback Traces (§3).
* **Milestone 4C (Clinical Depth):** OpenFDA/RxNorm Pharmacology Tool (`/drug`) (§4) and DUA-Compliant Synthetic Longitudinal Lab Trends (`/labs`) (§5).

---

## 0. Phase 3 Carry-Over Hygiene (Status: Completed & Committed)
**Objective:** Close the three known items carried over from the Phase 3 sign-off before modifying the streaming pipeline.
* **Agent & Site Repository Git Hygiene (`7bc9d31`, `c65e3f9`):** Committed `scripts/agent/retention.py`, the consolidated `layer-08-memory-session-state.md` policy document, and the workshop plan of record (`04-advanced_ux.md`)[cite: 5]. *(Note: Cloud Build `includedFiles` path filters will be added during Phase 5 Production Hardening so doc/script commits do not trigger service rebuilds[cite: 4].)*
* **Constant Patient Name Sidebar Label (`ff00550`):** Enforced the rule that the patient name is the sole, constant thread label[cite: 2]. Removed the unreachable `Conversation.title` field, rename route, and 5 legacy rename tests via migration `0010_remove_conversation_title`[cite: 2].
* **Sub-700px Responsive Sidebar Drawer & Accessibility (`c2c0def`):** Wired the top-left toggle icon (`◧`) to open the sidebar as an overlay drawer below `700px` and renamed its accessible labels to `"Open navigation"` / `"Close navigation"` / `"Collapse navigation"` / `"Expand navigation"` so it never collides with the top-right `"Open menu"` account disclosure.

---

## 1. Real-Time Streaming, Safety Swap Protocol & Client Pacing
**Objective:** Achieve instant Time-To-First-Token (TTFT) and deliberate, legible visual pacing without compromising post-generation clinical guardrails[cite: 1, 3].

### 1.1 Lookahead-Gated Streaming Architecture (Status: Completed — `1c84d67`)
* **ReAct Lookahead Buffer (`PROSE_LOOKAHEAD_CHUNKS = 1`, lowered from 3 on 2026-10-15):** In `graph.py`, `agent_node` inspects `tool_call_chunks` across the leading chunks of a turn. If a tool call is present, routing deliberation remains completely masked behind the Stage 1 stepper[cite: 1]. If no tool call appears, the held prefix releases immediately and streams live via `answer_delta`[cite: 1]. The hold was lowered because everything it holds is released as one burst: at three chunks it put 72% of an answer on screen in the first paint, leaving the client cadence almost nothing to pace. See §1.4 for the measurement. One chunk still keeps routing off the screen, because a turn that calls a tool opens with the call.
* **Prose, not the envelope (`8ba5dbf`):** The announced text is the chunk's own text, read out of the content blocks Gemini actually answers in. Before this the announcement was the *repr* of the envelope — measured live on `/ehr` as **6,116 characters announced for a 1,021-character answer**. The commit is part of 4A-1 rather than a follow-up to it: 4A-1 streamed the wrong string correctly.
* **Command Stream Separation (`streams_prose` vs. `prose_is_model_output`):** In `routing.py` and `deterministic.py`, `/ehr`, `/pubmed`, and free-text paths stream model prose via `astream`, whereas deterministic `/meds` extraction is structurally excluded from streaming raw JSON to the screen[cite: 1].

### 1.2 Dual Swap Protocol & Transport Isolation (Status: Completed)
* **Dedicated `prose_settled` Frame:** Rather than polluting the `answer` payload or `Turn.presentation` persistence in Postgres, post-hoc guardrail reconciliation emits a dedicated `prose_settled` SSE frame immediately prior to the terminal `answer` frame whenever streamed prose differs from guarded prose.
* **Silent Swap (`"silent"`):** Systemic envelope unwrapping, whitespace normalization, or citation formatting reconciles silently in the DOM.
* **Visible Safety Swap (`"visible"`):** When `guardrail.py` strips or modifies an unverifiable medication dose, frequency, ungrounded risk figure, or unresolvable citation, the client replaces the provisional text with the guarded output and triggers a visible verification notice in the **Evidence-Check** badge[cite: 1, 3].

### 1.3 4-Stage Progressive Rendering Sequence
1. **Stage 1 — Live Cognitive Stepper:** Immediately render and update the collapsible stage progress header (*Analyzing query* → *Searching clinical records & literature* → *Checking answer against evidence*).
2. **Stage 2 — Token-by-Token Prose Streaming (`answer_delta`):** Stream the synthesized clinical narrative into the active Turn Block[cite: 1].
3. **Stage 3 — Structured A2UI Surface Reveal:** Smoothly mount structured A2UI surfaces (`MedsGrid`, `RiskWidget`, `LiteratureList`, `LabTrendCard`, and evidence matrices) with an **Export / Copy** action toolbar in the top-right corner.
4. **Stage 4 — References & Follow-Ups Reveal:** Reveal the collapsible **References** accordion drawer and the **Follow-Up Questions** list at the foot of the turn.

### 1.4 Client-Side Streaming Cadence & Pacing Queue (Milestone 4A-1b — Next Up)
* **Why Client-Side Only:** Buffering on the agent or Django proxy degrades TTFT. Bytes must arrive in the browser at full wire speed while `copilot_canvas.js` paces the DOM painting.
* **Four Pacing Controls in `copilot_canvas.js`:**
  1. **Character Budget & Tick Interval:** Drain queued `answer_delta` characters at a deliberate, human-legible reading cadence (target chars/sec).
  2. **Queue Catch-Up Multiplier:** Dynamically accelerate the drain rate when the queued buffer exceeds a threshold length so long syntheses never lag far behind model completion.
  3. **Immediate Flush on Settle:** When `prose_settled` / `fillTurn` arrives, stop the animation timer and immediately render the authoritative settled text.
  4. **Accessibility Bypass:** Respect `prefers-reduced-motion: reduce` by skipping the pacing animation and rendering chunks directly.
* **Clinical Legibility Note:** Because a paced draft is easier to read while streaming, clinicians are more likely to read provisional spans before `prose_settled` fires—making the **Visible Safety Swap** and **Evidence-Check** badge notice essential safety anchors[cite: 1, 3].

**Measured on the running console, 2026-10-15.** Instrumented with a `MutationObserver` on the turn's prose node, so these are *draw* times rather than timings inferred from the wire. The first column exposed the problem; the second is after it was fixed.

| Observation | Before | After `PROSE_LOOKAHEAD_CHUNKS = 1` |
|---|---|---|
| Characters in the first paint | **426** | **102** |
| Total settled characters | 593 | 535 |
| First paint as a share of the answer | **72%** | **19%** |
| Draw events across the stream | 55 | 131 |
| Growth after the first paint | ~50 chars/sec | ~50 chars/sec, against a 45 target |
| Asterisks or `^[n]` visible in the draft | not measured | **0** |

**The first measurement is what set the fix.** A three-chunk hold released 426 characters in one burst — 72% of the answer on screen before the cadence had anything to do — so the lever was the hold depth, not the client rate. That is a two-character change in `graph.py`, and the agent suite passes unchanged at 709 tests. At one chunk the first paint is a single chunk and the cadence governs the remaining 81%, which is legible in the draw-event count rising from 55 to 131.

The cost is named in `graph.py` and is real: one chunk covers a turn that *opens* with a tool call, which is how ReAct routing behaves, but not a model that opens with prose and calls the tool later. `_prose_settled` reports that case on every streamed turn rather than it passing unnoticed.

---

### 1.5 Evidence Is Guaranteed, Not Requested (4A-1b)

**Found while verifying §1.4, 2026-10-15. It outranks everything above it, and it is not a canvas problem.** An answer about the patient's chart was served with **no citations at all**, because the turn never retrieved anything.

| Turn | `tool_calls` | `citations` | `guardrail_flags` |
|---|---|---|---|
| 36 | `rag_search_sections` | 3 sections | none |
| 38 | `rag_search_sections` | 4 sections | none |
| **34** | **none** | `[]` | `citation_out_of_range:^1, ^2, ^3` |
| **40** | **none** | `[]` | `citation_out_of_range:^1, ^2, ^3` |

Turns 34 and 40 answered a discharge-summary question having called no tool, so the presentation composed with zero citation slots, and the `^[1]..^[3]` the model wrote anyway were stripped by the citation guard — correctly, since they pointed at nothing. The reader got a confident summary of the chart with no evidence attached.

**The same stripping is what showed the Evidence-Check badge.** The badge fires on a *visible* swap, which requires the guardrails to have removed something; on these turns the thing removed was the citation. So the badge was reporting a lost citation, and reading it as a guardrail catching an unverifiable dose — which §1.4's note invited — was wrong. That misreading is recorded here rather than quietly corrected because it is the more dangerous of the two: the badge was right and the system was wrong beneath it.

`routing.py` names the gap: *"Free text is the one path where the model chooses the tool."* Every other path is deterministic — `/ehr` always calls `rag_search_sections` — so free text was the only path that could answer without reading the chart. It is intermittent: turns 34 and 36 asked the same question 73 seconds apart and only one retrieved.

**The fix, and the wrong fix first.** Binding the first turn with `tool_choice="any"` produces citations and is wrong, which two existing tests caught. A free-text risk or medication question is answered by *naming* the command that owns it and calling no tool — that boundary is precisely why those two tools are not bound — so forcing a call made the loop retrieve and answer the very question the boundary refuses. What shipped instead:

- the first turn stays on `AUTO`, so a redirect is still a redirect;
- if a run ends with text, no tool called, and no command named, the graph **demands** a retrieval once — `tool_choice="any"` for that turn only — and re-answers;
- a turn with no text is a failure (truncated, filtered) and is reported as one rather than re-prompted, because demanding a retrieval there replaced the real failure with a second one.

Verified live: the chart question returns **4 citation controls and no badge** (nothing stripped, `guardrail_flags: []`), and the risk question returns **0 citations** while still naming `/risk`. Agent suite **709 passed, 1 skipped**, with the redirect tests unchanged.

**Deferred, not skipped (4A-2, with the dual-scope classifier).** A safety net so a turn ending with no resolvable citations cannot be served as chart synthesis. It waits because the check must know whether a turn is *Patient Chart Synthesis* or *General Medical Evidence* — a general-medical answer legitimately cites nothing — and that classifier is §2's. Until then the citation guard's stripping, plus the demand above, is what stands in for it.

---

## 2. OpenEvidence-Style Follow-Ups, Inline Citations & Reference Drawers (Milestone 4A-2)
**Objective:** Replace primitive button chips with authoritative, low-friction clinical navigation and source verification.
* **OpenEvidence-Style Follow-Up Questions Table:**
  * Extend the agent response contract (and `Turn.presentation`) to emit 3 contextual follow-up inquiries tailored to the active patient and current turn.
  * Render a clean **Follow-Up Questions** section at the bottom of the newest turn block: full-width clickable rows separated by subtle horizontal hairlines (`1px solid #e2e8f0`), each displaying the natural-language question on the left and a right-pointing chevron (`>`) on the right. Clicking any row immediately submits that question as the next turn in the thread.
* **Inline Citation Pills & Collapsible References Drawer:**
  * Format inline citations as scannable pills (e.g., `[1]`, `[Discharge Note · HPI]`, or `[JAMA +1]`) that highlight or preview the source on hover/click.
  * Consolidate full source metadata at the bottom of each turn inside a collapsible **References** accordion drawer (numbered items with clickable PMID/DOI links or EHR section provenance).
* **Dual-Scope Turn Badging (Patient Chart vs. General Medical Evidence):**
  * While keeping the session pinned to the active `hadm_id`, allow the orchestrator to answer general medical or condition-level inquiries. Badge the turn header subtly as **Patient Chart Synthesis** vs. **General Medical Evidence** so the clinician never confuses general literature with a claim from the patient's chart.

---

## 3. Unified EHR + Risk Synthesis, Multi-Source Literature Matrix & Feedback Traces (Milestone 4B)
**Objective:** Pair predictive patient scores directly with clinical narrative, synthesize condition-level readmission risk across medical literature, and capture granular clinician feedback traces for Phase 6 Evals[cite: 1].
* **Combined `/ehr` + `/risk` View:** Running `/ehr` (or opening a comprehensive clinical summary) automatically renders the 30-day readmission **Risk Chip & Top Drivers** alongside the cited discharge narrative so the clinician sees *what* the risk is and *why* in one unified turn[cite: 1].
* **Clickable Risk Factors (Direct Manipulation):** Make individual risk driver bars in the `RiskWidget` interactive—clicking a specific driver (e.g., *Length of Stay*, *Prior Admissions*, *Comorbidity Burden*) auto-submits a targeted follow-up query investigating that clinical factor across the patient's notes and medications[cite: 1].
* **OpenEvidence-Style Multi-Source Condition Risk Matrix (`/pubmed` + `LiteratureList` Upgrade):**
  * When a clinician queries readmission risk, complication rates, or outcomes for a specific condition (e.g., *severe lumbar spinal stenosis with neurogenic claudification*, *post-op arthroplasty*, or *acute decompensated heart failure*), synthesize and render a **Structured Literature Risk Matrix Table** across 6–12 retrieved studies:
    * **Study Title & First Author**
    * **Source Journal & Publication Year**
    * **Evidence Tier Badge** (`Guideline`, `Meta-Analysis`, `RCT`, `Observational`)[cite: 1]
    * **Extracted Risk / Rate (%) & Cohort Thresholds**
* **Reference-Level & Turn-Level Feedback Traces (`👍` / `👎` for Phase 6 Evals):**
  * Render unobtrusive thumbs-up / thumbs-down (`👍` / `👎`) feedback buttons on each individual reference row (in both the Literature Matrix and References drawer) as well as on the overall turn synthesis.
  * Persist feedback events with `(conversation_id, turn_id, trace_id, pmid_or_source_id, rating)` so retrieval relevance and synthesis quality are directly measurable in the **Phase 6 Evaluation Suite**.

---

## 4. Clinical Curiosity & Pharmacology Integration (Milestone 4C)
**Objective:** Satisfy point-of-care questions about unfamiliar medications, dosing, and safety warnings without forcing clinicians to parse raw primary research abstracts[cite: 1].
* **OpenFDA / DailyMed & RxNorm Pharmacology Tool (`/drug`):** Add an MCP tool querying structured federal drug monographs (Indications & Usage, Boxed Warnings, Renal/Hepatic Dosing, Contraindications, Drug Interactions) via public OpenFDA/RxNorm APIs[cite: 1].
* **Interactive `MedsGrid` Drill-Down:** Make medication rows in the `MedsGrid` interactive. Clicking an unfamiliar medication triggers a structured **Clinical Pharmacology Card** (or pre-filled follow-up turn) that cross-references the drug's FDA monograph against the active patient's chart[cite: 1].

---

## 5. DUA-Compliant Synthetic Longitudinal Labs & Vitals (`/labs`) (Milestone 4C)
**Objective:** Complete the inpatient clinical triad (Notes + Meds + Labs) using a PhysioNet DUA-compliant synthetic physiological trajectory generator (zero raw MIMIC-IV `labevents` redistribution).
* **Phenotype-Conditioned Synthetic Lab Generator:**
  * Build a deterministic, seeded synthesis script (`hash(hadm_id, analyte)`) that generates 4–6 serial admission-to-discharge lab timepoints per demo cohort admission (Creatinine, eGFR, Potassium, Sodium, Hemoglobin, WBC, and conditional NT-proBNP)[cite: 1].
  * Condition each patient's trajectories on their existing synthetic admission profile, discharge summary narrative, and medication list (e.g., post-op hemoglobin dip and recovery for surgical admissions; upward creatinine/potassium drift for high-risk cardiorenal admissions) using textbook reference intervals and bounded parametric walks.
* **`LabTrendCard` A2UI Surface:** Render compact lab tables with inline SVG sparklines, admission-to-discharge delta indicators, reference ranges, and abnormal flags (`H` / `L` / `Critical`) so the agent can correlate medication safety and readmission risk with objective lab trends[cite: 1].

---

## 6. Ergonomic Multi-Line Composer & Table Export Polish (Milestone 4A-2)
**Objective:** Match the input and data-handling ergonomics of frontier AI workspaces[cite: 1].
* **Auto-Expanding Multi-Line Composer (Implemented, uncommitted):** The composer is now a `<textarea rows="1">` that grows vertically upward as content wraps, to a **200px ceiling** declared once in the stylesheet and read from there by the script, past which it scrolls internally. Measured: **43px empty → 66px wrapped → 90px after one `Shift+Enter` → exactly 200px at 16 lines** (`scrollHeight` 762, `overflow-y: auto`), and back to 43px on submit.
* **Keyboard & Submit Mechanics (Implemented):** `Enter` sends and `Shift+Enter` breaks the line, in explicit script — a `textarea` does not submit on `Enter`, and silently losing that was the one real regression risk in this change. The handler is bound to the form, not the box, so it runs after the `@` and `/` menus and yields to them by reading `defaultPrevented`; a question is not sent while the reader is still choosing the patient it is about.
* **Pill and control anchoring (Implemented):** the pill and send button drop to the baseline once the box wraps, and the bar's 999px radius relaxes to 24px, so a tall composer is not a lozenge.
* **Table Export Affordance (Not started):** unchanged, and still owed on `MedsGrid`, `LabTrendCard` and the Literature Risk Matrix.