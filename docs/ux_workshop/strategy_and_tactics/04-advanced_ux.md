# Phase 4: Clinical Intelligence, Streaming Architecture & Advanced UX

## Executive Summary
Phase 4 elevates the Enterprise Clinical Copilot from a stateful chart reader into a real-time, point-of-care clinical decision support workspace that rivals OpenEvidence and Gemini. It closes the three remaining Phase 3 hygiene items, implements end-to-end token and component streaming with clinical safety swap protocols, introduces OpenEvidence-style follow-up navigation and citation drawers, unifies the EHR and Risk views, and expands the MCP toolset with authoritative pharmacology (`/drug`) and longitudinal lab trends (`/labs`).

---

## Execution Roadmap (Milestones)
To maintain velocity and verify core protocol changes before adding new tools, Phase 4 is executed in three sequential milestones:
* **Milestone 4A (Core Experience):** Phase 3 Carry-Over Hygiene (§0), Real-Time Streaming & Safety Swap Protocol (§1), OpenEvidence-Style Follow-Ups & Citation Drawers (§2), and Auto-Expanding Composer (§6).
* **Milestone 4B (Risk Synthesis):** Unified `/ehr` + `/risk` View & Interactive Driver Bars (§3).
* **Milestone 4C (Clinical Depth):** OpenFDA/RxNorm Pharmacology Tool (`/drug`) (§4) and Longitudinal Warehouse Lab Trends (`/labs`) (§5).

---

## 0. Phase 3 Carry-Over Hygiene (Immediate Closeout)
**Objective:** Close the three known items carried over from the Phase 3 sign-off before modifying the streaming pipeline.
* **Agent Repository Git Hygiene:** Commit and push the patched `scripts/agent/retention.py` (`--note-cache-dir` fix) and the consolidated `layer-08-memory-session-state.md` policy document so the agent repository starts Phase 4 with a clean working tree. Exclude/ignore untracked scratch workshop files.
* **Sidebar Thread Label Resolution (`Conversation.title`):** Wire the sidebar thread row label to render `conversation.title || conversation.patient_name`. By default, every patient thread displays the clean Patient Name; if a clinician explicitly renames the thread via the `⋮` hover menu, the custom `Conversation.title` is displayed and persists across reloads.
* **Sub-700px Responsive Sidebar Drawer:** Wire the top-left sidebar toggle icon (`◧`) so that on viewports `< 700px`, the sidebar collapses into an on-demand overlay drawer (`position: fixed`, backdrop scrim) rather than disappearing without a navigation affordance or crushing the composer.

---

## 1. Real-Time Streaming & Clinical Safety Swap Protocol
**Objective:** Achieve instant Time-To-First-Token (TTFT) and intentional, OpenEvidence-grade visual pacing without compromising post-generation clinical guardrails.
* **End-to-End Streaming Transport (Agent → Site Proxy → Browser):**
  * Stream Server-Sent Events (SSE / `ndjson` event frames) from the agent through the Django proxy to the browser, ensuring `Turn.presentation` continues to persist the final composed A2UI envelope in Postgres upon stream completion.
* **4-Stage Progressive Rendering Sequence:**
  1. **Stage 1 — Live Cognitive Stepper:** Immediately render and update the collapsible stage progress header (e.g., *Analyzing query* → *Searching MIMIC-IV warehouse & literature* → *Checking answer against evidence*).
  2. **Stage 2 — Token-by-Token Prose Streaming (`answer_delta`):** Stream the synthesized clinical narrative token-by-token into the active Turn Block across `/ehr`, `/pubmed`, and free-text paths (while keeping intermediate ReAct tool-routing thoughts masked behind the Stage 1 stepper, and keeping deterministic `/meds` extraction fast).
  3. **Stage 3 — Structured A2UI Surface Reveal:** Smoothly mount and fade in structured A2UI surfaces (`MedsGrid`, `RiskWidget`, `LiteratureList`, `LabTrendCard`, and comparison tables) with an **Export / Copy** action toolbar in the top-right corner of tabular surfaces.
  4. **Stage 4 — References & Follow-Ups Reveal:** Once synthesis completes, reveal the collapsible **References** accordion drawer and the **Follow-Up Questions** list at the foot of the turn.
* **Dual Swap Protocol (Reconciling Streaming with Post-Hoc Guardrails):**
  * **Silent Swap:** If post-generation processing only unwraps systemic JSON envelopes, normalizes whitespace/citations, or applies standard fallback formatting, reconcile the DOM silently without visual disruption.
  * **Visible Safety Swap:** If a post-hoc clinical guardrail strips or modifies an unverified medication dose, ungrounded risk figure, or hallucinated PMID/citation, replace the provisional streamed text with the guarded output and trigger a visible verification notice in the **Evidence-Check** badge.

---

## 2. OpenEvidence-Style Follow-Ups, Inline Citations & Reference Drawers
**Objective:** Replace primitive button chips with authoritative, low-friction clinical navigation and source verification.
* **OpenEvidence-Style Follow-Up Questions Table:**
  * Extend the agent response contract (and `Turn.presentation`) to emit 3 contextual follow-up inquiries tailored to the active patient and current turn.
  * Render a clean **Follow-Up Questions** section at the bottom of the newest turn block: full-width clickable rows separated by subtle horizontal hairlines (`1px solid #e2e8f0`), each displaying the natural-language question on the left and a right-pointing chevron (`>`) on the right. Clicking any row immediately submits that question as the next turn in the thread.
* **Inline Citation Pills & Collapsible References Drawer:**
  * Format inline citations as clean, scannable pills (e.g., `[1]`, `[Discharge Note · HPI]`, or `[JAMA +1]`) that highlight or preview the source on hover/click.
  * Consolidate full source metadata at the bottom of each turn inside a collapsible **References** accordion drawer (numbered items with clickable PMID/DOI links or EHR section provenance).
* **Dual-Scope Turn Badging (Patient-Specific vs. General Medical Reference):**
  * While keeping the session strictly pinned to the active `hadm_id`, allow the orchestrator to answer general medical or textbook inquiries. Badge the turn header or status line subtly as **Patient Chart Synthesis** vs. **General Medical Evidence** so the clinician never confuses general literature with a claim from the patient's chart.

---

## 3. Unified EHR + Risk Synthesis & Interactive Drivers
**Objective:** Eliminate standalone, context-free risk numbers by pairing predictive scores directly with clinical narrative and interactive drill-downs.
* **Combined `/ehr` + `/risk` View:** Running `/ehr` (or opening a comprehensive clinical summary) automatically renders the 30-day readmission **Risk Chip & Top Drivers** alongside the cited discharge narrative so the clinician sees *what* the risk is and *why* in one unified turn.
* **Clickable Risk Factors (Direct Manipulation):** Make individual risk driver bars in the `RiskWidget` interactive—clicking a specific driver (e.g., *Length of Stay*, *Prior Admissions*, *Comorbidity Burden*) auto-submits a targeted follow-up query investigating that clinical factor across the patient's notes and medications.

---

## 4. Clinical Curiosity & Pharmacology Integration (Beyond PubMed)
**Objective:** Satisfy point-of-care questions about unfamiliar medications, dosing, and safety warnings without forcing clinicians to parse raw primary research abstracts.
* **OpenFDA / DailyMed & RxNorm Pharmacology Tool (`/drug`):** Add an MCP tool querying structured federal drug monographs (Indications & Usage, Boxed Warnings, Renal/Hepatic Dosing, Contraindications, Drug Interactions) via public OpenFDA/RxNorm APIs.
* **Interactive `MedsGrid` Drill-Down:** Make medication names in the `MedsGrid` interactive. Clicking an unfamiliar medication triggers a structured **Clinical Pharmacology Card** (or pre-filled follow-up turn) that cross-references the drug's FDA monograph against the active patient's diagnoses and renal/hepatic status.
* **Evidence Tier Badging & Context-Aware `/pubmed`:**
  * Automatically parameterize `/pubmed` searches with relevant context from the active patient's problem list when appropriate.
  * Parse PubMed `PublicationType` metadata to display visual trust badges (`Clinical Guideline`, `Meta-Analysis / Systematic Review`, `RCT`, `Observational`) on every `LiteratureList` reference.

---

## 5. Synthetic Longitudinal Labs & Vitals (`/labs`)
**Objective:** Complete the inpatient clinical triad (Notes + Meds + Labs) using a DUA-compliant synthetic physiological trajectory generator (zero raw MIMIC-IV `labevents` redistribution).
* **Phenotype-Conditioned Synthetic Lab Generator:**
  * Build a deterministic, seeded synthesis script (`hash(hadm_id, analyte)`) that generates 4–6 serial admission-to-discharge lab timepoints per demo cohort admission (Creatinine, eGFR, Potassium, Sodium, Hemoglobin, WBC, and conditional NT-proBNP).
  * Condition each patient's trajectories on their existing synthetic admission profile, discharge summary narrative, and medication list (e.g., post-op hemoglobin dip and recovery for surgical admissions; upward creatinine/potassium drift for high-risk cardiorenal admissions) using textbook reference intervals and bounded parametric walks.
* **`LabTrendCard` A2UI Surface:** Render compact lab tables with inline SVG sparklines, admission-to-discharge delta indicators, reference ranges, and abnormal flags (`H` / `L` / `Critical`) so the agent can correlate medication safety and readmission risk with objective lab trends.

---

## 6. Ergonomic Multi-Line Composer & Table Export Polish
**Objective:** Match the input and data-handling ergonomics of frontier AI workspaces.
* **Auto-Expanding Multi-Line Composer:** Upgrade the bottom prompt composer `<textarea>` to grow vertically upward as content wraps (up to a `~200px` / 7-line max-height ceiling before transitioning to an internal scroll).
* **Keyboard & Submit Mechanics:** Support `Enter` to submit, `Shift+Enter` to insert a newline, and instant snap-back to the single-line baseline height upon submission while keeping the `@patient` pill and send button cleanly anchored.
* **Table Export Affordance:** Include a lightweight `Export` (CSV/Copy) button in the header of structured A2UI tables (`MedsGrid`, `LabTrendCard`) matching the OpenEvidence table pattern.