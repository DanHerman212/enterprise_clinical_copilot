# Phase 4: Clinical Intelligence, Pharmacology & Advanced UX

## Executive Summary
Phase 4 elevates the Copilot from a multi-turn chart reader into an interactive point-of-care clinical decision support system. It unifies the EHR and Risk views, restores real-time token streaming with safety swap protocols, introduces authoritative pharmacology and lab-trend tools to satisfy clinical curiosity beyond PubMed, and upgrades the composer and A2UI widgets for low-friction clinical workflows.

## 1. Real-Time Streaming & Safety Swap Protocol
**Objective:** Achieve instant Time-To-First-Token (TTFT) without compromising post-generation clinical guardrails.
* **Synthesis Token Streaming (`answer_delta`):** Stream the final synthesis turn token-by-token across `/ehr`, `/pubmed`, and free-text paths (keeping intermediate ReAct tool routing masked behind live stage chips, and keeping deterministic `/meds` code-generation instant).
* **Dual Swap Protocol:**
  * **Silent Swap:** Silently replace systemic envelope unwrapping or standard fallback strings.
  * **Visible Safety Swap:** If a guardrail strips or modifies an unverified dose, risk number, or hallucinated PMID, replace the provisional stream with the guarded text and trigger a visible notice in the **Evidence-Check** badge.

## 2. Unified EHR + Risk Synthesis & Interactive Drivers
**Objective:** Eliminate standalone, context-free risk numbers by pairing predictive scores directly with clinical narrative and interactive drill-downs.
* **Combined `/ehr` + `/risk` View:** Running `/ehr` (or opening a clinical summary) automatically renders the 30-day readmission **Risk Chip & Top Drivers** alongside the cited discharge narrative so the clinician sees *what* the risk is and *why* in one unified turn.
* **Clickable Risk Factors:** Make individual risk driver bars interactive—clicking a factor auto-submits a contextual follow-up query investigating that specific clinical driver in the patient's notes and meds.

## 3. Clinical Curiosity & Pharmacology Integration (Beyond PubMed)
**Objective:** Satisfy point-of-care questions about unfamiliar medications, dosing, and general medical concepts without forcing clinicians to parse raw PubMed abstracts.
* **OpenFDA / DailyMed Pharmacology Tool (`/drug`):** Add an MCP tool querying structured FDA drug monographs (Indications, Boxed Warnings, Renal/Hepatic Dosing, Contraindications, Interactions) via public OpenFDA/RxNorm APIs.
* **Interactive `MedsGrid` Drill-Down:** Make medication rows in the `MedsGrid` clickable. Clicking an unfamiliar drug opens a structured **Clinical Pharmacology Card** (or triggers a pre-filled follow-up) that cross-references the drug's monograph against the active patient's chart.
* **Evidence Tier Badging on `/pubmed`:** Parse PubMed `PublicationType` metadata to display visual trust badges (`Guideline`, `Meta-Analysis`, `RCT`, `Observational`) on every `LiteratureList` card, plus context-aware query expansion using the patient's active problems.

## 4. Warehouse Expansion: Longitudinal Labs & Vitals (`/labs`)
**Objective:** Complete the inpatient clinical triad (Notes + Meds + Labs) using MIMIC-IV warehouse data.
* **Curated Lab & Vital Trajectories:** Expose key admission-to-discharge lab trends (e.g., Creatinine/eGFR, Potassium, Sodium, Hemoglobin, WBC, NT-proBNP) for the cohort.
* **`LabTrendCard` A2UI Surface:** Render compact lab tables with inline SVG sparklines and abnormal range badges (`H` / `L` / `Critical`) so the agent can correlate medication safety and readmission risk with objective lab trends.

## 5. Ergonomic Composer & Direct-Manipulation Polish
**Objective:** Match the input ergonomics of frontier AI workspaces.
* **Auto-Expanding Multi-Line Composer:** Upgrade the bottom composer to grow vertically upward with content (up to a ~200px / 7-line ceiling before internal scrolling), supporting `Shift+Enter` for newlines, `Enter` to submit, and instant snap-back to single-line height on send.