# 04A-updated_advanced_ux_directives.md

## Master Architectural Directive: Phase 4 Advanced UX Closeout

### Executive Summary & Design Rationale
This sprint finalizes the clinical presentation layer (Milestone 4B closeout through Phase 4D) before shifting focus to the Langfuse evaluation harness and infrastructure hardening.

To prevent cognitive overload and avoid an unformatted vertical dump of components, the user experience follows a **3-Pillar Progressive Disclosure Model**:
1. **`/ehr` (Anchor Turn):** Unified clinical narrative (4 block sections) + quantitative risk chip & interactive driver bars (`RiskInfluencers`).
2. **`/pubmed` (Comparative Evidence Matrix — *The Killer App*):** Side-by-side benchmarking of the patient's proprietary ML prediction against peer-reviewed cohort baselines (JAMA, Circulation, NEJM), driver alignments, and delta synthesis.
3. **`/meds` (Medication Reconciliation):** Direct review via the structured `MedsGrid`.
4. **Contextual Drill-Downs (No Top-Level Slash Clutter):** `/labs` and pharmacology cards are not separate top-level slash commands; they render as context-aware drill-down surfaces triggered when clicking driver bars or suggested follow-up rows.
5. **Unauthenticated Sandbox (`/demo/`):** A zero-latency, fixture-backed interactive preview for anonymous users, protected by a client-side OAuth modal on free-text entry or explicit sign-in intent.

---

### Phase 1: Immediate `/ehr` Visual & Contract Fixes (15 mins)

1. **Fix Numeric Float Leak (`deterministic.py` / `routing.py`):**
   * In `_presented_prediction`, add pre-formatted display strings:
     * `probability_display = f"{val * 100:.1f}%"` (e.g., `"33.4%"`)
     * `threshold_display = f"{thresh * 100:.1f}%"` (e.g., `"11.0%"`)
   * Update `routing.EHR_FORMATTING` to instruct the model:
     > *"Reference readmission risk using only the provided `probability_display` and `threshold_display` strings or describe the risk qualitatively. Never output unrounded raw floating-point numbers."*

2. **Enforce Section Block Typography (`demo_shell.css`):**
   * Replace fragile markdown paragraph matching with an explicit block selector:
     ```css
     .cp-narrative p > strong:first-child {
       display: block;
       font-size: 0.95rem;
       font-weight: 600;
       margin-top: 1.25rem;
       margin-bottom: 0.25rem;
     }
     ```
   * Ensure the four clinical sections (*Hospital course*, *Discharge diagnosis*, *Discharge instructions*, *Readmission risk & trajectory*) render as distinct blocks with consistent paragraph spacing (`line-height: 1.6`, `margin-bottom: 0.75rem`).

3. **Slash Bar Rationalization:**
   * Hide the redundant `/risk` chip from the composer input bar in `a2ui_console.html` / `copilot_canvas.js`.
   * Keep `/risk` aliased and active in backend routing so all test assertions continue to pass without regression.

---

### Phase 2: The "Killer App" — Comparative Literature Risk Matrix on `/pubmed`

Transform `/pubmed` (and the follow-up prompt: *"Compare this patient's readmission risk against peer-reviewed clinical cohorts"*) into a structured comparative evidence surface: `<a2ui-comparative-risk-matrix>`.

1. **Data Contract & Curated Benchmarks:**
   * Attach structured benchmark metadata to the three captured literature fixture files (`literature_chf_readmission.json`, `literature_copd_exacerbation.json`, `literature_lumbar_stenosis.json`):
     * `reported_rate` (e.g., `"22.8%"`)
     * `cohort_summary` (e.g., `"Acute decompensated HF with prior hospitalization within 6 months"`)
     * `matched_drivers` (e.g., `["Prior inpatient days", "RDW"]`)
     * `evidence_tier` (e.g., `"Observational Cohort"`)
     * `pmid` / `url`

2. **Frontend Matrix Component (`a2ui_clinical_components.js`):**
   * Build `<a2ui-comparative-risk-matrix>`:
     * **Row 0 (Pinned Patient Baseline):**
       * Style with a subtle clinical accent border and highlight background.
       * Columns: `Source: Enterprise Clinical Copilot (XGBoost)` | `Risk Benchmark: 33.4% (Above 11.0% threshold)` | `Cohort: Index Admission` | `Matched Drivers: Prior Inpatient Days, RDW, RBC` | `Provenance: Model v2.4`.
     * **Rows 1..N (Peer-Reviewed Cohorts):**
       * Columns: `Source: Journal · Year` + clickable `[PMID]` link | `Cohort Rate: Tabular %` | `Cohort Criteria` | `Matched Patient Risk Factors (Pills)` | `Abstract Excerpt`.
   * Wire the top-right `[📥 Export]` button to export this table cleanly as CSV/TSV.

3. **Narrative Risk Delta Synthesis:**
   * Update the synthesis prompt for comparative turns to explain the *risk delta* between the proprietary ML score and published literature ranges based on the patient's individual drivers.

---

### Phase 3: Contextual Lab & Medication Drill-Downs

Avoid top-level slash clutter. Attach rich clinical cards to interactive actions across both fixture and live modes:

1. **Driver Bar Clicks $\rightarrow$ `LabTrendCard`:**
   * Clicking a lab-related driver bar (e.g., *RDW* or *Lowest/recent RBC*) dispatches to `askFollowUp` and renders:
     * Narrative explanation of the biomarker trajectory.
     * A compact synthetic `LabTrendCard` displaying admission, peak, and discharge values, normal reference ranges, and abnormal direction flags.

2. **Suggested Follow-Up Clicks $\rightarrow$ Clinical Callout / `MedsGrid`:**
   * Ensure clicking clinical follow-up rows (e.g., *Dual antiplatelet therapy regimen* or *Glycemic control transition*) renders the synthesized clinical guidance paired with a structured `MedsGrid` or medication timeline table.

---

### Phase 4: Unauthenticated Interactive Showcase at `/demo/`

Deliver an immediate, zero-latency showcase for hiring managers, recruiters, and evaluators without burning GCP quotas:

1. **Serve Interactive Fixture Mode:**
   * Route `/demo/` to render `a2ui_console.html` pre-configured with `fixture_mode = True`.
   * Enable full client interaction: switching `@patient` profiles, running `/ehr`, `/pubmed`, `/meds`, clicking driver bars, and selecting suggested follow-up rows.
   * Maintain realistic streaming pacing without making external LLM or database calls.

2. **Conversion Gate for Free-Text Queries:**
   * Render a subtle status banner:
     `⚡ Interactive Demo Preview (Simulated Data) · [Sign in for Live Agent — 20 Free Credits]`
   * Intercept free-text submissions or clicks on `[Sign in]` with a modal offering:
     * `[Continue with Google]`
     * `[Continue with Microsoft]`
   * Point authentication routes to the social OAuth endpoints established in the infrastructure hardening track.

---

#### Execution Order & Verification
Execute **Phase 1 → Phase 2 → Phase 3 → Phase 4**. Pause after Phase 2 to verify the Comparative Risk Matrix in the browser before proceeding to Phases 3 and 4.