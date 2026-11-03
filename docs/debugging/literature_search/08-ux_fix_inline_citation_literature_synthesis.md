# UX & Clinical Presentation Directive: OpenEvidence-Parity Inline Citations, Literature Risk Synthesis, Visible References Block, and Multi-Page Print Support

## Context & Observed UX Gaps (Comparing ECC vs. OpenEvidence PDFs)
1. **Multi-Page Print Clipping (`1/1` Page Cutoff):**
   - When printing `/demo/a2ui/` (`Cmd+P` / Save as PDF), the fixed `100vh` app shell and inner scroll container clip the output at the bottom of Page 1, and the left patient sidebar and bottom composer input bar print over the document.
   - In contrast, OpenEvidence prints the full clinical narrative, comparative rate table, risk factors, numbered References list, and Follow-Up Questions cleanly across multiple pages (`1/3`, `2/3`, `3/3`).
2. **Repetitive, Bulky Inline Citation Badges:**
   - In `/ehr` prose, sentences repeatedly render long inline badges such as `2 · Brief hospital course` three times in a single paragraph, breaking reading flow.
   - We need compact, scannable inline citation pills, deduplication of redundant consecutive same-source badges within a paragraph when consecutive sentences share the same single source, and clean grouping when multiple sources back a statement.
3. **Bridging Individual ML Risk & Verified Literature Cohorts in Prose (`Readmission risk & trajectory`):**
   - Currently, the `/ehr` synthesis prompt receives `rag_search_sections` and `predict_readmission`, but does not reference the verified PubMed 30-day comparator cohorts (`comparators`) that appear right below it in `ComparativeRiskMatrix`.
   - When verified `comparators` exist on `/ehr`, the `Readmission risk & trajectory` section should concisely synthesize both the patient's individual ML risk + drivers AND the published 30-day readmission benchmark range/rates from the verified cohorts (citing their source numbers and noting that published cohorts represent population benchmarks alongside the patient's individualized estimate). When `comparators` is empty, it should focus cleanly on the individual estimate and drivers without inventing literature figures.
4. **Visible End-of-Turn `References` Section:**
   - Currently, full reference metadata is only visible if the user opens the citations drawer, making printed PDFs and full-page reads lack a bibliography.
   - Render a clean, numbered **References** section at the bottom of the composed turn (matching OpenEvidence's References block: showing the numbered note sections and full PubMed study citations with title, journal, year, and PMID) so the page is self-contained on screen and in print.

---

## Authorized Scope
- `services/agent/engine/deterministic.py` & `services/agent/engine/prompts.py` (to pass verified `comparators` into `/ehr` synthesis context and update `Readmission risk & trajectory` prompt instructions)
- `services/agent/postprocess/citations.py` (if citation deduplication or inline label formatting lives here)
- `services/agent/presentation/` (`a2ui.py`, `canvas.py`, `domain.py`)
- Frontend A2UI templates, CSS, and JS files (wherever citation pills, the A2UI component stack, the References list, and `@media print` styles are rendered)
- Unit test files under `tests/agent/` and frontend tests (excluding `tests/agent/test_layer_separation.py` and `evaluation/`, which remain read-only until Stage Gate 3).

---

## Required Changes

### 1. Multi-Page Print Stylesheet (`@media print`)
- Add a comprehensive `@media print` block to the A2UI stylesheet:
  - Hide navigation chrome, the left patient/conversation sidebar, top action buttons, composer input bar, and floating controls (`display: none !important`).
  - Reset `html`, `body`, the app shell, main layout containers, and chat scroll containers to `height: auto !important; max-height: none !important; overflow: visible !important; position: static !important; display: block !important;` so the browser paginates the entire conversation turn across as many pages as needed.
  - Add `break-inside: avoid; page-break-inside: avoid;` on structured cards (`RiskInfluencers`, `ComparativeRiskMatrix`, table rows, and individual reference entries) so tables and cards do not split awkwardly mid-row across page breaks.

### 2. Compact Inline Citation Pills & Redundancy Cleanup
- Refine inline citation pills so they are visually compact and clinical (muted pill styling that sits cleanly inline without bloating line height).
- Shorten overly verbose inline section labels (or display compact numbered/short-label pills inline with the full section/study title on hover/click), and avoid repeating the exact same single citation badge on every consecutive sentence in a single paragraph when the entire paragraph is drawn from that same section—while preserving strict citation accuracy and all existing reference invariants.

### 3. Literature Benchmark Synthesis in `/ehr` Narrative (`deterministic.py` & `prompts.py`)
- In `run_command` (`/ehr`), make the verified `comparators` (with their source numbers/PMIDs, cohort summaries, and verified 30-day rates) available to the `/ehr` synthesis model call so the model can cite them in `Readmission risk & trajectory`.
- Update the `/ehr` synthesis prompt so `Readmission risk & trajectory` weaves together:
  1. The patient's individualized ML 30-day readmission probability and top chart-grounded drivers (with their structured values).
  2. When verified literature comparators are present, a 1–2 sentence comparison placing the patient's individualized estimate alongside the published 30-day cohort rate(s) with inline citation(s), noting that published cohorts reflect population benchmarks.

### 4. Self-Contained Numbered `References` Block at Bottom of Turn
- Ensure the composed presentation / frontend renders a clean, collapsible or visible numbered **References** list at the bottom of the turn (always visible in `@media print` and cleanly presented on screen before Follow-Up Questions) showing each cited source in numerical order (`1..N`), distinguishing chart sections from PubMed literature citations (Title, Journal, Year, PMID).

---

## Verification & Deliverable
1. Run `pytest tests -m "not live"` (and any frontend test suite) and confirm 100% pass.
2. Verify on a live `/ehr` turn (e.g., `90000090` or `90000069`) that:
   - The narrative cites both the note sections (`1..k`) and the verified literature comparator rows (`k+1..N`) in `Readmission risk & trajectory`.
   - Inline citation pills are compact and non-repetitive.
   - The numbered References list renders cleanly, and `@media print` paginates the entire turn without clipping at Page 1.
3. STOP and report your changes and verification results.