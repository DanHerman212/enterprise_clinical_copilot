# UAT & Remediation Plan — Session 2026-08-25

**Status:** draft for review (read on the way to the airport)
**Trigger:** 10-patient live walkthrough of the production demo found 5 issues
after a UAT was already considered done. Mandate: **we cannot ship a publicly
facing demo that shows a UAT was done but leaves known issues in it.**

**Scope:** the live demo (Django site + agent harness + synthetic data + Vertex
serving). Both repos: `danielmherman` (site) and `enterprise_clinical_copilot`
(harness / data / prompts).

Raw walkthrough evidence: `/memories/session/demo-walkthrough-2026-08-25.md`.
Companion docs: `docs/code_review_plan.md` (by-section review),
`danielmherman/docs/hybrid_demo_test_agenda.md` (sprint structure).

## Progress log — 2026-08-25 (in flight)

**Done:**
- Built `projects/agent-harness/scripts/coherence_scan.py` — flags cohort-inclusion
  violations (neonate/infant/toddler notes + filled age < 18), sex/pronoun
  mismatches, and redaction artifacts ("Dr. X").
- Root cause confirmed for B1: `fill_features._parse_age` grabs the FIRST age in
  a note — in obstetric/neonatal notes that is the mother's (e.g. note 2581
  "born to a 26-year-old … lady" → patient displayed 26F).
- Ran the scan; manually validated every flagged note (caught + removed two
  regex false positives: obstetric-history "gravida/para" and "history of being
  a preemie twin").
- **Pruned 10 inclusion-violating patients** from `hybrid_cohort.json`,
  `hybrid_notes.json`, `data/hybrid/provenance.json` (108 → 98) via
  `projects/agent-harness/scripts/prune_inclusion_violations.py`; originals
  backed up as `*.pre-inclusion-prune.json`.
  Removed: 90000006, 90000007, 90000021, 90000038, 90000042, 90000062,
  90000066, 90000083, 90000085, 90000098.
- **Second filter — training-cohort exclusions (per user):**
  - **AMA discharges** (feature `discharge_location_ama`): removed 90000072,
    90000089, 90000096.
  - **Elective admission with planned hospital return** (text-detected:
    elective/scheduled + re-admission language): removed 90000093 (scheduled
    follow-up → re-admit for debridement/revision). Two regex false positives
    (return-to-diet, standard "return to ER if…" precaution) excluded by
    validation.
  - Cohort now **94 patients** (108 − 14 total).
- Verified: 94/94/94, ids consistent across all three files, 0 under-18,
  0 inclusion flags, 0 AMA flags, 0 planned-return flags. Backups preserved
  (108-patient originals).

**HOLDING (per user):** demo cohort has NOT been reseeded yet — user has an
additional filter to apply before `seed_synthetic_demo_cohort.py` runs.

**Teardown (2026-08-25, before user left for flight):**
- Ran `projects/agent-harness/scripts/teardown.py --yes` — removed BOTH billable
  Vertex endpoints: prediction (`readmission-endpoint`, n1-standard-2) and the
  RAG index endpoint (`readmission-rag-index`, e2-standard-2).
- Kept (free): model registry (`readmission-final-*`, `readmission-cpr-*`),
  the Vector Search index resource, GCS bundles, BigQuery, Cloud Run services.
- Background RAG poller stopped. Demo site is live-mode but endpoints are down
  until the RAG index is rebuilt (expected — normal between-session state).

**To resume (~1 hr later, updated after the 8/25 working session):**
1. **Cohort filters — DONE**: inclusion ≥18, AMA, elective-planned-return,
   hospice, pediatric slip-throughs (90000032, 90000080), sex-mismatch
   (90000061). Cohort now **89** (108 − 19). `demo_cohort.json` reseeded
   (harness + site), BigQuery `hybrid_notes/split/features` cleaned via
   `prune_rag_datapoints.py`.
2. **RAG index — NO rebuild needed for function** (retrieval is hadm-restricted;
   orphaned vectors invisible). Deploy the kept index as-is tomorrow
   (`deploy_synthetic_rag.py`) + prediction (`deploy_cpr.py`). ⚠️ Finding: the
   kept index was built WITHOUT StreamUpdate → incremental deletes are blocked
   (400). **Next index build should set `IndexUpdateMode.STREAM_UPDATE`** so
   future prunes are cheap `remove_datapoints`, not rebuilds.
3. **Re-deploy site** (seed runs with --prune → the 19 removed patients drop
   off production).
4. **Code fixes — DONE**: A1 (prompt threshold-only, no band words),
   C1/A2 (`feature_labels.py` `humanize` camelCase/snake_case → Title case on
   FactorBars + fixture prose; prompt FEATURE NAMES + deterministic driver
   sentence; 50 site tests pass).
5. **Task 7 — redaction decision: ACCEPTED as-is (2026-08-25).** "Dr. X" /
   redacted names stay in the notes — they visibly demonstrate PHI is
   redacted, which is a positive signal for a clinical demo. No sanitization,
   which also means **no RAG rebuild is required** (kept index deploys as-is).
6. **Re-run full UAT gate** (Part B) before public demo.

**Still open (next session):**
- Apply user's additional filter, then reseed demo_cohort.json + site copy.
- Rebuild RAG index over pruned notes + redeploy RAG endpoint; re-deploy site
  (seed runs with --prune).
- Fix A1 (narrative/band mismatch) in agent prompt.
- Fix C1 + A2 (feature labels, driver format).
- Investigate 3 sex/pronoun flags: 90000061 (23M/1F, likely real gender
  mismatch), 90000032 (7M/1F), 90000039 (2M/1F).
- REDACTION "Dr. X" is pervasive in the MTSamples source (70 instances) —
  decide: post-process notes or accept (flagged in Part A/C).
- Re-run full UAT gate (Part B) before public demo.

---

## Part A — What the walkthrough found

Five issues, grouped into three categories. The categories matter more than the
individual bugs: each is a *class* of failure the next UAT must actively hunt.

### Category 1 — Numeric fidelity (agent's words vs. what the UI shows)
- **A1. Narrative / band mismatch.** Alan Marchetti (risk 12.1%, just above the
  0.12 threshold): the agent prose said "above the threshold → the model
  predicts **high** risk of readmission," but the canvas band rendered
  **borderline**. A clinician reading the prose and looking at the band gets
  two different answers. This is a trust-breaker on the main screen.
- **A2. Driver format inconsistency.** Some turns render the SHAP drivers as
  prose ("The risk is primarily increased by …"), others as a bulleted list
  ("medication_count: increases risk (contribution: 0.5757)"). Same feature,
  different voice — reads like two different products.

### Category 2 — Synthetic data coherence
- **B1. Age/note mismatch.** Alicia Kowalski is listed **26F**, but her
  discharge note describes a **neonate** (group B strep prophylaxis,
  phototherapy for jaundice, ~30 mL feeds). This is the same class of bug as
  the gender/sex coherence issue already fixed once (2026-08-22) — it proves
  the generator still does not cross-check note content against patient
  demographics.
- **B2. Redaction artifacts.** Notes contain bare placeholder tokens such as
  "follow-up with **Dr. X** in 2 days." Intended for privacy, but a bare "X"
  reads as a mistake to a reviewer.

### Category 3 — Terminology consistency
- **C1. Feature-name leak / phrasing drift.** The prose says "medication order
  count," the canvas driver bar says `medication_order_count`. The narrative
  and the data layer use different labels for the same feature.

**What went right (so we don't "fix" the wrong thing):** 10/10 live risk
assessments returned correct probabilities + SHAP drivers + section-correct
citations. The yesterday citation-link, meds-citation, and summarize-numbering
bugs are **all confirmed fixed**. The problems above are fidelity/coherence
issues, not a broken pipeline.

---

## Part B — How we should look at UAT (methodology)

### Why the last UAT missed these
The last UAT was **screen-by-screen without category-based assertions or exit
criteria**. It verified "does the screen work" but not "does every screen agree
with every other surface on the same fact." A1, A2, B1, C1 are all *cross-
surface consistency* bugs: rail vs. canvas vs. prose vs. data. You cannot find
them by checking one screen in isolation — you have to check the *same fact
across surfaces* for every sampled patient.

### The fix: a UAT gate with consistency checklists, not screen walks

For each sampled patient, assert **the same fact across surfaces**:

| Check class | What it asserts | Example check |
|---|---|---|
| **Data coherence** | Note content matches patient demographics | Age/sex match the note's pronouns, neonatal terms, pregnancy; admission facts (LOS, admission type) match the rail |
| **Numeric fidelity** | The number agrees everywhere | rail % == canvas % == prose number; band label == threshold semantics (high/borderline/low) |
| **Citation correctness** | The cited section is the shown section | `^[n]` section == SourceCard section; numbering matches SUMMARY_SECTIONS order; clicking `[n]` changes the card |
| **Presentation consistency** | Same feature looks the same every turn | driver list format identical turn-to-turn; feature labels identical in prose and canvas; no raw keys leaked |
| **Operational** | The system is behaving like production | `agent-composed · live` indicator; quota decrements; no console errors; latency sane |

**Stratified sampling (not "first 10"):** sample across every band (high,
borderline, low) **and** demographic edge cases (age extremes — pediatric /
neonatal / geriatric, both sexes, varied admission types). A random or
alphabetical sample will miss exactly the coherence bugs we found. Target
12–15 patients per full gate.

**Deterministic checks where possible:** numeric fidelity and citation
mapping are *checkable in code*, not just by eye. Build a **UAT assertion
script** that hits the live `/ask` (or the deployed agent) for each sampled
patient and asserts the invariants (band == band_for(probability); source
section == cited section; labels consistent). Eyeballing is reserved for what
only a human can judge (readability, clinical tone).

**Evidence capture per patient:** one record per patient per chip (risk /
meds / summarize) — probability, band, drivers, source section, and a
screenshot. This is what makes "we looked" auditable. (The 8/25 walkthrough
file is the template.)

**Exit criteria (definition of "public-ready"):**
1. All five check classes pass on a stratified 12–15 patient sample.
2. Zero Critical / Major issues open. Minor issues either fixed or
   consciously accepted with a documented reason (not silently left).
3. No console errors across the whole pass.
4. Evidence captured for every sampled patient.

**Freeze → fix → re-verify:** once a UAT pass produces a finding, the fix goes
through the *same gate* again before the demo is shown. No "we'll fix it
later" — later is when the demo is public.

---

## Part C — Remediation plan for the specific issues

Ordered by dependency (data first, then the clinically visible one, then
polish). Each item: root-cause area → fix → how we verify it's gone.

### C.1 — Fix B1 (age/note coherence) + B2 (redaction artifacts) — *data*
- **Root cause:** synthetic cohort generation
  (`enterprise_clinical_copilot` scripts — `build_hybrid_fixtures.py` /
  `seed_demo_cohort.py`) assigns notes to patients without a coherence check.
- **Fix:** add a **coherence assertion at generation time** + a standalone
  validation script over the full cohort that scans notes for: neonatal /
  pediatric / geriatric terms vs. age band, gendered pronouns vs. sex,
  pregnancy/obstetric content vs. sex+age, and bare placeholder tokens ("X").
  Reassign or regenerate the offending notes (Alicia Kowalski et al.). For B2,
  replace bare "X" placeholders with a plausible phrase (e.g., "follow-up with
  the primary care provider").
- **Verify:** coherence scan over all 108 patients = 0 mismatches; re-seed +
  re-deploy; re-run gate Category "Data coherence."

### C.2 — Fix A1 (narrative/band mismatch) — *clinical trust, high visibility*
- **Root cause:** the agent's prose (harness `agent/prompts.py`) decides
  "above/below threshold" and says "high risk," while the site's band function
  (`demo/fixtures.py` `band_for`) uses low/borderline/high zones. Two different
  band semantics, one screen.
- **Fix:** one source of truth. Have the prediction tool return the **band
  label**, and instruct the prompt to describe the risk using that label
  (e.g., "above the operating threshold — **borderline** risk") instead of
  free-form "high." If we prefer threshold-only language, soften the prompt and
  keep the band; decide once and encode it.
- **Verify:** re-run the just-above-threshold cohort (Marchetti 12.1%,
  Kowalski 12.2%); assert prose band == canvas band on every turn.

### C.3 — Fix C1 (feature labels) + A2 (driver format) — *presentation*
- **Root cause:** the human-readable feature labels live in the agent prompt,
  while the canvas uses raw feature keys; the driver list format is
  prompt-authored and drifts turn to turn.
- **Fix:** a **single feature-label map** (feature key → display label) shared
  by the prompt guidance and the canvas; and a **deterministic driver
  presentation rule** in the prompt (always prose or always the same bullet
  schema), or better — compose the driver list from the tool's structured SHAP
  payload so the voice cannot drift.
- **Verify:** N-turn consistency check across ≥10 patients; assert prose label
  == canvas label and identical list shape every turn.

### C.4 — Make the fixes stick (guardrails)
- Add the UAT assertion script (Part B) to CI or the deploy checklist so a
  re-shipped demo must pass the invariants.
- Fold "cross-surface consistency" into the existing
  `docs/code_review_plan.md` by-section review (agent prompts + A2UI canvas
  sections) and the `hybrid_demo_test_agenda.md` Sprint D.

---

## Part D — Suggested run order

1. **Data:** coherence scan → fix B1 + B2 → regenerate/re-seed → re-deploy.
2. **Fidelity:** fix A1 (band-aware prose) in the agent prompt.
3. **Presentation:** feature-label map (C1) + deterministic drivers (A2).
4. **Re-run the full UAT gate** (stratified 12–15 patients, all 5 check
   classes, evidence captured). Exit criteria must all pass.
5. **Only then:** public demo.

**How to resume:** open this file → start Part D step 1 (coherence scan on the
108-patient cohort) → then the fixes in C.1–C.3, each verified by its check
class before moving on.

---

## References
- Walkthrough evidence: `/memories/session/demo-walkthrough-2026-08-25.md`
- UAT sprint structure: `danielmherman/docs/hybrid_demo_test_agenda.md`
- Code review plan (by-section, risk-ordered): `docs/code_review_plan.md`
- Synthetic data policy: `danielmherman/docs/synthetic_cohort_curation.md`
