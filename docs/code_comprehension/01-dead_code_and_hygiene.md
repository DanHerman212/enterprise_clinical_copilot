# Phase 1: Dead Code & Hygiene Remediation

**Status:** `COMPLETE (Step B executed; all three gates recorded)`  
**Target Layers:** Layer 1 (`01-client.md`) & Layer 12 (`12-delivery_ci_cd.md`)  
**Objective:** Quarantine and prune superseded Phase 1 artifacts, legacy shims, and unreachable code blocks across both repositories so subsequent phases only refactor active production code.

---

## 1. Step A — Audit Findings (Read-Only)

### 1.1 `danielmherman` (Layer 1 — Frontend & BFF Hygiene)

Cross-referencing `demo/urls.py`, `demo/views.py`, `demo/templates/demo/*.html`, `vite.config.js`, and `static/js/` against `01-client.md` ("One Console, One Shell"):

1. **Superseded Phase 1 Split-Pane & Q3 Spike Scripts (4 Tracked Files — 70,445 bytes total):**
   * **`static/js/demo_flow.js` (Lines 1–737, 38,188 bytes):** Implements the retired Phase 1 split-pane patient rail (`#patient-list`, `#patient-search`), chaptered thread (`#thread`), and starter-chip bar (`#composer-chips`). Zero templates in `demo/templates/demo/` contain these DOM IDs or load this script.
   * **`static/js/demo_a2ui.js` (Lines 1–285, 12,642 bytes):** Imports `createDemoFlow` from `./demo_flow.js` (line 20) and mounts into `#a2ui-root` / `#a2ui-host` / `#canvas-switch`. None of those IDs exist in `a2ui_console.html` or `landing.html`.
   * **`static/js/a2ui_renderer.js` (Lines 1–24, 1,014 bytes):** Legacy Rollup entry re-exporting `buildRiskCatalog` from `./a2ui_risk_components.js`. Explicitly superseded by `static/js/a2ui_clinical.js` (whose header comment on lines 10–14 notes: *"Why a separate entry from `a2ui_renderer` at all: the legacy components are being replaced..."*).
   * **`static/js/a2ui_risk_components.js` (Lines 1–378, 18,601 bytes):** Defines the retired Q3 spike components (`RiskBar`, `FactorBars`, `TraceCard`, and legacy `SourceCard`/`LiteratureList`). Furthermore, `copilot_canvas.js` (lines 714–718) explicitly documents that the legacy risk catalog is no longer loaded.
   * **Proposed Action:** Delete all four tracked source files (`static/js/demo_flow.js`, `static/js/demo_a2ui.js`, `static/js/a2ui_renderer.js`, `static/js/a2ui_risk_components.js`).

2. **Superseded Vite Bundler Entries & Ignored Build Outputs (`vite.config.js` & `static/js/bundled/`):**
   * **`vite.config.js` (Lines 20–24):** Currently builds three entries:
     ```javascript
     input: {
       a2ui_renderer: resolve(__dirname, 'static/js/a2ui_renderer.js'),
       a2ui_clinical: resolve(__dirname, 'static/js/a2ui_clinical.js'),
       demo_a2ui: resolve(__dirname, 'static/js/demo_a2ui.js'),
     }
     ```
   * **Ignored Build Artifacts (`static/js/bundled/`):** `static/js/bundled/a2ui_renderer.js` (772,020 bytes) and `static/js/bundled/demo_a2ui.js` (38,151 bytes) sit on disk alongside `static/js/.DS_Store` (6,148 bytes).
   * **Proposed Action:** Remove `a2ui_renderer` and `demo_a2ui` from `vite.config.js` `input` (leaving `a2ui_clinical` as the sole entry), and delete `static/js/bundled/a2ui_renderer.js`, `static/js/bundled/demo_a2ui.js`, and `static/js/.DS_Store`.

3. **Dead `SourceCard` Component in `static/js/a2ui_clinical_components.js`:**
   * **Lines 762–816 & Line 902:** Defines and registers `SourceCard` (`a2ui-source-card`). Its own comment on lines 768–770 states: *"Phase 2 replaces it with citations in the narrative itself; until that lands, the card has to keep working."*
   * **Verification:** In `enterprise_clinical_copilot/services/agent/a2ui.py`, the agent only emits `CitedNarrative`, `RiskInfluencers`, `LiteratureList`, `ComparativeRiskMatrix`, `LabTrendCard`, and `MedicationGrid`. And in `danielmherman/demo/tests.py` (line 440), `test_retired_surface_components_are_not_registered` already asserts `'RiskBar'` and `'FactorBars'` are absent from the clinical catalog.
   * **Audit Note / Decision Point:** We can either delete `SourceCard` from `a2ui_clinical_components.js` now (and add `'SourceCard'` to `test_retired_surface_components_are_not_registered`) OR keep it if any stored `Turn` rows or fixture files still contain `SourceCard` envelopes. **Recommendation:** Check `demo/fixtures/` during Step B; if zero fixtures emit `SourceCard`, prune it and rebuild `static/js/bundled/a2ui_clinical.js`.

4. **Legacy Split-Pane Stylesheet (`static/css/demo_splitpane.css`):**
   * **Finding:** `demo/templates/demo/guide.html` (line 24) still loads `<link rel="stylesheet" href="{% static 'css/demo_splitpane.css' %}?v=7">` solely for `.demo-header` rules (which `guide.html` overrides inline on lines 28–55 anyway).
   * **Proposed Action:** Inline the minimal `.demo-header` base flex rules into `guide.html` (or `demo_shell.css`) and delete `static/css/demo_splitpane.css` if no other template references it.

5. **`demo/urls.py` & `demo/views.py` Route Hygiene:**
   * **Finding:** All 9 URL patterns in `demo/urls.py` (`landing`, `a2ui_console`, `patient_autocomplete`, `a2ui_ask`, `conversation_history`, `conversation_detail`, `conversation_pin`, `conversation_delete`, `feedback`, `guide`) and all helper functions in `demo/views.py` are actively wired and covered by `demo/tests.py`. Zero dead routes in `urls.py` or `views.py`.

---

### 1.2 `enterprise_clinical_copilot` (Layer 12 — Repo & MLOps Hygiene)

1. **Unreachable Duplicate Code Block (`mlops/evaluation/validate_data.py`):**
   * **Lines 353–598:** Immediately following `if __name__ == "__main__": sys.exit(main())` on line 351, an entire legacy v0.4.x script is pasted at the bottom of the file, referencing undefined symbols (`ColumnMapping`, `DatasetMissingValues`, `DataDriftTable`).
   * **Proposed Action:** Delete lines 353–598.

2. **Superseded Single-Variant Note Renderers (`scripts/agent/generate_synthetic_notes.py`):**
   * **Lines 67–349:** Contains 9 unused functions (`_routine_short`, `_observation`, `_minor_elective`, `_diabetic_foot`, `_ckd_pneumonia`, `_postop_infection`, `_elderly_chf`, `_oncology_infection`, `_copd_readmission`) explicitly marked `LEGACY … do not use` in a banner at line 59, with zero call sites anywhere in the repository.

---

## 2. Step B — Execution Record

**Executed:** 2026-10-25. Architect approved Step A and instructed execution.

### 2.1 Corrections to the Step A findings

The audit's conclusions held; five of its details did not. They are recorded here because
each one would have produced a wrong edit if followed literally.

| Step A said | The repository says |
|---|---|
| `validate_data.py` dead block is lines 353–598 | The `__main__` guard is at 388; the dead block is **391–708** (317 lines, file is 708 long) |
| The block is a pasted duplicate | It is a **drifted older revision** of the same module — 32 diff hunks, and it defines `build_column_mapping()`, which the live half does not. It still never runs: it sits after the `__main__` guard and calls `ColumnMapping`, which is never imported |
| `vite.config.js` builds three entries incl. `a2ui_clinical`; keep that one | `input` held **two** entries (`demo_a2ui`, `a2ui_renderer`) and never named `a2ui_clinical`. With both retired the config has no entries, so the console build was retired with it |
| `demo/tests.py:440` has `test_retired_surface_components_are_not_registered` | No such test. The existing guard is `test_retired_presentation_modules_are_gone`, and `assertNotIn('SourceCard', types)` already appears at lines 238 and 1070 |
| `demo_splitpane.css` is kept only for `.demo-header` | It also carries `.demo-header-left/right`, `.demo-brand`, `.ecc-logo`, `.demo-brand-name`, `.demo-guide-link` and `.demo-logout` — all of which `guide.html` renders |

### 2.2 `danielmherman` — executed (commit `61eeea6`)

1. Deleted `static/js/demo_flow.js`, `demo_a2ui.js`, `a2ui_renderer.js`, `a2ui_risk_components.js` (70,445 bytes). Verified: no template loads them; `a2ui_console.html:263` loads `js/bundled/a2ui_clinical.js` only.
2. Deleted `vite.config.js`; `package.json` scripts reduced to `build` → `build:clinical`, `dev` → clinical watch. `Dockerfile` COPY list follows, so `npm run build` has one config to read.
3. Deleted `static/js/bundled/a2ui_renderer.js`, `static/js/bundled/demo_a2ui.js`, `static/js/.DS_Store`; rebuilt `bundled/a2ui_clinical.js` (801.54 kB, was 803 kB).
4. `SourceCard` removed from `a2ui_clinical_components.js` (component, export, catalog entry). Verified first: no fixture emits it and the agent names it only in prose.
5. `demo_splitpane.css` deleted; its header rules moved into `demo_guide.css`, which the guide already loads, and `guide.html` now loads one stylesheet.
6. Added `test_retired_frontend_scripts_stay_gone` so the four scripts and the component cannot return unnoticed.

### 2.3 `enterprise_clinical_copilot` — executed

1. `mlops/evaluation/validate_data.py`: 708 → 389 lines. The pasted revision after the `__main__` guard is gone.
2. `scripts/agent/generate_synthetic_notes.py`: 718 → 356 lines. The nine `LEGACY` single-variant renderers are gone; the variant `TEMPLATES` path and `_note` are untouched.

### 2.4 Gates

| Gate | Result |
|---|---|
| `pytest tests` (`enterprise_clinical_copilot`) | **1349 passed, 1 skipped** |
| `manage.py test` (`danielmherman`, `ENVIRONMENT=development`) | **233 tests, OK** |
| `manage.py check --deploy` (`ENVIRONMENT=production`) | **`System check identified no issues (2 silenced).`** |

How the deploy gate was made runnable: production settings read `SECRET_KEY` from Secret
Manager unconditionally, so the env var cannot stand in for it — with `GOOGLE_CLOUD_PROJECT=dummy`
the import dies with `PERMISSION_DENIED: Secret Manager API has not been used in project
dummy`. The gate was therefore run against the real project for the secret lookup only,
with the remaining production variables given stand-in values (none of them is read by
`check`):

```
ENVIRONMENT=production \
GOOGLE_CLOUD_PROJECT=trim-icon-498815-a0 \
CLOUD_SQL_CONNECTION_NAME=dummy \
ALLOWED_HOSTS=localhost \
CSRF_TRUSTED_ORIGINS=https://localhost \
GS_BUCKET_NAME=dummy \
DEMO_AGENT_URL=https://localhost \
./.venv/bin/python manage.py check --deploy
→ System check identified no issues (2 silenced).
```

The "2 silenced" are the checks silenced deliberately in settings; nothing is being hidden
by this run.

### 2.5 Open items — all resolved

1. **`demo/verify_live_citations.py` — RESOLVED by repair, not deletion.** It is an active
   live smoke script: it posts to the real `a2ui_ask` view with the deployed agent behind
   it, which `demo/tests.py` cannot do because every test there mocks the agent. So it stays.
   The prescribed one-line rename (`SourceCard` → `CitedNarrative`) would not have worked:
   `CitedNarrative` carries `{index, source, title, label, excerpt}` — human labels, no raw
   section name — so `card["section"]` would have raised. The assertions now read the turn's
   published `sources` (`{cite, section, text}`), which is where the section names live, and
   `CitedNarrative` presence is asserted separately to prove the prose reached the page. Two
   of its sentinels were already stale before this pass: `"not available"` and `"not found"`
   were `SourceCard` section names that no longer exist, so the "unavailable" and
   "empty_passages" cases now test the answer text and the absence of citations.
2. **`npm run build` has no console half** — accepted. A second bundle would need a second
   config and an explicit entry; the shared-chunk failure documented in
   `vite.clinical.config.js` is the reason the two were never merged.
3. **`manage.py check --deploy` — executed.** See §2.4.

### 2.6 Final ECC hygiene (same pass)

| Item | Action |
|---|---|
| `mlops/training/_export_sample_data.py` (617 bytes) | **Deleted.** Hardcoded `/Users/danherman/Desktop/enterprise_clinical_copilot/...` paths, mutated `sample_data.parquet` in place, zero references anywhere in the tree |
| `docs/code_comprehension/00-master_code_audit_strategy.md` | **Already gone from disk** — the master plan was renamed to `00-master_alignment_plan.md`. The rename is recorded in git so the empty blob left by the earlier commit does not linger |