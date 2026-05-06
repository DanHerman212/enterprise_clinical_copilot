# Open Questions & Ambiguities — MIMIC-IV 30-Day Readmission Pipeline

Living document. Each item is a decision required before (or during) the data
representation layer is finalized. Resolve in place by replacing the **Status**
line with `RESOLVED — <decision> — <date>` and adding a short rationale.

Source of record: [MIMIC-IV Readmission Prediction Research.txt](../MIMIC-IV%20Readmission%20Prediction%20Research.txt)

---

## 1. MIMIC-IV version & BigQuery dataset names
- **Why it matters:** Dataset paths and a few schema details differ between
  v2.2 and v3.1 (e.g. `physionet-data.mimiciv_hosp` vs
  `physionet-data.mimiciv_3_1_hosp`). Every downstream SQL reference depends
  on this.
- **Options:** v2.2 · v3.1 · both (pin one for training, validate on the other).
- **Status:** RESOLVED — v3.1 — 2026-05-04. Confirmed available datasets:
  `mimiciv_3_1_hosp`, `mimiciv_3_1_icu`, `mimiciv_3_1_derived`,
  `mimiciv_ed` (unversioned), `mimiciv_note` (unversioned).

## 2. Exact `admission_type` membership for acute vs planned
- **Why it matters:** The outline lists a partial set
  (`URGENT`, `EMERGENCY`, `EW EMER.` as acute; `ELECTIVE`, `OBSERVATION`,
  `SURGICAL SAME DAY ADMISSION` as planned). MIMIC-IV actually contains
  additional values such as `DIRECT EMER.`, `DIRECT OBSERVATION`,
  `OBSERVATION ADMIT`, `AMBULATORY OBSERVATION`. Mis-classifying these
  shifts label prevalence and pollutes the positive class.
- **Decision needed:** Final allow-list for the **positive label trigger**
  (acute readmission) and the explicit **exclusion list** (planned).
- **Status:** RESOLVED — 2026-05-04.
  - **Acute (positive trigger):** `URGENT`, `EMERGENCY`, `EW EMER.`,
    `DIRECT EMER.`, `DIRECT OBSERVATION`, `EU OBSERVATION`,
    `OBSERVATION ADMIT`, `AMBULATORY OBSERVATION`.
  - **Planned (excluded as triggers):** `ELECTIVE`,
    `SURGICAL SAME DAY ADMISSION`.

## 3. "First eligible index admission per patient" — selection order
- **Why it matters:** Two valid interpretations produce different cohorts:
  - (a) **Filter then rank:** apply all inclusion/exclusion rules, then take
    the earliest remaining admission per `subject_id`.
  - (b) **Rank then filter:** take each patient's first-ever admission, then
    drop the patient entirely if it fails any rule.
  Option (a) yields more patients; option (b) is more conservative against
  selection bias.
- **Status:** RESOLVED — (a) FILTER then RANK — 2026-05-04. Apply all
  exclusions to every admission per `subject_id`, then take the earliest
  surviving admission as the index. Aligns with standard healthcare quality
  reporting logic and preserves more viable patients.

## 4. Lookback windows under MIMIC-IV's de-identified timeline
- **Why it matters:** MIMIC-IV dates are randomly shifted per patient
  (`anchor_year_group`). The shift is *consistent within a patient*, so
  intra-patient `DATE_DIFF` is valid for prior-365-day admission counts and
  prior-180-day ED counts. No real calendar reasoning is possible.
- **Decision needed:** Confirm we treat the per-patient shifted timeline as
  continuous (standard practice) and document this assumption next to every
  rolling-window feature.
- **Status:** RESOLVED — ACCEPT — 2026-05-04. MIMIC-IV applies a single
  consistent random date shift per `subject_id`, so intra-patient
  `DATE_DIFF` is exact. Document this assumption next to every rolling-window
  feature.

## 5. ED visit source — `mimiciv_ed` module vs proxy
- **Why it matters:** LACE's `E` component and several utilization features
  require ED visit counts. The `mimiciv_ed` module is separately credentialed
  on PhysioNet.
- **Options:**
  - Use `mimiciv_ed.edstays` for true ED encounters (preferred).
  - Proxy via `admissions.edregtime IS NOT NULL` if ED module is unavailable.
- **Status:** RESOLVED — use `mimiciv_ed.edstays` — 2026-05-04. Module access
  confirmed in project `physionet-data`.

## 6. Prediction time anchor — at discharge vs at admission
- **Why it matters:** The HOSPITAL score and CMS HRRP framing both anchor at
  **discharge**. All point-in-time-correct features must therefore use data
  strictly before `dischtime`. An admission-time anchor would change the
  feature set materially (no last-24h vitals, no last labs pre-discharge).
- **Status:** RESOLVED — AT DISCHARGE (`dischtime`) — 2026-05-04. Matches
  HOSPITAL score / CMS HRRP framing and preserves last-24h vitals and last
  pre-discharge labs as predictive signal.

## 7. `mimiciv_note` access for unstructured text branch
- **Why it matters:** Discharge summaries drive the multimodal AUROC lift
  (~0.75–0.79 per outline). The notes module requires separate PhysioNet
  credentialing.
- **Decision needed:** Confirm access; if unavailable, the text branch is
  deferred and the v1 model is tabular-only.
- **Status:** RESOLVED — access available — 2026-05-04. Text branch is in
  scope for v2; v1 remains tabular-only to keep the first pipeline minimal.

## 8. Censoring policy for transfers
- **Why it matters:** Patients discharged to `OTHER FACILITY` / `ACUTE HOSPITAL`
  are excluded from the cohort (right-censored, unobservable). Confirm we do
  *not* attempt to include them with a survival/competing-risk treatment in v1.
- **Status:** RESOLVED — EXCLUDE in v1 — 2026-05-04. Patients discharged to
  `OTHER FACILITY` / `ACUTE HOSPITAL` leave the BIDMC network and their
  30-day outcome is unobservable; including them would inject false negatives.
  Survival / competing-risk treatment deferred to v2.

## 9. Missingness policy for labs/vitals
- **Why it matters:** Outline calls for **median imputation + boolean
  missingness indicators**. We need to fix:
  - Imputation computed on the **training fold only** (no leakage).
  - Indicator column naming convention (`<feat>__is_missing`).
  - Where this lives: BigQuery view vs Vertex pipeline component.
- **Status:** OPEN.

## 10. Cross-validation / split strategy
- **Why it matters:** Outline warns about "hierarchical bias and heavily
  correlated sequential records." Even with one row per patient, we should
  confirm: split by `subject_id` (group-aware), stratified on the label,
  with a held-out temporal slice for drift evaluation.
- **Status:** RESOLVED — group-by `subject_id`, chronological 70/15/15 —
  2026-05-04. Splits ordered by index `admittime` to simulate prospective
  deployment; no patient appears in more than one fold.

---

## Resolution log
_Append entries here as items move from OPEN → RESOLVED._
