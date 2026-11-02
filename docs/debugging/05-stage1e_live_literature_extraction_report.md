# Stage Gate 1E Report — Population Guard (F3) and Two-Arm Rates (F4)

**Date:** 2026-11-02
**Directive:** Stage Gate 1E (close F3 and F4, re-run the eight-admission suite)
**Status:** Stage Gate 1E complete. Backend only at this point — nothing wired into `/ehr`; `presentation/`, `turn.py`, `evaluation/` untouched.

---

## 1. Result

**Seven rows across four admissions** — the same count as Gate 1D, with the paediatric error removed and four rows now publishing two-arm contrasts.

| hadm | 1 | 1B | 1C | 1D | **1E pass 1 (cand / aligned / verified)** | wide | **1E rows** |
|---|---|---|---|---|---|---|---|
| 90000090 | 0 | 2 | 2 | 2 | 24 / 10 / 0 | 17 / 10 / 2 | **2** |
| 90000069 | 1 | 0 | 2 | 2 | 18 / 4 / **2** | — | **2** |
| 90000081 | 1 | 0 | 0 | 1 | 22 / 10 / **1** (5 excluded) | 8 / 7 / 0 (1 excl) | **1** |
| 90000001 | 0 | 0 | 0 | 0 | 1 / 0 / 0 | 18 / 10 / 0 | 0 |
| 90000019 | 2 | 1 | 2 | 2 | 15 / 10 / **2** | — | **2** |
| 90000037 | 0 | 0 | 0 | 0 | 20 / 10 / 0 | 15 / 4 / 0 | 0 |
| 90000058 | 0 | 0 | 0 | 0 | 3 / 1 / 0 (1 excluded) | 0 | 0 |
| 90000082 | 0 | 0 | 0 | 0 | 12 / 8 / 0 | 7 / 7 / 0 | 0 |
| **total** | 4 | 3 | 6 | 7 | | | **7** |

Gates: `pytest tests -m "not live"` → **1702 passed**, 1 skipped. Live → **1 passed**.

## 2. Verified rows

| hadm | pmid | journal | rate | kind |
|---|---|---|---|---|
| 90000090 | 25550095 | Arch Orthop Trauma Surg 2015 | **11.9%** | point |
| 90000090 | 41004383 | Anesth Analg 2026 · 62,022 geriatric hip arthroplasty, neuraxial vs general | **4.60% vs. 4.97%** | comparison |
| 90000069 | 40073748 | Clin Neurol Neurosurg 2025 · 808 laminoplasties | **3.7%** | point |
| 90000069 | 28142261 | Neurosurg Focus 2017 · 6,077 ACDF (5,590) vs CDR (487) | **2.6% vs. 0.4%** | comparison |
| 90000081 | 37065558 | J Thorac Dis 2023 · 182 decortication patients, RAPID score | **50.0% vs. 16.3%** | comparison |
| 90000019 | 33422393 | J Arthroplasty 2021 · conversion (8,369) vs primary (25,081) THA | **7.3% vs. 3.3%** | comparison |
| 90000019 | 32742739 | Geriatr Orthop Surg Rehabil 2020 · over-80 THA, n=7,730 | **4.9%** | point |

## 3. F3 — closed at both layers

- **Pool filter.** `compatible_cohorts` runs on the 25-record pool before ranking and before the ten-record truncation. `90000081` reports `population_excluded=5` on pass 1 and `1` on the wide pass; the paediatric cohort Gate 1D served no longer reaches the model, and the row that replaced it is an adult cohort.
- **Verification.** `verify_comparator` refuses any row whose title, journal, abstract or `cohort_summary` marks a paediatric population when the target is not paediatric.
- The filter lives in `deterministic.py` rather than in the PubMed query so the population vocabulary has one spelling, in the layer that knows what the chart said.

## 4. F4 — closed

`normalise_rate` returns `(rate, kind)` for three shapes: `point`, `range`, `comparison`. A comparison is republished with the abstract's own connector — `"7.8% and 11.5%"` → `"7.8% vs. 11.5%"`; a range is left byte-for-byte. Four of the seven rows are comparisons, and four were previously compressed to one arm (`28142261` as `2.6%` alone, `33422393` as `3.3%` alone).

## 5. Residuals

**Run-to-run variance.** `90000090` pass 1 verified 1 row in 1D and 0 in 1E on identical code, identical query and temperature 0 — PubMed's relevance order and retrieved set move between calls. The wide pass recovered two rows both times.

**Related cohort, not the patient's procedure.** `37065558` is adults with complicated effusion and/or empyema undergoing decortication, where this patient had thoracentesis. The population guard cannot see a procedure mismatch, and the model accepted it as a sub-cohort. This is the gap Stage Gate 2's procedure-invasiveness rule addresses.

**Two-arm rule read permissively.** The prompt asks for the patient's own arm alone when the other is a comparison arm; the model reported both in `28142261` and `33422393`, where both arms are the same procedure under different conditions.

**`90000001` unchanged** — 1 candidate on pass 1, 18 on the wide pass, none verified.

## 6. Stop

Stage Gate 1E ends here.
