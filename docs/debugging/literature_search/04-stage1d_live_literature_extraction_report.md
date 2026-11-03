# Stage Gate 1D Report — Provider Safety Blocks (F1) & Ranking Dilution (F2)

**Date:** 2026-11-02
**Directive:** Stage Gate 1D (systemic resolution of F1 and F2, re-run the eight-admission suite)
**Status:** Stage Gate 1D complete. Backend only — nothing wired into `/ehr`; `presentation/`, `turn.py`, `evaluation/` untouched.

---

## 1. Result

**Seven verified rows across four admissions**, up from six. All eight admissions now derive a target; none is withheld.

| hadm | Gate 1 | 1B | 1C | **1D pass 1 (cand / aligned / verified)** | wide | **1D rows** |
|---|---|---|---|---|---|---|
| 90000090 | 0 | 2 | 2 | 24 / 10 / 1 | 16 / 10 / 2 | **2** |
| 90000069 | 1 | 0 | 2 | 18 / 4 / **2** | — | **2** |
| 90000081 | 1 | 0 | 0 | 22 / 10 / 0 | 7 / 7 / **1** | **1** ⬆ |
| 90000001 | 0 | 0 | 0 | 1 / 0 / 0 | 18 / 10 / 0 | 0 |
| 90000019 | 2 | 1 | 2 | 15 / 10 / **2** | — | **2** |
| 90000037 | 0 | 0 | 0 | 20 / 10 / 0 | 15 / 3 / 0 | 0 |
| 90000058 | 0 | 0 | 0 | 3 / 1 / 0 | 0 | 0 |
| 90000082 | 0 | 0 | 0 | 12 / 8 / 0 | 7 / 7 / 0 | 0 |
| **total** | 4 | 3 | 6 | | | **7** |

Gates: `pytest tests -m "not live"` → **1693 passed**, 1 skipped. Live → **1 passed**.

## 2. Verified rows

| hadm | pmid | journal | cohort | rate |
|---|---|---|---|---|
| 90000090 | 37120515 | BMC Musculoskelet Disord 2023 | 199 unilateral femoral neck fractures, two care models | **7.8% vs 11.5%** at 30 days |
| 90000090 | 25550095 | Arch Orthop Trauma Surg 2015 | 1,081 aged 65+, level 3 trauma centre | **11.9%** |
| 90000069 | 40073748 | Clin Neurol Neurosurg 2025 | 808 cervical laminoplasties for spondylosis, NSQIP | **3.7%** |
| 90000069 | 28142261 | Neurosurg Focus 2017 | 6,077 single-level ACDF/CDR, NSQIP | **2.6%** (ACDF arm) |
| 90000081 | 36805137 | J Pediatr Surg 2023 | **paediatric** complicated effusion/empyema, n=5,424 | **15% / 12%** |
| 90000019 | 33422393 | J Arthroplasty 2021 | primary THA, n=25,081, PearlDiver | **3.3%** |
| 90000019 | 32742739 | Geriatr Orthop Surg Rehabil 2020 | over-80 THA, n=7,730, NSQIP | **4.9%** |

## 3. Fix-by-fix

### F1 — provider safety blocks: resolved at the provider layer

`model.build_extraction_llm` passes `BLOCK_NONE` for all four configurable categories; `build_llm` and it share one `_client` construction, and a test asserts the served path still carries the product thresholds (`BLOCK_MEDIUM_AND_ABOVE`). Result: zero withheld replies across eight admissions.

`90000058` — withheld on both prompts at Gate 1C — now derives **serotonin syndrome** and returns a genuine empty: three candidates, none stating a 30-day readmission rate.

`fallback_target_from_sections` exists and is tested (numbering, leading date, comorbidities, all-caps block headers, rejection of a sentence about the patient) but was **not needed live**; the provider fix removed the condition it exists for.

### F2 — query dilution: resolved, and it is why `90000081` returned a row

Pass 1 now asks one plain phrase (`condition AND outcome`); the anchor OR-set moved to the wide pass; `rank_by_specificity` orders the pool (condition phrase → multi-word anchor → single-word anchor → nothing) with PubMed order as the tie-break; only the top 10 reach the model.

Candidate pools rose from 6–10 to 12–24, and `90000081`'s adult hydrothorax-class cohort came back on the wide pass after two gates of nothing.

### R2 fired live for the first time

`28142261`'s quote — *"The readmission rates were 2.6% for ACDF and 0.4% for CDR (p = 0.003)"* — states no horizon; its title, *"Thirty-day readmission and reoperation rates…"*, does. Accepted on that basis.

## 4. Two new defects

### F3 — a paediatric cohort was admitted as this adult's comparison

`36805137`, *Journal of Pediatric Surgery*, patients under 18: passed the anchor gate (contains "pleural effusion"), passed verification (30-day rate, verbatim sentence, digits match), and was accepted by the model. Nothing in the pipeline sees age, and R4's sub-cohort permission made the model more willing to accept a narrower population. R4 recovered `32742739` (octogenarian THA — the same operation in a narrower band) and admitted a different population entirely.

### F4 — `rate_kind` has a third shape it does not mark

`37120515` reports `"7.8% and 11.5%"` and `36805137` reports `"15% and 12%"`; both are two-arm comparisons, both are marked `point`, and both would render as a single figure beside one predicted probability. The abstract's own wording for the first is `7.8% vs. 11.5%`, so the row's rate text is also a compression the abstract did not make.

### Residual

`90000001` returns 1 candidate on pass 1 and 18 on the wide pass, and verifies none: tibial plateau fracture / ORIF has no 30-day readmission cohort in the retrieved window.

## 5. Stop

Stage Gate 1D ends here. `presentation/domain.py` still supplies the comparative table; nothing is wired into `run_command`.
