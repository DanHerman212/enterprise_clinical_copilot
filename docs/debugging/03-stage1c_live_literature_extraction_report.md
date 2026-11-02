# Stage Gate 1C Report — R1–R4 Fixes, Verified Rows, and Two New Findings

**Date:** 2026-11-02
**Directive:** Stage Gate 1C (fix R1–R4 regressions, re-run the eight-admission suite)
**Status:** Stage Gate 1C complete. Backend only — nothing wired into `/ehr`; `presentation/`, `turn.py`, `evaluation/` untouched.

---

## 1. Result

**Six verified rows across three admissions**, up from four (Gate 1) and three (Gate 1B).

| hadm | Gate 1 rows | Gate 1B rows | Gate 1C pass 1 (candidates / aligned / **verified**) | wide pass | **Gate 1C rows** |
|---|---|---|---|---|---|
| 90000090 | 0 | 2 | 10 / 10 / **2** | — | **2** |
| 90000069 | 1 | 0 | 10 / 10 / **2** | — | **2** ⬆ |
| 90000081 | 1 | 0 | 10 / 9 / 0 | 3 / 3 / 0 | 0 |
| 90000001 | 0 | 0 | 1 / 0 / 0 | 1 / 1 / 0 | 0 |
| 90000019 | 2 | 1 | 10 / 7 / **2** | — | **2** ⬆ |
| 90000037 | 0 | 0 | 10 / 6 / 0 | 6 / 3 / 0 | 0 |
| 90000058 | 0 | 0 | withheld, both prompts | — | 0 |
| 90000082 | 0 | 0 | 10 / 10 / 0 | 5 / 5 / 0 | 0 |
| **total** | 4 | 3 | | | **6** |

Gates: `pytest tests -m "not live"` → **1685 passed**, 1 skipped. Live → **1 passed**.

## 2. Verified rows

### 90000090 — femoral neck fracture / hemiarthroplasty

| pmid | journal | cohort | rate |
|---|---|---|---|
| 25550095 | Arch Orthop Trauma Surg 2015 | 1,081 patients aged 65+, level 3 trauma centre | **11.9%** at 30 days |
| 33866715 | Singapore Med J 2022 | acute fragility hip fracture on an integrated care pathway, five years | **3.2% → 1.6%** (`range`) at 30 days |

> 25550095 — *"129 patients (11.9 %) were readmitted to the hospital within 30 days of their initial discharge date."*
> 33866715 — *"(d) reduced 30-day readmission rate from 3.2% to 1.6%; and"*

### 90000069 — cervical spondylosis / cervical decompression

| pmid | journal | cohort | rate |
|---|---|---|---|
| 26872307 | Spine 2018 | 3,401 posterior cervical fusions, NSQIP | **6.20%** at 30 days |
| 37716549 | Spine J 2023 | 33,092 ambulatory cervical spine surgery patients, NSQIP | **3.37%** at 30 days |

> 26872307 — *"Rate of 30-day readmission was 6.20%."*
> 37716549 — *"Incidences of 30-day readmission were 3.37% and 3.07% among cervical and lumbar patients, respectively."*

### 90000019 — total hip arthroplasty

| pmid | journal | cohort | rate |
|---|---|---|---|
| 33422393 | J Arthroplasty 2021 | primary THA, n=25,081, national all-payer database | **3.3%** at 30 days |
| 32742739 | Geriatr Orthop Surg Rehabil 2020 | over-80 THA, n=7,730, NSQIP | **4.9%** at 30 days |

## 3. Fix-by-fix

| fix | verdict |
|---|---|
| **R1** list-charts | partial — `90000081` derives *Pleural effusion / Thoracentesis* again; `90000058` still yields nothing (see §4, F1) |
| **R2** horizon from title | not exercised live — none of the title-only-horizon records (`24108298`, `33049410`, `31528438`) appeared in the 1C result sets. Three hermetic tests cover the rule; unproven against NCBI. |
| **R3** disease over symptom | worked — `cervical radiculopathy` → `Cervical spondylosis`, 9→10 aligned, 0→2 verified |
| **R4** sub-cohorts, temperature | worked — `32742739` accepted on its stated qualifier; `build_llm` already pins `temperature=0`, now asserted by a test |

## 4. Findings

### F1 — the provider's content filter withheld the target reply, reported as an empty chart

`90000090` returned `finish_reason=SAFETY`, `HARM_CATEGORY_DANGEROUS_CONTENT`, empty content, reproducibly across three calls with the 1C prompt. Gate 1's prompt had answered the same note. The report said *"the chart named no condition"* for a chart whose only diagnosis is a femoral neck fracture.

Mitigations applied:

* `derive_target` returns a `TargetResult` carrying the reason, and retries once with `LITERATURE_TARGET_FALLBACK_PROMPT` — the same rules, shorter frame. Recovered `90000090` and `90000001`.
* `_structured_reply` distinguishes `withheld` from `not_json`, so a withheld extraction reply is no longer reported as a malformed one.

`90000058` is withheld on **both** prompts, so this is not purely a prompt-length artefact. Its note is a serotonin-syndrome course — a plausible filter trigger on `DANGEROUS_CONTENT`.

### F2 — truncation order, not the query, is the binding constraint

`90000081` lost `31441096` (33% hepatic hydrothorax — Gate 1's row) because the OR'd anchors rank empyema, lobectomy and chest-wall papers into the top ten while the hydrothorax study falls outside the pool that survives filtering. `90000001` returned a single candidate for the same class of reason. No prompt change fixes this: it needs a ranking rule, not a wording rule.

## 5. Stop

Stage Gate 1C ends here. `presentation/domain.py` still supplies the comparative table; nothing is wired into `run_command`.
