# Multi-turn chip answerability — before and after

- Before: `agent-00175-ttr` — 4/21 GOOD, 17 BAD
- After: `agent-00176-ngw` — 21/21 GOOD, 0 BAD

## Before

Bad reasons: {'no resolving ^[n] citation': 10, 'follow_ups repeat an asked question': 7}

| Record | Turn 1 | Chip | Turn 2 tools | Citations | Verdict |
|---|---|---|---|---|---|
| 90000015 | /ehr | What is the dosing and taper schedule for oxycodone SR at discharge? | discharge_medications | — | BAD — no resolving ^[n] citation |
| 90000015 | /ehr | What is the discharge dosing and monitoring plan for Coumadin? | discharge_medications | — | BAD — no resolving ^[n] citation |
| 90000015 | /ehr | What outpatient follow-up appointments or clinic visits are scheduled … | rag_search, rag_search, rag_search, discharge_… | 1 resolved | BAD — follow_ups repeat an asked question |
| 90000015 | /meds | What is the dosing and taper schedule for oxycodone SR at discharge? | rag_search_sections, rag_search, rag_search, r… | 1 resolved | BAD — follow_ups repeat an asked question |
| 90000015 | /meds | What is the discharge dosing and monitoring plan for Coumadin? | rag_search, rag_search, rag_search, rag_search… | 1 resolved | BAD — follow_ups repeat an asked question |
| 90000015 | /meds | What outpatient follow-up appointments or clinic visits are scheduled … | rag_search | 1 resolved | BAD — follow_ups repeat an asked question |
| 90000023 | /ehr | What post-discharge activity restrictions, diet instructions, or warni… | rag_search | 1 resolved | BAD — follow_ups repeat an asked question |
| 90000023 | /ehr | What outpatient follow-up appointments or clinic visits are scheduled … | rag_search_sections | — | BAD — no resolving ^[n] citation |
| 90000023 | /ehr | Why does oncology history increase this patient's readmission risk dri… | get_lab_trends | 1 resolved | GOOD |
| 90000023 | /meds | What post-discharge activity restrictions, diet instructions, or warni… | rag_search_sections | — | BAD — no resolving ^[n] citation |
| 90000023 | /meds | What outpatient follow-up appointments or clinic visits are scheduled … | rag_search_sections | — | BAD — no resolving ^[n] citation |
| 90000105 | /ehr | What post-discharge activity restrictions, diet instructions, or warni… | rag_search | — | BAD — no resolving ^[n] citation |
| 90000105 | /ehr | What outpatient follow-up appointments or clinic visits are scheduled … | rag_search | — | BAD — no resolving ^[n] citation |
| 90000105 | /meds | What post-discharge activity restrictions, diet instructions, or warni… | rag_search_sections | — | BAD — no resolving ^[n] citation |
| 90000105 | /meds | What outpatient follow-up appointments or clinic visits are scheduled … | rag_search_sections | — | BAD — no resolving ^[n] citation |
| 90000105 | /meds | What are the discharge dosing instructions for omeprazole and Lortab? | rag_search, discharge_medications, rag_search,… | 1 resolved | GOOD |
| 90000106 | /ehr | Why does red cell distribution width (RDW) increase this patient's rea… | get_lab_trends | — | GOOD |
| 90000108 | /ehr | Why does oncology history increase this patient's readmission risk dri… | rag_search | 1 resolved | GOOD |
| 90000108 | /meds | What post-discharge activity restrictions, diet instructions, or warni… | rag_search_sections, rag_search | 1 resolved | BAD — follow_ups repeat an asked question |
| 90000108 | /meds | What outpatient follow-up appointments or clinic visits are scheduled … | rag_search | 1 resolved | BAD — follow_ups repeat an asked question |
| 90000108 | /meds | What are the discharge dosing instructions for Phenergan and Duragesic… | discharge_medications | — | BAD — no resolving ^[n] citation |

## After

Bad reasons: {}

| Record | Turn 1 | Chip | Turn 2 tools | Citations | Verdict |
|---|---|---|---|---|---|
| 90000015 | /ehr | What is the dosing and taper schedule for oxycodone SR at discharge? | discharge_medications | 1 resolved | GOOD |
| 90000015 | /ehr | What is the discharge dosing and monitoring plan for Coumadin? | discharge_medications | 1 resolved | GOOD |
| 90000015 | /ehr | What outpatient follow-up appointments or clinic visits are scheduled … | rag_search, rag_search, rag_search, discharge_… | 1 resolved | GOOD |
| 90000015 | /meds | What is the dosing and taper schedule for oxycodone SR at discharge? | rag_search_sections, rag_search, rag_search, r… | 1 resolved | GOOD |
| 90000015 | /meds | What is the discharge dosing and monitoring plan for Coumadin? | rag_search, rag_search, rag_search, rag_search… | 1 resolved | GOOD |
| 90000015 | /meds | What outpatient follow-up appointments or clinic visits are scheduled … | rag_search | 1 resolved | GOOD |
| 90000023 | /ehr | What post-discharge activity restrictions, diet instructions, or warni… | rag_search | 1 resolved | GOOD |
| 90000023 | /ehr | What outpatient follow-up appointments or clinic visits are scheduled … | rag_search_sections | 1 resolved | GOOD |
| 90000023 | /ehr | Why does oncology history increase this patient's readmission risk dri… | get_lab_trends | 1 resolved | GOOD |
| 90000023 | /meds | What post-discharge activity restrictions, diet instructions, or warni… | rag_search_sections | 1 resolved | GOOD |
| 90000023 | /meds | What outpatient follow-up appointments or clinic visits are scheduled … | rag_search_sections | 1 resolved | GOOD |
| 90000105 | /ehr | What post-discharge activity restrictions, diet instructions, or warni… | rag_search | 1 resolved | GOOD |
| 90000105 | /ehr | What outpatient follow-up appointments or clinic visits are scheduled … | rag_search | 1 resolved | GOOD |
| 90000105 | /meds | What post-discharge activity restrictions, diet instructions, or warni… | rag_search_sections | 1 resolved | GOOD |
| 90000105 | /meds | What outpatient follow-up appointments or clinic visits are scheduled … | rag_search_sections | 1 resolved | GOOD |
| 90000105 | /meds | What are the discharge dosing instructions for omeprazole and Lortab? | rag_search, discharge_medications, rag_search,… | 1 resolved | GOOD |
| 90000106 | /ehr | Why does red cell distribution width (RDW) increase this patient's rea… | get_lab_trends | — | GOOD |
| 90000108 | /ehr | Why does oncology history increase this patient's readmission risk dri… | rag_search | 1 resolved | GOOD |
| 90000108 | /meds | What post-discharge activity restrictions, diet instructions, or warni… | rag_search_sections, rag_search | 1 resolved | GOOD |
| 90000108 | /meds | What outpatient follow-up appointments or clinic visits are scheduled … | rag_search | 1 resolved | GOOD |
| 90000108 | /meds | What are the discharge dosing instructions for Phenergan and Duragesic… | discharge_medications | 1 resolved | GOOD |

## Three-turn progression — before

| Record | Turn 1 | Turn 2 clicked | Turn 2 offered | Turn 3 clicked | Turn 3 offered | Advanced? |
|---|---|---|---|---|---|---|
| 90000015 | /ehr | What is the dosing and taper schedule for oxyc… | What is the discharge dosing and monitoring plan for Coumadin?; What o… | What is the dosing and taper schedule for oxyc… | What is the dosing and taper schedule for oxycodone SR at discharge?; … | NO — repeated |
| 90000015 | /meds | What is the dosing and taper schedule for oxyc… | What is the discharge dosing and monitoring plan for Coumadin?; What o… | What is the dosing and taper schedule for oxyc… | What is the dosing and taper schedule for oxycodone SR at discharge?; … | NO — repeated |
| 90000023 | /ehr | What post-discharge activity restrictions, die… | What outpatient follow-up appointments or clinic visits are scheduled … | What post-discharge activity restrictions, die… | What post-discharge activity restrictions, diet instructions, or warni… | NO — repeated |
| 90000023 | /meds | What post-discharge activity restrictions, die… | What outpatient follow-up appointments or clinic visits are scheduled … | What post-discharge activity restrictions, die… | What post-discharge activity restrictions, diet instructions, or warni… | NO — repeated |
| 90000105 | /ehr | What post-discharge activity restrictions, die… | What outpatient follow-up appointments or clinic visits are scheduled … | What post-discharge activity restrictions, die… | What post-discharge activity restrictions, diet instructions, or warni… | NO — repeated |
| 90000105 | /meds | What post-discharge activity restrictions, die… | What outpatient follow-up appointments or clinic visits are scheduled … | What post-discharge activity restrictions, die… | What post-discharge activity restrictions, diet instructions, or warni… | NO — repeated |
| 90000106 | /ehr | Why does red cell distribution width (RDW) inc… | How does the abnormal discharge RDW (23.7 %) compare to its reference … | How does the abnormal discharge RDW (23.7 %) c… | How does the abnormal discharge RDW (23.7 %) compare to its reference … | yes |
| 90000106 | /meds | — | no chip offered | — | — | — |
| 90000108 | /ehr | Why does oncology history increase this patien… | What post-discharge activity restrictions, diet instructions, or warni… | What post-discharge activity restrictions, die… | What post-discharge activity restrictions, diet instructions, or warni… | yes |
| 90000108 | /meds | What post-discharge activity restrictions, die… | What outpatient follow-up appointments or clinic visits are scheduled … | What post-discharge activity restrictions, die… | What post-discharge activity restrictions, diet instructions, or warni… | NO — repeated |

## Three-turn progression — after

| Record | Turn 1 | Turn 2 clicked | Turn 2 offered | Turn 3 clicked | Turn 3 offered | Advanced? |
|---|---|---|---|---|---|---|
| 90000015 | /ehr | What is the dosing and taper schedule for oxyc… | What outpatient follow-up appointments or clinic visits are scheduled … | What is the discharge dosing and monitoring pl… | What is the discharge dosing and monitoring plan for Coumadin?; What o… | yes |
| 90000015 | /meds | What is the dosing and taper schedule for oxyc… | What outpatient follow-up appointments or clinic visits are scheduled … | What is the discharge dosing and monitoring pl… | What is the discharge dosing and monitoring plan for Coumadin?; What o… | yes |
| 90000023 | /ehr | What post-discharge activity restrictions, die… | What were the primary diagnosis and hospital course documented in the … | What outpatient follow-up appointments or clin… | What outpatient follow-up appointments or clinic visits are scheduled … | yes |
| 90000023 | /meds | What post-discharge activity restrictions, die… | How does the abnormal discharge Red blood cells (3.33 x10^6/uL) compar… | What outpatient follow-up appointments or clin… | What outpatient follow-up appointments or clinic visits are scheduled … | yes |
| 90000105 | /ehr | What post-discharge activity restrictions, die… | What were the primary diagnosis and hospital course documented in the … | What outpatient follow-up appointments or clin… | What outpatient follow-up appointments or clinic visits are scheduled … | yes |
| 90000105 | /meds | What post-discharge activity restrictions, die… | What were the primary diagnosis and hospital course documented in the … | What outpatient follow-up appointments or clin… | What outpatient follow-up appointments or clinic visits are scheduled … | yes |
| 90000106 | /ehr | Why does red cell distribution width (RDW) inc… | What were the primary diagnosis and hospital course documented in the … | How does the abnormal discharge RDW (23.7 %) c… | How does the abnormal discharge RDW (23.7 %) compare to its reference … | yes |
| 90000106 | /meds | — | no chip offered | — | — | — |
| 90000108 | /ehr | Why does oncology history increase this patien… | What outpatient follow-up appointments or clinic visits are scheduled … | What post-discharge activity restrictions, die… | What post-discharge activity restrictions, diet instructions, or warni… | yes |
| 90000108 | /meds | What post-discharge activity restrictions, die… | What were the primary diagnosis and hospital course documented in the … | What outpatient follow-up appointments or clin… | What outpatient follow-up appointments or clinic visits are scheduled … | yes |
