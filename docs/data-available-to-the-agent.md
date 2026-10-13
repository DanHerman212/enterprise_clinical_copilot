# Data available to the agent

Written 2026-10-11 for the UX workshop. Two stores are involved, and they hold
different kinds of thing:

- **The site database** (Cloud SQL, Postgres) holds *the demonstration's own
  state* — who the patients are, what was asked, what was answered, and quota.
  It holds no clinical data.
- **The warehouse** (BigQuery, `trim-icon-498815-a0.readmission`, us-east1) holds
  the *clinical data the tools read*: the model features, the discharge notes,
  the cohort boundary. These are the tables the MCP tools query.

---

## 1. Recommended patients for the UX session

Three admissions, chosen to exercise a different **decision state**, carry a
**dense note** for retrieval and citation, and tell a different story in the
summary line. Probabilities are from the current model (operating threshold
0.11).

| | Patient | `hadm_id` | Probability | State | Note | Why this one |
|---|---|---|---|---|---|---|
| 1 | Bruce Hollingsworth | 90000023 | 0.1148 | **above** (barely) | 3,443 chars · 12 sections | The most instructive alert: it clears the threshold by 0.005, so the risk widget has to explain *why*, not merely colour a bar. Sections include hospital course, discharge labs, procedures and discharge instructions. 60M · elective · 2 prior admissions · oncology history · 11-day stay · discharged home. |
| 2 | Susan Okafor | 90000019 | 0.0977 | **below** (barely) | 4,572 chars · **21 sections, the most in the cohort** | The reassurance case that needs the most care, and the richest section coverage for the source card and "show full section". 51F · elective · 2 prior admissions · 5-day stay · discharged home. |
| 3 | Cynthia Petrov | 90000015 | 0.0627 | **clearly below** | **6,814 chars, the densest in the cohort** · 18 sections | The unambiguous reassured state with the best retrieval material available. 61F · elective · 1 prior admission · 8-day stay · discharged to rehab. |

Together they span the decision boundary from both sides, three lengths of stay,
two discharge dispositions, and the two densest notes in the cohort.

### Alternates, with the cost stated

- **Randall Okafor — 90000108, p=0.2105.** The only *strongly* above-threshold
  patient, nearly twice the threshold, so the alert state is unmistakable. The
  cost is a thin note: 2,697 characters across **3 sections** (history of present
  illness, discharge medications, plan), so retrieval and citation have little to
  work with. It does carry discharge medications, so the medications journey
  works on it.
- **Curtis Whitfield — 90000095, p=0.0916.** 4,973 characters, 19 sections, and
  the only dense **emergency** admission: 45M, oncology history, 4-day stay,
  discharged to rehab. Take it if the workshop wants an emergency journey — all
  three picks above are elective admissions.
- **Chelsea Rasmussen — 90000020, p=0.1119.** 36F, just above threshold, 1,820
  characters across 10 sections. The thinnest of the near-threshold set, but the
  youngest patient in it.

### The shape of the cohort

Every one of the 87 admissions in the allowlist can be scored. (Two rows exist
in local development that were excluded from production and cannot be scored, so
a local run against those returns the generic failure message.) Against the
operating threshold of 0.11 the cohort is heavily weighted toward reassurance:
**83 below, 4 above**. Only four admissions exercise the alert state, which is
why it is worth choosing deliberately rather than by chance.

Do not use the `band` labels (low / borderline / high) that appear in
`hybrid_notes.json` and `hybrid_cohort.json`: they belong to the stale 0.12 run
and disagree with the current model's decisions.

---

## 2. Site database — tables the demo uses

Read by the Django application; the agent never sees these directly.

### `demo_demopatient` — the allowlist and the `@` selector
| Column | Type | Notes |
|---|---|---|
| `hadm_id` | bigint | **PK** — the admission id used everywhere else |
| `display_name` | varchar | what the `@` selector shows |
| `age` | smallint | |
| `sex` | varchar | |
| `summary` | varchar | one-line human summary, e.g. "60F · elective admission · 1 procedure · 8-day stay · discharged to rehab" |
| `split_name` | varchar | always `test` |
| `updated_at` | timestamp | |

87 rows in production. Membership is the authorisation boundary: a request for an
admission outside this table is refused before the agent is called.

### `demo_conversation` — one thread, one patient
| Column | Type | Notes |
|---|---|---|
| `id` | bigserial | **PK** |
| `user_id` | bigint | FK → `auth_user` |
| `patient_id` | bigint | FK → `demo_demopatient.hadm_id` |
| `created_at` | timestamp | |
| `last_turn_at` | timestamp | nullable |
| `expires_at` | timestamp | retention |
| `turn_count` | smallint | |

### `demo_turn` — one question and its answer
| Column | Type | Notes |
|---|---|---|
| `id` | bigserial | **PK** |
| `conversation_id` | bigint | FK → `demo_conversation` |
| `ordinal` | smallint | position in the thread; unique per conversation |
| `role` | varchar | `user` or `agent` |
| `question` | text | what was asked |
| `answer` | text | the composed answer |
| `model` / `code_revision` | varchar | what produced it |
| `langfuse_trace_id` | varchar | pointer to the trace |
| `citations` | jsonb | `[{cite, section, query}]` — identities only, passages re-derived |
| `tool_calls` | jsonb | `[{name, args, payload}]` — payload kept only where not re-derivable |
| `guardrail_flags` | jsonb | which guardrails fired, by name |
| `error` | varchar | |
| `created_at` | timestamp | |

### `demo_demoquota`, `demo_profile`, `auth_user`
Quota: `user_id`, `daily_limit`, `used`, `refunds`, `period_start`. Profile:
`display_name`, `avatar_url`. Standard Django `auth_user` for identity.

---

## 3. Warehouse — the clinical data

### `readmission.hybrid_features` — 53 columns (the model's inputs)
The `predict_readmission` tool reads this. `hadm_id` is the entity key.

**Utilisation and stay (7)** — `age`, `prior_admission_count`,
`prior_inpatient_days`, `recent_ed_visits`, `index_los_days`, `procedure_count`,
`medication_count`, `medication_order_count`

**Labs (9)** — `rbc_last`, `rbc_min`, `rdw_max`, `monocytes_min`,
`hemoglobin_min`, `sodium_last`, `sodium_max`, `sodium_min`, `oncology_flag`

**Demographics (9)** — `gender`, `race_white`, `race_black`, `race_hispanic`,
`race_asian`, `race_amind`, `race_nhpi`, `race_unknown`, `has_procedure`

**Admission type (8, one-hot)** — `admission_type_ew_emer`, `_eu_obs`,
`_obs_admit`, `_urgent`, `_direct_emer`, `_ambulatory_obs`, `_direct_obs`,
`_unknown`

**Discharge location (10, one-hot)** — `discharge_location_home`,
`_home_health`, `_snf`, `_rehab`, `_ltac`, `_hospice`, `_ama`, `_psych`,
`_assisted_living`, `_unknown`

**Insurance (5, one-hot)** — `insurance_medicare`, `_medicaid`, `_private`,
`_other`, `_unknown`

**Bookkeeping** — `subject_id` (bigint), `split_name` (string), `readmission_30d`
(integer, the label).

Those 49 features are a deliberate slice. MIMIC-IV carries far more — diagnoses,
procedures, chart events, prescriptions, microbiology, services, transfers, and
the free-text notes in full. Nothing else from it is loaded today, which is the
opportunity if we want to widen the tools.

### `readmission.hybrid_notes` — the retrieval corpus
| Column | Type | Notes |
|---|---|---|
| `hadm_id` | integer | joins to the features and the allowlist |
| `note_id` | string | the note identity within the admission |
| `text` | string | the discharge note, sectioned |

Three columns only. The note text is public MTSamples transcriptions re-keyed to
demonstration admission ids; the RAG index is built from this table, chunked by
section. Recommended sections for retrieval are defined in the chunking module
(`history_of_present_illness`, `hospital_course`, `discharge_diagnoses`,
`discharge_medications`, and so on).

### `readmission.demo_cohort` — the authorisation boundary (53 columns)
Same shape as `hybrid_features`. Derived from the corpus the tools actually
serve, so the boundary and the data cannot drift apart.

### `readmission.hybrid_split` — 2 columns
`hadm_id`, `split_name`.

### Present but out of scope for the demo
`analytics_dataset`, `analytics_dataset_encoded` (the real MIMIC-derived table —
never carries the demonstration admissions), `hybrid_features_v2` (a regeneration
staging table), `synthetic_features`, `synthetic_notes`, `synthetic_split`
(superseded by the hybrid set), `hospital_score` (the HOSPITAL baseline),
`test_predictions`.

---

## 4. Which tool reads what

| Tool | Reads |
|---|---|
| `predict_readmission` | `hybrid_features`, via the Vertex endpoint |
| `rag_search` | the Vector Search index (built from `hybrid_notes`); passage text re-resolved from `hybrid_notes` |
| `rag_search_sections` | `hybrid_notes` |
| `discharge_medications` | `hybrid_notes`, the discharge medications section |
| `search_literature` | PubMed (NCBI E-utilities) — outside the warehouse entirely |

---

## 5. Two things worth knowing before quoting numbers

**Ignore the threshold 0.12 figures.** Two captured probability sets exist and
they come from different training runs. `demo_fixtures/cohort_risk.json`
(threshold 0.11) is the **current** model and is the one to quote;
`evaluation/agent/results/hybrid_cohort.json` (threshold 0.12) is a **stale**
run and should be disregarded. For the same admission they disagree — 90000001
reads 0.0432 in the current set and 0.0649 in the stale one — so no number
should be carried into a document from the 0.12 set, including the curated `band`
labels that accompany it.

**Every row in the warehouse is demonstration data.** The admissions are
`90000001+`, the notes are re-keyed public transcriptions, and the features are
derived from them. Nothing here is real patient data, and the authorisation table
exists so that even a request bypassing the site could only ever reach this
cohort.
