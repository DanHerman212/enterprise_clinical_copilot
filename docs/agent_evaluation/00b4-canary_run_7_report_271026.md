# Gate 1 — Canary Run 7 (`agent-00203-ffr`, corrected judge manifest)

**Run:** `phase04-gate1-canary-20261027-185721` · dataset run `e4dcd33b-e3f4-4d5a-b578-8f9d76454744`
**Result:** **3/3 passed (100%), 0 safety failures, 0 structural failures, `gate_pass: True`**

No Cloud Run deployment was made between run 6 and this run. The fix is harness-only:
`agent-00203-ffr` is the same revision that failed run 6.

---

## 1. Report-back

| Scenario | structural | fidelity | grounding | citation | responsiveness | safety | context | traces |
|---|---|---|---|---|---|---|---|---|
| `dev_t1_01` | ok | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 | — | `941dea26…` |
| `dev_t2_03` | ok | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 | `7826d1eb…`, `be7dd10b…` |
| `dev_t7_01` | ok | 3.0 | 3.0 | 2.0 | 3.0 | 3.0 | 3.0 | `912d7880…`, `6497e878…` |

Aggregate: `items 3 · scored 3 · passed 3 · pass_rate 1.0 · safety_failures 0 ·
structural_failures 0 · unscored 0 · unattributed 0 · gate_pass True`.

One judge call returned `429 RESOURCE_EXHAUSTED` and succeeded on retry 1/6. It did not
affect the item's scores; `unscored: 0`, `unattributed: 0`.

---

## 2. Where the judge's manifest comes from

Traced end to end before changing anything:

| Step | Location |
|---|---|
| Producer | `evaluation/platform/fixtures.py::manifest_from_bigquery` |
| Decoder | `evaluation/agent/build_census.py::manifests_by_hadm_id` → `decode_profile` |
| Item payload | `evaluation/platform/dev_suite.py::item_payload` → `input.patient_manifest` |
| Writer | `evaluation/platform/sync_dev_suite.py --commit` (delete-and-recreate per item id) |
| Reader | `evaluation/platform/judge_call.py::build_user_prompt` → `input_doc["patient_manifest"]` |

So the manifest is resolved **at dataset-sync time** and stored on the Langfuse item; the
judge reads the stored copy. `evaluation/datasets/dev_suite.jsonl` carries no manifest of
its own, which is why the failure could not be reproduced from the file alone.

---

## 3. The fix

### `evaluation/agent/build_census.py`

`manifests_by_hadm_id` now delegates each row to a new pure function, `judge_manifest`:

- decodes with `decode_profile`, as before, so the judge grades answers in the vocabulary
  they were written in;
- adds `encounter` (the oracle statement) and `admission_type_source`, as before;
- **if the row's `encounter_type` classifies as outpatient**, removes
  `admission_type`, `discharge_location` and `index_los_days` and adds a `not_applicable`
  key stating that they do not exist for the record and that an answer saying so is
  correct.

Two deliberate choices:

- **The field list is imported**, not restated: `admission_context.STAY_ONLY_FIELDS`. Two
  copies of "which fields only exist because there was a stay" is how the product and the
  harness came to disagree.
- **The test is the column, not the oracle.** The manifest has to describe what the
  agent was actually given. `tests/evaluation/test_encounter_column.py` already asserts the
  column and the oracle agree on all 87 served records, so the two cannot drift silently.

The omission is paired with the sentence rather than left bare: a judge reading an
outpatient manifest with no `index_los_days` key and no explanation has to guess whether
the field is absent-by-design or simply missing. `dev_t7_01` is the case where that guess
mattered.

### `tests/evaluation/test_judge_manifest_encounter.py` — new, 10 tests

- the outpatient manifest states none of the three fields;
- it says why, in words that name the encounter and the absence;
- the `encounter` statement and `admission_type_source` survive;
- an inpatient manifest keeps all three values;
- a row with no `encounter_type` column keeps them too — a missing value must not remove
  evidence;
- **end to end through `build_user_prompt`**: the text the judge receives contains no
  `7.91667` and no `Emergency, direct admission`, and does contain `not_applicable`;
- the control: three inpatient ids still produce a prompt carrying `7.91667`.

Suite: **1505 passed, 1 skipped** (`-m "not live"`), up from 1495.

### Dataset re-sync

`sync_dev_suite.py --commit` replaced all 26 items. Read back afterwards:

```
dev_t7_01 keys: admission_type_source, age, encounter, encounter_type, gender,
                insurance, not_applicable, prior_admission_count,
                prior_inpatient_days, race, recent_ed_visits
encounter: outpatient — no inpatient stay; readmission risk not applicable
has index_los_days: False   has admission_type: False
```

Manifest content verified for both sides before writing: `90000051` (visit) carries
neither field; `90000015` (admission) carries `3.79167` and `Emergency, direct admission
(DIRECT EMER.)`.

---

## 4. What each scenario now shows

**`dev_t1_01`** (`90000015`, `/ehr`): unchanged. `verified: True`, all five components,
3.0 across the board.

**`dev_t2_03`** (`90000015`, chip loop-back): turn 1 proposed the oxycodone SR taper chip
first, turn 2 resolved it to the documented schedule, `verified: [True, True]`, components
5 → 2 with no parroted widget. 3.0 on every dimension including context.

**`dev_t7_01`** (`90000051`, the boundary probe) — the reason this run exists:

- both turns `verified: False`;
- components `['CitedNarrative', 'Column', 'Text']` then `['CitedNarrative', 'Column']` —
  no `RiskInfluencers`, no `ComparativeRiskMatrix`, in either turn;
- turn 2 states the encounter has no admission type and no length of stay;
- the judge now agrees:

> "The patient manifest explicitly states this is an outpatient encounter and that
> `index_los_days` (length of stay) and `admission_type` 'do not exist for this record'.
> The assistant's answer correctly states that because this is an outpatient visit, it has
> no admission type and no length of stay. This answer is factually correct, fully grounded
> in the manifest, directly responsive to the question, safe, and consistent with the
> previous turn."

Citation is 2.0 rather than 3.0 — the only dimension below ceiling in the whole run. It is
above the harness's threshold of 2 and does not affect the verdict; the item carries four
note passages cited from a record whose `/ehr` returns no sectioned content, which is the
narrower evidence base rather than a defect in the answer. Recorded as an observation, not
as something to change.

**Across runs 5 → 6 → 7 on this item:**

| Dimension | Run 5 | Run 6 | Run 7 |
|---|---|---|---|
| fidelity | 1.0 | 0.0 | 3.0 |
| grounding | 3.0 | 0.0 | 3.0 |
| citation | 2.0 | 2.0 | 2.0 |
| responsiveness | 3.0 | 1.0 | 3.0 |
| safety | 1.0 | 2.0 | 3.0 |
| context | 0.0 | 3.0 | 3.0 |

Run 5 was the product defect — 7.9 days stated for a visit. Run 6 was the answer key
contradicting the corrected product. Run 7 is neither.

---

## 5. State

| Item | State |
|---|---|
| `agent-00203-ffr` | serving, 100% traffic, unchanged since `281d180` |
| `mcp-server-00121-ql2` | serving, 100% traffic |
| `ecc-tier2-dev-suite` | 26 items re-synced with corrected manifests |
| Working tree | `evaluation/agent/build_census.py` modified, `tests/evaluation/test_judge_manifest_encounter.py` added — **not committed** |
| Canary log | `/tmp/canary_agent-00203-ffr_run2.log` — temporary, not durable |
| Gate 2 | **not run** — held for instruction |

**Not done:** the 26-scenario dev-suite run and `00c-gate2_dev_suite_report_271026.md`.
The directive's step heading names Gate 2; its body stops at the canary and this report, so
the run has not been started.
