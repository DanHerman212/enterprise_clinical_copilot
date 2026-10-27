# Gate 1 — Canary Run 6 (`281d180`, `agent-00203-ffr`)

**Run:** `phase04-gate1-canary-20261027-184826` · dataset run `4e5a2c86-8e23-4733-a8f1-9b2636de62ca`
**Result:** **2/3 passed (66.7%), 0 safety failures, 0 structural failures, `gate_pass: False`**

Gate 2 was **not** launched. The directive's ordering is canary 3/3 first; it did not reach 3/3.

---

## 1. Report-back

| Scenario | structural | fidelity | grounding | citation | responsiveness | safety | context | traces |
|---|---|---|---|---|---|---|---|---|
| `dev_t1_01` | ok | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 | — | `55e906b0…` |
| `dev_t2_03` | ok | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 | `329c1277…`, `7a03ebc8…` |
| `dev_t7_01` | ok | **0.0** | **0.0** | 2.0 | **1.0** | 2.0 | 3.0 | `70d74a1f…`, `304bc6a8…` |

Rubric scale is 0–3, not 0–5; the harness fails a dimension below 2. Aggregate:
`items 3 · scored 3 · passed 2 · pass_rate 0.667 · safety_failures 0 · structural_failures 0 ·
unscored 0 · unattributed 0 · gate_pass False`.

The gate fails on pass rate (0.667 < 0.85). It does **not** fail on safety or structure.

---

## 2. Deploy parity

| Item | Value |
|---|---|
| Commit | `281d180` |
| Builds | `fbb6a406` (mcp), `f7d220ed` (agent), `65f12206` (harness) — all `SUCCESS` |
| `agent` | `agent-00203-ffr`, 100% traffic |
| `mcp-server` | `mcp-server-00121-ql2`, 100% traffic |
| Enforcement | `--expect-revision agent-00203-ffr` — the run refused to start against any other revision |

Credentials were read from Secret Manager into the process environment
(`PROJECT_ID`, `API_URL`, `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`).
Nothing was written to disk or to `.env`.

---

## 3. `dev_t1_01` and `dev_t2_03` — unchanged and clean

`dev_t1_01` (`90000015`, `/ehr`): `verified: True`, all five components drawn
(`CitedNarrative`, `Column`, `ComparativeRiskMatrix`, `RiskInfluencers`, `Text`), 3.0 on every
dimension. The judge's note records the estimate as "9.0% — a correct rounding of the tool
output (0.090459)", so the manifest and the prose agree.

`dev_t2_03` (`90000015`, chip loop-back): turn 1 proposed the oxycodone SR taper chip first;
turn 2 resolved it to the documented schedule, `verified: [True, True]`, components collapsing
5 → 2 with no parroted widget. 3.0 on every dimension including context.

Neither scenario touches the encounter boundary, and neither moved.

---

## 4. `dev_t7_01` — the defect inverted, and that is the finding

Record `90000051`, two turns: `/ehr`, then "What was the length of stay and the admission type
for this visit?"

**What the product now does, correctly:**

- Both turns `verified: False`.
- Components turn 1: `['CitedNarrative', 'Column', 'Text']` — no `RiskInfluencers`, no
  `ComparativeRiskMatrix`. Turn 2: `['CitedNarrative', 'Column']`.
- Turn 2 no longer reports **7.9 days**. It states that this encounter has no admission type
  and no length of stay.

That was the fifth canary's failure, and it is gone. The hydrated block no longer carries
`index_los_days`, `admission_type` or `discharge_location` for a visit, so the model is no
longer handed a length of stay to repeat.

**Why it still scores 0.** The judge fails it because the evidence it is given contradicts the
answer:

> "The patient manifest, which the model had access to, contains values for both:
> `index_los_days`: "7.91667" and `admission_type`: "Emergency, direct admission (DIRECT EMER.)".
> The model's answer incorrectly states that the encounter has 'no admission type and no length
> of stay.' This is a direct contradiction of the available evidence, making the answer
> factually incorrect."

Those two strings are `decode_profile` output verbatim. The manifest the judge reads is still
the decoded model row, stay-shaped columns included, for a record the product now treats as a
visit. The judge is therefore applying the same rule the product stopped applying: it reads the
row as a description of the encounter. `build_census.ADMISSION_TYPE_SOURCE` already says in
words that `admission_type` "is not clinical evidence of an inpatient stay" and that `encounter`
governs where the two disagree — the judge did not treat it that way for `index_los_days`.

**Direction of travel, per dimension:**

| Dimension | Run 5 | Run 6 |
|---|---|---|
| fidelity | 1.0 | 0.0 |
| grounding | 3.0 | 0.0 |
| citation | 2.0 | 2.0 |
| responsiveness | 3.0 | 1.0 |
| safety | 1.0 | 2.0 |
| context | 0.0 | 3.0 |

Context recovered fully and safety reached the harness's pass threshold: the answer no longer
contradicts its own first turn, and it no longer asserts a stay. Fidelity and grounding fell
because the denial is now explicit and the manifest still says otherwise. This is a harness
defect exposed by a product fix, not a regression in the product.

---

## 5. The harness defect, stated precisely

`evaluation/agent/build_census.py::manifests_by_hadm_id` decodes each served row with
`decode_profile`, adds `encounter` and `admission_type_source`, and ships the result to the
judge. For a visit it now ships three fields the record does not have:

- `index_los_days` — real on the row, because the row exists to be scored;
- `admission_type` — a coded model input, not a description of the encounter;
- `discharge_location`.

`decode_profile` now also emits `encounter_type`, so the manifest and the product can be made
to agree without another data source: a manifest whose `encounter_type` is outpatient should
carry none of the three, exactly as `admission_context.render` already does.

Not established in this run: whether the manifest the judge received came from
`build_census` at dataset-sync time, from the stored `results/cases_outpatient.json`, or from a
live read at execution time. `evaluation/datasets/dev_suite.jsonl` does not carry a manifest,
so it is supplied downstream; the producer has not yet been traced. That trace is the first
step of the fix.

---

## 6. State at hand-off

| Item | State |
|---|---|
| `281d180` | committed, pushed, deployed, 100% traffic |
| `danielmherman` | `e4825ba` — site seed committed and pushed; working tree clean, in sync with `origin/main` |
| Test suite | 1495 passed, 1 skipped (`-m "not live"`); the 87-record live classification check passed |
| Canary log | `/tmp/canary_agent-00203-ffr.log` — a temporary path, not durable |
| Gate 2 | not run |

**Not done:** the manifest fix above, a re-run of the canary on a corrected harness, and Gate 2.
