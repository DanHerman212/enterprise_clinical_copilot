# Encounter type — implementation record (2026-10-27)

What was executed under the directive authorising the `encounter_type` schema fix, the
deletion of `OUTPATIENT_HADM_IDS`, and Gate 1 → Gate 2. This is a record of work done,
not a ruling or a result. The canary and Gate 2 have **not** been run.

Session began after an editor crash; the working tree was inspected before anything was
changed, so the first section distinguishes what had already landed from what this session
completed.

---

## 1. State inherited from the interrupted session

| Item | State on inspection |
|---|---|
| `readmission.hybrid_features` | `encounter_type` present, backfilled — 82 `inpatient`, 5 `outpatient` |
| `readmission.demo_cohort` | same column, same split |
| The five outpatient ids | `90000028 90000040 90000051 90000054 90000073` — identical to the retired frozenset |
| `services/agent/routing/encounter.py` | written, untracked |
| `services/agent/{engine,ingress,routing}`, `services/mcp/.../profile.py` | modified, uncommitted |

The BigQuery work was therefore already complete and was re-verified rather than repeated.
Both counts and the id list above come from queries run in this session.

Two defects were left by the crash and are the only edits in this session to code that had
already been written:

1. `services/agent/ingress/http.py` called `encounter.current_is_outpatient()` at two call
   sites with **no import for `encounter`**. Every `/ask` and `/ask/stream` returned 500.
   This was the cause of 31 test failures, not a modelling problem.
2. `services/agent/engine/deterministic.py` imported `OUTPATIENT_ENCOUNTER_NOTICE` twice.

---

## 2. Code changed in this session

### Deletion of the id set

- `services/agent/routing/routing.py` — the frozenset is gone. The comment in its place
  states that the encounter kind is read from the row, and the two references to the old
  name in prose were rewritten.
- `evaluation/outpatient_oracle.py` — **new**. The five ids, frozen, with a docstring
  explaining that it is the cohort builder's decision kept as an answer key and that
  nothing in `services/` imports it.
- Importers repointed to the oracle: `evaluation/agent/build_census.py` (3 sites),
  `evaluation/inventory/cohort_provenance.py`, `evaluation/inventory/dev_followup_matrix.py`.
  `cohort_provenance.py` lost its now-unused `routing` import.
- `scripts/agent/build_hybrid_fixtures.py` imports the oracle so a regenerated seed still
  knows which records were built as visits.

### A third copy of the same ids, found and removed from the runtime path

`services/agent/presentation/domain.py` carried `NON_ADMISSION` — the same five ids — and
`is_matrix_eligible()` decided from it, so the canvas was still branching on `hadm_id`
after the rest of the runtime had stopped. The predicate now reads
`encounter.current_is_outpatient()`; `NON_ADMISSION` remains as the phenotype map's
record for the audit tooling, and a test asserts it agrees with the oracle.

The import is at module scope, not inside the function: `test_layer_separation` refuses
function-local `services.agent` imports as concealed cycles.

### Unchanged from the interrupted session, reviewed

- `services/mcp/dependencies/features/profile.py` — `decode_profile` states
  `encounter_type` first; `feature_order()` still excludes it, so it never reaches the
  booster. The MCP query is `SELECT *`, so no SQL changed.
- `services/agent/routing/encounter.py` — the predicate, the classify/unknown rules, and
  the request-scoped ContextVar.
- `services/agent/routing/admission_context.py` — `hydrate` sets the value
  unconditionally (including `None`); `render` drops `admission_type`,
  `discharge_location` and `index_los_days` on a visit and states the absence.
- `services/agent/engine/graph.py`, `deterministic.py` — the notice and the command gate
  read the context variable instead of an id.
- `services/agent/ingress/http.py` — `outpatient=` on both routes reads the same state
  (plus the missing import, above).

### Repo seed artifacts

`data/agent/demo_cohort.json`, `data/agent/demo_fixtures/demo_cohort.json` and the site's
`demo/data/demo_cohort.json` now carry `encounter_type` on all 87 records. The five
outpatient `summary` strings — which read `3-day stay · discharged to rehab` for a visit
with no stay — were rewritten. **The site file is edited but not committed**; it belongs
to the sibling repository and was outside the directive's deploy scope.

---

## 3. Tests

| Command | Result |
|---|---|
| `pytest tests -m "not live"` | 1495 passed, 1 skipped, 43 deselected |
| `pytest tests/evaluation/test_encounter_column.py -m live` | 1 passed |

New file `tests/evaluation/test_encounter_column.py` covers:

- classification from the row alone, with the record's id belonging to no cohort
  (`99999999`), positive and negative;
- absent or unrecognised value → `None`, and `None` keeps the old behaviour rather than
  becoming "inpatient";
- the rendered block for a visit carries no `length_of_stay_days`, no `admission_type`
  and no `discharge_location`, and says there is no stay; the inpatient and unknown cases
  keep every field they had;
- `hydrate` records the value and clears the previous turn's;
- **live**: all 87 served records classify from the column and agree with the oracle
  87/87, read through `get_admission_profile` — the pre-flight call itself.

New `tests/conftest.py` resets the encounter ContextVar around every test. Without it a
test could pass on the previous test's answer; production sets it every turn, tests do not.

Also updated: `test_admission_context`, `test_free_text_routing`, `test_layer_separation`
(the new module added to the pinned manifest), `test_manifest_encounter_contract`,
`test_outpatient_ehr_notice`, `test_outpatient_risk_gate`, `test_phenotype_coverage`,
`test_prompt_contract`, `test_tier1_cohort_sweep`. The three files that call the command
gate directly now set the encounter state explicitly, with a comment saying the column
itself is verified live elsewhere.

---

## 4. Deploy

| Item | Value |
|---|---|
| Commit | `281d180` |
| Push | `01ea5a3..281d180 main -> main` |
| Builds | `fbb6a406` (mcp, SUCCESS), `f7d220ed` (agent, SUCCESS), `65f12206` (harness, SUCCESS) |
| `mcp-server` | `mcp-server-00121-ql2`, 100% traffic |
| `agent` | `agent-00203-ffr`, 100% traffic |

A fourth build, `c71fd011` on the `deploy-on-push` trigger, was still running when this was
written and will redeploy the site.

---

## 5. Not done

1. **Gate 1 canary** (`--canary`, scenarios `dev_t1_01`, `dev_t2_03`, `dev_t7_01`) — not
   run. It needs `--expect-revision agent-00203-ffr`, and the harness needs
   `API_URL`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` and
   `PROJECT_ID` in the environment. The Langfuse keys are not in the repository `.env` and
   were not supplied.
2. **Gate 2** (26-scenario dev-suite run) and its scorecard — not run; it follows a 3/3
   canary by the directive's own ordering.
3. **Site seed commit** — `danielmherman/demo/data/demo_cohort.json` is modified in the
   working tree of the sibling repository, uncommitted.
