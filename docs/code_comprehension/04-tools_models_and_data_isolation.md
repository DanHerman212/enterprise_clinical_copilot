# Phase 4: Tools, ML Inference & Data Isolation Audit

**Status:** `COMPLETE`  
**Target Layers:** Layer 5 (`docs/architecture/05-tools.md`), Layer 6 (`docs/architecture/06-custom_ml_models.md`), & Layer 7 (`docs/architecture/07-data_and_indexes.md`)  
**Repositories in Scope:** `enterprise_clinical_copilot` (primary) & `danielmherman` (cohort alignment check)  
**Objective:** Audit and verify strict `hadm_id` cross-patient isolation (R1), deterministic degraded/error contracts across all 7 MCP tools, ML feature-order and threshold single-source-of-truth parity across `services/mcp/` and `mlops/`, and strict online-vs-offline separation across Layers 5, 6, and 7.

---

## 0. Governance Protocol for This Spoke

* **Step A — The Audit Pass (Read-Only — CURRENT STEP):**
  * Inspect the target files in Section 1 against `docs/architecture/05-tools.md`, `06-ml_inference.md`, and `07-data_and_storage.md`.
  * Record exact file paths, line numbers, findings, and any proposed surgical fixes in **Section 2 (Step A — Audit Findings)**.
  * **STOP.** Do NOT modify any application or test code during Step A. Await explicit Architect approval.
* **Step B — The Execution Pass (Write + Verify — LOCKED UNTIL STEP A APPROVED):**
  * After Architect approval of Step A, execute the approved changes, run all verification gates, and record the results in **Section 3 (Step B — Execution Record)**.

---

## 1. Step A Audit Scope & Questions to Answer

### 1.1 Layer 5 — MCP Tool Contracts, R1 `hadm_id` Isolation & Degraded Flags (`services/mcp/`)
* **Architectural Standard (`05-tools.md`):**
  * Every patient-scoped tool (`predict_readmission`, `rag_search`, `rag_search_sections`, `discharge_medications`, `get_lab_trends`, `get_admission_profile`) must enforce `hadm_id` isolation at the query/lookup boundary (never via post-retrieval filtering across patients).
  * `search_literature` must remain strictly patient-agnostic (no `hadm_id` parameter, and guarded against patient identifier leakage).
  * External/enrichment tools (`search_literature`) must set deterministic `degraded: True` flags on upstream timeout/rate-limit/error rather than silently returning an empty list indistinguishable from "0 articles matched".
* **Audit Tasks:**
  1. Trace all 7 tools registered in `services/mcp/server.py` to their underlying implementation functions and data stores.
  2. Verify how each patient-scoped tool enforces `hadm_id` scoping at the SQL/Parquet/Chroma/dict lookup boundary, citing exact file paths and line numbers.
  3. Verify error and `degraded` payload shapes across all 7 tools against `services/agent/contracts.py` (`validate_tool_payload` / tool schema validators).
  4. Identify any dead code, unused configuration constants, or schema drift inside `services/mcp/`.

### 1.2 Layer 6 — ML Inference Boundary & Feature Contract Parity (`services/prediction/` & `mlops/`)
* **Architectural Standard (`06-ml_inference.md`):**
  * Online inference (`services/prediction/`) must validate its input feature schema, enforce a single canonical feature ordering shared with training (`mlops/`), and return calibrated probabilities alongside signed SHAP attributions and the model's operating threshold.
  * Online serving must not import `mlops/` training modules at runtime.
* **Audit Tasks:**
  1. Inspect `services/prediction/` and compare its feature list, threshold handling, and response schema against `services/mcp/` (`predict_readmission`) and `mlops/`.
  2. Check whether feature names/orders or thresholds are duplicated across files without a parity test locking them together.
  3. Check for any dead helpers, stale model artifact paths, or missing input validations in `services/prediction/`.

### 1.3 Layer 7 — Data & Storage Hygiene & Cohort Alignment (`services/agent/served_admissions.py`, `services/mcp/`, & `danielmherman/demo/`)
* **Architectural Standard (`07-data_and_storage.md`):**
  * Online services read immutable, pre-built cohort artifacts and never mutate clinical records.
  * The served admission cohort (`served_admissions.py` in `enterprise_clinical_copilot` and `DemoPatient` fixture/seed data in `danielmherman`) must agree so an admission offered in the BFF `@` autocomplete is never rejected as unserved by the agent (or lacking features/notes in MCP).
* **Audit Tasks:**
  1. Trace where `served_admissions.py`, the RAG note store, the feature store, lab trends, and `danielmherman`'s `DemoPatient` seed/migration data load their `hadm_id` sets.
  2. Check whether an automated test verifies that the BFF demo cohort and the Agent/MCP served cohort match.
  3. Identify any Layer 7 storage or artifact-loading drift against `07-data_and_storage.md`.

---

## 2. Step A — Audit Findings & Proposed Surgical Plan

Read-only pass. No application or test code modified. Line numbers against the tree at `2bae659`.

### 2.0 Summary

| § | Question | Verdict |
| :--- | :--- | :--- |
| 1.1 | Is R1 `hadm_id` isolation enforced at every lookup boundary? | **Yes for six of seven tools, and enforced twice on the retrieval path.** `get_lab_trends` is the one patient-scoped tool that never applies the shared request check. |
| 1.1 | Are `degraded` flags deterministic and total? | **Yes.** Every upstream failure in `search_literature` has a named code and `degraded: True`; the "nothing matched" case is a different shape on purpose. |
| 1.1 | Do all seven payloads pass a contract validator? | **Five of seven.** `discharge_medications` validates on the tool side but has no `TOOL_CONTRACTS` entry, so the copy the model reads is unchecked. `get_admission_profile` has neither, but is pre-flight-only and read defensively downstream. |
| 1.2 | Is there a feature-contract single source of truth? | **Yes, and it is pinned by a test.** Order comes from the bundle manifest written by training; threshold comes from `threshold.json` with no fallback; neither is duplicated in `services/`. |
| 1.2 | Does online serving ever import training code? | **No.** `services/` has zero `mlops` imports. Nothing tests that, though. |
| 1.3 | Do the BFF cohort and the agent's served cohort agree? | **They agree exactly today — 87 = 87, zero drift in either direction — but no test keeps them agreeing.** |

---

### 2.1 Layer 5 — MCP tools, R1 isolation, degraded flags

#### 2.1.1 The seven tools, traced

`services/mcp/server.py:95-101` registers them; `services/mcp/tools/__init__.py` exports the same seven.

| Tool | Entry | Implementation | Store | `hadm_id` scoped at |
| :--- | :--- | :--- | :--- | :--- |
| `predict_readmission` | `tools/prediction.py:147` | `_predict` (`:73`) | BigQuery `readmission.hybrid_features` via `FeatureSource.fetch`, then Vertex endpoint via `predict_one` | `bigquery_source.py:35-38` — `WHERE hadm_id = @hid` |
| `rag_search` | `tools/retrieval.py` (tool fn) | `_search` (`:274`) | Vertex Vector Search index + discharge-note table | `retrieval.py:320` — `filter=[Namespace("hadm_id", [str(hadm_id)])]` **plus** `:224` re-check |
| `rag_search_sections` | `retrieval.py:628` | `_search_sections` (`:526`) | discharge-note table (re-parsed, deterministic chunker) | `:537` + `_fetch_note_row` `:245-247` || `discharge_medications` | `retrieval.py:683` | `_search_medication_sections` (`:652`) → `_search_section_chunks` (`:580`) | discharge-note table | `:592` |
| `search_literature` | `literature.py:358` | `_fetch` (`:297`) | NCBI E-utilities over HTTP | **no patient scope by design** |
| `get_lab_trends` | `labs.py:40` | `lab_trends` (`dependencies/labs/synthetic.py:358`) | the admission's feature row | `labs.py:58` `source.exists` only — **no `valid_hadm_id`** |
| `get_admission_profile` | `admission_profile.py:69` | `_profile` (`:38`) | the same feature row, read back as words (`fetch_profile:63`) | `:40` |

#### 2.1.2 R1 is enforced twice on the retrieval path, once everywhere else

The index is written with a per-datapoint allow-list (`services/mcp/retrieval/embed.py:54` — `{"namespace": "hadm_id", "allow": [str(hadm_id)]}`), and the reader filters on the same namespace (`retrieval.py:320`). Writer and reader share the constant (`embed.py:23` `RESTRICT_NAMESPACE`), so the two cannot drift by editing one of them.

That alone is the standard the layer document asks for — server-side restrict, never post-filtering. `retrieval.py` goes further: `_fetch_texts` (`:203-231`) re-checks every resolved row's `hadm_id` and refuses to serve on a mismatch, with the reason written into the code (`"index restrict defect; refusing to serve"`, `:231`). That is a genuine defence-in-depth against a rebuild that drops the namespace, and it is the only place in the service that assumes the index could be wrong.

#### 2.1.3 The one gap: `get_lab_trends` does not apply the shared request check

`valid_hadm_id` (`tools/_validation.py:11-17`) is the one definition of a well-formed admission id, and its own docstring names the trap it exists for — "bool is an int subclass, reject". Five of the six patient-scoped tools call it:

- `prediction.py:75`, `retrieval.py:283`, `:537`, `:592`, `admission_profile.py:40`
- `labs.py` — **absent**. The tool goes straight to `await asyncio.to_thread(source.exists, hadm_id)` (`:58`).

Two consequences, both observable:

1. `hadm_id=True` binds as `INT64 1`, finds nothing, and answers `unknown_patient` — a wrong code for a malformed request, and the exact case `_validation.py` documents.
2. A non-numeric id raises inside `BigQuerySource.exists` (`bigquery_source.py:25-33`, no `try`), outside any handler in the tool, so the failure surfaces as a transport-level tool error rather than the `bad_request` every sibling returns.

The read itself is still scoped — `exists` and `fetch` both use the parameterised `WHERE hadm_id = @hid` — so this is a contract-consistency defect, not an isolation one.

#### 2.1.4 `search_literature` is patient-agnostic, and its degraded flags are total

Signature (`literature.py:358-380`): `query`, `max_results`. No `hadm_id` parameter, and the schema description tells the model the terms must be de-identified. `tool_error` omits `hadm_id` when not supplied, and the docstring at `contracts.py:108-130` makes the absence the privacy boundary: a tool never handed an admission cannot put one on the wire.

| Condition | Code | `degraded` | Site |
| :--- | :--- | :--- | :--- |
| Timeout | `literature_timeout` | `True` | `:337-338` |
| HTTP 429 | `literature_rate_limited` | `True` | `:339-342` |
| Other HTTP error | `literature_failed` | `True` | `:339-342` |
| Unreadable body / XML | `literature_failed` | `True` | `:344`, `:349` |
| `NCBI_EMAIL` unset | `literature_unavailable` | `True`, `misconfiguration` | `:412-417` |
| Search ran, nothing matched | — (`_empty`, `:120`) | absent | `:328`, `:352` |

The last row is the point of the flag: an empty result and a search that never happened are different facts, and the payload says which. `contracts.py:234-235` type-checks `degraded` as a bool when present.

#### 2.1.5 Payload validation: five of seven, and the split matters

Two independent boundaries validate tool payloads, and they do not cover the same tools.

| Tool | Self-validates before returning | In `TOOL_CONTRACTS` (agent side) |
| :--- | :--- | :--- |
| `predict_readmission` | `:165` | yes |
| `rag_search` | `:703` | yes |
| `rag_search_sections` | `:641` | yes |
| `discharge_medications` | `:703` | **no** |
| `search_literature` | **no** | yes |
| `get_lab_trends` | `:108` | yes |
| `get_admission_profile` | **no** | **no** |

`TOOL_CONTRACTS` (`contracts.py:335-342`) holds five; `validate_tool_result` (`:344-352`) returns an unregistered payload *unchanged*, with a docstring that justifies the leniency as a partial-deploy accommodation. Two tools have been permanently absent rather than temporarily.

- **`discharge_medications` reaches the model unvalidated.** `validate_retrieval_result` exists and this tool calls it at `:703`, so the tool side is covered; what is missing is the agent-side registry entry, which is the boundary that sees the payload the model reads. One line fixes it.
- **`get_admission_profile` is unvalidated on both sides**, but it is pre-flight only (`admission_context.PREFLIGHT_TOOLS`, `admission_context.py:48`) and its consumer reads named keys through `_FIELDS` with defaults, so a malformed payload degrades the manifest rather than crashing the turn. Lower severity, still a hole.

#### 2.1.6 Dead code and config drift inside `services/mcp/`

- **`config.py:336` `API_ENDPOINT`** — defined, referenced nowhere in `.py`, `.yaml`, `.sh` or `.md`.
- No module in `services/mcp/` is unimported; the tool package, `retrieval/`, `pipelines/` and both dependency packages all have callers.
- The `TOOL_CONTRACTS` shortfall above is the substantive drift; it is a completeness defect rather than dead weight.

#### 2.1.7 Proposed surgical plan

| # | Change | Files |
| :--- | :--- | :--- |
| 1 | `get_lab_trends` applies `valid_hadm_id` first and returns `bad_request`, matching its five siblings | `tools/labs.py` (~4 lines) |
| 2 | Register `discharge_medications` → `validate_retrieval_result` in `TOOL_CONTRACTS` | `contracts.py:335` (1 line) |
| 3 | Delete `API_ENDPOINT` | `config.py:336` |
| 4 | Add tests: a malformed id (`True`, `"90000015"`, `-1`) to each of the six patient-scoped tools returns `bad_request`; every name in `tools.__all__` has a `TOOL_CONTRACTS` entry | `tests/agent/test_tool_contracts.py`, new cases |
| 5 | **DECISION:** also give `get_admission_profile` a contract (new `validate_admission_profile_result`, ~25 lines) or record it as a documented pre-flight exemption | `contracts.py`, `tests/agent/test_tool_contracts.py` |
| 6 | **DECISION:** whether `search_literature` should self-validate before return, or rely on the agent boundary. Today it relies on the agent; the asymmetry with its siblings is not documented anywhere | `tools/literature.py:421` |

Item 4's second half is the one that keeps drift from recurring: it is exhaustive over `tools.__all__`, so a new tool cannot be added without deciding its contract.

---

### 2.2 Layer 6 — ML inference and feature-contract parity

#### 2.2.1 `services/prediction/` does not exist

The layer document's target directory is not in the repository. The prediction path is:

| Concern | Actual location |
| :--- | :--- |
| Tool entry, payload assembly | `services/mcp/tools/prediction.py` (165 lines) |
| Feature read + column restriction | `services/mcp/dependencies/features/bigquery_source.py` |
| Feature order (which bundle to read) | `services/mcp/dependencies/features/manifest.py:81` |
| Canonical order (training-side definition) | `mlops/data/encoding.py:169-179` (`feature_order`), `:182` (`groups`) |
| Endpoint handle | `services/mcp/dependencies/model_endpoint.py:29` (`predict_one`) |
| Serving container | `mlops/serving/cpr/predictor.py` |

The audit below is run against that path.

#### 2.2.2 Feature order has one definition and one artefact

The chain, end to end:

1. `mlops/data/encoding.py:169` defines the order from `NUMERIC_FEATURES + BINARY_FEATURES + one-hot blocks`.
2. Training writes it into the serving bundle's `manifest.json` (`mlops/training/training_pipeline.py:64-66` imports it as `SELECTED_FEATURES`).
3. Serving reads it back out of the deployed bundle (`mlops/serving/cpr/predictor.py:82`).
4. The MCP tool reads it from the **same** bundle (`dependencies/features/manifest.py:81`), via `_discover`, whose predicate for "the bundle to read" is deliberately the same predicate deployment uses (`FINAL_MODEL_PREFIX` + `stage` label).
5. `prediction.py:77` holds it for the turn and uses it twice: to build the instance (`to_instance`, `base.py:58`) and to detect an incomplete row (`prediction.py:96-106`).

No list is written out a second time in `services/`. The one historical exception is pinned by a test whose docstring records it: `tests/agent/test_feature_contract.py` asserts the three synthetic-data generators *derive* their column list from `mlops.data.encoding.feature_order` rather than restating it, and that the feature source distinguishes an absent column from a null value.

#### 2.2.3 The operating threshold has one source, and it fails loud

`mlops/serving/cpr/predictor.py:86-96` refuses to start if `threshold.json` is missing from the bundle, with the reason stated (ECC-68 — never fall back to 0.5 on a recall-weighted F2 optimum). The decision is computed at `:198` and returned in the response. `prediction.py:138` copies `pred["threshold"]` into the tool payload rather than holding a number of its own; the agent renders that value.

So: one artefact, no fallback, no duplicate. Nothing in `services/` names a threshold.

#### 2.2.4 Online serving does not import training code

`grep` for `import mlops` / `from mlops` across `services/`: zero hits. The dependency runs the other way only — `tests/agent/test_feature_contract.py` imports `mlops.data.encoding` to compare against it, which is a test, not runtime.

Nothing pins it. The standard is currently true by discipline; a one-line test would make it a rule.

#### 2.2.5 Input validation on the predict path

Four checks, all present: `valid_hadm_id` (`:75`); unknown admission → `unknown_patient` (`:80-86`); incomplete row → `incomplete_features` with the missing column names in the server-side detail (`:96-106`); response outside contract → `invalid_tool_response` (`:165`). `fetch` restricts the row to `feature_order()` (`bigquery_source.py:56-62`), so label and bookkeeping columns cannot reach the model input.

No dead helpers found in `prediction.py`; `_model_identity` (`:57`) is used for both the payload and the log line.

#### 2.2.6 Proposed surgical plan

Layer 6 is conformant. Two additions, both cheap and both about keeping it that way:

| # | Change | Files |
| :--- | :--- | :--- |
| 7 | A test asserting no module under `services/` imports `mlops` — walks the `services` tree with `ast`, fails with the file and import that broke the separation | new `tests/agent/test_layer_separation.py` |
| 8 | A test asserting the threshold reaches the payload from the response rather than a literal — `predict_readmission`'s payload `threshold` equals the endpoint's `threshold` for a stubbed prediction that deliberately disagrees with 0.5 | `tests/agent/test_predict_identity.py` or `test_tool_contracts.py` |

---

### 2.3 Layer 7 — Data, storage and cohort alignment

#### 2.3.1 The cohort chain: five places that must agree

| Place | What it holds | Source |
| :--- | :--- | :--- |
| `services/agent/served_admissions.py:26-40` | `SERVED_HADM_IDS`, 87 ids | a literal, regenerated from the evaluation census; the module docstring carries the command |
| `tests/agent/test_eval_contract.py:406-419` | pins the literal to `evaluation/agent/results/cases_82.json` + `cases_outpatient.json`, asserts 87 and the two excluded orphans | — |
| Vector Search index | one datapoint per note chunk, `hadm_id` namespace | `services/mcp/retrieval/embed.py:54` |
| `readmission.hybrid_features` | one row per admission, the model's 49 features | written by `mlops`/`scripts/agent/generate_hybrid_features_v2.py` |
| `danielmherman` `DemoPatient` | the BFF's cohort, loaded from `demo/data/demo_cohort.json` by `seed_demo_patients.py` (`DEFAULT_PATH`, `:24`) | produced by the harness (`agent-harness/scripts/seed_demo_cohort.py`) |

The agent refuses an unserved admission before any work runs (`turn.py`, `unserved_admission`), so a site patient missing from the literal is a request the BFF offers and the agent answers with "Admission not found".

#### 2.3.2 Measured today: exact agreement

Comparing `demo/data/demo_cohort.json` against `SERVED_HADM_IDS`: **87 site ids, 87 served ids, `site − served` empty, `served − site` empty.** The two cohorts are the same set right now.

#### 2.3.3 What keeps them agreeing: nothing

- The agent side is pinned twice (the literal against the eval census, and the sweep against the literal).
- The site side is pinned to nothing. `grep` for `served_admissions` or `SERVED_HADM_IDS` across `danielmherman/`: zero hits. There is no test asserting a seeded `DemoPatient` is servable, and no test that the seed artifact and the literal were generated from the same census.

This is the same shape as the `CORE_TOOLS` drift closed in Phase 2: two repositories, one invariant, and no check that crosses them. The agreement is currently a coincidence of generation order, not a property.

#### 2.3.4 Storage hygiene

- **The site writes only its own tables.** `DemoPatient`, `Conversation`, `Turn`, `DemoQuota` — no clinical record is mutated anywhere in `demo/`.
- **The agent and MCP write nothing.** Every tool is a read; `predict_readmission` is the only one whose result is *stored*, and it is stored in the site's `Turn.tool_calls`, not in a clinical store.
- **Cohort and note artefacts are pre-built and read-only at runtime** — the feature table, the index, the seed JSON, and `SERVED_HADM_IDS` itself.
- `evaluation/inventory/cohort_provenance.py` is a read-only inventory over the same four patient-scoped tools plus the phenotype map; it decides nothing and mutates nothing.

#### 2.3.5 Proposed surgical plan

| # | Change | Files |
| :--- | :--- | :--- |
| 9 | A cohort-parity test: the site's seed ids are a **subset** of `SERVED_HADM_IDS`, read from the agent's literal with `ast` and guarded by `skipUnless(sibling.exists())` — the Phase 2 pattern, so an isolated CI runner still runs its own suite | new `danielmherman/demo/test_cohort_parity.py` |
| 10 | The same assertion from the agent side, against the seed file, so the check runs in whichever repository is checked out with a sibling | `tests/agent/test_eval_contract.py` (new case, same guard) |
| 11 | **DECISION:** subset or equality. Subset is the invariant that matters — a site patient the agent refuses is the bug. Equality would also fail when the served set legitimately outgrows the demo cohort, which is the expected direction of drift | — |

---

### 2.4 Doc-vs-code mismatches recorded during the audit

| Spoke says | Actual |
| :--- | :--- |
| `services/prediction/` (§1.2, twice) | does not exist; the path is `services/mcp/tools/prediction.py`, `services/mcp/dependencies/{features,model_endpoint}.py`, `mlops/serving/` |
| `docs/architecture/06-ml_inference.md`, `07-data_and_storage.md` | actual filenames are `06-custom_ml_models.md`, `07-data_and_indexes.md` |
| "`validate_tool_payload` / tool schema validators" (§1.1.3) | the real names are `validate_tool_result` and `TOOL_CONTRACTS` |
| "all 7 MCP tools" for the degraded-flag contract (§1.1) | `degraded` is a literature-only concept; the other six signal failure as `{"error": code, "message": …}` and have no partial-result state to degrade |

---

### 2.5 Proposed Step B order

1. Items 1–4 (Layer 5): lab-id validation, contract registry entry, `API_ENDPOINT`, the two exhaustive tests. Suite green.
2. Items 7–8 (Layer 6): the separation and threshold tests. No production code touched.
3. Items 9–10 (Layer 7): the guarded cohort-parity tests in both repositories. No production code touched.
4. Item 5–6 depending on the decisions below.
5. §3 record and tracker update.

Items 1–3 change production code (four lines in total); 4 and 7–10 are tests. Nothing in this phase touches the agent's request path, the prompt, or the wire contract.

---

### 2.6 Decisions for the Architect

| # | Decision | Recommendation |
| :--- | :--- | :--- |
| 1 | `get_admission_profile`: add a payload contract, or document a pre-flight exemption | **Add the contract.** It is 25 lines and it closes the last tool with no validation at either boundary; the "pre-flight, therefore exempt" argument holds only as long as `_FIELDS` stays defensive. |
| 2 | `search_literature` self-validating before return, or agent-boundary only | **Agent-boundary only, documented.** Adding a second validation of the same payload on the same hop buys nothing; the comment should say the split is deliberate so a future reader does not "fix" the asymmetry. |
| 3 | Cohort parity: subset or equality | **Subset.** Equality fails in the expected direction of drift (served grows, demo cohort does not). |
| 4 | Cohort test placement: both repositories (guarded) or site only | **Both**, guarded. The site test catches the bug where it is created; the agent test catches it when the literal is regenerated. |


## 3. Step B — Execution Record

All four §2.6 recommendations approved as proposed; items 1–10 executed in the §2.5
order. Production code changed in four files (three lines in `labs.py`, one registry
entry plus one new validator, one dead constant deleted); the rest are tests.

### 3.1 Executed Changes

| Item | File | Change | Kind |
| :--- | :--- | :--- | :--- |
| 1 | `services/mcp/tools/labs.py` | `valid_hadm_id` applied first, before the feature source is acquired, returning `bad_request` | prod, +11 |
| 2 | `services/mcp/contracts.py` | `discharge_medications` → `validate_retrieval_result` in `TOOL_CONTRACTS` | prod, +1 |
| 3 | `services/mcp/config.py` | `API_ENDPOINT` deleted (referenced nowhere) | prod, −1 |
| 5 | `services/mcp/contracts.py` | `validate_admission_profile_result` added (~30 lines) and registered | prod, +32 |
| 6 | `services/mcp/tools/literature.py` | the single-boundary choice documented at the return site | prod, comment |
| 4 | `tests/agent/test_tool_boundary_parity.py` | **New** (23 tests): malformed id × 6 tools × 3 shapes → `bad_request`; registry covers `tools.__all__`; the lenient path stays narrow; the profile contract accepts errors and rejects non-strings | test |
| 7 | `tests/agent/test_layer_separation.py` | **New** (3 tests): no module under `services/` imports `mlops`, walked with `ast`; plus a vacuity guard and a second look at `model_runtime.py` | test |
| 8 | `tests/agent/test_predict_identity.py` | The payload's `threshold` equals the endpoint's `0.077`, not 0.5 | test, +22 |
| 10 | `tests/agent/test_eval_contract.py` | The site's seed ids are a subset of `SERVED_HADM_IDS`, guarded on the sibling | test, +35 |
| 9 | `danielmherman/demo/test_cohort_parity.py` | **New** (3 tests): the seed artifact names patients, every one is served, and the two unscoreable orphans are not seeded | test |

Suite arithmetic: agent 1,349 → **1,377** (+28); site 243 → **246** (+3).

### 3.2 Two deviations from the approved text

1. **`validate_admission_profile_result` lives in `services/mcp/contracts.py`, not
   `services/agent/contracts.py`.** §2.6 item 1 said the agent's contracts module. That
   module holds the *request and response* contract; the four tool-payload validators
   and `TOOL_CONTRACTS` are in `services/mcp/contracts.py`, and the agent imports them
   from there (`services/agent/mcp_client.py:35`). Putting the fifth validator in the
   agent would have separated it from the registry that dispatches to it — the registry
   is keyed to validators that must share a module with it, or "a tool and its contract
   cannot ship apart" stops being true. Recorded rather than silently obeyed.
2. **Item 4's tests are a new file, not additions to `tests/agent/test_tool_contracts.py`.**
   That file is about the shape of individual payloads; the new tests are about *coverage*
   — every tool, every malformed id. Keeping them apart means the exhaustive-over-`__all__`
   property is readable in one place instead of buried among per-payload cases.

### 3.3 What the audit got wrong, and what the run found

- **The labs check had to go before `get_feature_source()`.** Putting it where the other
  tools put theirs — after the source is in hand — would build a BigQuery client for a
  request that is already known to be malformed. Moved above the acquisition.
- **`SERVED_HADM_IDS` is an annotated assignment** (`frozenset[int] = frozenset({...})`,
  `ast.AnnAssign`, not `ast.Assign`). The site-side reader in item 9 failed on the first
  run for exactly that reason; the reader now handles both forms.
- Nothing else in the audit was contradicted. The `TOOL_CONTRACTS` shortfall, the dead
  constant, the unpinned cohort and the unpinned `services → mlops` separation were all
  as described.

### 3.4 Behaviour changes

1. **A malformed admission id is refused identically by all six patient-scoped tools.**
   `get_lab_trends` no longer answers `unknown_patient` for `hadm_id=True`, and no longer
   raises inside the BigQuery client for a string id.
2. **`discharge_medications` and `get_admission_profile` payloads are validated at the
   agent boundary.** A malformed one now raises `ToolContractError` where it previously
   reached the prompt or the manifest unchecked. No well-formed payload changes.
3. Nothing else. No prompt, no routing, no wire field, no model or threshold.

### 3.5 Verification Gates

| Gate | Result |
| :--- | :--- |
| `pytest tests` (`enterprise_clinical_copilot`) | **1377 passed, 1 skipped** (211s) |
| `python manage.py test` (`danielmherman`, `ENVIRONMENT=development`) | **246 tests OK** |
| `python manage.py check --deploy` (`ENVIRONMENT=production`) | **System check identified no issues (2 silenced)** |

### 3.6 Deliberately not done

- **`search_literature` still returns unvalidated**, per Decision #2, with the reason now
  written at the return site so a future reader does not "fix" the asymmetry without
  removing the registry entry in the same change.
- **No layer-document edits.** `05-tools.md`, `06-custom_ml_models.md` and
  `07-data_and_indexes.md` describe responsibilities and name no modules; the mismatches
  in §2.4 are in this Spoke's own §1, not in the layer documents themselves.
- **The cohort test reads the seed artifact, not the `DemoPatient` table.** The artifact
  is the only path that populates the table (`seed_demo_patients.py`), so it is the
  honest place to compare — a DB assertion would additionally require the seeder to run
  inside the suite, and would then be testing the seeder rather than the cohort.
