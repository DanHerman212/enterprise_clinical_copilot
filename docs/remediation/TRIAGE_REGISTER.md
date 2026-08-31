# Consolidated Triage Register — both repos

One row per finding, merged from `enterprise_clinical_copilot/docs/REVIEW_BACKLOG.md`
(ECC-xx) and `danielmherman/docs/REVIEW_BACKLOG.md` (S*-xx). Full finding text +
remediation detail stays in those files and in `docs/adversarial_code_review/` —
this register is the working triage view. Sorted severity → cluster → ID.

**Snapshot (2026-08-31):** 124 findings — **4 Critical**, **41 Major** open, 65 Minor
open, 14 resolved. Fix clusters are defined in `REMEDIATION_ROADMAP.md`.

Clusters: **A** config fail-closed · **B** agent answer integrity · **C** spend/quota/DoS ·
**D** chunker determinism · **E** fail-loud build pipeline · **F** isolation & retrieval (R1) ·
**G** info disclosure · **H** site auth hardening · **I** deps & supply chain ·
**J** XSS/sanitization/CSP · **K** public content & contact form · **L** MLOps gate integrity ·
**M** deploy/infra ops · **N** front-end correctness · **Q** SQL parameterization · **Z** misc

## Critical (4 open, 1 resolved)

| ID | Repo | Cluster | Category | Location | One-liner | Status |
|---|---|---|---|---|---|---|
| ECC-46 | ECC+site | A | ops/security | cloudbuild*.yaml, settings.py | Runtime env unversioned; `ENVIRONMENT` fails open to development — recreating a service yields a public debug site | **resolved 2026-08-31** |
| ECC-04 | ECC | B | correctness | agent/guardrail.py, graph.py | No guard that a stated risk number matches (or even came from) a predict call — fabricated risk served with 200 | open |
| ECC-20 | ECC | C | ops/spend | mcp_server/tools/rag_search.py | Anchored retry recurses with no depth guard — thousands of billed embed/search/BQ calls from one query | open |
| ECC-32 | ECC | D | correctness | rag_search.py vs chunk pipeline | Index built with `pack_to=700`, serving re-chunks with `None` — silent whole-note fallback leaks metadata into citations | open |
| ECC-64 | ECC | L | correctness | pipelines/components/load_data.py | No train/val/test patient-disjointness assertion — every gate potentially leakage-contaminated | open |

## Major (46)

| ID | Repo | Cluster | Category | Location | One-liner | Status |
|---|---|---|---|---|---|---|
| ECC-15 | ECC | A | security/ops | config.py + deploy scripts | Real project IDs/URLs committed as fail-open defaults — local runs silently hit prod | **resolved 2026-08-31** |
| S1-01 | site | A | security/ops | settings.py, cloudbuild.yaml | `ENVIRONMENT` defaults to development and deploy never sets it — prod can run `DEBUG=True` | **resolved 2026-08-31** |
| S1-02 | site | A | ops | settings.py, demo/views.py | Fixture mode defaults **true** — a prod deploy omitting the var silently serves fixtures as live | **resolved 2026-08-31** |
| S9-01 | site | A | ops | settings.py | No prod fail-fast on missing env (Cloud SQL host, ALLOWED_HOSTS, CSRF origins) — opaque runtime failures | **resolved 2026-08-31** |
| ECC-05 | ECC | B | security | agent/prompts.py, graph.py | No data/instruction separation — question + note passages reach the model with zero injection mitigation | open |
| ECC-10 | ECC | B | correctness | agent/graph.py, mcp_client.py | MCP input schemas never reach the model (no `args_schema`) — model guesses arg names; `kwargs` heuristic fragile | open |
| ECC-11 | ECC | B | correctness | agent/guardrail.py | Dose removal via `str.replace` — removing an unsupported "5 mg" corrupts a supported "2.5 mg" | open |
| ECC-12 | ECC | B | correctness | agent/graph.py, server.py | `final_text` falls back to stale earlier AI text when the final turn is empty — served with 200 | open |
| ECC-02 | ECC | C | ops/spend | agent/graph.py, server.py | Unbounded per-request spend: no recursion limit, no tool cap, 180s tool timeout exceeds upstream 120s deadline | open |
| ECC-09 | ECC | C | ops | agent/mcp_client.py | Synchronous ID-token fetch inside async path blocks the entire event loop on every request | open |
| ECC-25 | ECC | C | security | mcp_server/server.py | MCP HTTP transport has no app-layer auth/authorization — one `--allow-unauthenticated` exposes data + spend | open |
| S1-09 | site | C | security | demo/views.py, models.py | Refund loop defeats the spend cap: guaranteed-tool-error requests always refunded; `hadm_id` never validated | open |
| ECC-41 | ECC | D | correctness | rag/sections.py | Any bare single-word alias line becomes a section boundary — truncation + corpus-wide chunk-id shifts | open |
| ECC-23 | ECC | D | correctness | rag_search.py | Unknown section id yields a null-text citation; `_KNOWN_SECTIONS` is a stale 13-name copy of a ~29-name vocab | open |
| ECC-38 | ECC | E | correctness | embed_chunks.py (script + component) | `zip()` silently truncates short embed responses — chunks vanish from the ingest with no error | open |
| ECC-39 | ECC | E | ops | scripts/embed_chunks.py | Inverted exit codes: failed batches exit 0 (and still upload); a clean run exits nonzero | open |
| ECC-40 | ECC | E | ops | scripts/deploy_index.py | Count verification asserts hardcoded `555770` — wrong for every rebuild | open |
| ECC-42 | ECC | E | ops | scripts/prune_rag_datapoints.py | Index-removal failure ignored; BQ rows deleted anyway — stale vectors left with no source text | open |
| ECC-19 | ECC | F | security | rag_search.py `_fetch_texts` | R1 isolation rests on the index restrict alone — no hadm_id re-check when resolving passage text | open |
| ECC-24 | ECC | F | correctness | rag_search.py | Retry replaces a non-empty original result with an empty retried one — real hits discarded | open |
| ECC-06 | ECC | G | ops/security | agent/server.py | 502 body + `/health` leak internal topology (MCP_URL, project, IAM detail) to any direct caller | open |
| S1-03 | site | G | security | demo/views.py | 502 `detail` returns `str(exc)` to the browser — private agent host leaks verbatim | open |
| S1-05 | site | H | security/ops | /accounts/login/, /admin/ | No throttling/lockout on login or admin; admin at the default path | open |
| S6-13 | site | H | security | content/views.py | Inactive projects publicly served — detail/section views never filter `is_active` | open |
| S1-16 | site | I | security | requirements.txt | 72 known CVEs across Django/pillow/sqlparse/bleach pins | **resolved 2026-08-31** |
| ECC-55 | ECC | I | ops | serving/cpr/Dockerfile, deploy_cpr.py | Content-hash image tag is false — mutable base + unpinned SDK mean same tag, different image | open |
| ECC-61 | ECC | I | security/ops | predictor.py load, cpr cloudbuild | No bundle checksum, `:latest` provenance, `_TAG` defaults to latest — mutable supply chain end to end | open |
| S6-01 | site | J | security | content templates, models | Stored XSS: `\|safe` rendering with no server-side sanitization anywhere | open |
| S6-06 | site | J | security | content/sectioning.py | Sectioning round-trip un-escapes entities — author-escaped text becomes live markup | open |
| S6-02 | site | K | ops | ContactView.post | Public contact form: no rate limiting, unbounded TextField — spam/DB bloat | open |
| S6-03 | site | K | correctness | ContactView.post | No server-side validation — long inputs raise `DataError` → 500 on prod Postgres | open |
| S6-07 | site | K | correctness | content/sectioning.py | Drill-down silently drops all content before the first `<h2>` | open |
| ECC-60 | ECC | L | security | cpr/predictor.py | Zero input validation: typo'd/missing keys become NaN — confident, silently wrong clinical probability | open |
| ECC-65 | ECC | L | correctness | training_pipeline.py, gates | Gate baseline is a runtime parameter — submit `0.0` and both performance gates are neutralized | open |
| ECC-66 | ECC | L | correctness | evaluate_test.py, optuna_hpo.py | Val→test stability check is warn-only with a biased reference — an overfit model still registers | open |
| ECC-36 | ECC | M | security | deploy_index.py, deploy_rag_endpoint.py | Vector Search endpoint is public (no PSC/VPC) — DUA/data-governance exposure with ECC-53 | open |
| ECC-47 | ECC | M | ops | mlops/scripts/deploy_cpr.py | Undeploys ALL models before deploying the new one — guaranteed outage window, no rollback | open |
| ECC-53 | ECC | M | security/ops | deploy_rag_endpoint.py defaults | Defaults deploy the real 555k MIMIC index (~$270/mo) to a public endpoint | open |
| ECC-54 | ECC | M | ops | deploy_endpoint.py, teardown.py | Default path creates an always-billing FeatureOnlineStore that teardown never removes | open |
| ECC-56 | ECC | M | ops | site cloudbuild.yaml | Migrations + destructive `--prune` run before deploy — failed deploy leaves old revision on new schema | open |
| S7-06 | site | N | correctness | demo_flow.js | Concurrent asks overwrite the last turn — answer A attributed to question B | open |
| S7-07 | site | N | correctness | demo_flow.js | Patient-switch race paints the previous patient's thread under the new patient (R1 display bleed) | open |
| S7-08 | site | N | correctness | demo_flow.js, demo_splitpane.js | Unguarded payload derefs outside try/catch — stuck spinner on any failed tool | open |
| S7-09 | site | N | correctness | demo_splitpane.js | Custom demo maps `^[n]`→`passages[n-1]` with no intent resolution — wrong source card (A2UI path already fixed) | open |
| ECC-37 | ECC | Q | security | pipelines/components/chunk_notes.py + scripts | Pipeline params f-string'd into SQL (`split_name`, table refs) — CWE-89 surface | open |
| ECC-63 | ECC | Q | security | pipelines/components/load_data.py | All SQL inputs interpolated with no query parameters — exfiltration possible via crafted split name | open |

## Minor (65 open) + resolved

| ID | Repo | Cluster | Category | Location | One-liner | Status |
|---|---|---|---|---|---|---|
| ECC-48 | ECC | A | security | register_serving_model.py | Real bundle URI + project hardcoded as defaults | **resolved 2026-08-31** |
| ECC-51 | ECC | A | security | settings.py (get_secret) | Secrets pinned to `latest`, no rotation procedure; rotation takes effect only on redeploy | open |
| ECC-57 | ECC | A | correctness | deploy/teardown scripts | `PROJECT_ID` honored by some scripts, hardcoded in others — teardown can target the wrong project | **resolved 2026-08-31** |
| ECC-72 | ECC | A | ops | register_model.py, cpr cloudbuild, rag pipeline | Project ID hardcoded in 3 more files outside `src.config` | **resolved 2026-08-31** |
| S1-10 | site | A | security | settings.py | Real private agent URL committed as the env default | **resolved 2026-08-31** |
| S1-18 | site | A | ops | manage.py check --deploy | 6 warnings, all corroborating S1-01/S1-12 — add as CI gate | open |
| S9-02 | site | A | ops | config | No documented required-env manifest (`.env.example`) | **resolved 2026-08-31** |
| S9-03 | site | A | correctness | settings.py | ALLOWED_HOSTS/CSRF origins split without strip/filter — whitespace entries never match | **resolved 2026-08-31** |
| ECC-13 | ECC | B | correctness | agent/graph.py | Tool results serialized as Python repr (not JSON) into ToolMessage | open |
| ECC-14 | ECC | B | correctness | agent/guardrail.py | Citation check is range-only + advisory — out-of-range markers served intact | open |
| ECC-35 | ECC | B | correctness | guardrail.py, chunking.py | Redaction guard covers only age — all other `___` fields rely on the prompt alone | open |
| ECC-07 | ECC | C | security | agent/server.py | No app-level auth on `/ask` — relies wholly on Cloud Run IAM | open |
| S1-07 | site | C | correctness | demo/models.py | UTC-midnight quota rollover; refund not tied to the period it debited | open |
| S1-08 | site | C | ops | deployment_strategy.md | No global daily budget / kill switch (per-user quota only) | open |
| S1-11 | site | C | correctness | demo/views.py, agent_client.py | Unvalidated agent response shape — 500 after credit consumed, refund path skipped | open |
| S1-15 | site | C | architecture | agent_client.py, Dockerfile | Blocking 120s `requests.post` in sync views can exhaust the thread pool and stall the whole site | open |
| ECC-26 | ECC | D | correctness | rag_search.py | Chunk-id miss silently falls back to the entire note; `or` also swallows legitimately-empty chunks | open |
| ECC-33 | ECC | D | maintainability | build_chunks.py, rag_search.py, sections.py | Section vocabulary duplicated in three hand-maintained lists | open |
| ECC-34 | ECC | D | correctness | rag/chunking.py | Fixed-width fallback cuts mid-word | open |
| ECC-43 | ECC | D | correctness | rag_search.py (sections) | `rag_search_sections` cites datapoint ids that don't exist in the index, with a fake `score: 1.0` | open |
| ECC-44 | ECC | D | correctness | rag/chunking.py (`_pack`) | Span merging re-includes filtered redaction-only pieces in packed chunks | open |
| S7-02 | site | D | maintainability | demo_flow.js | Client `SECTION_ALIASES` is a 4th hand-maintained copy of the section vocabulary | open |
| ECC-45 | ECC | E | ops | rag/notes.py | Chunks cache has no manifest/count check; notes verification only fires on full exhaustion | open |
| ECC-70 | ECC | E | correctness | rag_ingest_pipeline.py | KFP cache key ignores table contents — re-submission silently reuses stale chunks/embeddings | open |
| ECC-22 | ECC | F | correctness | predict.py, rag_search.py | No demo-cohort restriction server-side — any hadm_id present in the tables works | open |
| ECC-27 | ECC | F | correctness | rag_search.py | `LIMIT 1` without `ORDER BY` — nondeterministic note selection for multi-note admissions | open |
| ECC-28 | ECC | F | correctness | predict.py, rag_search.py | Inconsistent `hadm_id` validation across the three tools (bool passes isinstance int) | open |
| ECC-30 | ECC | F | correctness | rag_search.py | Retry trigger reduces to "not rank 1" — broader and costlier than the intended "absent from top-k" | open |
| ECC-08 | ECC | G | architecture | agent/server.py | Full `tool_calls` payloads forwarded to the browser in live mode | open |
| ECC-21 | ECC | G | security | mcp_server/tools/* | Tool errors embed raw exception text that reaches the model + Langfuse | open |
| S1-04 | site | H | security | settings.py | Default 2-week session cookie; survives browser close | open |
| S1-06 | site | H | ops | urls.py | `auth.urls` exposes password-reset routes with no templates — those paths 500 | open |
| S1-12 | site | H | security | settings.py | No HSTS | open |
| S1-14 | site | H | ops | Dockerfile | Container runs as root — no `USER` directive | open |
| ECC-16 | ECC | I | ops | agent/Dockerfile | Mutable base tag; no hash-pinned requirements | open |
| ECC-31 | ECC | I | security | mlops components | `joblib.load` with no artifact integrity verification (CWE-502) | open |
| ECC-59 | ECC | I | correctness | register_serving_model.py | Models registered against a mutable `:latest` serving container | open |
| ECC-69 | ECC | I | ops | _image.py, components | Unpinned xgboost/optuna/sklearn/pandas — irreproducible runs; CPR parity unenforced | open |
| S6-09 | site | I | security | content/base.html | CDN scripts without SRI; Mermaid on floating `@10` | open |
| S9-05 | site | I | security/ops | requirements.txt | Dead `django-ckeditor` (CKEditor 4, known CVEs) + stale bleach in prod image; Django pinned to bare 6.0 | **resolved 2026-08-31** |
| S1-13 | site | J | security | settings.py | CKEditor `htmlSupport` allows everything; public upload route; `publicRead` default ACL | open |
| S6-10 | site | J | ops | settings.py | No Content-Security-Policy anywhere (the backstop for `\|safe` + CDN scripts) | open |
| S6-11 | site | J | security | content_extras.py | `first_sentence` wrongly declares `is_safe=True` | open |
| S7-01 | site | J | security | demo_flow.js | `showEmpty` innerHTML sink — safe today, latent XSS for any future dynamic caller | open |
| S7-03 | site | J | ops | demo JS | No CSP backstop for the demo surface (shared with S6-10) | open |
| S7-10 | site | J | security | a2ui vendor + demo_a2ui.js | DOMPurify defaults allow `<a>`/`<img>` — phishing/beacon channel inside the clinical canvas | open |
| S6-04 | site | K | correctness | content/models.py | Duplicate title → slug collision → 500 on admin save | open |
| S6-08 | site | K | security | content/urls.py | Section slugged `preview` shadowed by the staff preview route | open |
| S6-12 | site | K | correctness | ContactView.post | Error path loses user input, returns 200, accepts whitespace-only values | open |
| ECC-67 | ECC | L | correctness | calibrate_threshold.py, train_final.py | Threshold selected from CV OOF but applied to a different refit model — off its F-beta optimum | open |
| ECC-68 | ECC | L | correctness | cpr/predictor.py | Missing `threshold.json` silently falls back to 0.5 — flips recall-weighted decisions | open |
| ECC-71 | ECC | L | architecture | training_pipeline.py, register_model.py | Register not gated on shap/fairness; no `parent_model` lineage; gate metrics never persisted | open |
| ECC-73 | ECC | L | correctness | cpr/predictor.py | SHAP contributions are log-odds but the response/comment imply probability units | open |
| ECC-49 | ECC | M | ops | scripts/teardown.py | Prefix-matched teardown could delete future non-demo endpoints | open |
| ECC-58 | ECC | M | security | all cloudbuild files | No dedicated build `serviceAccount` — any build runs as an identity that can deploy prod | open |
| S7-04 | site | N | robustness | demo_splitpane.js, demo_flow.js | Non-numeric probability renders `NaN%` and a wrong band | open |
| S7-05 | site | N | correctness | demo_flow.js | Markdown regexes run on escaped text — entities display literally in code spans | open |
| S7-11 | site | N | correctness | demo_flow.js | Mixed/reversed citation ranges silently disappear | open |
| S7-12 | site | N | correctness | demo_flow.js, a2ui_canvas.py | Section start-match not line-anchored — mid-sentence aliases truncate extraction (client + server copies) | open |
| S7-13 | site | N | ops | templates | Cache-bust version skew — custom demo runs `demo_flow` 5 revisions stale | open |
| S7-14 | site | N | correctness | console templates, demo_flow.js | Unscored patients render "NaN%" (no `\|default` on data-probability) | open |
| S7-15 | site | N | correctness | demo_splitpane.js | Source lookup by query equality — repeated questions surface the wrong turn's passages | open |
| S7-16 | site | N | correctness | demo_a2ui.js | Stale trace pane leaks the previous patient's envelope JSON (R1 bleed in the trace surface) | open |
| S7-17 | site | N | correctness | demo_a2ui.js | Missing SourceCard → cite click no-op; failed extraction mislabels the whole note as a section | open |
| ECC-29 | ECC | Q | security | bigquery_source.py, config.py | Table-name env vars interpolated into SQL with no validation/allowlist | open |
| ECC-17 | ECC | Z | ops | agent/graph.py | `handler.last_trace_id` unverified attribute — possible AttributeError after full spend | open |
| ECC-50 | ECC | Z | ops | .env.lanfuse | Typo'd filename (gitignored, no secret leaked) — future loaders silently miss it | open |
| S1-17 | site | Z | security | bandit scan | 10 Low findings — verified false positives / low risk; awareness only | open |
| S6-05 | site | Z | ops | content/tests.py | No tests for the public content surface (auth gating, publishing filters, contact form) | open |
| S9-04 | site | Z | ops | settings.py | Legacy `publicRead` ACL breaks under UBLA; `MEDIA_URL` disagrees with storage output | open |
| S9-06 | site | Z | ops | settings.py (LOGGING) | LOGGING omits the `demo` app logger — the surface that most needs telemetry | open |
| S9-07 | site | Z | correctness | urls.py | DEBUG static served from stale `staticfiles/` instead of WhiteNoise | open |
| ECC-01 | ECC | — | security | bandit scan | B608 verified false positive — all user values parameterized | **resolved** |

## How to work this register

- Fix by **cluster**, not by individual ID — the roadmap orders the clusters.
- When a finding is fixed, flip Status here **and** in the source backlog.
- Definition of done (from the review plan): **zero Critical/Major open**.
