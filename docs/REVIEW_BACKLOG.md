# ECC Review Backlog — enterprise_clinical_copilot

Strategy + phases: `docs/code_review_plan.md`. Decisions locked 2026-08-29 —
adversarial model: **Claude Fable 5** (subagent override); primary pass: main
assistant (DeepSeek V4 Flash); **review-only first**; done = **zero Critical/
Major open**. Cross-repo risk order spans both repos; §1 (auth + quota) lives in
`danielmherman/docs/REVIEW_BACKLOG.md` and is reviewed first.

## Section status

| # | Section | Location | Status | Protocol | Notes |
|---|---|---|---|---|---|
| §2 | Agent harness | `projects/agent-harness/agent/` | **review complete** (primary + adversarial) — ECC-02…ECC-17 | Both | graph, prompts, server, mcp_client. Core of the system. |
| §3 | MCP + cross-patient isolation | `projects/agent-harness/mcp_server/` | **review complete** (primary + adversarial) — ECC-19…ECC-30, ECC-01 closed | Both | **R1 crown jewel** — cross-patient isolation in retrieval. |
| §4 | RAG / citations | `projects/agent-harness/rag/` + `scripts/` (index build) | **review complete** (primary + cross-check adversarial) — ECC-32…ECC-45 | Cross-check | index build, retrieval, citation groundedness. |
| §5 | IAM / secrets / deployment | `projects/agent-harness/scripts/` (deploy path) + cloudbuild + Dockerfiles + mlops deploy | **review complete** (primary + blind adversarial) — ECC-46…ECC-59 | Blind + scanners | 7.4k LOC bucket, but most scripts are dev helpers — review focused on the production deploy path. |
| §8 | MLOps | `projects/mlops/pipelines/` + `projects/agent-harness/pipelines/` | **review complete** (primary + cross-check adversarial) — ECC-60…ECC-73 | Cross-check | training/HPO/serving/feature store. Eval suite anchors understanding. |
| Deps | Dependencies (CVE) | `requirements.txt` (this + site repo) | **done 2026-08-30 — ECC clean** | Scanner | pip-audit on all 4 ECC manifests: **no known CVEs**. (Site repo: 72 CVEs — see site backlog S1-16.) |

## §2 Understand — Agent harness

**Scope:** `projects/agent-harness/agent/` — the live agent that turns a question
into a grounded, cited answer by driving Gemini + MCP tools. It is what Django
proxies to (§1). The tools it calls (predict, RAG) are §3.

**Entry points:**
- `agent/server.py` — Starlette HTTP surface (the deployed Cloud Run service):
  - `GET /health` — shallow (no Vertex/MCP/BigQuery) → status, project, model, transport.
  - `POST /ask` — `{"question": str}`; validates (non-empty → 400; >2000 chars → 413); opens a **fresh MCP session per request**; runs the graph; composes A2UI card + guardrails; returns `{question, answer, guardrail_flags, tool_calls, a2ui, model, mcp_transport}`.
- `agent/run.py` — CLI: `python -m agent.run "question" [--transport http|stdio] [--trace]`.
- `agent/Dockerfile` — python:3.12-slim, `USER 1000`; image ships only `agent/` + `mcp_server/config.py` (deliberately **no** BigQuery/Vertex libs — the agent reaches them through the MCP server).

**The graph (`graph.py`) — explicit LangGraph, not `create_react_agent`:**
- Control flow `START → agent → tools → agent → END`: the agent turns once to emit tool calls, the tool node runs them, the agent turns *again* to narrate — the second turn is where the prompt's guardrails apply.
- `AgentState`: `messages` + `tool_calls`, both `operator.add` (append).
- `agent_node`: `ChatGoogleGenerativeAI` (Gemini via Vertex, `temperature=0`, `max_output_tokens` from config, `max_retries=3`) bound with the tools.
- `tool_node`: per call, flattens a `kwargs` nesting Vertex can emit, calls `toolbox.call(name, args)` (never raises — returns dict), records `{name, args, response}` into `tool_calls`, appends a `ToolMessage`.
- `route`: last message is an AIMessage with tool_calls → tools, else END.
- **No explicit max-iteration cap in `build_graph`** — relies on LangGraph's default recursion limit.
- `ask()` builds a graph fresh per question, seeds `SystemMessage(SYSTEM_PROMPT)` + `HumanMessage(question)`, runs with a Langfuse `CallbackHandler` (no-op stand-in when `LANGFUSE_*` keys are absent), publishes `langfuse_trace_id` on the returned state.
- `final_text()` — last non-empty AI text (handles content-block lists).

**Tool wiring (`mcp_client.py`) — hand-rolled MCP→Gemini adapter:**
- Why: `langchain-mcp-adapters` is incompatible with `mcp==2.0.0` (imports `mcp.server.fastmcp`, removed in 2.0). `MCPToolbox` wraps a `ClientSession`; `load()` lists tools; `call(name, args)` executes with a 180s read timeout, **failures become structured `{"error": ...}` payloads, never exceptions** (the model reports errors instead of collapsing the graph).
- `_clean_schema`: strips Pydantic-emitted JSON-Schema keys (`title`, `$schema`, `additionalProperties`, `default`) Gemini would 400 on.
- Transports by env (`MCP_TRANSPORT`, default `stdio`): `stdio` = subprocess `python -m mcp_server.server --transport stdio` (no network/auth); `http` = `streamable_http_client` to `{MCP_URL}/mcp` with an **ID token** (audience = service URL; metadata server in prod, `gcloud` subprocess locally), using the SDK's vendored `httpx2`.
- `MCPToolbox._tools` is a public dict read by `graph._tools()`.

**System prompt (`prompts.py`) — Tier 2 guardrails (the behavioral contract):**
A large `SYSTEM_PROMPT` encoding hard rules: must call `predict_readmission` for risk and `rag_search`/`rag_search_sections` for notes; never answer from model knowledge of this patient; `rag_search_sections` **once** for fixed sections; flowing prose, no fabricated sections, empty is a real answer; every note claim cited `^[n]` to a returned passage (global numbering, each passage cited at most once, never stacked, meds cite the `discharge_medications` passage often `^[3]` not `^[1]`); report probability/threshold exactly and never invent a risk band in prose; attribute only `top_factors` (TreeSHAP logit-space, human labels); never fill a `___` redaction; reproduce med name/dose/freq verbatim; surface score-vs-note conflicts without revising the prediction; decision-support framing.

**Deterministic post-hoc guardrails (`guardrail.py`) — “LLM proposes, code disposes”:**
Pure functions of (answer, evidence), applied in `server.py` after the run; the served answer is the guarded one, `flags` returned for observability:
1. **Redacted-age** — drop an invented specific age when the source age is redacted (`___`).
2. **Medication verifier** — every asserted dose+unit/frequency must appear in the retrieved evidence; unverifiable doses dropped, unmatched freqs flagged (not auto-edited); plus **per-med** freq verification against that med's own `discharge_medications` entry (name+dose matched; conservative — skips multi-med chunks).
3. **Citation range** — every `^[n]` must point at a returned passage (else flagged).
Conservative: drops only on a clear mismatch, never rewrites positive content; regression-checked against passing answers.

**A2UI composition (`a2ui.py`):**
Turns a `predict_readmission` payload into **A2UI v0.9 messages** (`createSurface → updateComponents → updateDataModel`; `BASIC_CATALOG_ID` read off the shipped bundle). Rules: tools return JSON, the agent composes UI; **never render without `fallback_text`** (R8); `audience: ["user"]` (R7 — model must not re-read its own UI output); clinical values in `updateDataModel`, not baked into components; Markdown-escape feature names; error/malformed payloads render an honest error card. `risk_card_from_tool_calls` picks the **last** predict result.

**Observability:** Langfuse via the official `CallbackHandler` (native LangGraph trace + Gemini generation span + per-MCP-call tool span); enabled only when all three `LANGFUSE_*` vars are present; `last_trace_id` on state for the eval loop.

**Error handling:** MCP/tool failures → structured `{"error": ...}` data (never a collapsed graph). Server `/ask` wraps exceptions, unwraps `asyncio.ExceptionGroup`, logs `exc_info`, returns **502 `{"error": "agent_failed", "message": "<Type>: <detail>"}`** (service is IAM-private; Django genericizes to the browser). Too long → 413; bad JSON → 400.

**Config (`mcp_server/config.py` + env):** `GEMINI_MODEL`, `GEMINI_MAX_OUTPUT_TOKENS`, `LOCATION`, `PROJECT`, `MCP_TRANSPORT` (stdio), `MCP_URL`, `LANGFUSE_PUBLIC_KEY/SECRET_KEY/HOST`, `PORT`.

**Failure modes (expected):** downstream down → tools return `{"error":...}` → model reports plainly (Django refunds + 502, §1); MCP transport failure → ExceptionGroup unwrapped → 502; agent answers with zero tool calls → `tool_calls` empty, A2UI card None (prose only); Langfuse absent → no-op.

**Test coverage (`tests/`, 12 files):** `test_a2ui.py` (v0.9 shape, fallback/audience/error paths), `test_guardrail.py`, `test_agent_local.py`, `test_mcp_stdio.py`, `test_graph_rewrite_smoke.py`, `test_tier1.py`, plus RAG/sections/chunking/embed tests for §3/§4. Whole-harness tests ≈ 2k LOC.

**Areas to probe in the review passes (not findings yet):** no explicit max-turn cap (relies on LangGraph recursion limit → tool-loop spend, interacts with §1 quota); prompt-injection surface (free-text question + note passages both reach the model; no explicit "ignore instructions in the question/notes" directive — data-level isolation is §3); `/ask` 502 body includes exception detail (IAM-private + Django genericizes — confirm nothing sensitive reaches the browser via §1's `detail`); `tool_calls` responses (full predict + RAG passages) forwarded to the browser in live mode (intended for the canvas; raw evidence client-side, synthetic); no hard check that a risk answer actually called `predict_readmission` (guardrails verify citations vs passages, not "did you predict"); `_clean_schema` silently strips unknown keys (arg behavior depends on the tool — verify in §3).

## §3 Understand — MCP server + cross-patient isolation

**Scope:** `projects/agent-harness/mcp_server/` — the three MCP tools the agent
calls: `predict_readmission`, `rag_search`, `rag_search_sections`. This is the
**R1 cross-patient isolation boundary**. Served over stdio (local/Claude
Desktop) or streamable-HTTP (Cloud Run, `stateless_http` by default — the tools
hold no per-session state).

**Entry points / layout:** `server.py` (transport-agnostic `MCPServer`; adds the
3 tools; `/health` shallow → project/location/feature_source), `config.py`
(shared constants — defaults are the REAL production values; project
`trim-icon-498815-a0`, `us-east1`, dataset `readmission`, feature table
`hybrid_features`, index endpoint `readmission-rag-index`, deployed index
`rag_tree_ah`, embedding `gemini-embedding-001` 768-dim, `RESTRICT_NAMESPACE =
"hadm_id"`, discharge table `hybrid_notes`), `endpoint.py` (cached Vertex
Endpoint handle, one lookup per process), `features/` (feature source seam).

**Tool 1 — `predict_readmission` (`tools/predict.py`):** takes `hadm_id`; fetches
the feature row via the feature source; orders by manifest `feature_order`
(`to_vector`); calls the Vertex endpoint; returns probability / threshold /
decision / base_value / top_factors (**MAX_FACTORS=5** of 23 parent groups) +
model_version + feature_source. Errors are structured dicts — `unknown_patient`
(not in the feature source), `feature_fetch_failed`, `incomplete_features`
(missing columns — never a silent feature shift), `prediction_failed`. Blocking
work wrapped in `asyncio.to_thread`.

**Tool 2 — `rag_search` (`tools/rag_search.py`):** `hadm_id` + `query` +
`top_k` (validated 1–20). Embeds the query (gemini-embedding-001, 768-dim,
`RETRIEVAL_QUERY`); queries the Vector Search index with the **hadm_id
restrict applied IN the index query** via `Namespace("hadm_id",
[str(hadm_id)])` — server-side, **not** a post-filter (R1). Passage text is
resolved by `note_id` (recovered from datapoint id `"{note_id}_{section}_
{ordinal}"`) in one batched BigQuery query, then **deterministically re-chunked**
so the returned text is the exact section chunk, not the whole note (whole-note
text leaks unrelated sections into a citation). **Section-anchored retry:** if the
query clearly targets a section (`_section_for_query` keyword map) and that
section didn't rank first, retry with the section's ACTUAL body text as the
query (fixes query-side drift / wrong-rank). Empty is a real answer
(`{returned: 0, passages: []}`); `missing_text` errors if the index returns an
id not in BigQuery. Validates hadm_id (positive int), query non-empty, top_k
1–20. `asyncio.to_thread` wrapper.

**Tool 3 — `rag_search_sections`:** `hadm_id` only. **Deterministic** section
coverage: fetches the note by `hadm_id` (`WHERE hadm_id = @hadm_id`), re-parses
+ re-chunks with the same deterministic chunker that built the index, returns
ONE passage per major section (fixed `SUMMARY_SECTIONS` order: hospital course,
discharge diagnosis, discharge meds, discharge instructions, discharge
summary) with `score: 1.0` — 100% recall by construction, no top-k luck.
Isolated by the hadm_id fetch. Honest empty if no summary sections exist.

**Feature source (`features/`):** `base.py` Protocol seam (`fetch(hadm_id)` →
dict of manifest columns, missing → None). BigQuery only (Feature Store removed
2026-08-03 for cost). `bigquery_source.py`: `SELECT * FROM {table} WHERE
hadm_id = @hid LIMIT 1` — value parameterized; result restricted to
`feature_order` columns (SELECT * would let the label/bookkeeping columns reach
the model — a silent correctness bug, so it's filtered). `endpoint.py`:
`predict_one` **raises** if the endpoint returns an empty prediction list
(never a silent success).

**Isolation model (the crux):**
- predict — isolated per hadm_id (single-row fetch).
- rag_search_sections — isolated per hadm_id (note fetched by hadm_id).
- rag_search — isolated via the **index Namespace filter (in-query)**. Text
  resolution **trusts the index restrict**: `note_ids` → text with no re-check
  that the note belongs to the requested hadm_id (single layer).
- Synthetic/hybrid data: `hybrid_features` + `hybrid_notes` (MT-* synthetic
  notes); real MIMIC tables explicitly out of scope. The tools do **not**
  restrict to the `demo_cohort` table — any hadm_id present in the tables works
  (ties to §1 S1-09).

**Failure modes (expected):** endpoint down → `prediction_failed`; index down →
`search_failed`; embed fail → `embed_failed`; missing text → `missing_text`;
unknown patient → `unknown_patient`; empty result → honest empty. Errors are
structured payloads the agent narrates; never exceptions (see §2
`MCPToolbox.call`).

**Test coverage:** `tests/test_rag_search.py` (fakes for index/embed/bigquery),
`test_sections.py`, `test_chunking.py`, `test_concepts.py`, `test_tier1.py`
(Tier 1 acceptance incl. R1).

**Areas to probe in the review passes (not findings yet):**
- **ECC-01 B608 verification:** all user values look parameterized (`@hid`,
  `@hadm_id`, `@note_ids`); interpolations are config constants only — likely a
  false positive; verify each site.
- The section-anchored **retry recursion** in `_search` (same-body re-entry, no
  depth guard).
- Single-layer isolation in `_fetch_texts` (no note_id→hadm_id re-check).
- Tools don't restrict to the demo cohort (ties to S1-09).
- Error messages embed exception text (`type: exc`) → reach the model + Langfuse.
- `/health` discloses project/location/feature_source (same theme as ECC-06).
- No app-level auth (same as ECC-07) — IAM is the only gate.

## §4 Understand — RAG (index build, retrieval, citation grounding)

**Scope:** the retrieval stack behind the notes evidence. `rag/` (pure stdlib —
no cloud calls at import) + the index-build `scripts/`. §3 already covered the
serving tool (`rag_search`); §4 is **how the index is built, how passages are
chunked/grounded, and how consistency between build and serving is guaranteed**.

**1. Section parsing (`rag/sections.py`) — the foundation.** Discharge notes are
semi-structured; everything depends on where section boundaries fall. `parse_note`
splits ONLY on an explicit `KNOWN_HEADINGS` allowlist (~29 canonical sections +
aliases incl. MTSamples variants). A heading-shaped line absent from the list is
recorded as `unknown_headings` and LEFT in the enclosing body — under-splitting is
the safer error (dilutes rather than truncates). `ParsedNote` reports
`unknown_headings` + `coverage` so a parse failure is loud. `_build_lookup` raises
on duplicate aliases. Two regexes: colon-form (`_HEADING_RE`) + the MTSamples
numbered-list bare-heading form (`_BARE_HEADING_RE`).

**2. Chunking (`rag/chunking.py`) — the retrieval unit.** One chunk = one Vector
Search datapoint = one citable passage. Rules: split on section boundaries;
section ≤ `max_chars` (1500) = one chunk; longer splits paragraph → sentence →
fixed-width fallback. **Redaction-only / empty chunks are dropped** (they cost
embedding money and can never support a citation). `chunk_id =
"{note_id}:{section}:{ordinal}"` is **deterministic** → re-indexing is idempotent
and never orphans chunks. `pack_to` greedily merges pieces for line-oriented
sections. `Chunk` carries char offsets within the section body.

**3. Embedding (`rag/embed.py`) — pure shared helpers.** gemini-embedding-001 @
768 dims; `RETRIEVAL_DOCUMENT` at index time, `RETRIEVAL_QUERY` at query time.
`datapoint_id` folds `:` → `_` (Vector Search id charset `[A-Za-z0-9_-]`).
`vector_search_record` attaches the **`hadm_id` restrict namespace** per datapoint
(the R1 filter §3 relies on). Shared by build and serving so they can't drift.

**4. Note cache (`rag/notes.py`).** Test-split notes cached under `~/.cache`
(deliberately NOT the repo — the repo sits in iCloud and MIMIC text must not
sync). Gzipped JSONL + a **manifest-verified count** so a truncated/corrupt cache
fails loudly instead of silently analyzing half a corpus.

**5. Concept tagging (`rag/concepts.py`).** medspaCy + ConText assertion handling
(negated/hypothetical/historical/family) over seed phrase lists. **Precomputed at
ingestion, never at query time**; runs only under `.venv-nlp` (numpy 2.x vs
harness numpy 1.26). Labeled test data (`tests/data/concept_sentences.json`) is
the durable quality bar. Hedging is reported but never gates; sections scoped by
caller. (Feeds the eval/evidence gate, not serving.)

**6. Index build (`scripts/`).** The pipeline: `build_chunks.py` → `embed_chunks.py`
→ `deploy_index.py` (→ `chain_rag_deploy.py`/`deploy_rag_endpoint.py`/`wait_rag_deploy.py`).
- `build_chunks.py`: chunks the note cache, filtered to a **section whitelist**
  (`DEFAULT_SECTIONS` — the 13 narrative/assessment sections); reports zero-chunk
  notes + section distribution. Idempotent (deterministic ids).
- `embed_chunks.py`: resumable batch embedding (skips already-done ids), writes
  ingest JSONL.
- `deploy_index.py`: **two-index approach (D3)** — a small BRUTE_FORCE index for
exact ground truth at pennies, the full TREE_AH index for serving. `verify()`
blocks until READY and **asserts the datapoint count == expected** (silent
shortfall = dropped chunks → fail loudly). `count_hybrid_chunks.py` replicates
the chunker locally to compute the expected vector count.
- `validate_rag.py`/`verify_rag_query.py`: §9 validation trials against the
deployed index.
- `prune_rag_datapoints.py`: removes datapoints for retired patients (index +
BigQuery). `teardown.py`: cost teardown of the index endpoint.

**7. Retrieval + grounding (serving, §3 covers the tool).** `rag_search` re-runs
the deterministic chunker to return the exact section chunk a citation points at
(never the whole note); falls back to the whole note only if the id can't be
reproduced (ECC-26); section-anchored retry for wrong-rank (ECC-20/24/30);
citations `^[n]` range-checked by the guardrail (ECC-14).

**Failure modes (expected):** unknown heading → stays in enclosing body + counted;
zero-chunk note → reported; index shortfall → deploy fails loudly; redaction-only
chunk → dropped; corrupt cache → manifest check raises; retrieval miss → honest
empty.

**Test coverage:** `tests/test_sections.py` (parser incl. alias/unknown-heading
cases), `test_chunking.py` (determinism, redaction-drop, long-body splits),
`test_concepts.py` + labeled concept sentences, `test_embed.py` (datapoint id,
ingest rows).

**Areas to probe in the review passes (not findings yet; cross-check protocol —
the adversarial pass will read THIS doc and try to falsify it):**
- Consistency between the three section lists: `build_chunks.DEFAULT_SECTIONS`
  (13), `rag_search._KNOWN_SECTIONS` (13), and `parse_note`'s ~29 canonical
  sections. Any mismatch = dropped or unparseable sections (ties ECC-23).
- **Chunker-parameter drift:** `build_chunks` accepts `--max-chars/--pack-to`;
  serving `_chunk_texts_for` calls `chunk_note` with DEFAULTS. If the deployed
  index was built with non-default params, serving re-chunking can't reproduce
  ids → systematic whole-note fallback (ECC-26). Verify the deployed index used
defaults or the params are recorded.
- Count verification (EXPECTED_VECTORS): is the datapoint-count assert enforced
  in the CI/pipeline path or only a manual script step?
- Deterministic chunk_id format vs the serving regex parse (ECC-23): note_id/section
  charset.
- Redaction: partially-redacted sections still yield chunks containing `___`
  (guardrails handle ages; other redactions rely on the prompt).
- Citation grounding is range-checked (ECC-14) but not semantically verified.

## §5 Understand — IAM / secrets / deployment

**Scope:** the production deployment topology across both repos. Per Dan
(2026-08-30): the 49-file `scripts/` bucket is **mostly one-off development
helpers** — review effort goes to the **production deploy path**; the dev helpers
are low-risk tooling. Future direction: automate deployment of the right
resources (currently largely manual/console-state).

**Deployment topology — 3 Cloud Run services (scale-to-zero, free idle):**
- **Site** (`danielmherman`) — the ONLY public surface. Deployed by the site
  `cloudbuild.yaml`: build → push → migrate job → seed-demo-patients job →
  `gcloud run deploy` with `website-sa` service account. **No env vars set in the
  deploy step** (S1-01 — `ENVIRONMENT`/DEBUG fail-open on recreate). Secrets via
  Secret Manager (`SECRET_KEY`, `db-password`) in prod settings.
- **Agent** (`agent/server.py`) — private, IAM-invoker-gated. Image built by
  `cloudbuild.agent.yaml` (`docker build -f agent/Dockerfile`, harness-root
  context).
- **MCP server** (`mcp_server/server.py`) — private. Image via
  `cloudbuild.mcp.yaml` (ships `mcp_server/` + `rag/`).
- **Agent/MCP runtime deploy + env config are NOT committed** (console /
  deploy-trigger state) — recreating the service from the build files loses
  `GEMINI_MODEL`, `MCP_URL`, `LANGFUSE_*`, `PROJECT_ID` (same recreate-from-file
  class as S1-01).

**2 Vertex billable-by-hour endpoints (`teardown.py` is the cost lever):**
- **Prediction** `readmission-endpoint` (n1-standard-2 ~$80/mo) — deployed by
  `deploy_cpr.py`: **content-addressed image** (hash of Dockerfile/predictor.py/
  requirements.txt → rebuild only on source change), registers `readmission-cpr-*`
  model, **undeploys ALL stale models**, deploys; auto-discovers the newest
  `readmission-final-*` serving bundle from Model Registry provenance.
- **RAG** `readmission-rag-index` (e2-standard-2 ~$68/mo) — deployed by
  `deploy_synthetic_rag.py` (safe values). `deploy_rag_endpoint.py` defaults are
  the **555k real index on e2-standard-16 (~$270/mo)** — `launch_endpoints.sh`
  explicitly warns to use the synthetic script. `public_endpoint_enabled=True`
  (ECC-36).
- `launch_endpoints.sh` — parallel stand-up (deploy_cpr.py +
  deploy_synthetic_rag.py). `teardown.py` — undeploy+delete the two endpoints
  (keeps Model Registry, the Vector index, GCS, BQ; `--dry-run`/`--only`).

**IAM / service accounts:**
- Site: `website-sa` stated explicitly in cloudbuild (least-privilege; avoids the
  default compute account); migrate + seed jobs use it. ID tokens minted via the
  metadata server for agent/mcp access.
- Agent/MCP: rely entirely on Cloud Run IAM invoker + audience; **no app-layer
  auth** (ECC-25/ECC-07).
- **No codified IAM bindings / secret automation in the repo** (console-managed)
  — the "future automation" direction Dan flagged.

**Secrets:**
- Site: Secret Manager (`SECRET_KEY`, `db-password`) via `get_secret()` (prod only).
- Langfuse: `.env.lanfuse` in the harness working tree — **typo'd name for
  `.env.langfuse`** (missing 'g'); **gitignored** (`.gitignore:24 .env.*`) + mode
  600 → not committed, but misnamed (hygiene; a loader expecting the correct
  name won't find it). `LANGFUSE_*` are read from env vars (`graph.py`); the file
  is sourced manually.
- Hardcoded prod project `trim-icon-498815-a0` scattered as defaults across
  `config.py`, `deploy_cpr.py`, `register_serving_model.py`, `teardown.py`,
  `deploy_index.py`, `deploy_rag_endpoint.py`, `build_images.sh` (ECC-15).

**Deploy-path scripts (reviewed):** `deploy_cpr.py`, `deploy_rag_endpoint.py`,
`deploy_synthetic_rag.py`, `deploy_index.py`, `chain_rag_deploy.py`,
`wait_rag_deploy.py`, `rag_endpoint_status.py`, `teardown.py`,
`register_serving_model.py`, `deploy_endpoint.py`, `build_images.sh`,
`build_rag_image.sh`, `launch_endpoints.sh`, `smoke_test.py`,
`integration_test_live.py`, `verify_mcp_live.py`, `verify_rag_query.py`,
`validate_rag.py`, `watch_pipeline.py`, `inspect_failed_pipeline.py`,
`submit_recall_job.py`.

**Dev-cycle helpers (low review priority per Dan):** build_*/load_*/generate_
synthetic_*/seed_*/probe_*/explore_*/crawl_mtsamples/clean_mtsamples/
fill_features/check_coherence/coherence_scan/coverage_report/fetch_note_cache/
select_notes/score_test_split/audit_features/drive_live_chips/
capture_synthetic_rag_fixtures/build_golden_sample_108/build_hybrid_*/
prune_inclusion_violations/count_hybrid_chunks/embed_chunks/check_gemini.

**Dockerfiles:** site (runs as root — S1-14), agent + mcp (`USER 1000`). All
`python:3.12-slim` mutable tag, no hash-pinned deps (ECC-16).

**Failure modes (expected):** endpoint down → `smoke_test.py` discovers bundle;
recreate-from-file loses env config (S1-01); deploy_cpr undeploys stale models
before deploy (brief empty-serving window); teardown is destructive (`--yes`).

**Test/validation:** `smoke_test.py` (live endpoint check), `integration_test_live.py`,
`verify_rag_query.py`, `validate_rag.py` — live validation against deployed
resources (not unit tests).

**Areas to probe in the review passes (not findings yet):**
- Agent/MCP runtime env config not committed — how are they actually deployed today?
- `.env.lanfuse` typo'd filename (gitignored but misnamed).
- `deploy_cpr.py` undeploys ALL stale models on the endpoint before deploy (serving window).
- `teardown.py` matches index endpoints by prefix `readmission` (collision risk).
- `register_serving_model.py` default bundle URI embeds real resource IDs.
- Hardcoded `PROJECT` scattered (ECC-15); no secrets rotation; Secret Manager
  `latest` version usage.
- No codified IAM/secret automation (the future-automation direction).

## §8 Understand — MLOps (training / HPO / serving / feature store)

**Scope:** the model side — `projects/mlops/` (training pipeline + components +
CPR serving) + `projects/agent-harness/pipelines/` (RAG ingest) + the eval
suite. Feature store = **BigQuery** (Vertex AI Feature Store removed 2026-08-03
for cost; features static in `analytics_dataset_encoded` / `hybrid_features`).
Protocol: cross-check.

**The training DAG (`pipelines/training_pipeline.py`):** `load_data →
validate_data → benchmark_xgboost → benchmark_gate → optuna_hpo →
train_final → calibrate_threshold → evaluate_test / shap_explain /
fairness_audit → register_model`. Explicit gates:
- **`validate_data`** — Evidently drift + data-quality; **hard-fails** the
  pipeline when the drifted-column share exceeds `max_drifted_share` (0.2).
- **`benchmark_gate`** — the benchmark XGBoost must beat the **HOSPITAL baseline
  AUCPR** (read at compile time from `artifacts/hospital_baseline.json`).
- **`evaluate_test`** — the honest pre-test estimate is the **HPO validation
  AUCPR** (not `train_final`'s combined fit); test-set AUCPR recorded.
- **`fairness_audit`** — audit on the test split.
- **`calibrate_threshold`** — F-beta-optimal operating threshold on out-of-fold
  train predictions (metadata for the decision layer only; the model returns
  calibrated probabilities).
- **`register_model`** — publishes a **versioned serving bundle** (`model.bst` +
  `manifest.json` + `threshold.json`) to GCS + records a Vertex Model Registry
  provenance entry (`readmission-final-<ts>`).

**Feature contract:** all encoding static in BigQuery (one-hot + missingness
policy); the model consumes a fixed-order numeric vector; `CAT_FEATURES` empty.

**CPR serving (`pipelines/serving/cpr/predictor.py`):** `ReadmissionPredictor` —
downloads the bundle (`prediction_utils.download_model_artifacts`), loads
`model.bst` via xgboost (native format, not pickle), parses requests (positional
lists or named dicts; JSON null → NaN = native missing), returns probability +
threshold decision + native-TreeSHAP attributions aggregated to parent groups
(top 10). Endpoint is private; the MCP `predict_readmission` tool is the gate
(§3).

**HPO:** optuna (50 trials, 2700s timeout) into Vertex Experiments.

**RAG ingest pipeline (`agent-harness/pipelines/`):** chunk_notes → embed_chunks
→ build_index (verified count) — covered in §4 (ECC-32/37/38/39/40).

**Eval suite (`agent-harness/eval/`, 22 files):** golden eval + LLM-judged
scoring with Langfuse — the rigor layer, not the serving path.

**Failure modes (expected):** gate failure → pipeline halts (drift / benchmark);
model artifact is native xgb (not pickle — lower deserialization risk than
ECC-31's joblib path); endpoint input-shape errors → tool-level structured
error.

**Test coverage:** `pipelines/tests/` (`test_train_final`, `test_pipeline`),
`_export_sample_data.py`, `pytest.ini`. Partial.

**Probe areas (not findings yet):**
- CPR `preprocess`: **no instance-count/shape validation**; `float(v)` accepts
  NaN/Inf/garbage → garbage predictions or endpoint errors (endpoint private;
  MCP is the gate — ties ECC-28).
- Model artifact (`model.bst`) downloaded with **no hash/integrity check** (ties
  ECC-31 / CWE-502).
- `DEFAULT_CPR_SERVING_IMAGE` hardcoded `:latest` (ties ECC-59).
- `load_data`/components: SQL interpolation of pipeline params (ECC-37 class —
  verify).
- Positive: the gates (benchmark / drift / fairness / threshold) are strong — a
  well-built pipeline.

## Findings backlog (severity-ranked)

*Status 2026-08-30: CVE scan done (clean). **CWE scan done**. §1 lives in the
site backlog. **§2 complete** (ECC-02…ECC-17). **§3 complete** (ECC-19…ECC-30).
**R1 verified.** **§4 complete** (ECC-32…ECC-45). **§5 complete** (ECC-46…
ECC-59). **§8 complete 2026-08-30** (primary + cross-check adversarial; merged
ECC-60…ECC-73 = 1 Critical, 5 Major, 7 Minor). Cross-check falsified "gates are
strong" (baseline bypassable) and found ECC-64 train/test leakage.*

| ID | Section | Severity | Category | Location | Finding | Remediation | Status |
|---|---|---|---|---|---|---|---|
| ECC-01 | §2/§3 | Minor | security | bandit scan (ECC) | 55 Medium / 433 Low / 0 High. **B608 SQL-string construction** in `mcp_server/features/bigquery_source.py:22` and `mcp_server/tools/rag_search.py:125,135`. **RESOLVED 2026-08-30 (verified false positive):** every user-controlled value is parameterized (`@hid`, `@hadm_id`, `@note_ids` — `ScalarQueryParameter`/`ArrayQueryParameter`); the only interpolated parts are config constants (`TABLE`, `DISCHARGE_TABLE`), never user input. No SQL injection. Other bandit signals: B104 bind-all-interfaces on both server entrypoints (normal behind Cloud Run ingress), B101 asserts ×396 (mostly tests — verify no serving-code asserts under `-O`), B108 tmp + B311 pseudo-random in `eval/` (low). | Deep-triage the B101/B104 items with §5; B608 closed. | resolved |
| ECC-02 | §2 | Major | ops/spend | `agent/graph.py`, `agent/mcp_client.py` (180s), `agent/server.py` (no deadline), §1 `DEMO_AGENT_TIMEOUT=120` | Spend per request is effectively unbounded: no explicit `recursion_limit` (relies on LangGraph default ~25), no per-turn tool-call cap, each MCP call up to 180s, no request deadline. Worst case ~12 billed Gemini calls + many 180s tool calls in one request; the upstream (Django 120s) aborts + refunds while the agent keeps billing to completion — **refunded AND billed**. (Folds prior ECC-03; adversarial corroborates.) | Set explicit low `recursion_limit` + per-turn tool cap; reduce tool timeouts to ≤ upstream deadline; wrap `ask()` in `asyncio.timeout` under the proxy timeout. | **resolved 2026-08-31** — `recursion_limit=10`, `MAX_TOOL_CALLS_PER_TURN=5` (structured `tool_call_limit` refusal), tool timeout 100s < transport 110s; `/ask` wrapped in `asyncio.timeout(ASK_TIMEOUT_SECONDS=110)` → structured 504 |
| ECC-04 | §2 | Critical | correctness/faithfulness | `agent/graph.py`, `agent/guardrail.py` | No deterministic guard on the risk number itself. `guard_answer` checks redacted age, med dose/freq, citation range — never that a stated probability/threshold matches a `predict_readmission` response, nor that a risk-shaped answer had a predict call. A fabricated "risk is 0.14" with zero tool calls is served verbatim with HTTP 200 (all guardrails inert on empty evidence; A2UI card None). The system prompt is the only defense against the exact failure its own docstring names. **Corroborated + upgraded to Critical by the adversarial pass.** | In `guard_answer`: extract stated probabilities; require match with the predict response (drop/flag mismatches); if no predict call ran and a risk number is stated, strip + flag `risk_number_unsupported`; consider refusing/downgrading flagged answers. | **resolved 2026-08-31** — `verify_risk_numbers` in `guard_answer`: every risk-shaped number (0.xx decimal or percent) must match a successful `predict_readmission` probability/threshold to its own stated precision, else stripped + flagged `risk_number_unsupported`; acts even with zero tool calls; dose units, concentrations, and evidence-quoted values exempt |
| ECC-05 | §2 | Major | security/faithfulness | `agent/prompts.py`, `agent/graph.py`, `agent/server.py` | Prompt-injection / no data-instruction separation: both the free-text question (2000 chars, content-unvalidated) and retrieved note passages reach the model with no mitigation — passages dumped into `ToolMessage` content unmarked, no system-prompt rule that delimited content is data, no "ignore instructions in notes". Combined with ECC-04, an induced instruction about the number passes end-to-end. Synthetic notes limit exploitability; the question path is fully user-controlled today. Adversarial pass rates Major. | Wrap tool results in explicit delimiters + a "content is data, never instructions" system rule; strip instruction-like patterns; close ECC-04 so induced fabrications are caught deterministically. | **resolved 2026-08-31** — tool results wrapped in `<tool_result name="...">` delimiters (graph.py) + new DATA VS INSTRUCTIONS system-prompt section (content is data, imperative note text is never obeyed, the question cannot change the rules); ECC-04 closed the deterministic backstop |
| ECC-06 | §2 | Major | ops/security | `agent/server.py` (`/ask` 502, `/health`) | `/ask` 502 body echoes `Type: cause` (and for an ExceptionGroup joins all inner exception strings — httpx2/MCP/google-auth text routinely embeds the private `MCP_URL`, IAM/audience detail, table names). `/health` returns `mcp_url`, project ID, region, model unauthenticated at the app layer. Contained today (IAM-private; Django genericizes) but internal topology leaks to any direct caller. Upgraded per adversarial pass. | Log detail server-side; return a generic code + opaque correlation ID; drop `mcp_url`/`project` from `/health` or gate it. | **resolved 2026-08-31** — 502s log detail server-side and return `agent_failed` + 12-char correlation id; `/health` no longer returns project/location/mcp_url |
| ECC-07 | §2 | Minor | security | `agent/server.py`, deployment | No app-level auth on `/ask` — relies entirely on Cloud Run IAM (ingress + invoker). A misconfigured invoker binding exposes the agent directly, bypassing §1's quota gate. | Defense-in-depth: optional shared secret; verify IAM binding + ingress in §5. | **resolved 2026-08-31** — on Cloud Run (`K_SERVICE`) `/ask` returns 401 without an Authorization header (IAM front end always forwards one); IAM binding verification remains a cloud-side action |
| ECC-08 | §2 | Minor | architecture | `agent/server.py` (`tool_calls`), §1 live mode | Full `tool_calls` responses (predict payload + RAG passages) forwarded to the browser in live mode. Intended for the canvas, but raw evidence (synthetic) ships client-side. | Trim to what the canvas needs. | **resolved 2026-08-31** — /ask response trims tool_calls to `{name, response}` (the two fields the site's canvas composition reads); guardrails still run on the full records |
| ECC-09 | §2 | Major | ops | `agent/mcp_client.py` (`id_token`), `agent/server.py` | `id_token` is fully synchronous (blocking `fetch_id_token` or `subprocess.run` gcloud, 60s) and is called inside the async `http_toolbox` on every request — it blocks the **entire uvicorn event loop** for the token fetch, stalling all concurrent `/ask` and `/health`. | Wrap in `asyncio.to_thread`; cache the token until near expiry instead of minting per request. | **resolved 2026-08-31** — `_cached_id_token` (45 min TTL per audience) called via `asyncio.to_thread` |
| ECC-10 | §2 | Major | correctness | `agent/graph.py` (`_MCPTool`, `_tools`), `agent/mcp_client.py` (`gemini_tool`) | The MCP tools' input schemas never reach the model: `_MCPTool` sets no `args_schema`, so `bind_tools` declares empty/inferred parameters and the model must guess arg names from prose; the cleaned `MCPToolbox.gemini_tool` is dead code on this path. The fragile `kwargs`-unwrapping heuristic exists because of this — and would itself mangle a legitimate argument literally named `kwargs`. Args pass to `toolbox.call` with zero validation. | Convert each MCP `input_schema` into a Pydantic `args_schema`; drop the `kwargs` heuristic; validate required args before calling. | **resolved 2026-08-31** — `_MCPTool` declares `args_schema=_clean_schema(tool.input_schema)` (langchain-core accepts a JSON-schema dict), so `bind_tools` advertises real parameter names/types; kwargs heuristic removed, arguments pass through verbatim |
| ECC-11 | §2 | Major | correctness | `agent/guardrail.py` (`verify_med_tokens`) | Unsupported dose tokens are removed with `cleaned.replace(surface, "")` — a plain substring replace that ignores the regex lookarounds used to find the match. An unsupported "5 mg" also destroys a supported "2.5 mg" (→ "2."). A medication-fidelity guardrail can corrupt a correct dose in the served answer. | Remove by match span (`_remove_spans`) instead of `str.replace`. | **resolved 2026-08-31** — `verify_med_tokens` collects match spans and removes via `_remove_spans`; regression test proves "2.5 mg" survives dropping "5 mg" |
| ECC-12 | §2 | Major | correctness | `agent/graph.py` (`final_text`), `agent/server.py` | `final_text` returns the LAST non-empty AI text. When the final agent turn is empty (the documented MAX_TOKENS failure that raises nothing), it silently falls back to an earlier AI message — typically a pre-tool preamble — and serves that stale fragment (or empty string) with HTTP 200. | Only accept text from the final AI message; if empty/MAX_TOKENS, return `answer_unavailable` instead of 200. | **resolved 2026-08-31** — `final_text` accepts only the final message (and only if AIMessage); empty → server returns 502 `answer_unavailable` and the site refunds the quota debit |
| ECC-13 | §2 | Minor | correctness | `agent/graph.py` (`tool_node`) | Tool results are serialized into `ToolMessage` as `str(payload)` (Python repr: single quotes, `None`, `True`), not JSON — the model sees a format the prompt doesn't describe, and exact-match reasoning (citations/numbers) is less reliable. | `json.dumps(payload, ensure_ascii=False)` for ToolMessage content. | **resolved 2026-08-31** — ToolMessage content is now JSON inside the ECC-05 `<tool_result>` delimiters |
| ECC-14 | §2 | Minor | correctness | `agent/guardrail.py` (`check_citations`), `agent/server.py` | Citation verification is range-only and advisory-only: out-of-range `^[n]` markers are flagged but never stripped from the served answer (rendering as citations to nothing); no support check between claim and passage; with zero passages, an answer containing `^[1]` is served intact. | Strip out-of-range markers from the answer; consider a lexical-overlap support check. | **resolved 2026-08-31** — `check_citations` now strips out-of-range `^[n]` markers (returns cleaned answer + flags); lexical-overlap support check remains future work |
| ECC-15 | §2/§5 | Major | security/ops | `mcp_server/config.py`, `deploy_cpr.py`, `register_serving_model.py`, `teardown.py`, `deploy_index.py`, `deploy_rag_endpoint.py`, `danielmherman/settings.py` | **Real production identifiers committed throughout** (upgraded per adversarial pass): GCP project ID `trim-icon-498815-a0`, project number `778397675435`, bucket, Artifact Registry paths, the numeric Vector Search index ID, a concrete GCS bundle URI with pipeline run IDs, and the private agent's actual Cloud Run URL as `DEMO_AGENT_URL` default (S1-10). Full internal topology reconstructable from source, and it's **fail-open**: a container started without env vars silently operates against the real project + real BigQuery tables (config.py says intentional for local convenience) — local mistakes run against prod by default. | Make `PROJECT_ID`/`DEMO_AGENT_URL` required (fail fast) — never default to production; move real IDs to an untracked/env file; if the repo becomes public, audit IAM accordingly. **RESOLVED 2026-08-31:** config.py requires PROJECT_ID (env or untracked repo-root .env, no default); deploy-path scripts route through shared config; DEMO_AGENT_URL default removed site-side (S1-10); .env.example placeholder only. | resolved |
| ECC-16 | §2 | Minor | ops | `agent/Dockerfile` | Mutable base tag `python:3.12-slim` (no digest pin) → non-reproducible rebuilds; `pip install` without `--require-hashes`. (Non-root `USER 1000` is correctly set.) | Pin base image by digest; hash-pin requirements (`pip-compile --generate-hashes`). | open |
| ECC-17 | §2 | Minor | ops | `agent/graph.py` (`ask`, langfuse) | When Langfuse is enabled, `ask()` reads `handler.last_trace_id` off the official `langfuse.langchain.CallbackHandler`; unverified that current handlers expose it. If absent → `AttributeError` after the full run completes (request 502s after all Vertex/MCP spend) in the observability-enabled config that dev runs never exercise. Low confidence. | `getattr(handler, "last_trace_id", None)`; add a test with a real/stub enabled handler. | open |
| ECC-19 | §3 | Major | security | `mcp_server/tools/rag_search.py` (`_fetch_texts`) | Isolation in `rag_search` rests on a **single layer**: the index `Namespace("hadm_id", …)` filter. `_fetch_texts` then resolves passage text by `note_id` with **no re-check** that the note belongs to the requested hadm_id. Fail-closed if the index restrict is misconfigured (missing namespaces exclude datapoints), but any defect upstream (wrong/missing restrict token, stale index, `_parse_datapoint_id` truncation) flows straight through to another patient's section text. Adversarial pass rates **Major** (this is R1). | Add `AND hadm_id = @hadm_id` to `_fetch_texts` (fetch `note_id, hadm_id, text`, parameterized) and hard-error `isolation_violation` if any resolved hadm_id differs — a second independent enforcement of R1 at the BigQuery layer. | **resolved 2026-08-31** — `_fetch_texts(note_ids, hadm_id)` fetches `note_id, hadm_id, text` and raises `IsolationViolation` on any foreign row; `_search` surfaces it as a structured `isolation_violation` error |
| ECC-20 | §3 | Critical | ops/spend (DoS) | `mcp_server/tools/rag_search.py` (`_search` anchored retry, L205–233) | The section-anchored retry recurses with **no depth guard**: `_search(hadm_id, body, top_k)` re-enters with the same section body when the anchored query persistently fails to rank the target chunk first (the body itself contains the trigger word → same branch). `_section_bodies` re-fetches the note **uncached** each level. Every level is a billed embed + Vector Search query + 1–2 BigQuery queries; a single adversarial/unlucky query can generate **thousands of billable calls** then hit Python's RecursionError. **Upgraded to Critical by the adversarial pass** (spend amplification / DoS, attacker-triggerable). | Pass an explicit `depth`/`is_retry` flag — permit exactly ONE anchored retry, skip anchor computation on the retry path; cache `_section_bodies` per request. | **resolved 2026-08-31** — `_search(..., is_retry=True)` skips anchoring on the retry path, structurally bounding the chain at one retry; regression test proves a self-triggering section body stops at 2 index calls |
| ECC-21 | §3 | Minor | security | `mcp_server/tools/*` (`_error` messages) | Tool error messages embed raw exception text (`feature_fetch_failed`/`prediction_failed`/`embed_failed`/`search_failed` all carry `f"{type(exc).__name__}: {exc}"`). That can include internal table names, project ids, URLs — and it reaches the **model** (which narrates it) and Langfuse. Same theme as ECC-06. | Log detail server-side; return stable error codes + short generic messages to the tool contract. | **resolved 2026-08-31** — all four `_error` paths log `exc_info` server-side and return stable codes + generic messages |
| ECC-22 | §3 | Minor | correctness | `mcp_server/tools/predict.py`, `tools/rag_search.py` | The tools do **not** restrict to the `demo_cohort` table — any hadm_id present in `hybrid_features`/`hybrid_notes` works. A user (via §1) can probe admissions beyond the 24-patient demo cohort. Data is synthetic, but the cohort boundary is enforced nowhere server-side. Ties to S1-09. | Enforce the demo-cohort membership as the authorization boundary (or document that the tables hold only demo rows). | **resolved 2026-08-31** — documented (config.py): the site enforces cohort membership pre-agent (S1-09) and hybrid_features/hybrid_notes hold only synthetic hybrid rows (90000001+, MT-*) by construction — a bypassed request can never reach real MIMIC data |
| ECC-23 | §3 | Major | correctness | `mcp_server/tools/rag_search.py` (`_parse_datapoint_id`/passage build) | A datapoint id that doesn't match the known-section pattern yields `note_id=None`/`section=None`, and the passage is appended with `text=None` (the `if note_id and text is None` error path is skipped because note_id is None). `_KNOWN_SECTIONS` (13 names) is a hand-maintained copy of the chunker's vocabulary while `parse_note` recognizes ~29 sections — any indexed section outside the 13 hits this path. A malformed/foreign index id reaches the model as a **null-text citation** instead of an error — the exact "silent drop" the `missing_text` guard exists to prevent. Upgraded per adversarial pass. | Return a structured error (or skip + flag `unparsed`) when note_id is None; derive `_KNOWN_SECTIONS` from `rag.sections` canonical names instead of a duplicate literal. | **resolved 2026-08-31** — unparsed ids now return a structured `unparsed_datapoint` error; `_KNOWN_SECTIONS` derived from `rag.chunking.INDEX_SECTIONS` |
| ECC-24 | §3 | Major | correctness | `mcp_server/tools/rag_search.py` (L206–214) | The wrong-rank retry replaces a **non-empty** original result with an **empty** retried result: `retried` is returned whenever `not retried.get("error")`, but `{"returned": 0, "passages": []}` has no error key — so real hits from the original query are discarded and the caller gets zero passages. The retry was meant to improve ranking, not destroy recall. | Only return `retried` when `retried.get("returned", 0) > 0`; otherwise fall through to the original neighbors. | **resolved 2026-08-31** — exactly that guard; empty retry falls through to the original hits (regression test) |
| ECC-25 | §3 | Major | security | `mcp_server/server.py` (HTTP transport), `config.py` | The MCP HTTP transport has **no app-layer auth and no per-principal authorization** — nothing binds a caller to a permitted hadm_id; isolation is per-request, not per-principal. The whole posture rests on Cloud Run IAM being right at deploy; a single `--allow-unauthenticated` exposes the full dataset **and** billable Vertex/BigQuery calls (compounding ECC-20 into an unauthenticated cost-amplification vector). Extends §2 ECC-07. | Enforce no-unauthenticated deployment (verify platform-injected identity header in middleware); for anything beyond a demo, add a caller→permitted-admissions authorization mapping. | **resolved 2026-08-31** — K_SERVICE-gated ASGI guard 401s header-less non-/health requests on the HTTP transport (defense-in-depth; IAM binding verification stays cloud-side); per-principal authorization deferred as out of demo scope |
| ECC-26 | §3 | Minor | correctness | `mcp_server/tools/rag_search.py` (L256) | `body = chunk_texts.get(note_id or "", {}).get(nb.id) or text` falls back to the **entire note** whenever the chunk id fails to reproduce — which happens systematically if the index was built with different chunker parameters (not recorded anywhere; `_chunk_texts_for` calls `chunk_note` with defaults), and the `or` also treats a legitimately-empty chunk as missing. The result is exactly the "whole-note leakage into a citation" the module docstring says it prevents — silently, with no granularity marker. | Use `.get(nb.id)` with an explicit `is None` check; tag fallbacks (`"granularity": "note"`) or error; pin/record chunker params used at index build. | **resolved 2026-08-31** — explicit `is None` chunk lookup; fallback logs a warning and tags the passage `"granularity": "note"`; build params pinned via `DEFAULT_PACK_TO` |
| ECC-27 | §3 | Minor | correctness | `mcp_server/tools/rag_search.py` (`_fetch_note_row`, `_fetch_note`) | Both note queries use `LIMIT 1` with **no `ORDER BY`** — BigQuery gives no ordering guarantee, so if an admission has more than one note row, the selected note is nondeterministic. `rag_search_sections` could cite a different note across calls, and the section-anchoring retry could anchor on a note different from the one the index matched. | Add a deterministic `ORDER BY note_id` (or aggregate all of the admission's notes); consider asserting the single-note assumption. | **resolved 2026-08-31** — both queries now `ORDER BY note_id LIMIT 1` |
| ECC-28 | §3 | Minor | correctness | `mcp_server/tools/predict.py`, `tools/rag_search.py` (sections) | `predict_readmission` and `rag_search_sections` perform **no `hadm_id` validation**, unlike `rag_search` (positive-int check). A non-int (or `bool`, which passes `isinstance(x, int)`) reaches the BigQuery binding and surfaces as an opaque `feature_fetch_failed` exception string instead of a clean `bad_request`. Validation is inconsistent across the three tools. | Extract the `hadm_id` validation into a shared helper; apply in all three entry points (and reject `bool`). | **resolved 2026-08-31** — shared `tools/_validation.valid_hadm_id` (rejects bool) applied in `_search`, `_search_sections`, and `predict._predict`; all return structured `bad_request` |
| ECC-29 | §3 | Minor | security | `mcp_server/features/bigquery_source.py`, `tools/rag_search.py`, `config.py` | Table identifiers (`FEATURE_TABLE`, `DISCHARGE_TABLE` env vars) are interpolated into SQL with **no validation/allowlist**. User values are parameterized (verified), but an operator-controlled env var could inject arbitrary SQL or — more realistically — silently repoint the demo at the **real MIMIC tables** the config comments say must never be served. | Validate table names against a strict pattern at import, and/or allowlist permitted tables for the deployment. | **resolved 2026-08-31** — `config._validated_table_ref` validates FEATURE_TABLE (dataset.table) and DISCHARGE_TABLE (project.dataset.table) at import: dot-separated `[A-Za-z0-9_-]` segments only, so an env var carrying SQL fails the boot loudly |
| ECC-30 | §3 | Minor | correctness | `mcp_server/tools/rag_search.py` (L206–209) | The wrong-rank trigger `(not top_sections or top_sections[0] != anchor or anchor not in top_sections)` reduces to just `top_sections[0] != anchor` (the other clauses are dead). As written the expensive retry fires whenever the anchor merely isn't rank 1 (even at rank 2) — a broader, costlier trigger than the comment's "absent from top-k" intent. | Decide the intended trigger (absent-from-top-k vs not-rank-1), implement it explicitly, delete dead clauses. | **resolved 2026-08-31** — intended trigger is not-rank-1 (recall@1 rationale); implemented explicitly, dead clauses deleted |
| ECC-31 | §8 | Minor | security | `projects/mlops/pipelines/components/{shap_explain,evaluate_test,fairness_audit}.py`, tests | **CWE-502 scan (2026-08-30):** model artifacts loaded via `joblib.load` (pickle) with **no integrity verification** — deserialization of potentially untrusted data if the artifact store/CI output can be tampered with (arbitrary code execution on load). Deployed serving uses xgboost `model.bst` (native format, lower risk); the training/eval components are the exposure. `yaml.safe_load` in tests is safe; `exec()` only in `Archive/` (archived). | Verify artifact integrity (hash/signature) before `joblib.load`; prefer native formats; treat the artifact store as trusted-and-verified. Owned by §8. | open |
| ECC-32 | §4 | Critical | correctness | `mcp_server/tools/rag_search.py` (`_chunk_texts_for`) vs `pipelines/components/chunk_notes.py` + `rag_ingest_pipeline.py` (pack_to=700) | **Build/serving chunker mismatch (confirmed; upgraded to Critical by the cross-check):** the deployed index is built with `pack_to=700`, but serving `_chunk_texts_for` re-chunks with default `pack_to=None` → different chunk_ids on any packed (long) section → `chunk_texts.get(...).get(nb.id)` misses → **silent whole-note fallback** returning the ENTIRE note — including non-indexed metadata sections (name, unit no, dates) — as the "cited passage". `count_hybrid_chunks.py`'s own numbers prove the corpora diverge (307 unpacked vs 136 packed chunks). No log/flag on the fallback. | Thread build-time `pack_to`/`max_chars` through serving config (single shared constant); log/flag the whole-note fallback so drift is observable. | **resolved 2026-08-31** — `rag.chunking.DEFAULT_PACK_TO = 700` single-sourced; `_chunk_texts_for` chunks with build params (`DEFAULT_MAX_CHARS`/`DEFAULT_PACK_TO`); fallback logged + tagged. NOTE: chunker fixes (ECC-41/34/44) shift chunk ids — requires one index rebuild + mcp redeploy before live citations resolve to chunks again |
| ECC-33 | §4 | Minor | maintainability | `scripts/build_chunks.py` (`DEFAULT_SECTIONS`), `mcp_server/tools/rag_search.py` (`_KNOWN_SECTIONS`), `rag/sections.py` (`KNOWN_HEADINGS`) | The section vocabulary is **duplicated in three hand-maintained lists**: build whitelist (13), serving `_KNOWN_SECTIONS` (13), parser canonicals (~29). They can silently diverge → sections dropped at build or unparseable at serving (ties ECC-23). | Derive all three from one source (e.g. `rag.sections` canonical names + a single whitelist constant); add a consistency test. | **resolved 2026-08-31** — `rag.chunking.INDEX_SECTIONS` is the single source (validated against `KNOWN_HEADINGS` at import + in tests); build whitelist, pipeline component, scripts, and serving all import it |
| ECC-34 | §4 | Minor | correctness | `rag/chunking.py` (`_emit_fragment` fixed-width fallback) | The fixed-width fallback for run-on sentences cuts at `max_chars` **mid-word** (no word-boundary awareness), producing chunks that start/end mid-token — degraded retrieval and a citation that points at a text fragment beginning mid-word. | Split on the last whitespace ≤ max_chars before falling back to hard cuts. | **resolved 2026-08-31** — fixed-width fallback now cuts at the last whitespace in the window (hard cut only for single giant tokens) |
| ECC-35 | §4 | Minor | correctness | `agent/guardrail.py` (`redact_invented_age`), `rag/chunking.py` (redaction drop) | The deterministic redaction guard covers **only age**. Other `___` redactions (dates, names, facility, doses) reach the model via chunks and rely on the prompt alone ("never fill a redaction") — consistent with the ECC-04 theme (no hard backstop for the risk-critical fields that matter most). | Extend the redaction guard to other high-value fields (dates, names); at minimum flag when an answer contains a concrete value the source redacted. | **resolved 2026-08-31** — `flag_invented_dates`: when retrieved passages carry `___` redactions, any concrete calendar date in the answer not found verbatim in a passage is flagged `redacted_date_filled` (flag-only; names/facility remain prompt-governed) |
| ECC-36 | §4/§5 | Major | security | `scripts/deploy_index.py`, `scripts/deploy_rag_endpoint.py` (`public_endpoint_enabled=True`) | The Vector Search index endpoint is created with a **public endpoint** (no PSC/VPC) — internet-reachable, gated only by IAM, inconsistent with the private agent/MCP posture (upgraded per adversarial pass). Combined with ECC-53 (a code path can deploy the real MIMIC-derived index), a public endpoint is an unnecessary exposure and a DUA/data-governance risk. | Use a private endpoint (PSC/VPC), or enforce that only synthetic-note indexes can be deployed to the public endpoint (vector-count guard in the shared deploy path, not just the wrapper). | **resolved 2026-08-31** — shared `_deploy_guard.assert_synthetic_scale` refuses >100k-vector indexes to the public endpoint, now enforced in the shared deploy path (`deploy_rag_endpoint.py`) and `deploy_index.py`, not just the `deploy_synthetic_rag.py` wrapper |
| ECC-37 | §4 | Major | security | `pipelines/components/chunk_notes.py` (L55–61), `scripts/count_hybrid_chunks.py`, `scripts/prune_rag_datapoints.py` | **CWE-89 surface on the pipeline side:** `split_name`, `notes_table_ref`, `split_table_ref` are runtime pipeline params interpolated directly into SQL f-strings (`WHERE a.split_name = '{split_name}'`), unlike serving which parameterizes. Exploitability low (operator-controlled params at submit) but it's an unnecessary injection surface and repointing risk (ties ECC-29). | Use `ScalarQueryParameter` for `split_name`; validate table refs against a strict pattern/allowlist. | **resolved 2026-08-31** — chunk_notes binds `split_name` as a ScalarQueryParameter and validates both table refs against a strict project.dataset.table pattern; count_hybrid_chunks parameterized; prune_rag_datapoints binds hadm_ids as ARRAY<INT64> UNNEST; fake-client test proves the value never reaches SQL text |
| ECC-38 | §4 | Major | correctness | `scripts/embed_chunks.py`, `pipelines/components/embed_chunks.py` | `zip(batch, resp.embeddings)` **silently truncates** when the embed API returns fewer embeddings than inputs — chunks vanish from the ingest with no error; the per-record dim check runs after truncation and can't catch it. Only backstop is the index count assert (which is hardcoded — ECC-40). | `zip(..., strict=True)` or assert `len(resp.embeddings) == len(batch)`; fail the batch. | open |
| ECC-39 | §4 | Major | ops | `scripts/embed_chunks.py` (L148–154, L246–257) | **Inverted exit semantics:** failed embed batches print + `failed += 1` + continue (run uploads to GCS, exits as if fine), while `main()` returns a dict so `SystemExit(main())` exits **1 on success**. A run with dropped batches can look successful; a clean run exits nonzero — poison for wrapping automation. (The pipeline component raises correctly.) | `return 0 if failed == 0 else 1`; skip `upload_ingest()` when `failed > 0`. | open |
| ECC-40 | §4 | Major | ops | `scripts/deploy_index.py` (L166) | `expected = 555770` is a **hardcoded magic number** for the original MIMIC corpus. Wrong for the current hybrid corpus (136 chunks per `count_hybrid_chunks.py`) or any rebuild — the script either fails spuriously or, if the number ever coincides, passes wrongly. No `--expected` flag, no derivation from the ingest it just read. | Count records in the ingest source (already streamed for the brute-force sample) or require `--expected`. | open |
| ECC-41 | §4 | Major | correctness | `rag/sections.py` (`_BARE_HEADING_RE` + single-word aliases "History", "Medications", "Condition", "Recommendations", "Follow Up") | The second parsing pass turns **any** line consisting solely of a common single-word alias into a section boundary — reintroducing the truncation the module docstring says it avoids ("under-splitting is the safer error"), and changing chunk ordinals/ids corpus-wide when notes contain such lines. | Restrict the bare-heading pass to multi-word or uppercase-only aliases, or a per-alias opt-in list. | **resolved 2026-08-31** — bare-heading pass now requires fully-uppercase lines (the MTSamples shape); mixed-case alias lines stay in the body |
| ECC-42 | §4 | Major | ops | `scripts/prune_rag_datapoints.py` (L64–83) | `remove_datapoints` failure is caught+printed and execution **continues** to delete the BigQuery rows, then exits 0; removal is also submitted async with no completion verification. Result: stale vectors stay queryable (violating the "nothing excluded remains" contract) while source rows are gone → later retrieval trips `missing_text`. | Abort (nonzero) before the BQ deletes when index removal fails/can't be confirmed. | open |
| ECC-43 | §4 | Minor | correctness | `mcp_server/tools/rag_search.py` (`rag_search_sections`, L389–404) | `rag_search_sections` builds citation ids from the **unpacked** serving re-chunk (ECC-32), so for long sections the ids name datapoints that don't exist in the index — the agent surfaces citations that can't be traced to a retrieval result. `picks[0]` cites only the first fragment of any long section, and `score: 1.0` implies a full-confidence retrieval that never happened. | Chunk with build parameters; mark these passages as deterministic (non-retrieved) rather than `score 1.0`. | **resolved 2026-08-31** — re-chunks with build params so ids match the index; passages carry `"retrieval": "deterministic"` instead of a fabricated score |
| ECC-44 | §4 | Minor | correctness | `rag/chunking.py` (`_pack`) | Placeholder-only pieces are filtered before packing, but `_pack` merges spans as `(first start, last end)` — a filtered redaction-only piece **between** two kept pieces is re-included in the packed chunk text. Redacted runs still get embedded inside packed chunks. | Build packed spans from the kept pieces' boundaries only. | **resolved 2026-08-31** — `_pack` refuses to merge across a gap whose body text is non-whitespace (i.e. a filtered piece) |
| ECC-45 | §4 | Minor | ops | `rag/notes.py` (`iter_chunks`) | Only the **notes** cache is manifest-verified; `iter_chunks` has no count check — a truncated `chunks.jsonl.gz` reads as a short-but-valid corpus. Also `iter_notes`'s verification only fires on full exhaustion (an early `--limit` break skips it). | Add a manifest/count check to `iter_chunks`; verify on open, not only on exhaustion. | open |
| ECC-46 | §5 | Critical | ops/security | `cloudbuild.agent.yaml`, `cloudbuild.mcp.yaml`, site `cloudbuild.yaml`, `danielmherman/settings.py` | **Agent/MCP + site runtime env config is unversioned, and the prod/dev switch fails open** (upgraded to Critical per adversarial pass): `ENVIRONMENT` defaults to `development` → `DEBUG=True`, insecure fallback `SECRET_KEY`, SQLite on any deployment missing the var; and no deploy step sets any runtime env (`ENVIRONMENT`, `ALLOWED_HOSTS`, `CSRF_TRUSTED_ORIGINS`, `CLOUD_SQL_CONNECTION_NAME`, `GS_BUCKET_NAME`, `DEMO_AGENT_URL`, `DEMO_FIXTURE_MODE`). Recreating a service from the repo (the exact scenario the cloudbuild comment defends against) produces a public debug site / misconfigured private services. Site side = S1-01; agent/MCP config lives only in console state. | Fail closed (`ENVIRONMENT` default `production` or hard-fail if unset); declare all env vars in the deploy step via `--set-env-vars`/`--env-vars-file` (secrets via `--set-secrets`); codify agent/MCP deploy + env config in the repo; add a drift check. **RESOLVED 2026-08-31:** site — ENVIRONMENT required + validated, full runtime env declared on the deploy step and both jobs; ECC — cloudbuild.agent.yaml/cloudbuild.mcp.yaml now build AND deploy with SA, --no-allow-unauthenticated, and env codified (agent resolves MCP_URL at deploy time). | resolved |
| ECC-47 | §5 | Major | ops | `projects/mlops/scripts/deploy_cpr.py` (undeploy loop) | `deploy_cpr.py` undeploys **ALL** stale models on the endpoint before deploying the new one — a **guaranteed serving outage window on every redeploy**, and if the new deploy fails (bad bundle, quota, image), the endpoint is left empty with no rollback and the previous model's exact deployment config unrecorded (upgraded per adversarial pass). | Deploy the new model first (`traffic_percentage=100`), then undeploy stale only after the new deployment succeeds; add a rollback path. | **resolved 2026-08-31** — deploy_cpr.py deploys the new model at 0% traffic, shifts 100% via `ep.update(traffic_split=…)`, then undeploys stale; any failure leaves the previous deployment serving (rollback = no-op) |
| ECC-48 | §5 | Minor | security | `projects/mlops/scripts/register_serving_model.py` | Default `BUNDLE_URI` hardcoded with **real resource IDs** (pipeline-root path containing the project number, run id, model id) + hardcoded `PROJECT`. Internal topology ships in source (ties ECC-15). | Remove the hardcoded default bundle; require `BUNDLE_URI`. **RESOLVED 2026-08-31:** BUNDLE_URI required (arg or env); PROJECT via src.config. | resolved |
| ECC-49 | §5 | Minor | ops | `scripts/teardown.py` (`VECTOR_ENDPOINT_PREFIX="readmission"`) | Teardown matches Vector index endpoints by prefix `readmission` — a future non-demo resource whose name starts with that prefix would be undeployed+deleted by `teardown --yes`. Destructive with broad matching. | Match by exact display name (or an explicit allowlist); require `--yes` confirmation with a listing. | **resolved 2026-08-31** — teardown.py matches the Vector Search endpoint by EXACT display name (`readmission-rag-index`, from config), not the `readmission` prefix; the existing `--yes`/interactive confirmation still lists resources before deleting |
| ECC-50 | §5 | Minor | ops | `projects/agent-harness/.env.lanfuse` | **Typo'd filename** — should be `.env.langfuse` (missing 'g'). Gitignored (`.gitignore:24 .env.*`) + mode 600 → **no committed secret** (positive), but misnamed: confusing, and any future dotenv loader expecting the correct name silently won't find the keys. | Rename to `.env.langfuse`; keep it gitignored. | open |
| ECC-51 | §5 | Minor | security | `danielmherman/settings.py` (`get_secret` → `versions/latest`) | No secrets rotation strategy; Secret Manager reads pinned to `latest`, so a rotated secret takes effect only on the next redeploy, and there's no rotation workflow. | Pin secret versions where appropriate; document a rotation procedure; note the rotate-then-redeploy behavior. | open |
| ECC-53 | §5 | Major | security/ops | `scripts/deploy_rag_endpoint.py` (defaults), `scripts/launch_endpoints.sh` | **High-cost + DUA footgun defaults** (upgraded per adversarial pass): `INDEX_ID`/`INDEX_MACHINE_TYPE` default to the **555k-vector index built from real MIMIC-derived notes on e2-standard-16 (~$270/mo)**, and the documented bare usage deploys it to a **public** endpoint. This is both a cost footgun and a data-governance/DUA exposure for a repo that must never serve real MIMIC-derived text; the mitigation is a comment in a different file, not a safe default. | Remove the real-corpus defaults; require `INDEX_ID` explicitly (error if unset); default machine to `e2-standard-2`; add the same >100k-vector refusal guard `deploy_synthetic_rag.py` has. | **resolved 2026-08-31** — `INDEX_ID` is required (no default), machine defaults to `e2-standard-2`, and the shared >100k-vector guard runs in the shared deploy path; `launch_endpoints.sh` comment updated |
| ECC-54 | §5 | Major | ops | `projects/mlops/scripts/deploy_endpoint.py` (FeatureOnlineStore default), `scripts/teardown.py` | `deploy_endpoint.py`'s default path creates a Bigtable-backed **FeatureOnlineStore** (`min_node_count=1`, autoscale 3 — always-on hourly Bigtable meter ~$0.65+/hr) unless the operator passes `--skip-feature-view`; **`teardown.py` ("the single biggest cost lever") does not know about or tear it down**. An operator running the canonical teardown believes billing is stopped while Bigtable keeps metering. (CPR may have superseded this path — if so, delete the FeatureView path rather than keep it runnable-but-uncovered.) | Add FeatureOnlineStore teardown to `teardown.py` (and its docstring inventory), or remove the FeatureView path; make feature-view creation opt-in. | **resolved 2026-08-31** — the FeatureView path was REMOVED from deploy_endpoint.py entirely (the Vertex Feature Store was a retired dev experiment; BigQuery is the only feature source, `FEATURE_SOURCE=bigquery`) |
| ECC-55 | §5 | Major | ops | `projects/mlops/pipelines/serving/cpr/Dockerfile` + `deploy_cpr.py` (`image_tag`) | The CPR image's **content-addressing is false**: tag = hash of Dockerfile+predictor.py+requirements.txt, but the Dockerfile pulls a mutable base (`python:3.10`) and `--force-reinstall`s an unpinned SDK floor (`google-cloud-aiplatform[prediction]>=1.27.0`). Same content hash → materially different images; a stale cached image is silently reused forever; a `--force-build` silently changes serving behavior under the same tag. The reproducibility guarantee the tagging scheme claims does not hold. | Pin the base image by digest and the SDK to an exact version in the hashed requirements; drop `--force-reinstall`. | open |
| ECC-56 | §5 | Major | ops | `danielmherman/cloudbuild.yaml` (migrate + seed `--prune` before deploy) | The pipeline mutates the production DB **twice before the new revision is deployed**: migrations, then a destructive `--prune` cohort seed. If the deploy fails/cancels, production is left on the new schema + pruned cohort while the **old revision keeps serving**, with no rollback. The comment only considers the reverse ordering hazard. | Make migrations backward-compatible (expand/contract) or deploy `--no-traffic` → migrate/seed → promote; at minimum never prune before the revision that expects the pruned cohort is live. | **resolved 2026-08-31** — site cloudbuild.yaml now deploys `--no-traffic` first, then migrate + seed, then promotes via `update-traffic --to-latest`; rollback = don't promote |
| ECC-57 | §5 | Minor | correctness | `scripts/teardown.py`, `deploy_cpr.py`, `deploy_synthetic_rag.py`, `deploy_index.py` (vs `deploy_rag_endpoint.py`, `wait_rag_deploy.py`, `config.py`) | Environment-override behavior is **inconsistent**: some scripts honor `PROJECT_ID`, others hardcode it. An operator targeting a second project deploys there but `teardown.py` silently operates on the hardcoded project — resources keep billing in one project while teardown reports success against another. | Read project/location from one shared config module (or a required env) in every script. **RESOLVED 2026-08-31:** agent-harness scripts import mcp_server.config; mlops scripts use src.config.get_project_id() — no per-script hardcodes remain in the deploy path. | resolved |
| ECC-58 | §5 | Minor | security | `cloudbuild.yaml` (site), `cloudbuild.agent.yaml`, `cloudbuild.mcp.yaml` | **No `serviceAccount` in any Cloud Build config** → builds run as the project default SA. The site pipeline needs run.admin + `iam.serviceAccountUser` on `website-sa` on that default SA — so any build submission in the project can execute arbitrary code as an identity that deploys the production site and runs DB-mutating jobs. | Create a dedicated deploy SA with minimal roles; set `serviceAccount:` in each cloudbuild; keep image-only builds on a build SA with Artifact Registry write only. | **resolved 2026-08-31** — top-level `serviceAccount: cicd-deployer` added to site `cloudbuild.yaml`, `cloudbuild.agent.yaml`, and `cloudbuild.mcp.yaml` (roles documented: run.admin, iam.serviceAccountUser on the runtime SA, artifactregistry.writer, logging.logWriter); SA creation is in GCP_DEPLOYMENT_GUIDE §18c |
| ECC-59 | §5 | Minor | correctness | `projects/mlops/scripts/register_serving_model.py` (`DEFAULT_IMAGE`) | Registers models against a **mutable `:latest` serving container** (`xgboost-cpu.2-1:latest`). A model registered today and redeployed after teardown may serve on a different container than it was validated on, silently — undermining the provenance the model registry is explicitly kept for. | Pin the pre-built container to a versioned tag or digest. | open |
| ECC-60 | §8 | Major | security | `pipelines/serving/cpr/predictor.py` (`preprocess`) | **Zero input validation on the CPR endpoint** (upgraded per cross-check): dict instances with a typo'd/missing key silently become NaN → XGBoost treats NaN as "missing" and returns a **confident, silently wrong probability** (the worst failure for a clinical score); positional lists of wrong length → raw shape error / opaque 500; `float(v)` accepts `inf`/`nan`/booleans and raises unhandled `ValueError` on other garbage; no batch cap. | Validate per instance (dict keys ⊆ feature_order with unknown-key rejection; list length == feature_order; reject non-finite); 400-style structured errors; cap batch size. | **resolved 2026-08-31** — `preprocess` validates every instance: unknown dict keys rejected, list length must match, values must be finite numbers or null (bools rejected), batch capped at `MAX_BATCH=100`; structured ValueErrors with instance/feature context |
| ECC-61 | §8 | Major | security/ops | `pipelines/serving/cpr/predictor.py` (`load`), `register_model.py`, `serving/cpr/cloudbuild.yaml` | **End-to-end mutable supply chain** (merged ECC-62; upgraded per cross-check): `download_model_artifacts` fetches the bundle with **no checksum** (the bundle writes no hash manifest); provenance points at `:latest`; the CPR cloudbuild `_TAG` **defaults to `latest`** so a substitution-less build defeats the content-hash-tag design; unpinned base `python:3.10` + `google-cloud-aiplatform[prediction]>=1.27.0`. | Write SHA-256 digests into the bundle and verify in `load()`; record image by digest; make `_TAG` required (no default); pin base image. | open |
| ECC-63 | §8 | Major | security | `pipelines/components/load_data.py` | **All SQL inputs interpolated** (upgraded per cross-check): `full_table_ref`, `id_col`, feature cols, `label_col`, `split_col`, and the split-name string literals are f-string'd into BigQuery SQL with **no query parameters**. Operator-controlled params (runs as `PIPELINE_SA`), but a crafted `train_split` can exfiltrate any table the SA reads into training artifacts. | Pass split names via `query_parameters`; validate identifiers/table refs against a strict regex; feature cols already come from `encoding.feature_order()` (keep + document). | **resolved 2026-08-31** — splits bound as ARRAY<STRING> `UNNEST(@splits)`; `full_table_ref` + id/label/split cols validated against strict regexes before any client exists; feature-order provenance documented; injection tests added |
| ECC-64 | §8 | Critical | correctness (leakage) | `pipelines/components/load_data.py` (L59–90), whole pipeline | **No train/val/test patient-disjointness assertion.** The pipeline blindly trusts the upstream `split_name` column and never asserts subject_id disjointness across splits. All grouped-CV machinery protects only within-train/val; if any patient's admissions straddle train+test, the test AUCPR gate, stability check, and fairness audit are all contaminated. `id_col` is already available for every split — the assertion is one line of set arithmetic away. | In `run_load_data`, assert disjoint train/val/test id sets and hard-fail on overlap. | **resolved 2026-08-31** — `assert_patient_disjoint` runs on every load: any subject_id shared between two splits raises with pair + counts + sample ids |
| ECC-65 | §8 | Major | correctness (gate integrity) | `training_pipeline.py` (L104), `benchmark_gate.py`, `evaluate_test.py` | **The performance gates are bypassable:** `hospital_aucpr` is a runtime pipeline parameter — any submitter can pass `0.0` and neutralize both `benchmark_gate` and the eval hard-fail. The build-time source is an unversioned local JSON. No minimum margin/CI (a 0.0001 improvement passes). | Resolve the baseline inside the gate from a versioned, access-controlled artifact; record its provenance; require a margin. | **resolved 2026-08-31** — both gates `validate_baseline` (fail closed outside (0,1), so a `0.0` override hard-fails) and require `MIN_GATE_MARGIN=0.01`; build-time loader validates the versioned artifact (provenance fields: method, n_patients, generated_at_utc) |
| ECC-66 | §8 | Major | correctness (gate effectiveness) | `evaluate_test.py`, `optuna_hpo.py` | The val→test stability check is **warn-only** (no raise) — a badly overfit model still registers. The reference `hpo_val_aucpr` is `study.best_value` (max over ~50 trials — winner's curse, optimistically biased) on a 50% subsample / 3-fold CV — a different regime from the full-data final model, making the fixed 0.02 threshold statistically arbitrary. | Make stability a hard gate with a defensible reference (clean refit-CV estimate) or drop the "honest generalization" framing. | **resolved 2026-08-31** — stability now raises (hard gate). Defensibility: the reference is optimistically biased (winner's curse), so measured degradation is an upper bound — the gate can only over-trigger, never silently pass an overfit model |
| ECC-67 | §8 | Minor | correctness | `calibrate_threshold.py`, `train_final.py` | The operating threshold is selected from **OOF probabilities of CV models trained on train-only**, but applied to a **different model** refit on train+80% val with full `n_estimators` — the refit model's probability distribution shifts, so the served `threshold.json` is systematically off its F-beta optimum. | Recompute/verify the threshold on the final model's val predictions; report the F-beta the final model actually achieves at the shipped threshold. | **resolved 2026-08-31** — `evaluate_test` reports `fbeta_at_threshold` (final model at shipped threshold), `test_optimal_fbeta/threshold`, and the shortfall as diagnostics (threshold never re-tuned on test) |
| ECC-68 | §8 | Minor | correctness | `predictor.py` (`load`) | If `threshold.json` is absent (only written when `tuned_threshold is not None`), the CPR **silently falls back to 0.5** — for a recall-weighted F2 threshold (typically well below 0.5) this silently flips many readmit decisions to negative with no error/log. | Fail `load()` loudly if `threshold.json` is missing, or emit `"threshold_source": "default"` in every response. | **resolved 2026-08-31** — `load()` raises RuntimeError when `threshold.json` is absent; no 0.5 fallback |
| ECC-69 | §8 | Minor | ops | `_image.py`, component `packages_to_install` | Components install **unpinned** `xgboost`/`optuna`/`scikit-learn`/`pandas` on a `:latest` base (only Evidently pinned) — runs aren't reproducible over time; xgboost version parity with the CPR (pinned >=2.1,<2.2) is unenforced (doc-correction 6). | Use the pinned custom image as the default; pin versions; record package versions in the registry entry/manifest. | open |
| ECC-70 | §8 | Minor | correctness | `agent-harness/pipelines/rag_ingest_pipeline.py` (L149–156) | `enable_caching=True` on a pipeline whose first step reads BigQuery: the KFP cache key covers code+params, **not table contents** — if the source tables change, a re-submission silently reuses stale chunks/embeddings and builds an index that doesn't match the data. (The training pipeline correctly disables caching.) | Add a data-fingerprint parameter (table last_modified / row-count hash) to `chunk_notes` inputs so the cache key invalidates on data change. | open |
| ECC-71 | §8 | Minor | architecture | `training_pipeline.py` (register gated only by `.after(evalt)`); `register_model.py` | `shap_explain` and `fairness_audit` are **not upstream of `register_model`** — a crash in either leaves an already-registered model with no audit marker. Also `register_model` calls `Model.upload` with **no `parent_model`** (no version lineage) and the gate metrics are only **printed, never persisted** on the registry entry. | Add `.after(fairness)` (and shap) to `register_model` or record audit status/URIs; use `parent_model` for lineage; persist gate metrics as model labels/metadata. | **resolved 2026-08-31** — register waits on evalt + shap + fairness; optional `parent_model` pipeline param passed to `Model.upload`; gate metrics persisted as `gate_metrics.json` in the bundle + registry labels; YAML recompiled |
| ECC-72 | §8 | Minor | ops | `register_model.py`, `serving/cpr/cloudbuild.yaml`, `rag_ingest_pipeline.py` | GCP project ID `trim-icon-498815-a0` hardcoded in 3 more source files (image URIs + env fallbacks) while the training pipeline resolves from `src.config` — config split across sources (ties ECC-15). | Route all project/image references through `src.config` / required env with no baked defaults. **RESOLVED 2026-08-31:** register_model.py derives the CPR image from the run's project/location; cpr/cloudbuild.yaml _IMAGE has no default; rag_ingest_pipeline.py resolves via shared config. | resolved |
| ECC-73 | §8 | Minor | correctness | `predictor.py` (`postprocess`), `train_final.py` comment | `pred_contribs=True` on a `binary:logistic` booster returns contributions in **log-odds (margin) space**, but the response pairs them with a probability and `train_final.py`'s stale comment claims "probability units" — the response never declares units, and a consumer could misread attributions as probability deltas. | Declare attribution units in the response/manifest; fix the stale comment. | **resolved 2026-08-31** — every prediction carries `"attribution_units": "log_odds"`; module docstring + stale train_final comment corrected |

*(First findings land here. Severity: Critical / Major / Minor. Category:
security / correctness / architecture / ops.)*

## Cross-cutting bucket

Findings not tied to one section (shared utils, logging, error handling).

## Definition of done

All sections reviewed; **zero Critical/Major open** (fixed or documented
deferral with owner + date); scanners run; backlog tracked here.

## Cadence (per section)

1. Understand pass → **Understand doc** (written to be read: entry points, how
   it operates, data flow, config, failure modes).
2. **Dan reads + confirms** the mental model.
3. Primary review pass → findings.
4. Adversarial pass — Claude Fable 5, per the section's protocol (blind or
   cross-check per the mapping).
5. Triage into this backlog.

One section per sitting. Progress update this file as we go.
