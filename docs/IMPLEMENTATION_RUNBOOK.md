# Implementation Runbook — Enterprise Clinical Copilot

How to build and operate this system, step by step, from a fresh data warehouse
to a live demo. Each step records its goal, prerequisites, commands, and how to
verify success. This is the living record of the end-to-end path — written as
we execute it, and (in Step 8) folded into the repo's semantic layer so a new
engineer can reproduce the whole system from this document.

> **Status:** in progress. Steps are filled in as they are completed.
> **Last executed:** 2026-09-01.

---

## Step 0 — Finish the remediation sweep

**Goal:** close the 26 remaining Minor findings so Step 8 (cleanup) starts from
a fully-resolved adversarial code review (currently 0 Critical / 0 Major).

**Prerequisites:** clean working trees in both repos; nothing pushed.

**Scope:** clusters A, C, D, I, K, N, Z (26 items, grouped below).

### Z — misc (small)
- [x] ECC-17 — `agent/graph.py`: `handler.last_trace_id` via `getattr`.
- [x] ECC-50 — renamed `.env.lanfuse` → `.env.langfuse` (gitignored).
- [x] S1-17 — `# nosec` on the local-only gcloud subprocess; test passwords documented false positives.
- [x] S9-04 — UBLA caveat + bucket-IAM command documented (switch sequenced with IaC, Step 6).
- [x] S9-06 — LOGGING now includes the `demo` logger.
- [x] S9-07 — DEBUG static serves the live `static/` sources.

### A — config/ops
- [x] ECC-51 — Secret Manager rotation procedure documented (settings + GCP guide §7a).
- [x] S1-18 — `manage.py check --deploy --fail-level WARNING` added to the cloudbuild migrate job.

### C — spend
- [x] S1-08 — global budget / kill switch tracked in deployment_strategy.md; carried as a Step 7 gate before any public window.

### D — vocabulary
- [x] S7-02 — client `SECTION_ALIASES` documented as a copy of the ECC canonical; consolidation sequenced with Step 8.

### I — deps & supply chain
- [x] ECC-16 — agent/Dockerfile base pinned by digest; requirements exact-pinned.
- [x] ECC-31 — joblib artifacts hash-verified (dump/load sidecar).
- [x] ECC-59 — register_serving_model requires SERVING_IMAGE (no :latest).
- [x] ECC-69 — component deps pinned to major ranges; xgboost parity with CPR.
- [x] S6-09 — CDN assets SRI-pinned; mermaid pinned @10.9.3.

### K — content & contact form
- [x] S6-08 — `section_slug` reserves 'preview' (maps to 'preview-section').
- [x] S6-12 — contact whitespace/retention pinned by regression tests.

### N — front-end correctness
- [x] S7-04 — non-numeric probability renders `NaN%`.
- [x] S7-05 — markdown regexes run on escaped text.
- [x] S7-11 — mixed/reversed citation ranges silently disappear.
- [x] S7-12 — section start-match not line-anchored (client + server copies).
- [x] S7-13 — cache-bust version skew (custom demo 5 revisions stale).
- [x] S7-14 — unscored patients render `NaN%`.
- [x] S7-15 — source lookup by query equality.
- [x] S7-16 — stale trace pane leaks previous patient's envelope.
- [x] S7-17 — missing SourceCard → cite click no-op.

**Verify:** ECC `pytest` and site `manage.py test` suites stay green; register
snapshot shows 0 Critical / 0 Major / 0 Minor open.

---

## Step 1 — Rebuild the dataset (data warehouse)

**Goal:** regenerate the encoded analytics dataset and the demo-cohort tables
(hybrid notes/features/split) that everything downstream reads.

**Prerequisites:** BigQuery access as the pipeline/service account.

**Commands:**
```bash
# Regenerate the encoded view from the single source of truth (committed SQLX).
cd projects/mlops && ../../.venv/bin/python -m src.encoding --emit-sql

# Install Dataform deps (once; node_modules/ is gitignored) then run the ELT graph.
cd /Users/danherman/Desktop/enterprise_clinical_copilot
npx --yes @dataform/cli@3.0.0 install
npx --yes @dataform/cli@3.0.0 run
```

**Verify:** row counts match; encoding view regenerated from `src.encoding`.
- ✅ `cohort` = 352,699 rows (matches the expected count); split 70/14/14/1/1
  (train/validation/test/prod_test/demo) in `analytics_dataset_encoded`.
- ✅ all 3 assertions pass (`split_is_disjoint` + the two `rowConditions`).
- ✅ encoded view byte-identical to `python -m src.encoding --emit-sql` output.
- Hybrid demo-cohort tables (`readmission.hybrid_notes/split/features`) already
  in sync (89 rows = the canonical artifact); re-scored after Step 2 retrains.

---

## Step 2 — MLOps: train + deploy a new model

**Goal:** train on the rebuilt dataset and deploy a fresh CPR model to the
Vertex endpoint.

**Prerequisites:** rebuilt dataset; CPR image rebuilt (see below).

**Commands:**
```bash
.venv/bin/python projects/mlops/pipelines/training_pipeline.py submit   # or the KFP submit path
.venv/bin/python projects/mlops/scripts/deploy_cpr.py
```
**Note:** the next model registration writes `checksums.json` (verified by the
predictor at load, ECC-61). The CPR image must be rebuilt first — its base is
digest-pinned and the SDK is pinned to `==1.161.0` (ECC-55).

**Verify:** `smoke_test.py` returns a calibrated probability + attributions.

---

## Step 3 — RAG: rebuild + deploy a new index

**Goal:** chunk, embed, build, and deploy the Vector Search index.

**Prerequisites:** `rag_ingest_pipeline.yaml` recompiled (a `data_fingerprint`
param was added to the pipeline — ECC-70).

**Commands:**
```bash
.venv/bin/python projects/agent-harness/pipelines/rag_ingest_pipeline.py submit
.venv/bin/python projects/agent-harness/scripts/deploy_synthetic_rag.py
```
**Note:** `deploy_synthetic_rag.py` auto-picks the newest synthetic `rag-tree-ah-*`
index; the shared guard refuses anything >100k vectors (ECC-36/53).

**Verify:** `EXPECTED_VECTORS` matches; the index endpoint serves the demo cohort.

---

## Step 4 — Redeploy the agent + MCP images

**Goal:** rebuild and redeploy both private services from their cloudbuild configs.

**Prerequisites:** `cicd-deployer` build service account exists (Step 0/6 of the
deploy handoff); Artifact Registry access.

**Commands:**
```bash
gcloud builds submit --config projects/agent-harness/cloudbuild.mcp.yaml   --substitutions _IMAGE=us-east1-docker.pkg.dev/<PROJECT>/readmission/mcp-server:latest projects/agent-harness
gcloud builds submit --config projects/agent-harness/cloudbuild.agent.yaml --substitutions _IMAGE=us-east1-docker.pkg.dev/<PROJECT>/readmission/agent:latest projects/agent-harness
```
**Verify:** `/health` on both; `MCP_URL` resolved at agent deploy time.

---

## Step 5 — Site deploy (Cloud Build)

**Goal:** deploy the Django site with migrations, cohort seed, and traffic promote.

**Prerequisites:** `cicd-deployer` SA (cloudbuild pins it); push to `main`.

**Commands:** `git push` (triggers `cloudbuild.yaml`: build → push →
`--no-traffic` deploy → migrate → seed `--prune` → promote).

**Verify:** site loads; new requirements (nh3/django-csp/django-axes) installed;
migration 0003 applied.

---

## Step 6 — Automation (IaC + Prefect)

**Goal:** make the deploy reproducible (IaC) and orchestrate recurring work (Prefect)
where it removes manual toil.

**Scope:** Terraform/Bicep for GCP resources; Prefect for training/RAG workflows.
*(filled in as we execute)*

---

## Step 7 — Confirm + optimize

**Goal:** live-test the rebuilt system end to end and tighten cost/latency/config.

**Gate (S1-08):** before any public window, confirm the per-user quota AND the
kill-switch plan from `deployment_strategy.md` §3.2/Part 4 (budget alerts,
`DemoAccess.kill_switch`) are in place.

*(filled in as we execute)*

---

## Step 8 — Cleanup (repo + semantic layer)

**Goal:** repo hygiene and a coherent semantic layer — consistent naming,
structure, and docs, with this runbook integrated as the canonical guide.

**Carried items:** S7-02 — consolidate the client `SECTION_ALIASES` with the
server vocabulary (server-emitted JSON) so the section list lives in one place.

*(filled in as we execute)*

---

## Step 9 — Content

**Goal:** fill in the website's public content (articles, projects, resume).

**Prerequisites:** everything above confirmed working.

*(filled in as we execute)*
