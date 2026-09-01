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
- [ ] S7-02 — client `SECTION_ALIASES` is a 4th hand-maintained copy.

### I — deps & supply chain
- [ ] ECC-16 — `agent/Dockerfile` mutable base tag.
- [ ] ECC-31 — `joblib.load` with no artifact integrity (CWE-502).
- [ ] ECC-59 — `register_serving_model` registers against mutable `:latest`.
- [ ] ECC-69 — unpinned xgboost/optuna/sklearn/pandas in components.
- [ ] S6-09 — CDN scripts without SRI; mermaid on floating `@10`.

### K — content & contact form
- [ ] S6-08 — section slug `preview` shadowed by the staff preview route.
- [ ] S6-12 — contact error path loses input; whitespace-only values accepted.

### N — front-end correctness
- [ ] S7-04 — non-numeric probability renders `NaN%`.
- [ ] S7-05 — markdown regexes run on escaped text.
- [ ] S7-11 — mixed/reversed citation ranges silently disappear.
- [ ] S7-12 — section start-match not line-anchored (client + server copies).
- [ ] S7-13 — cache-bust version skew (custom demo 5 revisions stale).
- [ ] S7-14 — unscored patients render `NaN%`.
- [ ] S7-15 — source lookup by query equality.
- [ ] S7-16 — stale trace pane leaks previous patient's envelope.
- [ ] S7-17 — missing SourceCard → cite click no-op.

**Verify:** ECC `pytest` and site `manage.py test` suites stay green; register
snapshot shows 0 Critical / 0 Major / 0 Minor open.

---

## Step 1 — Rebuild the dataset (data warehouse)

**Goal:** regenerate the encoded analytics dataset and the demo-cohort tables
(hybrid notes/features/split) that everything downstream reads.

**Prerequisites:** BigQuery access as the pipeline/service account.

**Commands:** (filled in as we execute)

**Verify:** row counts match; encoding view regenerated from `src.encoding`.

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

*(filled in as we execute)*

---

## Step 9 — Content

**Goal:** fill in the website's public content (articles, projects, resume).

**Prerequisites:** everything above confirmed working.

*(filled in as we execute)*
