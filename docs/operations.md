# Operations

How to build, deploy, monitor, and tear down the system. The full step-by-step runbook is
`docs/IMPLEMENTATION_RUNBOOK.md`; this is the operational summary.

## Environment

- **Project:** `trim-icon-498815-a0` (GCP), region `us-east1`.
- **Python:** `.venv` (run GCP scripts with `env -u PROJECT_ID` to avoid a stale local
  `PROJECT_ID`).
- **Dataform:** `npx @dataform/cli@3.0.0` (node_modules gitignored).

## Build order (cold start)

1. **Dataset** — regenerate the encoded view from committed SQLX, then run the Dataform
   graph (`cohort → split → features → marts`). Assertions (`split_is_disjoint`) must pass.
2. **Model** — submit the Vertex training pipeline, then `deploy_cpr.py` to register and
   deploy the Custom Prediction Routine.
3. **Retrieval** — submit the RAG ingest pipeline, then `deploy_synthetic_rag.py` to deploy
   the Vector Search index.
4. **Agent + MCP** — rebuild both Cloud Run images via their `cloudbuild.*.yaml` configs.
5. **Site** — push to `main`; `cloudbuild.yaml` builds, migrates, seeds, and promotes.

## Key scripts

| Script | Purpose |
|---|---|
| `scripts/setup_environment.sh` | One-time environment bootstrap |
| `scripts/copy_mimic.sh` | Mirror MIMIC-IV tables into BigQuery |
| `scripts/launch_endpoints.sh` | Start local serving endpoints for development |
| `mlops/serving/deploy_cpr.py` | Register + deploy the risk model |
| `scripts/agent/deploy_rag.py` | Deploy the retrieval index |
| `scripts/agent/generate_hybrid_features_v2.py` | Regenerate demo-cohort features |

## Evaluation loop

1. **Collect** traces against the agent (`eval/collect.py`).
2. **Judge** with the LLM-as-judge against the versioned rubric (`eval/judge.py`) — writes
   `judged.jsonl` + a `golden_report.json`.
3. Attach scores to Langfuse (see the v4 note below).

## Monitoring

- **Model:** Vertex Model Monitoring on input + attribution drift; a scheduled job
  recomputes AUCPR on matured labels; a Pub/Sub topic carries the retraining signal.
- **Agent:** Langfuse traces (self-hosted) with rubric scores.

## Teardown

`scripts/agent/teardown.py` undeploys and deletes the billable Vertex
endpoints (predict endpoint + vector index endpoint) while keeping the model registry, index
artifacts, GCS, and BigQuery. Rebuild with the deploy scripts above.

## Langfuse (self-hosted, v4)

Runs as Cloud Run services (`langfuse-web`, `langfuse-worker`) fronted by
`observability.danielmherman.com`, with ClickHouse as the event store on a small VM.

> **Note:** the deployment is in v4 `events_only` mode, which disables the classic REST
> API. The eval score-attachment path (`judge.py` → `create_score`) predates v4 and is
> being migrated to the OTel-native scoring path — see
> `services/docs/langfuse_v4_migration_plan.md`. Until then, eval scores are
> durably archived as local JSONL/JSON.
