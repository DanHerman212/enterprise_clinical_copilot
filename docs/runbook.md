# Enterprise Clinical Copilot — Master Runbook

The shared **HOW (execution)** of the monorepo: one-time repository and cloud bootstrap that every project depends on. Project-specific steps live in each project's own runbook, written as each step is executed.

## Repository Bootstrap

Clone the repository and create a local Python environment:

```bash
git clone https://github.com/<owner>/enterprise_clinical_copilot.git
cd enterprise_clinical_copilot
python3 -m venv .venv
source .venv/bin/activate
```

## Cloud Bootstrap

One-time Google Cloud setup: enables the required APIs, creates the pipeline service account and its IAM roles, and provisions the storage areas (GCS bucket + BigQuery dataset).

1. Authenticate with Google Cloud:

   ```bash
   gcloud auth login
   ```

2. Create your config file and set the project ID:

   ```bash
   cp .env.example .env
   # edit .env and set PROJECT_ID to your GCP project
   ```

3. Run the setup script:

   ```bash
   ./scripts/setup_environment.sh
   ```

The script is idempotent — safe to re-run. Configuration (region, bucket, dataset, service-account name) is read from `.env`; see [.env.example](../.env.example) for the available variables.

## Per-Project Runbooks

| Project | Runbook |
|---|---|
| MLOps | [projects/mlops/docs/runbook.md](../projects/mlops/docs/runbook.md) |
| Agentic RAG | [projects/agentic-rag/docs/runbook.md](../projects/agentic-rag/docs/runbook.md) |
| Agent Engine + Harness | [projects/agent-harness/docs/runbook.md](../projects/agent-harness/docs/runbook.md) |

---
