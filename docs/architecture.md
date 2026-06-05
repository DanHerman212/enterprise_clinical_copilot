# Enterprise Clinical Copilot — Master Architecture

The shared **HOW (design)** of the monorepo: the common foundation every project builds on, and the seams where projects connect. Project-specific components live in each project's own architecture doc.

## Shared Foundation

_Placeholder — the infrastructure shared across all projects:_

- _GCP project and environment layout_
- _BigQuery (source datasets and shared tables)_
- _IAM, service accounts, and security boundaries_
- _Networking (VPC, VPC-SC)_
- _CI/CD and repository tooling_

## Project Seams

_Placeholder — the interfaces between projects: the model endpoint the agent calls, the vector index the RAG system serves, the contracts that connect them._

## Per-Project Architecture

| Project | Architecture |
|---|---|
| MLOps | [projects/mlops/docs/architecture.md](../projects/mlops/docs/architecture.md) |
| Agentic RAG | [projects/agentic-rag/docs/architecture.md](../projects/agentic-rag/docs/architecture.md) |
| Agent Engine + Harness | [projects/agent-harness/docs/architecture.md](../projects/agent-harness/docs/architecture.md) |

---
