# Security & Compliance

How this project handles a de-identified clinical dataset responsibly, and what is — and is
not — exposed publicly.

## Data governance (MIMIC-IV DUA)

The model is trained on **MIMIC-IV v3.1**, a de-identified critical-care dataset that
requires credentialed PhysioNet access and a signed Data Use Agreement. Core obligations:

- **No re-identification** of patients.
- **No redistribution** of raw data or derivatives.
- **No sharing of credentialed access.**
- **Research use** (commercial use requires permission).
- **Acknowledgment** of PhysioNet/MIMIC-IV in publications.

> This is a factual risk assessment, not legal advice. See `docs/mimic_dua_compliance.md`
> for the full checklist and caveats.

## Public surface = synthetic, real system

The public demo serves **synthetic data only** (a hybrid MTSamples cohort: real note text +
story-anchored features). MIMIC-IV feeds training and evaluation exclusively. The system
itself — agent pipeline, RAG index, predict path — is live; the **data** is not real MIMIC
content.

The deployed model (`model.bst`) is a trained function serving synthetic inputs only; it is
not a redistribution or derivative of MIMIC data under the DUA.

## No PHI in the repository

- Raw note passages, eval traces, and judged JSONL are **gitignored** (never committed).
- The repo contains code and schemas only; the `manifest.json` feature schema and aggregate
  stats are benign.
- `.df-credentials.json` was removed from tracking and gitignored (2026-09-05).
- Model binaries (`*.bst`) are gitignored; the served model lives in the Vertex Model
  Registry.

## IAM & secrets

- Secrets (keys, DB credentials, Langfuse keys) live in **Secret Manager**; nothing is
  committed.
- Deploys run as a least-privilege `cicd-deployer` service account.
- The agent and MCP server are private Cloud Run services; the Django BFF is the only
  public surface.

## Cross-patient isolation

- `rag_search` applies a **`hadm_id` restrict server-side on every query** — retrieval can
  never cross patients.
- `predict_readmission(hadm_id)` is a pure function of the admission ID; there is no
  patient-resolving endpoint exposed.

## Authentication & abuse control

- Django BFF handles login, per-user **quota**, and session state.
- `django-axes` rate-limits authentication; CSP middleware restricts content sources.
- A logout path is exposed; the agent is never directly reachable by the browser.

## Adversarial code review

A nine-part adversarial review lives in `docs/adversarial_code_review/`, covering auth/quota,
MCP cross-patient isolation, RAG citations, IAM/secrets/deployment, admin content, frontend
JS, and the Django config. Findings were triaged in `docs/remediation/TRIAGE_REGISTER.md`
(archived with the working notes).

## Compliance checklist

- [x] Public demo is synthetic-only; real MIMIC content never ships.
- [x] Raw note text is gitignored; eval JSONL stays out of the repo.
- [x] Secrets externalized to Secret Manager; credentials removed from tracking.
- [ ] Scrub/gitignore eval report JSONs containing `hadm_id`s + clinical fragments.
- [ ] Acknowledge PhysioNet/MIMIC-IV in any published write-up.
