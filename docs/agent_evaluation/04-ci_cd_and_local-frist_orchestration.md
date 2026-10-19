# 04 — CI/CD & Local-First Orchestration

## 1. Executive Summary & Problem Statement
In our earlier engineering cycles, our deployment and evaluation workflows suffered from four major operational failure modes:
1. **Cloud Build as a Debugging Loop:** Treating Cloud Run Jobs and container image builds as the primary way to test harness, prompt, or fixture changes resulted in 15-minute turnaround cycles and slow iteration.
2. **Unfiltered Service Triggers:** The 2nd-gen regional (`us-east1`) Cloud Build triggers `deploy-agent-on-push` and `deploy-mcp-on-push` lack path filters (`includedFiles`)[cite: 1]. Consequently, pure documentation commits (`docs/**`) or harness-only edits (`evaluation/**`) fire full container builds and mint unnecessary production Cloud Run revisions[cite: 1].
3. **Unpinned Harness Job Definitions:** While `deploy-harness-on-push` builds and pushes `us-east1-docker.pkg.dev/trim-icon-498815-a0/ecc-eval/harness:${COMMIT_SHA}`, the stored Cloud Run Job definitions (`eval-adversarial` and `eval-clinical`) require explicit updates or synchronized pipeline orchestration so they never execute stale code against a newer agent revision.
4. **Uncoordinated Cloud Mutations:** Executing cloud jobs or mutating remote state without a formal local preflight and clean git-tree guard creates uncalibrated trace noise in Langfuse and orphaned artifacts in Cloud Storage.

This module establishes a strict **Two-Tier Orchestration Workflow**: a fast, zero-build **Tier A Inner Loop** on the developer laptop, and a SHA-pinned, path-filtered **Tier B Outer Loop** on Cloud Run Jobs.

---

## 2. The Two-Tier Orchestration Architecture

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ TIER A: INNER LOOP (Local CLI → Live Cloud Run Agent — Zero Container Build) │
│  1. In-Process Contract Suite: pytest tests/ -q (~2s, 0 network calls)       │
│  2. Local Pilot Rehearsal:     collect_http.py + judge.py from laptop        │
│                                against $AGENT_URL (1–3 patients, ~2 mins)    │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │ (Code committed & git status clean)
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ TIER B: OUTER LOOP (Path-Filtered Cloud Build + SHA-Pinned Cloud Run Jobs)   │
│  1. Trigger Isolation:  agent, mcp, and harness only build when their        │
│                         respective source trees are modified.                │
│  2. Clean-Tree Guard:   Refuse official pipeline execution if git working    │
│                         tree is dirty or local HEAD != origin/main.          │
│  3. SHA-Pinned Jobs:    Update eval-adversarial & eval-clinical to exact     │
│                         harness:$COMMIT_SHA before executing certification.  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Tier A: The Local-First Inner Loop

Never build a container or trigger a Cloud Run Job to verify a code change, fixture edit, or rubric tweak. Every verification starts on the local machine.

### 3.1 Step 1: In-Process Deterministic Contract Gate
Before making any network or LLM calls, run the fast unit and contract suite:

```bash
pytest tests/ -q
```
* **Requirement:** 100% pass rate (~900+ tests in < 5 seconds).
* **Coverage:** Validates `_trace_metadata`[cite: 1], `propagate_attributes`[cite: 1], JSONL parse-then-emit merging, `_expected_refusal` scoping, float/bool score aggregation, modality-aware prompt formatting (`prose=False`), and Adaptive 2-of-3 consensus logic.

### 3.2 Step 2: Local Pilot Rehearsal Against `$AGENT_URL`
When a change touches agent behavior, collector wiring, or judge calibration, rehearse it locally from `evaluation/agent/` against the deployed Cloud Run service using your local identity token:

```bash
export AGENT_URL="https://<your-agent-service-url>"
export BEARER_TOKEN="$(gcloud auth print-identity-token)"
export RUN_TAG="local-rehearsal-$(git rev-parse --short HEAD)"

# 1. Collect 1-patient pilot + Suite 2 locally
python evaluation/agent/collect_http.py \
  --url "$AGENT_URL" \
  --max-patients 1 \
  --suite2 \
  --run-tag "$RUN_TAG" \
  --out "evaluation/agent/results/traces_${RUN_TAG}.jsonl"

# 2. Judge locally (without uploading uncertified artifacts to GCS)
python evaluation/agent/judge.py \
  --mode clinical \
  --in "evaluation/agent/results/traces_${RUN_TAG}.jsonl" \
  --out "evaluation/agent/results/judged_${RUN_TAG}.jsonl" \
  --report "evaluation/agent/results/report_${RUN_TAG}.json" \
  --run-tag "$RUN_TAG"
```
* **Why this matters:** Edit-to-feedback takes ~90 seconds instead of 15 minutes, costs pennies, and leaves Cloud Run Job definitions untouched.

---

## 4. Tier B: CI/CD Trigger Hygiene & Path Filtering

To stop documentation commits (`docs/**`) and evaluation harness edits (`evaluation/**`) from rebuilding and redeploying production Cloud Run services (`agent` and `mcp`), all regional triggers in `us-east1`[cite: 1] must enforce strict `includedFiles` and `ignoredFiles` rules.

### 4.1 Path Filter Specification for `us-east1` Triggers

| Trigger Name | Build Config | Required `includedFiles` | Required `ignoredFiles` |
| :--- | :--- | :--- | :--- |
| **`deploy-agent-on-push`** | `services/agent/cloudbuild.yaml`[cite: 1] | `services/agent/**` | `docs/**`, `evaluation/**`, `**/*.md` |
| **`deploy-mcp-on-push`** | `services/mcp/cloudbuild.yaml`[cite: 1] | `services/mcp/**` | `docs/**`, `evaluation/**`, `**/*.md` |
| **`deploy-harness-on-push`** | `evaluation/agent/cloudbuild.yaml` | `evaluation/agent/**`, `services/agent/**` | `docs/**`, `**/*.md` (except `evaluation/agent/rubric.md`) |

*(Note: `evaluation/agent/rubric.md` must **not** be ignored by `deploy-harness-on-push`, as rubric updates are runtime inputs to `judge.py`).*

### 4.2 Verifying Trigger Builds in `us-east1`
Because all triggers are 2nd-gen regional triggers in `us-east1`, always include `--region=us-east1` when inspecting build state[cite: 1]:

```bash
gcloud builds list \
  --project trim-icon-498815-a0 \
  --region us-east1 \
  --limit 5 \
  --format="table(id,status,substitutions.TRIGGER_NAME,substitutions.COMMIT_SHA)"
```

---

## 5. Outer-Loop Job Pinning & Clean-Tree Guards

### 5.1 Separating "Build/Update" from "Execute"
A Cloud Run Job (`eval-adversarial`, `eval-clinical`) is a stored execution template pointing to a container image tag. 
* **Updating a job definition** (`gcloud run jobs update ... --image ...`) changes which image will run next, without incurring LLM or compute charges.
* **Executing a job** (`gcloud run jobs execute ...`) launches a live run, writes traces and scores to Langfuse, and seals artifacts to `gs://ecc-eval-artifacts/`.

### 5.2 Clean-Tree & SHA-Pinning Script (`evaluation/agent/run_certified_job.sh`)
To guarantee that no official Cloud Run Job ever runs against uncommitted local changes or a mismatched harness image, wrap outer-loop job executions in a deterministic preflight guard:

```bash
#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:?Usage: $0 <adversarial|clinical> <run_tag> [max_patients]}"
RUN_TAG="${2:?Usage: $0 <adversarial|clinical> <run_tag> [max_patients]}"
MAX_PATIENTS="${3:-82}"
REGION="us-east1"
PROJECT="trim-icon-498815-a0"

# 1. Clean working tree guard
if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: Git working tree is dirty. Commit or stash changes before running a cloud job." >&2
  exit 1
fi

# 2. Sync check with origin/main
git fetch origin main --quiet
LOCAL_SHA="$(git rev-parse HEAD)"
REMOTE_SHA="$(git rev-parse origin/main)"
if [[ "$LOCAL_SHA" != "$REMOTE_SHA" ]]; then
  echo "ERROR: Local HEAD ($LOCAL_SHA) does not match origin/main ($REMOTE_SHA). Push first." >&2
  exit 1
fi

IMAGE="us-east1-docker.pkg.dev/${PROJECT}/ecc-eval/harness:${LOCAL_SHA}"

# 3. Verify the exact SHA-pinned harness image exists in Artifact Registry
if ! gcloud artifacts docker images describe "$IMAGE" --quiet >/dev/null 2>&1; then
  echo "ERROR: Harness image $IMAGE not found in Artifact Registry. Wait for deploy-harness-on-push." >&2
  exit 1
fi

JOB_NAME="eval-${PHASE}"

# 4. Pin the job definition to the exact commit SHA image
echo "Pinning ${JOB_NAME} to ${IMAGE}..."
gcloud run jobs update "$JOB_NAME" \
  --project "$PROJECT" \
  --region "$REGION" \
  --image "$IMAGE" \
  --quiet

# 5. Execute the job with explicit parameters
echo "Executing ${JOB_NAME} with RUN_TAG=${RUN_TAG}, MAX_PATIENTS=${MAX_PATIENTS}..."
gcloud run jobs execute "$JOB_NAME" \
  --project "$PROJECT" \
  --region "$REGION" \
  --wait \
  --update-env-vars "RUN_TAG=${RUN_TAG},MAX_PATIENTS=${MAX_PATIENTS},GIT_SHA=${LOCAL_SHA}"
```

---

## 6. Rules of Engagement & Change Control Contract

To prevent unapproved cloud state mutations and keep engineering cadence predictable, the following operational boundaries govern all engineering work:

1. **"Implement / Fix / Commit" = Local Only:** Any directive to implement a feature, fix a bug, or update a rubric authorizes local code edits, running `pytest`, and committing to git. It **never** authorizes `gcloud builds submit`, `gcloud run jobs update`, `gcloud run jobs execute`, `gsutil`/`gcloud storage` writes, or Langfuse trace deletions.
2. **Explicit Cloud Verbs Required:** Every cloud mutation requires an explicit instruction naming the exact operation:
   * *Publish Fixtures:* `bash evaluation/agent/sync_fixtures.sh`
   * *Update Triggers:* `gcloud builds triggers import --region=us-east1 ...`
   * *Pin Job Image:* `gcloud run jobs update ...`
   * *Execute Live Run:* `gcloud run jobs execute ...`
3. **Trace Hygiene in Langfuse:** Local exploratory checks must use a `local-` prefix in `run_tag` (e.g., `run:local-smoke-1`) so they can be filtered out or purged cleanly without touching certified gate runs (`run:gate2-pilot-*`, `run:gate3-census-*`).

---

## 7. Operational Directives for Phase 04

1. **Review & Audit Only First:** Read this `04-cicd_and_local_first_orchestration.md` specification, inspect the current trigger exports and `evaluation/agent/pipeline.yaml`, and report any discrepancies or open questions.
2. **Author Local Scripts & Tests:** Implement `evaluation/agent/run_certified_job.sh` (or equivalent dry-run flags in `entrypoint.py`) and prepare the trigger YAML definitions locally with `includedFiles` / `ignoredFiles` path filters so `docs/**` commits stop triggering `agent` and `mcp` rebuilds[cite: 1].
3. **Hold All Cloud Actions:** Do **not** import the updated triggers to Cloud Build, push to remote, or execute any Cloud Run Jobs until the Phase 04 implementation plan is reviewed and explicitly approved.