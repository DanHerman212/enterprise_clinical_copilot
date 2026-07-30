# Agent + MCP Build Guide

_Date: 2026-07-30 · Status: plan, not yet executed_

A step-by-step guide for building the clinical copilot demo: a **deployed MCP server**
exposing the readmission model as a tool, a **LangGraph agent** that calls it, and an
**A2UI-rendered demo page** on the Django website.

Companion to [architecture.md](architecture.md) (the *why*) and the website's
[GCP deployment guide](../../../../danielmherman/docs/GCP_DEPLOYMENT_GUIDE.md) (the
pattern this guide follows).

---

## Decisions recorded (2026-07-30)

| Question | Decision |
|---|---|
| Build order | Deploy endpoint → MCP → agent → UI → **then** full training pass |
| Endpoint cost | Deploy it; ship a teardown script alongside |
| Feature source | Build **both**; BigQuery for dev/CI/tests, Feature Store for live demos |
| MCP topology | **Separate Cloud Run service** (HTTP) in prod; stdio locally. One transport-agnostic codebase |
| Region | `us-east1` everywhere; Gemini via cross-region or global endpoint |
| Demo access | Django auth-gated, per-user quota in Postgres |
| Patient selection | Synthetic-name search over a curated cohort, **plus** advanced `hadm_id` entry |
| Repo / project | `projects/agent-harness`, GCP project `trim-icon-498815-a0` |

**Guiding principle:** walking skeleton first. Every step ends with something verifiable.
Do not start a step until the previous one's verification passes.

---

## Table of Contents

1. [Prerequisites and layout](#1-prerequisites-and-layout)
2. [Enable APIs and create service accounts](#2-enable-apis-and-create-service-accounts)
3. [Deploy the Vertex endpoint](#3-deploy-the-vertex-endpoint)
4. [Build the teardown script](#4-build-the-teardown-script)
5. [Feature source abstraction](#5-feature-source-abstraction)
6. [MCP server and the predict tool](#6-mcp-server-and-the-predict-tool)
7. [Verify MCP over stdio](#7-verify-mcp-over-stdio)
8. [Deploy the MCP server to Cloud Run](#8-deploy-the-mcp-server-to-cloud-run)
9. [Enable Gemini on Vertex](#9-enable-gemini-on-vertex)
10. [LangGraph agent (local)](#10-langgraph-agent-local)
11. [Agent to MCP over authenticated HTTP](#11-agent-to-mcp-over-authenticated-http)
12. [Deploy the agent to Cloud Run](#12-deploy-the-agent-to-cloud-run)
13. [Tier 1 acceptance tests](#13-tier-1-acceptance-tests)
14. [Demo cohort and synthetic names](#14-demo-cohort-and-synthetic-names)
15. [Django BFF: auth, quota, token exchange](#15-django-bff-auth-quota-token-exchange)
16. [A2UI rendering layer](#16-a2ui-rendering-layer)
17. [Assemble the demo page](#17-assemble-the-demo-page)
18. [Cost control](#18-cost-control)
19. [Troubleshooting](#19-troubleshooting)
20. [After the skeleton: full training pass](#20-after-the-skeleton-full-training-pass)

---

## 1. Prerequisites and layout

**Existing assets this guide depends on:**

- `projects/mlops/scripts/deploy_cpr.py` — registers and deploys the CPR
- `projects/mlops/scripts/smoke_test.py` — the BigQuery feature-fetch glue to reuse
- `projects/mlops/pipelines/serving/cpr/predictor.py` — the serving contract
- Existing venv at repo root: `.venv/bin/python`

**Target layout:**

```
projects/agent-harness/
  mcp_server/
    __init__.py
    server.py            # transport-agnostic entrypoint (stdio | http)
    tools/
      predict.py         # predict_readmission tool
    features/
      base.py            # FeatureSource protocol
      bigquery_source.py # default: dev, CI, tests
      feature_store.py   # demo: low-latency online lookup
    Dockerfile
    requirements.txt
  agent/
    __init__.py
    graph.py             # LangGraph definition
    mcp_client.py        # stdio (local) / HTTP + ID token (prod)
    a2ui/
      catalog.py         # component catalog definition
      render.py          # structured result -> A2UI payload
    Dockerfile
    requirements.txt
  scripts/
    teardown.py          # endpoint + feature store
    seed_demo_cohort.py  # curated patients + synthetic names
  tests/
    test_tier1.py
    fixtures/
      expected.json      # parameterized expected values
  docs/
    architecture.md
    BUILD_GUIDE.md       # this file
```

> **Why a `features/` package rather than a flag.** The architecture calls for a
> pluggable source. A `Protocol` with two implementations keeps the tool code identical
> across dev and demo, and makes the Feature Store path testable without billing.

---

## 2. Enable APIs and create service accounts

```bash
PROJECT_ID=trim-icon-498815-a0
REGION=us-east1
gcloud config set project ${PROJECT_ID}

gcloud services enable \
  aiplatform.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  bigquery.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com \
  bigtable.googleapis.com \
  bigtableadmin.googleapis.com
```

> **The two Bigtable APIs are not optional if you use Feature Store.** A Vertex
> online store is Bigtable underneath, and the sync is a BigQuery export job
> writing into it. Without them the sync fails with
> `Missing IAM permission: bigtable.tables.mutateRows` — which reads like an IAM
> problem and sends you to grant roles the service agent already has.
>
> **Then wait before syncing.** API enablement and service-agent propagation take
> several minutes. A sync started ~1 minute after enabling them ran for 52
> minutes reporting `code=0` and never finished or failed. Compare: the genuine
> permission failure surfaced in 0.1 min. A sync that neither completes nor
> errors is wedged, not slow.

Create **two** service accounts with distinct privileges:

```bash
gcloud iam service-accounts create mcp-server-sa \
  --display-name="MCP server (reads features, calls Vertex endpoint)"

gcloud iam service-accounts create agent-sa \
  --display-name="LangGraph agent (calls Gemini and the MCP server)"
```

Grant least privilege:

```bash
MCP_SA="mcp-server-sa@${PROJECT_ID}.iam.gserviceaccount.com"
AGENT_SA="agent-sa@${PROJECT_ID}.iam.gserviceaccount.com"

# MCP server: read features, call the prediction endpoint
for ROLE in roles/bigquery.dataViewer roles/bigquery.jobUser roles/aiplatform.user; do
  gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${MCP_SA}" --role="${ROLE}" --condition=None
done

# Agent: call Gemini only. No BigQuery, no endpoint access.
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:${AGENT_SA}" --role="roles/aiplatform.user" --condition=None
```

> **Why the agent has no data access.** The agent reasons; it never touches PHI-adjacent
> data directly. Everything flows through the MCP tool. This is worth doing even in a
> demo — it is the difference between "I used two service accounts" and "I could
> articulate a data-access boundary" in an interview.

---

## 3. Deploy the Vertex endpoint

Reuse the existing script:

```bash
.venv/bin/python projects/mlops/scripts/deploy_cpr.py
```

**Verify** with the known-good patient before going further:

```bash
.venv/bin/python projects/mlops/scripts/smoke_test.py 20924467
```

Expected (current model): probability ≈ **0.1314**, decision **1**, base value −1.3386.

> **Do not proceed until this passes.** Every later step assumes a working endpoint. A
> failure here is an MLOps problem, not an agent problem, and debugging it through two
> layers of MCP plumbing wastes hours.

### Record the expected values

Once it passes, capture the observed numbers — this is the moment they are
authoritative:

```bash
.venv/bin/python projects/mlops/scripts/smoke_test.py 20924467 --write-fixture
```

That writes `projects/agent-harness/tests/fixtures/expected.json`:

```json
{
  "generated_at": "2026-07-30",
  "generated_by": "projects/mlops/scripts/smoke_test.py --write-fixture",
  "model_version": "readmission-final-<run tag>",
  "patients": {
    "20924467": {
      "base_value": -1.3386,
      "decision": 1,
      "probability": 0.1314,
      "tolerance": 0.0001
    }
  },
  "threshold": 0.12
}
```

Four details earn their place:

- **`tolerance`** — never assert float equality against a served model. Serving-side
  numeric drift is real, and an exact-match failure reads like an integration bug
- **`model_version`** — when a test fails after §20, this says immediately whether you
  are looking at a stale fixture or a genuine regression
- **`patients` as a map** — §14 adds a cohort spanning the 0.12 boundary; borderline
  cases drop in without a schema change
- **`generated_by`** — points at the command that regenerates it

Re-running `--write-fixture` upserts one patient and leaves the others intact. If the
probability moved by more than `tolerance`, it prints the before → after so a model
change is visible rather than silent.

---

## 4. Build the teardown script

`projects/agent-harness/scripts/teardown.py` \u2014 written **before** you forget the
endpoint is running.

```bash
# always look first; changes nothing
.venv/bin/python projects/agent-harness/scripts/teardown.py --dry-run

# delete, with a confirmation prompt
.venv/bin/python projects/agent-harness/scripts/teardown.py

# delete unattended (CI, or the end of a demo script)
.venv/bin/python projects/agent-harness/scripts/teardown.py --yes
```

It removes the two things that bill: **deployed models + the endpoint**, and the
**Feature Store online store and its views** (\u00a75, matched by the
`readmission` prefix so this script already covers whatever \u00a75 names them). Feature
Store discovery is wrapped defensively \u2014 if the API is disabled or nothing exists yet,
it reports and moves on rather than failing the run.

It deliberately **keeps** the model registry entries (`readmission-final-*`), GCS
bundles, BigQuery tables, and Artifact Registry images. Those are free or near-free, and
`smoke_test.py` discovers the serving bundle through that registry provenance \u2014 deleting
it would break bundle discovery.

Rebuild the endpoint with `deploy_cpr.py`, which is the matching stand-up step; no
separate `standup.py` is needed. The CPR image is content-addressed, so a rebuild reuses
the existing image and only re-registers the model.

> **Costs while idle:** the endpoint (`n1-standard-2`) bills continuously \u2014 roughly
> **$50-70/month** if left up. Cloud Run services scale to zero and cost nothing. This
> script is the single biggest cost lever in the project.

---

## 5. Feature source abstraction

**`features/base.py`** — the seam:

```python
from typing import Protocol

class FeatureSource(Protocol):
    def fetch(self, hadm_id: int) -> dict[str, float | None]:
        """Return column -> value for one admission. Missing values are None."""
```

**`features/bigquery_source.py`** — lift directly from `smoke_test.py::_fetch_patient`
and `_assemble_features`. This is the default everywhere except a live demo.

**`features/feature_store.py`** — online lookup, entity = `hadm_id`.

Both must return **identical values** for the same `hadm_id`. Prove it:

```bash
.venv/bin/python -m pytest projects/agent-harness/tests/test_feature_parity.py
```

> **Write the parity test before the Feature Store implementation.** The failure mode
> here is subtle — a Feature Store view with stale or differently-typed values produces
> predictions that are *plausible but wrong*, which is far worse than an error. The
> parity test is the only thing standing between you and a demo that quietly lies.

Select via env var, defaulting to the cheap path:

```python
SOURCE = os.environ.get("FEATURE_SOURCE", "bigquery")  # "bigquery" | "feature_store"
```

### Feature Store serves the demo cohort, not the whole table

`scripts/build_demo_cohort.py` builds `readmission.demo_cohort`, and the feature
view is created over **that**, not over `analytics_dataset_encoded`.

The reason is arithmetic. The full table is 352,699 rows / 140.8 MB, and the
demo queries a few dozen patients — so a full sync provisions, bills, and
re-exports the entire dataset to serve ~0.01% of it. Measured, on one Bigtable
node:

| Scope | Rows | Sync |
|---|---|---|
| Full table | 352,699 | 52 min, never finished |
| Demo cohort | 41 | under one 20 s poll |

Cohort selection rules, all defensible on their own:

- **test split only** — a demo patient the model trained on proves nothing
- **balanced on `readmission_30d`** — so both a high- and low-risk case are on hand
- **deterministic** via `FARM_FINGERPRINT(hadm_id)`, which is stable across runs
  without the low-id bias of `ORDER BY hadm_id`
- **fixture patients pinned in**, and the script exits non-zero if one is absent

Ids outside the cohort simply fall back to BigQuery, which is the default source
anyway — so the advanced free-text `hadm_id` path still works.

> §14 owns final cohort selection. This is a defensible default, not a decision.

Provision and sync:

```bash
.venv/bin/python projects/agent-harness/scripts/build_demo_cohort.py
.venv/bin/python projects/agent-harness/scripts/setup_feature_store.py
```

`setup_feature_store.py` is idempotent, attaches to an already-running sync
rather than failing on `FailedPrecondition`, and **exits non-zero if the sync
reports a non-zero status or zero rows**. That last check exists because an
earlier version printed `Sync complete — 0 rows synced` for a sync that had died
with a permission error.

> **An unset protobuf `Timestamp` is the epoch, not `None`.** `if
> sync.run_time.end_time:` is truthy for a *running* sync, so every in-flight
> sync looks finished. Use `sync._pb.run_time.HasField("end_time")`.

---

## 6. MCP server and the predict tool

Use the `mcp` Python SDK (`pip install "mcp[cli]"`).

> **SDK 2.0 removed `mcp.server.fastmcp`.** The ergonomic server class is now
> `MCPServer`, imported from `mcp.server`. Same decorator/`add_tool` model, new
> name. Verified against `mcp==2.0.0` on 2026-07-30. Also note the wire types
> moved to snake_case: `tool.input_schema`, not `tool.inputSchema`.

**Files:**

| File | Role |
|---|---|
| `mcp_server/endpoint.py` | Cached `Endpoint` lookup + `predict_one()` |
| `mcp_server/tools/predict.py` | The tool: fetch → order → predict → shape |
| `mcp_server/server.py` | Transport-agnostic entrypoint |

`mcp_server/server.py` must be **transport-agnostic** — the transport is a flag,
not a fork in the code, so the local path and the deployed path cannot drift:

```python
from mcp.server import MCPServer
from .tools import predict_readmission

server = MCPServer(name="readmission", version="0.1.0", instructions="…")
server.add_tool(predict_readmission)

# --transport stdio  ->  server.run("stdio")
# --transport http   ->  server.run("streamable-http", host="0.0.0.0", port=port)
```

Registering with `add_tool` rather than a `@server.tool()` decorator keeps
`tools/predict.py` free of a server import — no circular dependency, and the
tool function stays directly callable from tests.

> **Under stdio the transport *is* stdout.** Anything printed to stdout corrupts
> the JSON-RPC stream. Diagnostics go to stderr.

`tools/predict.py` — the tool contract:

```python
async def predict_readmission(hadm_id: int) -> dict:
    """30-day readmission risk for one hospital admission.

    Returns probability, the decision at the operating threshold, and the
    top contributing factors (parent-aggregated TreeSHAP).
    """
    return await asyncio.to_thread(_predict, hadm_id)
```

**The tool is async, the work is not.** BigQuery and Vertex calls are
synchronous; under the HTTP transport a blocking tool stalls the event loop for
every concurrent caller. `asyncio.to_thread` costs one line now and is a
retrofit later.

**Cache the clients.** The endpoint lookup, the manifest, and the feature source
are all `lru_cache`d for the process lifetime. `smoke_test.py` refetches each
run, which is right for a CLI and wrong for a long-lived server — that would be
three API round-trips on every tool call.

**Return shape** (stable contract — the agent, the tests, and A2UI all depend on it):

```json
{
  "hadm_id": 20924467,
  "probability": 0.1314,
  "threshold": 0.12,
  "decision": 1,
  "base_value": -1.3386,
  "top_factors": [
    {"feature": "prior_inpatient_days", "contribution": 0.42, "direction": "increases"}
  ],
  "model_version": "readmission-final-20260723172647",
  "feature_source": "bigquery"
}
```

> **The tool returns plain JSON, never A2UI.** This is deliberate — see
> [a2ui_requirements.md](../../../docs/a2ui_requirements.md) R3. A tool that returns UI
> is coupled to one presentation layer and stops being reusable by Claude Desktop or CI.
> The agent composes A2UI in §16.

> **Include `model_version` and `feature_source` in every response.** When a demo
> produces a surprising number, these two fields answer "which model, reading from
> where?" immediately. Without them that question costs an hour.

**Error handling:** an unknown `hadm_id` must return a structured error, not raise. This
is a Tier 1 acceptance criterion (§13).

```json
{"hadm_id": 1, "error": "unknown_patient",
 "message": "No admission 1 in the feature source (bigquery).",
 "feature_source": "bigquery"}
```

Codes: `unknown_patient`, `feature_fetch_failed`, `incomplete_features`,
`prediction_failed`.

**Missing values are fine; a missing column is not.** A null feature is
legitimate — the model reads it as NaN by design, and patient 20924467 has 8 of
them. But `to_vector` fills absent keys with `None`, so a *short* row would
silently shift every feature after the gap and still return a plausible
probability. The tool therefore checks the fetched row against `feature_order()`
and returns `incomplete_features` rather than predicting on it.

> This is the same failure mode as the Feature Store sync that reported success
> while writing nothing (§5). Both times the symptom was silence, not an error.
> Prefer a loud failure over a plausible number.

---

## 7. Verify MCP over stdio

Run the server and inspect it with the official tool. Run it **from
`projects/agent-harness/`** — the directory name contains a hyphen, so it is not
an importable package path; `mcp_server` must be resolvable from the cwd:

```bash
cd projects/agent-harness
npx @modelcontextprotocol/inspector \
  ../../.venv/bin/python -m mcp_server.server --transport stdio
```

**Verify:**
- `predict_readmission` appears under List Tools with its docstring and schema
- Calling it with `hadm_id=20924467` returns the expected probability
- Calling it with `hadm_id=1` returns a graceful structured error

> If the first call times out, raise the request timeout in the inspector's
> Configuration panel before assuming a bug. A cold call does `Model.list` + GCS
> manifest + BigQuery + Vertex predict; the caches make every later call fast.

### Then automate it

The inspector proves the protocol works once, on one machine, and leaves nothing
behind. `tests/test_mcp_stdio.py` is the durable version — it spawns the server
as a real subprocess and talks JSON-RPC to it:

```bash
.venv/bin/python -m pytest projects/agent-harness/tests/test_mcp_stdio.py -v
# 5 passed in 7.82s
```

It covers what the in-process tests of §6 cannot: the initialize handshake,
`tools/list` as a model sees it, content serialisation, and — the one worth
having — **that stdout stays clean**. Under stdio the transport *is* stdout, so
a stray `print()` in any dependency corrupts the stream. That regression is
invisible until a client fails to parse.

Uses `asyncio.run` rather than pytest-asyncio, so it adds no plugin dependency.
The session is module-scoped: one spawn, one cold start, five assertions.

**Optional but worth doing once:** register the same command in Claude Desktop's MCP
config and ask it in natural language. This is the concrete proof of the reusability
claim — and it takes ten minutes.

---

## 8. Deploy the MCP server to Cloud Run

Three files, all in `mcp_server/`: `Dockerfile`, `.dockerignore`,
`requirements.txt`. The build context is the **package directory**, not the repo
root — `agent-harness` contains a hyphen and is not importable, so the package
is copied to `/app/mcp_server` where `python -m mcp_server.server` resolves.

`requirements.txt` is deliberately *not* the repo's full requirements. The
training stack (xgboost, pandas, sklearn) has no place in a serving image — the
model runs behind the Vertex endpoint, not in this container.

**Dockerfile lessons carried over from the website:**

- `python:3.12-slim`, no `gcc`/`libpq-dev` — nothing here compiles
- `.dockerignore` including `.venv/` and `__pycache__/`
- **JSON-form `CMD`** so the process is PID 1 and receives `SIGTERM` directly;
  with the shell form, `sh` swallows the signal and Cloud Run waits out the full
  grace period on every revision swap

```dockerfile
CMD ["python", "-m", "mcp_server.server", "--transport", "http"]
```

Note there is no `--port`. **Cloud Run injects `PORT`** and expects the
container to honour it, so `--port` defaults to `int(os.environ["PORT"])`.
Hard-coding 8080 works until the day it doesn't.

### Stateless HTTP is not optional here

Streamable-HTTP sessions live in the memory of one instance. Behind a Cloud Run
load balancer with no session affinity, a follow-up request can land on a
different instance and fail to find its session — an error that only appears
under concurrency, which is to say in front of an audience.

The server therefore runs `stateless_http=True` by default (`--stateful` opts
out for single-instance debugging). This tool holds no per-session state, so
there is nothing to lose. Verified locally: two independent sessions each
completed a full call, and the log shows `Terminating session: None` after each.

### Health check

`/health` is registered with `@server.custom_route` and is **deliberately
shallow** — it does not touch Vertex or BigQuery. A deep check would bill on
every probe of a scale-to-zero service and would mark the container unhealthy
whenever a dependency blipped, causing Cloud Run to recycle a process that is
fine. Per-request dependency failures already surface as the tool's structured
errors.

### Service account

The MCP server needs exactly three things. Grant no more:

```bash
gcloud iam service-accounts create mcp-server \
  --display-name "Readmission MCP server"

MCP_SA="mcp-server@${PROJECT_ID}.iam.gserviceaccount.com"

for ROLE in roles/aiplatform.user roles/bigquery.jobUser roles/bigquery.dataViewer; do
  gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member "serviceAccount:${MCP_SA}" --role ${ROLE}
done

# Serving bundle (manifest.json) only — not the whole MLOps bucket.
gsutil iam ch serviceAccount:${MCP_SA}:objectViewer gs://${PROJECT_ID}-mlops
```

`bigquery.jobUser` is separate from `dataViewer` and both are required: reading
a table is a *query job*, and a viewer without job rights fails at run time with
a permission error that reads like a data problem.

### Build and deploy privately

```bash
gcloud builds submit --tag ${REGION}-docker.pkg.dev/${PROJECT_ID}/readmission/mcp-server:latest \
  projects/agent-harness/mcp_server

gcloud run deploy mcp-server \
  --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/readmission/mcp-server:latest \
  --region ${REGION} \
  --no-allow-unauthenticated \
  --service-account ${MCP_SA} \
  --set-env-vars "^@^PROJECT_ID=${PROJECT_ID}@LOCATION=${REGION}@FEATURE_SOURCE=bigquery" \
  --min-instances 0 --max-instances 3 --memory 512Mi
```

> **Use the `^@^` delimiter and a single `--set-env-vars` flag.** Repeating the flag does
> not accumulate — the last occurrence silently replaces the rest. This cost us a
> debugging cycle on the website deploy.

`--min-instances 0` means cold starts. A cold call also warms three caches
(endpoint lookup, manifest, feature source), so the first request after idle is
slow and every one after is not. That is the right trade for a portfolio demo;
raise it to 1 only if a live audience is watching.

**Verify** with your own credentials before wiring the agent:

```bash
MCP_URL=$(gcloud run services describe mcp-server --region ${REGION} --format="value(status.url)")
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" "${MCP_URL}/health"
```

Then repeat the §7 protocol check against the deployed URL rather than a
subprocess — same assertions, different transport.

---

## 9. Enable Gemini on Vertex

```bash
gcloud services enable aiplatform.googleapis.com
```

Confirm the model is reachable from `us-east1`. If it is not, use the **global
endpoint** rather than relocating the agent — keeping the agent co-located with Cloud SQL
and the prediction endpoint matters more than a few milliseconds of LLM latency.

**Verify** with a trivial generate call before building the graph. A model-availability
error surfacing inside a LangGraph trace is much harder to read than the same error on
its own.

> **Compliance note.** Only tabular features and model outputs reach Gemini at this
> stage — no note text. That arrives with `rag_search` in Phase 3, at which point the
> PhysioNet DUA question must be settled. Staying in-GCP via Vertex is the defensible
> path; do not route MIMIC-derived text through a third-party API.

---

## 10. LangGraph agent (local)

Build against **stdio** first — no network, no auth.

Minimum viable graph:

```
START → agent (Gemini + tools) → tool node → agent → END
```

**System prompt requirements** (these are the Tier 2 guardrails, worth writing now):

- Report the probability and threshold decision exactly as returned; never round or
  restate them differently
- Attribute risk factors **only** from `top_factors`; never invent or infer others
- State plainly that this is a decision-support signal, not a diagnosis
- If the tool errors, say so — never fabricate a plausible number

**Verify:** ask *"What is the readmission risk for admission 20924467?"* and confirm the
agent calls the tool and reports `0.1314`.

> **Checkpointer keying.** When you add persistence, key the thread on
> **user + session**, never on `hadm_id`. Keying by patient means two concurrent
> demo users looking at the same patient share conversation state — they would see each
> other's messages. This is the single most likely multi-user bug in the design.

---

## 11. Agent to MCP over authenticated HTTP

Grant the agent permission to invoke the MCP service:

```bash
gcloud run services add-iam-policy-binding mcp-server \
  --region ${REGION} \
  --member="serviceAccount:${AGENT_SA}" \
  --role="roles/run.invoker"
```

`agent/mcp_client.py` fetches an ID token **audienced to the MCP service URL**:

```python
import google.auth.transport.requests
import google.oauth2.id_token

def _id_token(audience: str) -> str:
    req = google.auth.transport.requests.Request()
    return google.oauth2.id_token.fetch_id_token(req, audience)
```

Select transport by environment so local dev stays on stdio:

```python
MCP_TRANSPORT = os.environ.get("MCP_TRANSPORT", "stdio")  # "stdio" | "http"
```

> **The audience must be the service URL**, not the endpoint path. A mismatched audience
> produces a `401` that looks identical to a missing IAM binding. Check the audience
> first — it is the more common cause.

---

## 12. Deploy the agent to Cloud Run

```bash
gcloud run deploy agent \
  --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/readmission/agent:latest \
  --region ${REGION} \
  --no-allow-unauthenticated \
  --service-account ${AGENT_SA} \
  --set-env-vars "^@^PROJECT_ID=${PROJECT_ID}@LOCATION=${REGION}@MCP_TRANSPORT=http@MCP_URL=${MCP_URL}" \
  --min-instances 0 --max-instances 3 --memory 1Gi
```

**Verify** end to end with your own identity token, then confirm the chain in logs:
Django → agent → MCP → endpoint.

> **Two cold starts now sit in the path.** First request after idle pays agent + MCP
> startup. If a live demo feels sluggish, `--min-instances 1` on both services for the
> day costs a few dollars and is the right trade.

---

## 13. Tier 1 acceptance tests

The Phase 2 exit criterion. All four must pass:

1. **Known-good value** — agent given `hadm_id=20924467` returns the expected
   probability, decision, and non-empty `top_factors`
2. **Graceful error** — an invalid `hadm_id` produces a structured error, not a crash or
   an invented number
3. **Routing assertion** — the call actually went through MCP (assert on the tool-call
   trace, not just the answer)
4. **Schema contract** — the response validates against the §6 schema

```bash
.venv/bin/python -m pytest projects/agent-harness/tests/test_tier1.py -v
```

> **Do not hardcode `0.1314`.** Load expected values from
> `tests/fixtures/expected.json` (written by `smoke_test.py --write-fixture` in §3, and
> regenerated by the full training pass in §20). Compare within the fixture's
> `tolerance`, never by equality. Hardcoding guarantees that every test fails after
> retraining and that the failures look like integration bugs rather than an expected
> model change.

---

## 14. Demo cohort and synthetic names

`scripts/seed_demo_cohort.py` writes a Postgres table on the website's Cloud SQL
instance:

| column | notes |
|---|---|
| `hadm_id` | real MIMIC identifier |
| `display_name` | **synthetic**, deterministic, seeded |
| `age`, `sex` | from `feat_demographics` — the real record |
| `summary` | short clinical descriptor for the search result |

**Rules:**

- **20-40 patients**, deliberately spanning the decision boundary: clear highs, clear
  lows, and at least two **near 0.12** — the borderline cases are where a threshold and
  SHAP factors earn their keep
- Names must match the record's actual sex and plausible age
- The mapping is **stored, not generated at runtime** — otherwise names change on every
  deploy, breaking screenshots and your own demo script
- Descriptors must be **clinical, not outcome-based**: *"72F — CHF, 3 prior admissions"*,
  never *"high risk case"*, which spoils the prediction

> **Label synthetic names in the UI.** Every patient view shows something like
> *"Margaret Ellison · synthetic name · MIMIC-IV record 20924467"*. Assigning fake names
> does not violate the PhysioNet DUA — you are not re-identifying anyone — but a demo
> that *appears* to show named patients invites exactly the wrong question. One line of
> UI removes the ambiguity and demonstrates you understood the obligation.
>
> Only the **name** is synthetic. All clinical values are real de-identified MIMIC data,
> and the UI must never blur that.

---

## 15. Django BFF: auth, quota, token exchange

Three responsibilities, in the website project:

**1. Authentication** — `@login_required`. Accounts are issued, not self-registered.

**2. Per-user quota** in Postgres. The atomic-increment requirement is not optional:

```python
from django.db.models import F

updated = DemoQuota.objects.filter(
    user=request.user, used__lt=F("daily_limit")
).update(used=F("used") + 1)

if not updated:
    return JsonResponse({"error": "Daily demo limit reached."}, status=429)
```

> **Never read-modify-write the counter.** `q.used += 1; q.save()` loses increments
> under concurrency — two simultaneous requests both read `4`, both write `5`, and the
> limit silently doesn't hold. The `F()` expression pushes the increment into the
> database where it is atomic. Same reasoning as the earlier two-concurrent-users
> discussion.

**3. Token exchange** — mint an ID token audienced to the agent's Cloud Run URL and
proxy the request. The browser never talks to the agent, so there is no CORS, no public
agent, and no client-side credentials.

Grant the website's runtime service account invoker rights on the agent:

```bash
gcloud run services add-iam-policy-binding agent \
  --region ${REGION} \
  --member="serviceAccount:${PROJECT_NUM}-compute@developer.gserviceaccount.com" \
  --role="roles/run.invoker"
```

---

## 16. A2UI rendering layer

> **Spike verified 2026-07-30.** The open question — does the A2UI Lit renderer load
> from an ESM CDN with no build step? — is **answered: yes**. All nine checks pass in
> [a2ui_cdn_spike.html](../spikes/a2ui_cdn_spike.html): both packages resolve, Lit
> dedupes, the custom element registers, the v0.9 payload renders, and data binding
> resolves. Re-run it any time with
> `python3 -m http.server 8777` from the repo root.
>
> Full write-up, including the four failure modes and the duplicate-Lit investigation:
> [a2ui_spike_findings.md](a2ui_spike_findings.md).

### 16a. The verified import map

Copy this exactly. Every entry is load-bearing.

```html
<script type="importmap">
{
  "imports": {
    "lit": "https://esm.sh/lit@3.2.1",
    "lit/": "https://esm.sh/lit@3.2.1/",
    "zod": "https://esm.sh/zod@3.25.76",
    "zod/": "https://esm.sh/zod@3.25.76/",
    "@lit/context": "https://esm.sh/@lit/context@1.1.6?external=lit",
    "@a2ui/markdown-it": "https://esm.sh/@a2ui/markdown-it@0.1.0",
    "@a2ui/web_core/v0_9": "https://esm.sh/@a2ui/web_core@0.10.5/v0_9?external=lit,zod",
    "@a2ui/lit/v0_9": "https://esm.sh/@a2ui/lit@0.10.2/v0_9?external=lit,zod"
  }
}
</script>
```

Why each piece matters — all four of these were discovered by the spike failing:

| Detail | Consequence if wrong |
|---|---|
| `?external=lit,zod` | Without it esm.sh bundles a private Lit per package. Two `CustomElementRegistry` attempts, components never upgrade, **no error message** |
| Trailing-slash entries (`lit/`, `zod/`) | The renderer imports `lit/decorators.js`; web_core imports `zod/v3`. A bare-name mapping does **not** cover subpaths |
| `zod >= 3.25` | 3.24.1 has no `zod/v3` subpath → hard 404 at module resolution |
| Version ≠ spec version | npm `@a2ui/lit@0.10.2` implements **spec v0.9** via the `/v0_9` subpath. Do not "upgrade" to match a spec number |

### 16b. Wire it up

```js
const { MessageProcessor } = await import('@a2ui/web_core/v0_9');
const { basicCatalog, Context } = await import('@a2ui/lit/v0_9');
const { ContextProvider } = await import('@lit/context');
const { renderMarkdown } = await import('@a2ui/markdown-it');

// Text properties are Markdown. Without this, `variant: 'h2'` shows the user a
// literal "## Heading". The renderer is injected via Lit context, not a global.
new ContextProvider(host, { context: Context.markdown, initialValue: renderMarkdown });

const processor = new MessageProcessor([basicCatalog]);
processor.onSurfaceCreated(s => {
  const el = document.createElement('a2ui-surface');
  el.surface = s;                 // property, not attribute
  host.replaceChildren(el);
});
processor.processMessages(messages);
```

### 16c. The v0.9 payload shape

**This is the trap most likely to cost time.** Nearly every example online is v0.8, and
v0.9 is not backward compatible. In v0.9 `component` is a **string** and properties sit
**inline**:

```js
// v0.9 — correct
{ id: 'root',  component: 'Card',   child: 'card-body' }
{ id: 'title', component: 'Text',   text: 'Readmission risk', variant: 'h2' }
{ id: 'prob',  component: 'Text',   text: { path: '/probability' } }

// v0.8 — silently fails
{ id: 'title', component: { Text: { text: { literalString: 'Readmission risk' } } } }
```

Feeding v0.8 shapes to the v0.9 renderer produces **no exception** — just a console
warning, `Component implementation not found for type: [object Object]`, and an empty
box. Watch for that string.

Three more v0.9 differences:

- `updateDataModel` takes `path` + `value`. The v0.8 `contents` adjacency list of
  `valueString`/`valueNumber` entries is gone
- `createSurface` has **no `root` property** — the component whose `id` is literally
  `"root"` is the tree root
- `Text.variant` values `h1`-`h5` are implemented by **prepending Markdown hashes**,
  which is why the markdown renderer is mandatory rather than cosmetic

Message order: `createSurface` → `updateComponents` → `updateDataModel`.

### 16d. Vendor before production

The spike uses esm.sh, which is right for local iteration. **Vendor the files into
`static/` before the demo goes live.** During the spike esm.sh returned
`ERR_CONNECTION_CLOSED` on a cold artifact and the page failed to boot — it recovered on
retry, but that is a third-party outage sitting directly in the render path of a demo
you may be showing to an employer. Vendoring also removes the supply-chain exposure of
executing CDN-served JS.

### 16e. Scope

**Step 5 is one fixed component** — a `RiskCard` showing probability, threshold,
decision, and `top_factors`, composed from `basicCatalog` (`Card`, `Column`, `Text`).
No custom catalog is needed. The agent does not choose component shapes yet; that starts
in Phase 3 when `rag_search` gives it something to choose between.

Keep payload construction in **one adapter module** so the v1.0 migration
(`theme` → `surfaceProperties`) is a single-file change.

Remaining points from [a2ui_requirements.md](../../../docs/a2ui_requirements.md): emit
**fallback text** alongside every payload (R8), and annotate `audience: ["user"]` so raw
JSON stays out of the LLM's context on later turns (R7).

---

## 17. Assemble the demo page

Route: `www.danielmherman.com/projects/clinical_copilot/demo`

Flow:

1. Login-gated page with a **search box** over the demo cohort (synthetic names)
2. Optional **advanced** field: raw `hadm_id` entry, validated against the holdout set
3. Selection posts to the BFF → agent → MCP → endpoint
4. Agent returns A2UI + fallback text; the Lit renderer draws the `RiskCard`
5. Page displays the synthetic-name disclaimer and a model-version footer

**Streaming:** start with a single response. If you add streaming later, use **SSE with
an async generator** — the website already logs a warning about synchronous iterators
under ASGI, and a sync generator will block a worker.

---

## 18. Cost control

| Resource | Idle cost | Notes |
|---|---|---|
| Vertex endpoint (`n1-standard-2`) | **~$50-70/mo** | The main lever — tear down when not demoing |
| Online Feature Store | **bills continuously** | Enable only for live demos |
| Cloud Run (3 services) | ~$0 | Scales to zero |
| Cloud SQL | ~$7-10/mo | Already running for the website |
| Gemini Flash | per-token | Negligible at demo volume; quota caps abuse |

**Daily habit while building:**

```bash
.venv/bin/python projects/agent-harness/scripts/teardown.py
```

Set a **billing budget alert** rather than relying on remembering.

---

## 19. Troubleshooting

**`401` from the MCP service** — check the ID token *audience* before checking IAM. The
audience must be the service URL, not the path. Mismatched audience and missing binding
produce identical errors.

**Agent returns a plausible number the tool never produced** — the system prompt is not
constraining hard enough, or `top_factors` is being paraphrased. This is a Tier 2
faithfulness failure and the most important thing to catch. Test it deliberately.

**Feature Store and BigQuery disagree** — stop and fix before demoing. Run the §5 parity
test. A wrong-but-plausible prediction is worse than an outage.

**Predictions changed unexpectedly** — check `model_version` and `feature_source` in the
tool response. That is what those fields are for.

**Cold-start latency on first request** — expected; two services are waking. Use
`--min-instances 1` for a scheduled demo.

**Two users see each other's conversation** — the checkpointer is keyed by `hadm_id`
instead of user + session. See §10.

---

## 20. After the skeleton: full training pass

Only once §13 passes end to end:

1. Run the full training pipeline
2. Redeploy the CPR bundle (`deploy_cpr.py` reuses the image; only `artifact_uri` changes)
3. **Regenerate the fixture** from the new model \u2014 once per patient already in it:
   ```bash
   .venv/bin/python projects/mlops/scripts/smoke_test.py 20924467 --write-fixture
   ```
4. Re-run Tier 1 \u2014 it should pass with new numbers, no code changes
5. Update the demo cohort if the risk distribution shifted enough that the borderline
   cases are no longer borderline

> This is the payoff for parameterizing expected values in §13. If Tier 1 needs code
> edits at this point, the tests were coupled to a specific model rather than to the
> contract.

---

## Open items

- PhysioNet DUA / LLM-use position — required before Phase 3 (`rag_search`), not before
  this guide
- MCP HTTP server auth beyond Cloud Run IAM (per-user attribution, if ever needed)
- Conversational multi-turn session model — deferred until the demo needs it
- Whether to promote the demo cohort table into its own app rather than living in
  `content/`
