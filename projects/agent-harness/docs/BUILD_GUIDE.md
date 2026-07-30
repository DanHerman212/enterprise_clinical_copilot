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
  secretmanager.googleapis.com
```

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

Record the value you actually get into `tests/fixtures/expected.json` — see §13 for why
it must not be hardcoded.

---

## 4. Build the teardown script

Write `projects/agent-harness/scripts/teardown.py` **now**, before you forget the
endpoint is running. It must handle:

1. Undeploy models from `readmission-endpoint`, then delete the endpoint
2. Delete the online Feature Store / feature view (§5) if it exists
3. Print what it found and what it removed; exit cleanly if nothing exists
4. Accept `--dry-run`

Add a matching `scripts/standup.py` (or reuse `deploy_cpr.py`) so the cycle is
symmetrical. A teardown you cannot reverse in one command will not get used.

> **Costs while idle:** the endpoint (`n1-standard-2`) bills continuously — roughly
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

---

## 6. MCP server and the predict tool

Use **FastMCP** (bundled with the `mcp` Python SDK).

`mcp_server/server.py` must be **transport-agnostic**:

```python
import argparse
from mcp.server.fastmcp import FastMCP

app = FastMCP("readmission")

# tools registered here (see tools/predict.py)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    app.run(transport=args.transport, port=args.port)
```

`tools/predict.py` — the tool contract:

```python
@app.tool()
def predict_readmission(hadm_id: int) -> dict:
    """30-day readmission risk for one hospital admission.

    Returns probability, the decision at the operating threshold, and the
    top contributing factors (parent-aggregated TreeSHAP).
    """
```

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

---

## 7. Verify MCP over stdio

Run the server and inspect it with the official tool:

```bash
npx @modelcontextprotocol/inspector \
  .venv/bin/python -m projects.agent_harness.mcp_server.server --transport stdio
```

**Verify:**
- `predict_readmission` appears under List Tools with its docstring and schema
- Calling it with `hadm_id=20924467` returns the expected probability
- Calling it with `hadm_id=1` returns a graceful structured error

**Optional but worth doing once:** register the same command in Claude Desktop's MCP
config and ask it in natural language. This is the concrete proof of the reusability
claim — and it takes ten minutes.

---

## 8. Deploy the MCP server to Cloud Run

`mcp_server/Dockerfile` — apply the lessons already learned on the website:

- `python:3.12-slim`, no `gcc`/`libpq-dev` unless something actually needs compiling
- `.dockerignore` including `.venv/`
- **JSON-form `CMD`** so the process receives `SIGTERM` as PID 1

```dockerfile
CMD ["python", "-m", "mcp_server.server", "--transport", "http", "--port", "8080"]
```

Build and deploy **privately** — no `--allow-unauthenticated`:

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

**Verify** with your own credentials before wiring the agent:

```bash
MCP_URL=$(gcloud run services describe mcp-server --region ${REGION} --format="value(status.url)")
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" "${MCP_URL}/health"
```

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
> `tests/fixtures/expected.json`, which the full training pass (§20) regenerates.
> Hardcoding guarantees that every test fails after retraining and that the failures
> look like integration bugs rather than an expected model change.

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

> **Expansion pending.** Sections 16-17 are intentionally at design level. Step-by-step
> instructions will be written after the CDN spike below returns a result, because a
> negative result changes the approach entirely (vendor the renderer into `static/`, or
> hand-write the component against the A2UI schema). Run the spike early — it is
> independent of §3-15 and can proceed in parallel.

Follow [a2ui_requirements.md](../../../docs/a2ui_requirements.md). The load-bearing
points:

- **Lit renderer**, loaded natively from a **pinned** CDN URL or import map — no Vite, no
  bundler, no Node in the image (R2)
- **Pin spec v0.9.1** behind a single adapter module (R4)
- The **agent** composes A2UI via `a2ui-agent-sdk`; the MCP tool stays plain JSON (R3)
- Always emit **fallback text** alongside the payload (R8)
- Annotate rendered payloads `audience: ["user"]` so the raw JSON stays out of the LLM's
  context on later turns (R7)

**Step 5 scope is one fixed component** — a `RiskCard` showing probability, threshold,
decision, and `top_factors`. The agent does not choose component shapes yet; that starts
in Phase 3 when `rag_search` gives it something to choose between.

**Verify the renderer in isolation first** — a static HTML file with a hardcoded A2UI
payload, no agent involved. This answers the one open question from the A2UI research:
whether the A2UI Lit renderer package (distinct from Lit core, with its own dependency
tree) loads cleanly from an ESM CDN.

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
3. **Regenerate `tests/fixtures/expected.json`** from the new model
4. Re-run Tier 1 — it should pass with new numbers, no code changes
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
