# Production Hardening, Observability, Load Testing & Unit Economics Specification

This document outlines the production hardening checklist, concurrency architecture, observability strategy, load-testing suite, and unit economics for the fully serverless GCP architecture (Cloud Run, BigQuery, Vertex AI) using a Django backend and Lit/A2UI frontend. 

It is specifically tailored to enforce a strict ceiling of **100 maximum active users**, **20 request credits per user**, and an automated **Self-Service Demo / Request-More-Credits workflow**.

---

## 1. Architecture Decision Record: Cloud Run vs. GKE for 100 Concurrent Users

**Verdict: Stay on Cloud Run. Do not migrate to GKE.**

For a workload capped at 100 concurrent users and 2,000 total bounded requests, Google Kubernetes Engine (GKE) introduces unnecessary operational overhead, VPC IP-exhaustion management, and slow node-pool scaling. 
* **Concurrency Capacity:** Cloud Run scales in seconds, bills to the nearest 100ms, and natively supports up to **1,000 concurrent requests per container instance**.
* **I/O-Bound Workload Profile:** Because the application relies on I/O-bound orchestration (waiting on Vertex AI and BigQuery), an async Python server (Gunicorn with `UvicornWorker` or Django ASGI) on Cloud Run configured with `--concurrency=25` and `--max-instances=5` will effortlessly absorb 100 simultaneous connections while acting as a hard cost ceiling.

---

## 2. Augmented Production Hardening & Pre-Launch Checklist

Complete these steps after locking in the final UX and before driving public traffic (e.g., via a LinkedIn post).

### Phase 1: Edge & Content Delivery (The Front Door)
- [ ] **Configure Cloud CDN:** Ensure Cloud CDN is enabled on the Global Load Balancer backend service pointing to the frontend Lit application. Verify static assets (JS, CSS, A2UI components) are cached at the edge to minimize Cloud Run egress costs.
- [ ] **Tune Cloud Armor Rate Limiting:** Verify WAF/rate-limiting rules are active (e.g., max 30 requests/minute per IP) to prevent bot floods before they reach Cloud Run or trigger authentication flows.
- [ ] **Enforce HTTPS Redirects:** Ensure the Global Load Balancer is strictly redirecting all HTTP traffic to HTTPS.

### Phase 2: Authentication, Subscriber Cap & Cost Guardrails
- [ ] **Implement Google OAuth & Atomic Subscriber Gate ($N \le 100$):** Integrate `django-allauth` and test the "Continue with Google" flow. Override the `DefaultSocialAccountAdapter.pre_social_login` hook to run an atomic count check (`User.objects.filter(is_active=True, is_staff=False).count() >= 100`) before creating a local Django `User`. If the limit is reached, intercept the signup and return an A2UI modal displaying:
  > *"The current subscriber base is over subscribed, please reach out directly if you're interested in getting a self-service demo account to dan@danielmherman.com"*
- [ ] **Enforce Atomic 20-Credit Ledger:** Replace daily token buckets with a persistent `credits_remaining = models.PositiveIntegerField(default=20)` field on the user profile. In the request allowance middleware, decrement credits using atomic database expressions (`F('credits_remaining') - 1`) paired with `select_for_update()` so rapid parallel clicks cannot bypass the 20-request cap via race conditions.
- [ ] **Build the "Request More Credits" Button & Flow:** When `credits_remaining == 0`, return HTTP `402 Payment Required` (or `429`) with an A2UI rate-limit warning payload that disables the chat input and renders a **"Request More Credits"** CTA button:
  - **Option A (Lead-Gen / Controlled Approval):** Clicking the button fires a POST to `/api/credits/request-topup/`, flagging the user in Django Admin, sending an instant notification to `dan@danielmherman.com` with a one-click signed approval link to grant +20 credits, and displaying a confirmation banner to the user.
  - **Option B (Instant Self-Service Demo Top-Up):** Allow a one-time self-service click that automatically grants +10 or +20 bonus credits (`topup_count < 1`) while logging high engagement, after which subsequent clicks prompt the user to email `dan@danielmherman.com`.
- [ ] **Fallback for Unauthenticated Users:** Ensure visitors without active sessions receive pre-rendered, zero-cost sandbox data instead of triggering live model inference or BigQuery operations.

### Phase 3: Compute & Scaling (Cloud Run & Django)
- [ ] **Switch Django to ASGI for I/O Concurrency:** Run Django under Gunicorn using `uvicorn.workers.UvicornWorker` (2 workers, 2 vCPU, 2 GiB RAM) so each container can hold 25–40 concurrent connections waiting on Vertex AI and BigQuery network I/O without thread starvation.
- [ ] **Right-Size Cloud Run Concurrency & Instance Caps:** Set `--concurrency=25` and `--max-instances=5` on both the Django backend and MCP orchestration services. Five instances at 25 concurrency caps capacity at 125 simultaneous requests—accommodating 100 concurrent users while preventing runaway cloud spend.
- [ ] **Enable CPU Boost:** Turn on startup CPU boost for the MCP orchestration and Django Cloud Run containers to minimize "cold start" latency when scaling up for new users.
- [ ] **Configure Min Instances (Launch Week Only):** Temporarily set `--min-instances=1` (or 2) on critical backend services during the initial launch window to ensure the first wave of users hits warm containers.
- [ ] **Review Django DB Connections:** Ensure Django's database connection settings (e.g., `CONN_MAX_AGE = 60` or Django connection pooling) are optimized for serverless environments so 5 containers under sudden load do not exhaust PostgreSQL connection limits.

### Phase 4: AI Orchestration & Data Warehouse
- [ ] **Vertex AI Quota Check (100 Burst Verification):** Check Google Cloud regional Quota limits for the target Gemini models (Requests Per Minute and Tokens Per Minute). Ensure quotas exceed **300 RPM** and **1M TPM** to absorb 100 concurrent users firing `/risk` within the same 60-second window.
- [ ] **Implement Exponential Backoff:** Ensure the orchestration layer gracefully handles HTTP `429 (Resource Exhausted)` errors from Vertex AI with automated jittered retries (initial 1s delay, factor 2, max 3 retries) rather than dropping the user's chat request.
- [ ] **Optimize BigQuery SQL Agent:** Ensure the Text-to-SQL agent explicitly injects the `@patient_id` parameter to limit BigQuery query scopes, minimizing slot contention and processing overhead, and enforce `maximum_bytes_billed` (e.g., `100 MB`) per query in the client config.

### Phase 5: Telemetry, Observability & Alerting (Cloud Operations & OpenTelemetry)
- [ ] **Structured JSON Logging & Trace Correlation:** Configure Django (`python-json-logger`) and the MCP orchestration layer to emit structured JSON logs to `stdout` so GCP Cloud Logging automatically indexes severity and payload fields. Extract the `X-Cloud-Trace-Context` header on incoming requests and attach `logging.googleapis.com/trace` to every log entry so a single user `/risk` command correlates logs across Django, MCP, Vertex AI, and BigQuery.
- [ ] **Distributed Tracing (OpenTelemetry + Cloud Trace):** Instrument the Django backend and MCP service with the OpenTelemetry SDK (`opentelemetry-exporter-gcp-trace`). Wrap critical spans to isolate latency bottlenecks:
  - `span:auth_and_credit_check` (PostgreSQL lock & decrement time)
  - `span:mcp_intent_and_sql_gen` (Vertex AI Turn 1 latency)
  - `span:bigquery_patient_execution` (BigQuery job execution & slot wait time)
  - `span:mcp_clinical_synthesis_a2ui` (Vertex AI Turn 2 latency)
- [ ] **Custom Log-Based Metrics (Business & Quota KPIs):** Create structured log events and register GCP Log-Based Metrics for critical business guardrails:
  - `subscriber_cap_reached_count`: Emitted whenever a 101st+ user hits the oversubscribed gate (`event_type="OVERSUBSCRIBED_BLOCK"`).
  - `credit_depletion_count`: Emitted when a user hits `credits_remaining == 0` (`event_type="CREDITS_EXHAUSTED"`).
  - `credit_topup_requested`: Emitted when a user clicks the **"Request More Credits"** button (`event_type="TOPUP_REQUESTED"`).
  - `vertex_token_consumption`: Log `input_tokens`, `output_tokens`, and `model_name` on every LLM turn to track real-time token burn per user.
- [ ] **BigQuery `INFORMATION_SCHEMA` Telemetry:** Create a lightweight monitoring view over `region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT` filtered by your service account labels (`agent:text_to_sql`) to monitor `total_bytes_billed`, `total_slot_ms`, and failed/blocked SQL queries in real time.
- [ ] **Cloud Monitoring Dashboards & Proactive Alerting:** Build a unified "Launch Command Center" dashboard in GCP Cloud Monitoring and wire notification channels (Email to `dan@danielmherman.com` + Slack/Webhook) for the following alert policies:
  - **High-Intent Lead Alert:** Immediate notification when `event_type="OVERSUBSCRIBED_BLOCK"` or `event_type="TOPUP_REQUESTED"` fires, including the user's email address.
  - **Vertex AI 429 Quota Warning:** Alert if Vertex AI `429 Resource Exhausted` retry events exceed 5 occurrences within a 1-minute rolling window.
  - **Latency & Error SLA Breach:** Alert if Cloud Run $P_{95}$ latency on `/api/chat/command` exceeds `8,000ms` for 5 consecutive minutes or HTTP `5xx` rate exceeds `1%`.
  - **GCP Budget & Billing Alert:** Set programmatic GCP Billing Budget alerts at **$25**, **$50**, and **$100** thresholds to guarantee zero billing surprises.
- [ ] **Frontend Real User Monitoring (Lit / A2UI Error Boundary):** Add a lightweight global error boundary (`window.onunhandledrejection` and A2UI render error catchers) in the Lit frontend that POSTs client-side rendering failures or broken stream payloads to `/api/telemetry/client-error` for visibility into browser-side issues.

### Phase 6: Live Load Testing (Test-in-Prod Strategy)
- [ ] **Prepare Synthetic Accounts:** Create a script to generate test users in the Django DB with a specific prefix (`loadtest_user_XX@gmail.com`) for easy post-test cleanup.
- [ ] **Execute Boundary & Step-Load Tests:** Run the test suite during an off-peak window. First verify the 101st-user signup gate and 21st-credit depletion gate, then ramp from 10 concurrent users to 50, and hold at 100.
- [ ] **Validate Telemetry & Alerting Pipelines:** During the load test, verify that distributed traces render end-to-end in Cloud Trace, custom log-based metrics increment accurately, and the `OVERSUBSCRIBED_BLOCK` alert fires cleanly without dropping log entries.
- [ ] **Data Cleanup:** Run a SQL script to purge all `loadtest_` users, their chat histories, and associated logs from the production database and data warehouse.

---

## 3. Comprehensive Load & Boundary Testing Suite

### Stage A: Atomic Gate & Race-Condition Verification
1. **The 101st User Signup Test:** Pre-seed the database with 95 `loadtest_user_XX@gmail.com` accounts. Fire 15 concurrent OAuth signup callbacks at $t=0$.
   - **Pass Criteria:** Exactly 5 accounts succeed (bringing the total active non-staff users to 100). The remaining 10 requests receive HTTP `403 Forbidden` with the exact payload: `"The current subscriber base is over subscribed, please reach out directly if you're interested in getting a self-service demo account to dan@danielmherman.com"`, and emit 10 `OVERSUBSCRIBED_BLOCK` structured log events.
2. **The 21st Credit Depletion Test:** Use 1 synthetic user with `credits_remaining = 20` and fire 30 concurrent `/risk` requests at $t=0$.
   - **Pass Criteria:** Exactly 20 requests return HTTP `200 OK`, `credits_remaining` hits `0` (never negative), and 10 requests return HTTP `402/429` rendering the **"Request More Credits"** button.

### Stage B: Full 100-User Concurrency `k6` Script
Save as `tests/load/loadtest_100_concurrent.js` to simulate the complete user journey (`Authenticate -> Select @Andre Ashford -> Trigger /risk command`) with trace propagation enabled:

```javascript
import http from 'k6/http';
import { check, sleep } from 'k6';
import { SharedArray } from 'k6/data';

// Pre-generated session tokens for loadtest_user_001 through loadtest_user_105
const users = new SharedArray('users', function () {
  return JSON.parse(open('./synthetic_tokens.json'));
});

export const options = {
  scenarios: {
    // 1. Ramp to 100 concurrent active users executing clinical workflows
    concurrent_100_journey: {
      executor: 'ramping-vus',
      startVUs: 10,
      stages: [
        { duration: '1m', target: 50 },
        { duration: '2m', target: 100 }, // Hold at forecasted upper bound
        { duration: '30s', target: 0 },
      ],
      exec: 'clinicalJourney',
    },
    // 2. Test the 101st+ user oversubscription gate under full load
    oversubscribed_signup_gate: {
      executor: 'per-vu-iterations',
      vus: 5,
      iterations: 1,
      startTime: '1m30s', // Fire when system is at peak concurrency
      exec: 'attemptOverflowSignup',
    },
  },
  thresholds: {
    'http_req_duration{step:select_patient}': ['p(95)<500'],
    'http_req_duration{step:risk_command}': ['p(95)<8000'], // LLM + BQ SLA
    'http_req_failed{expected_response:true}': ['rate<0.01'],
  },
};

const BASE_URL = __ENV.TARGET_URL || 'https://app.danielmherman.com';

export function clinicalJourney() {
  const user = users[(__VU - 1) % 100];
  const headers = {
    Authorization: `Bearer ${user.token}`,
    'Content-Type': 'application/json',
    'X-LoadTest-Run': 'true', // Tag for filtering in Cloud Logging / Trace
  };

  // Step 1: Select @Andre Ashford
  const selectRes = http.post(
    `${BASE_URL}/api/context/patient`,
    JSON.stringify({ patient_handle: '@Andre Ashford' }),
    { headers, tags: { step: 'select_patient' } }
  );
  check(selectRes, { 'patient selected (200)': (r) => r.status === 200 });

  sleep(1);

  // Step 2: Trigger /risk command (consumes 1 credit)
  const riskRes = http.post(
    `${BASE_URL}/api/chat/command`,
    JSON.stringify({ command: '/risk', patient_handle: '@Andre Ashford' }),
    { headers, tags: { step: 'risk_command' } }
  );

  check(riskRes, {
    'valid response or clean credit cap': (r) =>
      r.status === 200 ||
      (r.status === 402 && r.body.includes('Request More Credits')),
  });

  sleep(2);
}

export function attemptOverflowSignup() {
  const res = http.post(
    `${BASE_URL}/api/auth/signup-callback`,
    JSON.stringify({ email: `overflow_user_${__VU}@gmail.com` }),
    { headers: { 'Content-Type': 'application/json', 'X-LoadTest-Run': 'true' } }
  );

  check(res, {
    'blocked with oversubscribed message': (r) =>
      r.status === 403 &&
      r.body.includes('The current subscriber base is over subscribed') &&
      r.body.includes('dan@danielmherman.com'),
  });
}
```

> **Cost-Control Tip for Dry Runs:** Pass a custom header (`X-LoadTest-Mode: mock-llm`) during initial infrastructure load tests so the MCP service simulates a `2.5s` async delay and executes a real parameterized BigQuery read without consuming 2,000 live Vertex AI requests. Run a final pass against live Vertex AI once Cloud Run scaling, database pooling, and Cloud Trace spans are verified.

---

## 4. Unit Economics & Budgeting Exercise

Below is the projected cost model for **100 maximum active users**, each consuming their full **20 request credits** (**2,000 total bounded requests**).

### Workload Assumptions per Request (`/risk` Command)
* **LLM Calls per Request:** 2 turns via MCP (Turn 1: Intent routing + Text-to-SQL generation; Turn 2: Clinical synthesis + A2UI payload generation).
* **Token Footprint per Request:** ~4,000 Input Tokens (system prompt, schema context, clinical records) + ~1,000 Output Tokens (reasoning + A2UI components).
  * *Per-User Total (20 requests):* **80,000 Input Tokens** / **20,000 Output Tokens**.
  * *Cohort Total (2,000 requests):* **8.0M Input Tokens** / **2.0M Output Tokens**.
* **BigQuery Footprint per Request:** 1 parameterized query scanning ~15 MB (billed at the 10 MB per-query minimum, or **$6.25/TB** on-demand).
* **Cloud Run Compute per Request:** 4 seconds wall-clock time across 2 services (Django + MCP) configured at 2 vCPU / 2 GiB RAM.
* **Telemetry & Observability Footprint:** ~15 KB of structured JSON logs and 4 OpenTelemetry trace spans per request (~30 MB total logs and 8,000 spans across 2,000 requests—100% covered by GCP Cloud Operations free tier limits of 50 GiB/mo for logs and 2.5M spans/mo for Cloud Trace).

### Variable Cost Breakdown: Per User (20 Credits) vs. Full Cohort (100 Users)

| Cost Component | Unit Pricing (GCP 2026 Rates) | Cost per User (20 Requests) | Total Variable Cost (100 Users / 2,000 Req) |
| :--- | :--- | :--- | :--- |
| **Vertex AI (Option A: Gemini 3.8 Flash)** | $0.75 / 1M In • $3.75 / 1M Out *(Intro)*<br>*($1.50 / $7.50 standard rate)* | **$0.135**<br>*($0.270 standard)* | **$13.50**<br>*($27.00 standard)* |
| **Vertex AI (Option B: Gemini 3.1 Pro)** | $2.00 / 1M In • $12.00 / 1M Out | **$0.400** | **$40.00** |
| **BigQuery On-Demand** | $6.25 per TB scanned (~20 MB/req) | **$0.0025** | **$0.25** |
| **Cloud Run Active Compute** | $0.000024/vCPU-s • $0.0000025/GiB-s | **$0.003** | **$0.30** |
| **Cloud CDN & Egress** | ~$0.08 / GB (~10 MB assets/user) | **$0.0008** | **$0.08** |
| **Cloud Logging, Monitoring & Trace** | Free tier (< 50 GiB logs / < 2.5M spans) | **$0.0000** | **$0.00** |
| **Total Variable Cost (Flash Tier)** | **Low-Latency Workhorse** | **~$0.14 / user** | **~$14.13 total** |
| **Total Variable Cost (Pro Tier)** | **Heavy Reasoning Tier** | **~$0.41 / user** | **~$40.63 total** |

### Fixed Baseline Infrastructure Cost (Launch Week vs. Monthly Run-Rate)
* **Global External Application Load Balancer + Cloud Armor:** ~$18.00/month base forwarding rule + ~$5.00/month for Cloud Armor WAF policy = **~$23.00/month** (~$5.75 for launch week).
* **Cloud Run `--min-instances=1` (Launch Week Only, 168 hours):** Keeping 1 warm container (1 vCPU, 2 GiB RAM with CPU throttled when idle) costs **~$3.50/week** per service (**~$7.00/week** across Django + MCP). Resetting `--min-instances=0` after launch week drops idle compute to **$0.00**.
* **Cloud SQL for PostgreSQL (`db-f1-micro` or `db-g1-small`):** **~$10.00 to $28.00/month**.

### Summary Metrics
* **Marginal Variable Cost per User (20 requests):** **$0.14** (Flash) to **$0.41** (Pro). Granting an additional 20 requests via the **"Request More Credits"** button costs less than **$0.41** per high-intent lead.
* **All-In Cost (Amortizing ~$40/mo fixed infra across 100 users):** **~$0.54 to $0.81 per user** for the entire 100-user launch cohort (**~$54 to $81 total spend**).