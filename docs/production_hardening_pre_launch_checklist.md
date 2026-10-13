# Production Hardening & Pre-Launch Checklist

This checklist is designed for a fully serverless GCP architecture (Cloud Run, BigQuery, Vertex AI) using a Django backend and Lit/A2UI frontend. Complete these steps after locking in the final UX and before driving public traffic (e.g., via a LinkedIn post).

## 1. Edge & Content Delivery (The Front Door)
- [ ] **Configure Cloud CDN:** Ensure Cloud CDN is enabled on the Load Balancer backend service pointing to the frontend Lit application. Verify static assets (JS, CSS, A2UI components) are being cached at the edge to minimize Cloud Run egress costs.
- [ ] **Tune Cloud Armor Rate Limiting:** Verify your WAF/rate-limiting rules are active. Set a reasonable threshold for IP-based rate limiting to prevent bot floods before they reach Cloud Run or trigger authentication flows.
- [ ] **Enforce HTTPS Redirects:** Ensure the Global Load Balancer is strictly redirecting all HTTP traffic to HTTPS.

## 2. Authentication & Cost Guardrails
- [ ] **Implement Google OAuth:** Integrate `django-allauth` and test the "Continue with Google" flow. Ensure it smoothly creates a local Django `User` and redirects back to the main canvas.
- [ ] **Enforce Token Buckets:** Verify the request allowance middleware is functioning. Ensure authenticated users are capped (e.g., 10 live queries/day) and that the system elegantly returns an A2UI rate-limit warning rather than a backend crash.
- [ ] **Fallback for Unauthenticated Users:** Ensure visitors without active sessions receive pre-rendered, zero-cost sandbox data instead of triggering live model inference or BigQuery operations.

## 3. Compute & Scaling (Cloud Run & Django)
- [ ] **Enable CPU Boost:** Turn on startup CPU boost for your MCP orchestration and Django Cloud Run containers to minimize "cold start" latency when scaling up for new users.
- [ ] **Configure Min Instances (Launch Week Only):** Temporarily set `--min-instances=1` (or 2) on critical backend services during the initial launch window to ensure the first wave of users hits warm containers.
- [ ] **Review Django DB Connections:** Ensure Django's database connection settings (e.g., `CONN_MAX_AGE`) are optimized for serverless environments so you don't exhaust local PostgreSQL connection limits under sudden load.

## 4. AI Orchestration & Data Warehouse
- [ ] **Vertex AI Quota Check:** Check your Google Cloud region's Quota limits for the specific Gemini Enterprise models you are using (Requests Per Minute and Tokens Per Minute).
- [ ] **Implement Exponential Backoff:** Ensure your orchestration layer gracefully handles HTTP 429 (Resource Exhausted) errors from Vertex AI with automated retries, rather than dropping the user's chat request.
- [ ] **Optimize BigQuery SQL Agent:** Ensure your Text-to-SQL agent explicitly injects the `patient_id` parameter to limit BigQuery query scopes, minimizing slot contention and processing overhead.

## 5. Live Load Testing (Test-in-Prod Strategy)
- [ ] **Prepare Synthetic Accounts:** Create a script to generate test users in the Django DB with a specific prefix (e.g., `loadtest_user_XX@gmail.com`) for easy post-test cleanup.
- [ ] **Write the Load Script:** Write a Locust or k6 script simulating the full user journey: Authenticate -> Select `@Andre Ashford` -> Trigger `/risk` command.
- [ ] **Execute Step-Load Test:** Run the script during an off-peak window. Start with 10 concurrent users, ramp to 50, then to 100.
- [ ] **Monitor Telemetry:** During the test, monitor Cloud Run Response Times (P95), Vertex AI Quota usage, and BigQuery execution times via GCP Cloud Logging.
- [ ] **Data Cleanup:** Run a SQL script to purge all `loadtest_` users, their chat histories, and associated logs from the production database and data warehouse.