# Layer 2 — Edge (the front door)

Status: audited 2026-09-11; implemented, verified live, and torn down 2026-09-13.
Decision D: the reference edge (load balancer + Cloud Armor) is defined in
Terraform, was brought up, was verified, and is now off — the certificate, DNS
zone and domain mapping remain, so `infra/edge/edge.sh up` restores it in
about ten minutes. Gaps 2 and 3 closed.

Sections 3–5 record the state **as audited**, before any change. Section 7
records what changed and the evidence.

---

## 1. What the edge is

The edge is everything a request passes through **before it reaches your
application code**. It is the first place an attacker, a bot, or a runaway
script meets your system, and it is the cheapest place to stop them, because
nothing behind it has spent any money yet.

```
  internet ──▶ [ DNS ] ──▶ [ TLS termination ] ──▶ [ ingress rules ] ──▶ [ auth ] ──▶ [ rate limit ] ──▶ app
                              who am I talking to?     who may connect?    who are you?   how often?
```

Terms used in this document:

| Term | Meaning |
|---|---|
| TLS termination | Where the HTTPS connection is decrypted. On Cloud Run, Google does this for you before the request reaches your container. |
| Ingress | A Cloud Run setting that controls *where* traffic may come from: anywhere on the internet (`all`), only from inside your Google Cloud project (`internal`), or only through a Google load balancer (`internal-and-cloud-load-balancing`). |
| Invoker permission | A Cloud Run IAM role. If `allUsers` has it, anyone can call the service. If only a service account has it, callers must present a Google identity token for that account. |
| Identity token | A short-lived signed token that proves which Google service account is making a request. Cloud Run checks it before your code runs. |
| External Application Load Balancer | Google's global front door. Gives you one public IP, a managed certificate, and a place to attach Cloud Armor. |
| Cloud Armor | Google's web application firewall. Attaches to a load balancer. Does rate limiting, IP allow/deny lists, and blocks common attack patterns. |
| Identity-Aware Proxy (IAP) | Google's login wall for internal users. Sits at the load balancer; only Google-account holders you list get through. |
| Rate limiting | Refusing requests beyond N per time window, per IP or per user. Protects against abuse and bounds cost. |
| `run.app` URL | The default public hostname Cloud Run gives every service. Reachable by anyone who knows it unless ingress or IAM says otherwise. |

**Layer 2 is: how the public reaches Django, and how Django reaches the
agent.** It is mostly configuration, not code.

---

## 2. What Google requires of this layer

From *Single-agent AI system using ADK and Cloud Run* (reviewed 2025-12-09),
security section, and *Well-Architected Framework, AI/ML perspective:
Security* (reviewed 2025-11-26), "Prevent malicious queries":

1. **Controlled ingress for the public service.** Disable the default
   `run.app` URL of the frontend and put an external Application Load Balancer
   in front of it. The load balancer owns the certificate and the public IP.
2. **Cloud Armor on that load balancer** for request filtering, DDoS
   protection, and rate limiting.
3. **User authentication at the door.** Internal users: Identity-Aware Proxy.
   External users: Identity Platform or Firebase Authentication, with the
   application handling sign-in and making authenticated calls.
4. **Service-to-service calls authenticated with identity tokens**, least
   privilege on who may invoke each private service.
5. **Rate limiting and throttling** per IP or per user; monitor request
   volume; alert on spikes and on unusual geographic or time-of-day patterns.
6. **Request validation at the boundary** — reject malformed or oversized
   input before it reaches the model.

Items 1–3 are Google's reference topology. Items 4–6 are stated as
requirements regardless of topology.

---

## 3. How this application implements the layer

Sections 3–5 are the audit as it stood on 2026-09-11, before any change. They
are kept as the "before" record; section 7 is what was done about them.

### 3.1 Deployed topology (verified with `gcloud`, project `trim-icon-498815-a0`, region `us-east1`)

| Service | Ingress | Who may invoke | Public hostname |
|---|---|---|---|
| `danielmherman` (Django) | `all` | `allUsers` | `danielmherman.com`, `www.danielmherman.com` via Cloud Run domain mapping; also `danielmherman-….run.app` |
| `agent` | `all` | `website-sa`, the project default compute service account, and one user account | `agent-….run.app` |
| `mcp-server` | `all` | (not audited this layer) | `mcp-server-….run.app` |
| `langfuse-web`, `langfuse-worker` | `all` | (observability; out of scope) | — |

No external Application Load Balancer, no Cloud Armor policy, no
Identity-Aware Proxy exist in the project. All five services are reachable
from the internet at the network level; the agent is protected only by the
invoker check.

Scaling limits, which are the cost ceiling at the door: every service has
`maxScale = 3` and `containerConcurrency = 80`; no minimum instances.

### 3.2 Public traffic → Django

Repository `danielmherman`. Everything the edge would normally do is done by
Django itself:

| Control | Where | What it does |
|---|---|---|
| Force HTTPS | `danielmherman/settings.py` 442–443 | Redirect HTTP→HTTPS; trust Cloud Run's `X-Forwarded-Proto`. |
| Strict transport | `settings.py` 450 | HSTS, 30 days. |
| Secure cookies | `settings.py` 444–445, 137 | Session and CSRF cookies HTTPS-only; session ends when the browser closes. |
| Host allow-list | `settings.py` 88; `cloudbuild.yaml` 55 | `ALLOWED_HOSTS` = `danielmherman.com, www.danielmherman.com, .run.app`. |
| Login required | `demo/views.py` 80, 107 | Both demo routes require a logged-in user. Accounts are issued; there is no signup route (`danielmherman/urls.py` 15–16). |
| Login lockout | `settings.py` 145–151, 273 | django-axes: 5 failed attempts locks the username for 1 hour. State in a database cache table so it holds across instances (`settings.py` 355–357). |
| Admin path hidden | `settings.py` 132 | Admin is at a non-default path from an environment variable. |
| Per-user quota | `demo/models.py` 55 (`DemoQuota`), `consume` 95, `refund` 128 | 10 agent calls per user per day (`settings.py` 102), refund on failure capped at 3/day (108). Claimed **before** the agent is called. |
| Input bounds | `demo/views.py` 22, 25, 119–136 | JSON object required; question ≤ 2000 chars; admission id must exist in the demo cohort; unknown chip rejected — all before quota is touched. |
| Concurrency bound | `demo/agent_client.py` 44 | At most 4 in-flight agent calls per instance. |
| CSRF | Django middleware; `demo_flow.js` 622 | Required on every POST. |

### 3.3 Django → agent

| Control | Where | What it does |
|---|---|---|
| Identity token per request | `demo/agent_client.py` 47 (`_id_token`) | Django mints a token for the agent URL from the Cloud Run metadata server. |
| Agent requires the token | `services/agent/http.py` 48, 69–75 | Rejects requests with no `Authorization` header when running on Cloud Run. (Cloud Run itself has already rejected anyone without a valid invoker token; this is a second check.) |
| Agent re-validates input | `services/agent/http.py` 40, 80–83; `services/agent/contracts.py` 53 | Same 2000-char bound and shape check, independently of Django. |
| Agent timeout | `services/agent/http.py` 92–99 | Bounded wall-clock per request; returns 504. |

### 3.4 What is absent

- No load balancer, Cloud Armor, or IAP.
- No per-IP rate limiting anywhere. The only limits are per authenticated
  user (quota) and per username (login lockout). An unauthenticated client can
  hit `/accounts/login/` or any public page as fast as Cloud Run will serve.
- No request ID. Cloud Run attaches `X-Cloud-Trace-Context` to every inbound
  request, but Django does not read it and does not forward it to the agent
  (`demo/agent_client.py` sends only `Authorization`), and the agent does not
  read it either. A single user action cannot be followed across the two
  services in logs.
- The default `run.app` hostname for Django is enabled and in
  `ALLOWED_HOSTS`, so the site is reachable at two hostnames.
- The project default compute service account and a personal user account
  hold invoker on the agent, in addition to `website-sa`.

---

## 4. Current state against the requirement

As audited. The state after the change is verified in 7.3; nothing in this
table is still true of an edge that is up.

| Google requires | What exists | Met? |
|---|---|---|
| Load balancer in front of the frontend; default URL disabled | Cloud Run domain mapping directly to the service; `run.app` URL live | **No** |
| Cloud Armor: filtering, DDoS, rate limiting | None. Per-user quota and per-username lockout only | **No** — the *rate-limiting intent* is partly met in-app for authenticated paths; nothing for anonymous traffic |
| User authentication at the door | Django login, issued accounts, lockout — in the app, not at the edge | **Partly** — authentication exists and is sound; it is not at the edge |
| Service-to-service identity tokens, least privilege | Token per request; agent checks it. Two extra invokers on the agent beyond `website-sa` | **Mostly** — trim invokers |
| Rate limit / throttle; monitor volume | Quota bounds spend per user; `maxScale=3` bounds spend globally. No per-IP limit; no volume alerting | **Partly** |
| Validate at the boundary | Django and agent both validate shape and size before any spend | **Yes** |
| Request ID stamped and propagated | None | **No** |

---

## 5. Gaps

**Gap 1 — No edge in front of Django.** Cloud Run's domain mapping is a
convenience, not a front door. There is no place to attach a WAF, no
per-IP rate limit, and the default hostname is exposed. Google's reference is
LB → Cloud Armor → Cloud Run with `run.app` disabled.

**Gap 2 — No request ID across the two services.** Every request already
carries `X-Cloud-Trace-Context`; nobody reads it. Django should forward it to
the agent; both should log it. Ten lines total. This is the prerequisite for
end-to-end tracing later.

**Gap 3 — Over-broad invoker on the agent.** The default compute service
account and a personal account can call the agent directly, bypassing quota.
Only `website-sa` should.

---

## 6. Design decisions

Gaps 2 and 3 are small, free, and unambiguous; they should be done regardless.
Gap 1 is the decision.

| Option | What it is | Monthly cost (approx.) | Trade-off |
|---|---|---|---|
| **A. Full reference topology.** | Global external Application Load Balancer + serverless NEG → Django. Cloud Armor policy with per-IP rate limit. `run.app` ingress restricted to `internal-and-cloud-load-balancing`. Managed certificate for both hostnames. | LB ≈ $18 (forwarding rule) + Cloud Armor ≈ $5/policy + $1/rule + per-request charges; call it **$25–30/month** | Exactly what Google draws. Strongest interview story. Ongoing cost for a demo that is mostly idle. |
| **B. Restrict ingress, defer LB.** | Set Django ingress to `internal-and-cloud-load-balancing` *without* creating an LB → site goes dark. Not viable alone; listed for completeness. | — | Not an option on its own. |
| **C. Harden what exists; document the LB as a deliberate deferral.** | Trim invokers (Gap 3). Forward request ID (Gap 2). Add per-IP rate limiting on `/accounts/login/` and `/demo/a2ui/ask/` inside Django (one middleware, database-backed like axes). Keep domain mapping. Record: "LB + Cloud Armor is the production step; deferred for cost on a portfolio demo; here is the exact `gcloud` sequence to enable it." | **$0** | Every *intent* Google lists is met; the *topology* is not. Interview answer becomes "I know exactly what the edge should be and why I haven't paid for it yet." |
| **D. A, but tear down after verification.** | Build A, verify live, screenshot, record the commands, then delete the LB and Cloud Armor policy and revert ingress to `all`. | ≈ $1–2 one-off | Proves you can do it; nothing left running. Site must be re-pointed twice (DNS to LB IP and back). |

Recommendation: **C now, with the LB sequence written down**, and revisit at
the end of the twelve layers when you decide what stays running. If you want
the topology to exist even briefly, D is the way to do it. A is the right
answer if this becomes a real product rather than a portfolio piece.

Decision (2026-09-12): **D, with two refinements.**

1. *Global* external Application Load Balancer, not regional. Google's
   reference architecture draws the regional one, which bills for proxy
   instances (about $80/month idle). The global one bills only the forwarding
   rule and Cloud Armor policy (about $25/month, or $0.03/hour when up). For
   a serverless backend in one region the two are functionally the same.
2. The edge is code, not a runbook. Terraform owns every edge resource and a
   single variable decides whether it exists. "Tear down" is `edge_enabled =
   false`; "stand up" is `true`. To make that repeatable, DNS moved from
   GoDaddy to Cloud DNS so the records can be flipped by the same apply.

Why D and not C: the interview answer "I know what the edge should be" is
weaker than "here is the edge, here is the evidence it worked, here is why it
is off right now." The cost of D over C is one afternoon and about a dollar.

---

## 7. What changed

Three pieces of work, in the order they were done. All infrastructure lives in
the `danielmherman` repository under `infra/`, because the edge fronts the
website; the agent repository is touched only for Gap 2.

### 7.1 DNS moved from GoDaddy to Cloud DNS

Terraform cannot flip records at GoDaddy. Without that, "stand up the edge"
would always end with a manual step, and the teardown could not be automated
at all. So the zone moved first.

| Step | Artifact |
|---|---|
| Exported GoDaddy's zone, then wrote the record set that was actually live (4 `A` and 4 `AAAA` to Google's `ghs` front end, `www` CNAME, MX and SPF for ImprovMX, DMARC, one site-verification TXT) | `infra/dns/danielmherman.com.zone` (22 lines) |
| Snapshot of what GoDaddy served, for comparison | `infra/dns/baseline-godaddy-2026-09-13.txt` |
| Snapshot of what Cloud DNS served after import, checked identical | `infra/dns/clouddns-2026-09-13.txt` |
| Cloud DNS public zone `danielmherman`, nameservers `ns-cloud-d1..d4.googledomains.com`; GoDaddy nameservers switched; propagation confirmed on 8.8.8.8, 1.1.1.1, 9.9.9.9 | (GCP resource) |

Record TTLs were lowered from 600 to 300 seconds so a flip takes five minutes
to reach every resolver, not ten.

### 7.2 The edge in Terraform (`infra/edge/`)

Seven files, about 510 lines including the script. State is in a versioned,
private GCS bucket so it survives a laptop (`versions.tf` 14–15).

Two booleans drive everything, because "the load balancer exists" and "the
public is being sent to it" are different facts and the interesting moment is
when they disagree:

| `edge_enabled` | `dns_points_at_edge` | State |
|---|---|---|
| `false` | `false` | nothing running; DNS on the Cloud Run domain mapping |
| `true` | `false` | load balancer built and serving, but no traffic sent to it yet — the drain state, used in both directions |
| `true` | `true` | live at the load balancer |

`dns_points_at_edge = true` with `edge_enabled = false` is refused at plan
time by a precondition (`dns.tf` 32): DNS cannot publish an address that does
not exist. Verified by asking for exactly that combination and getting
"Resource precondition failed".

| File | What it defines |
|---|---|
| `variables.tf` | `edge_enabled` (28) and `dns_points_at_edge` (37), both default `false`; domain, zone, service names; the three rate-limit numbers (55, 61, 67). |
| `certificate.tf` | Certificate Manager: DNS authorization for apex and `www` (10, 15), the `_acme-challenge` CNAMEs those need (22, 30), the managed certificate (38), a certificate map (50) and its two entries (54, 61). **Always present** — this is what lets the edge come up in minutes instead of waiting for a new certificate each time. |
| `edge.tf` | Everything that costs money, each with `count = edge_enabled ? 1 : 0` (13): static IP (16); serverless NEG pointing at the Cloud Run service (21); Cloud Armor policy (33) with two path-scoped limits; backend service in `EXTERNAL_MANAGED` scheme carrying the policy (114); HTTPS proxy using the certificate map (138); forwarding rules on 443 (145) and 80 (171); and an HTTP→HTTPS redirect URL map (155). |
| `dns.tf` | Driven by `dns_points_at_edge`, not by `edge_enabled`. The apex `A` record publishes the LB address or Google's four `ghs` addresses (29); the `AAAA` record exists only while DNS is on the domain mapping (41), because the load balancer has no IPv6; `www` is a CNAME to the apex or to `ghs.googlehosted.com.` (55). All TTL 300. |
| `outputs.tf` | `edge_enabled`, `dns_points_at_edge`, `edge_ip`, `certificate_state`, the live DNS answers. |
| `edge.sh` | `up` (34) / `down` (79) / `status` (94). Needed because the Cloud Run service is owned by `cloudbuild.yaml`, so its ingress setting is changed here with `gcloud`, and that change has to be ordered against the DNS change with a TTL wait between them. |

**Both directions drain, in three phases.** The first version of `down`
flipped DNS and destroyed the load balancer in a single apply, which meant a
resolver still holding the load-balancer address — up to one TTL — was sent to
an address with no forwarding rule behind it. `up` never had that problem
because it was already two steps; `down` now uses the same shape, which is why
the two booleans exist at all:

| | `up` | `down` |
|---|---|---|
| 1 | build the load balancer, DNS still on the domain mapping (39–41) | reopen ingress so the domain mapping can serve the moment DNS moves (83) |
| 2 | wait until the LB actually answers, abort if it never does (42–69) | move DNS back to the domain mapping, LB still running (86–87) |
| 3 | move DNS to the LB (72–73) | wait one TTL (88) |
| 4 | wait one TTL (74) | destroy the load balancer (90–91) |
| 5 | close ingress to the LB only (76) | — |

Neither direction ever leaves a client with an address that does not answer,
and neither depends on the other being run first.

### 7.3 Verified live, 2026-09-13

Baseline apply (`edge_enabled=false`): 8 resources added, 3 changed (TTL
only), 0 destroyed. Certificate went `PROVISIONING` → `ACTIVE` in about
five minutes.

`edge.sh up`: 10 resources added, 2 changed, 1 destroyed (the `AAAA`).
Backend service creation was the slow step, 2m30s. Then:

| Check | Result |
|---|---|
| `https://danielmherman.com/` resolved by public DNS | `200`, remote IP `136.68.137.130` (the LB), response headers `via: 1.1 google`, `server: Google Frontend`, `alt-svc: h3` |
| Certificate served | `CN=danielmherman.com`, issuer Google Trust Services WR3, valid 90 days |
| `https://www.danielmherman.com/` | `200` |
| `http://danielmherman.com/` | `301 → https://danielmherman.com:443/` (Google's redirect writes the explicit port; browsers treat it as canonical) |
| Cloud Armor: 90 requests to `/accounts/login/` in well under a minute | 57 × `200`, 33 × `429`. Load balancer log: `enforcedSecurityPolicy.name=danielmherman-edge`, `configuredAction=RATE_BASED_BAN`, `rateLimitAction.outcome=BAN_THRESHOLD_EXCEED`. All requests from that IP then `429` for the 300-second ban. |
| Default hostname `danielmherman-….run.app` after ingress restricted | `404` (Cloud Run answers 404, not 403, for traffic that fails the ingress rule — it does not confirm the service exists) |
| Old `ghs` path (a resolver with a stale answer) after ingress restricted | `404` — the domain mapping is still configured but the ingress rule refuses it |
| Cloud Run request log for a proxied request | `httpRequest.remoteIp` is the real client, `serverIp` is the LB. The LB passes the client address through, so Django's own logs and django-axes keep seeing real IPs. |

Two things observed that were not planned: the rate limit admitted 57, not
60, before banning — Cloud Armor's counting is approximate by design; and
during the 20 minutes the edge was up, the load balancer log showed a
third-party IP make 15 requests. There is real background traffic.

`edge.sh down` returns the site to the Cloud Run domain mapping and destroys
the ten billable resources. What remains: the Cloud DNS zone, the certificate
and its map, the DNS authorizations, and the `infra/edge` state. Bringing the
edge back is one command and about ten minutes, most of it the TTL wait.

### 7.3.1 The drain, measured

The whole point of the two-variable design is the window between "DNS has
moved" and "the load balancer is gone". That window was open for 330 seconds
during the teardown, and both paths answered throughout it:

| Path | What a client sees |
|---|---|
| A resolver that re-resolved — DNS now publishes the four `ghs` addresses | `https://danielmherman.com/` → **200**, answered over IPv6 (`2001:4860:4802:38::15`), because restoring `dns_points_at_edge = false` brings the `AAAA` record back and the load balancer never had one |
| A resolver still holding `136.68.137.130` | `https://danielmherman.com/` with that address pinned → **200**, `server: Google Frontend` — the load balancer is still up and still serving |
| `www`, re-resolved | **200** over IPv6 |

The three DNS changes that produced this were the only thing the drain apply
touched: apex `A` back to the four `ghs` addresses, `AAAA` recreated, `www`
back to `ghs.googlehosted.com.` — `Plan: 1 to add, 2 to change, 0 to destroy`,
with the load balancer untouched. Had the old single-apply `down` been used,
the second row of that table would have been a connection failure for up to
five minutes instead of a 200.

Once the TTL had expired, the destroy ran: **0 to add, 0 to change, 10 to
destroy**, about 2.5 minutes, the backend service (1m2s) and the two
forwarding rules (22s) being the slow parts. Afterwards, every list is empty —
forwarding rules, security policies, global addresses, backend services, both
URL maps, both proxies, and the serverless NEG. Still present: the certificate
(`ACTIVE`), its map and both DNS authorizations, the Cloud DNS zone, both
Cloud Run domain mappings, and the Terraform state.

Serving checks immediately after: apex `200` over IPv6, `www` `200` over
IPv6, authoritative answers back to four `A` and four `AAAA` records. One
detail worth noticing: `http://danielmherman.com/` now answers `302` from
Django's `SECURE_SSL_REDIRECT` rather than the `301` the load balancer's
redirect URL map used to send. The redirect moves from the edge to the
application when the edge comes down, which is exactly the layering the
`ALLOWED_HOSTS` and HSTS settings were always there to cover.

### 7.4 Gap 3 closed — agent invokers, and what that does not buy

Removed `roles/run.invoker` on `agent` from the default compute service
account and from the personal account. The service's IAM policy now names
exactly one invoker, `website-sa`.

That is narrower than it sounds, and the difference matters:

| Identity | Holds | Can invoke the agent? |
|---|---|---|
| `website-sa` | explicit `roles/run.invoker` | yes, by design |
| personal account | `roles/owner` | yes — Owner contains `run.routes.invoke` |
| default compute service account | `roles/editor`, `roles/run.admin` | yes — `roles/run.admin` contains `run.routes.invoke` |

So the binding list and the effective caller list are two different things.
The removal was still right — it deleted a redundant grant and made the
intended path visible — but "only the website can call the agent" is a claim
about the policy, not about reality. Genuine closure belongs to the security
layer: take `run.admin` and `editor` off the default compute service account,
and decide whether the agent should also refuse anything that did not arrive
through the load balancer (an ingress and network design, not an IAM binding).

Related finding, same theme. `cloudbuild.yaml` line 8 declares
`serviceAccount: cicd-deployer@…` and explains that builds must not run as the
project default service account. They do anyway: the trigger carries its own
service account, and build `a17ace19` reports
`778397675435-compute@developer.gserviceaccount.com`. The declaration in the
file is inert for triggered builds. One flag on the trigger aligns them, and
it belongs to the delivery layer.

### 7.5 Gap 2 closed — one request id across both services, verified in production

Cloud Run stamps `X-Cloud-Trace-Context: TRACE_ID/SPAN_ID;o=1` on every
inbound request. Django now reads it and forwards it; the agent logs it.

| Where | What |
|---|---|
| `demo/agent_client.py` 98 (`trace_id`) | Reads the header from the Django request, keeps the part before the slash. Empty locally and in tests. |
| `demo/agent_client.py` 110 (`ask`), 136–138 | `ask(question, trace='')`; when a trace is present it is sent to the agent as `X-Cloud-Trace-Context`. Failure log line carries it (160). |
| `demo/views.py` 148, 150, 157 | The view extracts the id, passes it, and logs it on the failure path. |
| `services/agent/http.py` 79 | Agent reads the header (or `-`). |
| `services/agent/http.py` 100, 120, 136, 171 | Every error log line for the request ends with `trace=<id>`. The 12-character `correlation_id` the client receives is unchanged; the trace pairs the services, the correlation id pairs client and server. |
| `services/agent/http.py` 178 | One `INFO` line on success: `agent /ask ok trace=… tools=… flags=…`. |
| Tests | `demo/tests.py` 349, 364; `tests/agent/test_error_disclosure.py` 70 |

This is plain-text logging, deliberately. Cloud Logging can correlate on a
structured `logging.googleapis.com/trace` field and show both services in one
trace view; that is the observability layer's job and will build on this.

One thing had to be fixed before any of this was observable. The agent logged
`INFO` and nothing in the process configured logging, so the root logger sat at
its default `WARNING` and the per-request line was discarded before it reached
stdout — a log line nobody can read is not observability.
`services/agent/http.py` 201–204 now calls `logging.basicConfig(level=
logging.INFO)` at process start. `uvicorn`'s own logging config leaves the root
logger alone, so it survives the call below it.

**Deployed and verified in production, 2026-09-13.** Both services were built
and deployed (Django `danielmherman-00098-88t` from commit `ab77773`, agent
`agent-00023-9db`), and two real questions were asked through the browser with
a logged-in user. Each produced one trace id that appears on both sides:

| Time (UTC) | Django request log | Agent log |
|---|---|---|
| 18:10:03 | `trace=b2eebc93510038c576f716d3cf5fd05b`, status 502, 43.45 s | `agent /ask ok trace=b2eebc93510038c576f716d3cf5fd05b tools=2 flags=0` |
| 18:13:43 | `trace=91164950ce9c4f3d0e69c8ff7b8b41d4`, status 502, 3.11 s | `agent /ask ok trace=91164950ce9c4f3d0e69c8ff7b8b41d4 tools=1 flags=0` |

The 502s are expected and unrelated to this change: the MCP tools call live
prediction and retrieval endpoints that are not running, so the agent returned
well-formed payloads whose tool results were errors and Django refunded the
quota credit. The point of the test is the request path, not the answer — the
id is stamped by Cloud Run, forwarded by Django, and logged by the agent, and
that is exactly what the two columns show. Verifying the full live answer path
needs those endpoints up, which belongs to the model-runtime and tools layers.

Cosmetic detail worth knowing: because the service starts as
`python -m services.agent.http`, the record's logger name is `__main__`, not
`services.agent.http`. Filtering agent logs by logger name will not match.

Test status: Django `manage.py test` 84 OK. Agent `pytest tests/agent`
240 passed, 6 skipped, 10 errors — the 10 are the live-Gemini tests, which
fail identically without this change when no cloud credentials are present.

### 7.6 The rate limit was wrong, and real use found it

The first version applied one Cloud Armor rule to every request: 60 per
minute per IP, and 300 seconds of ban once exceeded. That is fine arithmetic
and bad design, and it failed the moment the site was used properly. One load
of the A2UI console pulls dozens of ES modules out of `/static/vendor/` —
`lit`, `lit-html`, `zod`, `preact-signals`, `linkify-it` and their transitive
dependencies — so a single page load is most of the budget, and a refresh
tripped the ban. The load-balancer log shows it: ~40 module requests at
`18:03:28`, and by `18:03:38` every path, including `/favicon.ico`, returning
429 for the full ban.

Two lessons, both worth stating in an interview:

- **A request counter cannot tell a stylesheet from a login attempt.** The
  match is now scoped to the paths that can actually cost money or be abused —
  `/accounts/login` at 30/min and `/demo/a2ui/ask` at 60/min, per IP — and
  everything else is default-allow. Volumetric attacks were never going to be
  stopped by a per-IP request counter; that is Cloud Armor's always-on L3/L4
  protection plus Cloud Run's `maxScale`.
- **A ban is not path-scoped.** `rate_based_ban` bans the client IP for the
  whole policy, so exceeding a limit on the sign-in path also denies
  `/favicon.ico` and the homepage to that address. Scoping the *match* removes
  the false positives; it does not make the ban surgical.

Verified after the change: 120 rapid requests for a vendored module returned
120 × 200; 60 rapid homepage hits returned 60 × 200; the sign-in path was
limited when hammered; and immediately afterwards `/`, `/demo/a2ui/` and a
vendored module returned 200/302/200. Ordinary browsing is no longer counted.

**Now applied.** `throttle` is the calmer action: 429 the excess, keep no ban
state. The Terraform provider could not make that change in place —
`ban_duration_sec` is Optional+Computed, so the previous value is re-sent, and
the API rejects a ban duration on any action that is not `rate_based_ban`. The
policy had to not exist for the new action to be created clean, which was only
possible because the edge was already down. On the next build it came up as
`throttle` on the first apply.

Verified against the live policy on 2026-09-14, with the edge up:

| Check | Result |
|---|---|
| Rule set read back from the API | priorities `1000` and `1100` both `throttle`, `2147483647` default `allow`; matches `request.path.startsWith('/accounts/login')` at 30/min/IP and `request.path.startsWith('/demo/a2ui/ask')` at 60/min/IP; `banDurationSec` **absent** on every rule |
| 45 rapid requests to `/accounts/login` | 30 × `301`, then 15 × `429` — the limit fires at the threshold, and the 301s are Django's `APPEND_SLASH` redirects, so all 30 reached the application |
| Same client, immediately after | `/` → `200`, `/favicon.ico` → `404`, `/robots.txt` → `404` — real Django 404s, not 429s |

That last row is the regression closed. Under `rate_based_ban` every one of
those paths returned `429` for the full ban; now the excess is refused on the
matched path and the client is never denied the rest of the site.

A third lesson, about method rather than money: right after a policy change
the previous rules keep enforcing for a minute or two, and I read one of those
stale denials as evidence that a ban always takes down the whole site. It does
not follow, and the claim did not survive a measurement taken after
propagation.

### 7.7 Bring-up failed on a race the script created

`edge.sh up` aborted on its first run after the throttle change, with "LB not
serving; DNS untouched, nothing to undo". The load balancer was fine: minutes
later the same address returned `301` on port 80 and `200` on 443.

The probe was at fault, and it was two faults in one loop:

- **It was too impatient.** 24 attempts at 5-second intervals is about two
  minutes. A newly created forwarding rule is not reliably programmed at
  Google's edge inside that window, and the backend service in front of it had
  taken 2m20s to create in that same apply.
- **It waited on the slowest thing first.** The probe only spoke HTTPS on 443,
  which needs the forwarding rule *and* the certificate map to have
  propagated. Port 80 needs only the URL map's redirect, so it starts
  answering first, and it proves the whole forwarding path — rule, target
  proxy, URL map, backend service, NEG — is live without involving TLS at all.

Readiness is now two stages, in the order the pieces actually come up:

| Stage | Probe | Expects | Window |
|---|---|---|---|
| Forwarding path | `http://danielmherman.com/` on port 80 | `301` to HTTPS | 60 × 5s |
| Certificate | `https://danielmherman.com/` on port 443 | `200` | 60 × 5s |

(`edge.sh` 60–64 and 65–69, sharing the `ready_when` helper at 48–58.) Each
stage aborts on its own with a message naming which half failed, and both
still leave DNS untouched, so a failure has nothing to undo. `--resolve` is
used rather than `-H Host` so the SNI matches the managed certificate.

The re-run on 2026-09-14 completed all three phases: `0 added, 2 changed, 1
destroyed` for the phase-1 apply, which was a no-op because the LB already
existed; readiness passed immediately; DNS moved; ingress closed to
`internal-and-cloud-load-balancing`. Confirmed behaviourally rather than
declaratively — the default `run.app` hostname returned `404` while the domain
returned `200`, and both `run.googleapis.com/ingress` and its `ingress-status`
read `internal-and-cloud-load-balancing`.

Teardown was unchanged and symmetric: `0 added, 0 changed, 10 destroyed`,
ingress back to `all`, no forwarding rules, no security policies, no reserved
global address. During the TTL window both paths served at once — the domain
resolved to the four `ghs` addresses and returned `200` while the LB still
answered `200` at its own address — which is the drain working. The released
static IP answers `503` until DNS has finished draining.

Two `gcloud` details are worth recording, because they cost time and both read
as "there is no policy":

- There is no `security-policies rules list` subcommand in this version, and
  field paths like `--format='value(rules.priority)'` return empty. The rules
  come back from `gcloud compute security-policies describe <name>`, which
  returns a **one-element JSON array**, not an object.
- The Cloud Armor policy is named `danielmherman-edge`. `edge` is the
  Terraform resource name and resolves to nothing.

---

## 8. Interview questions this layer answers

**Where does the public internet meet your system?**
When the edge is up: a global external Application Load Balancer on a static
IP. TLS terminates there with a Certificate Manager certificate. Cloud Armor
rate-limits per client IP before the request reaches Cloud Run, and Cloud
Run's ingress rule refuses anything that did not come through the LB — the
default `run.app` hostname answers 404. When the edge is down, the same
site is served by Cloud Run's domain mapping and Django enforces HTTPS, HSTS,
host allow-list, login, and a per-user quota itself. The whole edge is
Terraform with one boolean, and I have the evidence from the last time it was
up.

**Why is it down?**
Cost on a demo that is idle most of the time — about $25 a month for the
forwarding rule and policy. The point of defining it in code was to make
"down" a decision I can reverse in ten minutes, not a shortcut.

**Why a global load balancer when Google's diagram shows a regional one?**
The regional one bills for proxy instances whether or not traffic flows;
the global one does not. With a single-region serverless backend they behave
the same. The global one also gets me an anycast IP and HTTP/3 for free.

**How do you bring it up without a dark window?**
Three phases. Build the LB and probe it while Cloud Run ingress is still open
and DNS still points at the domain mapping, so no client is affected at all.
Move DNS to the LB; resolvers holding the old answer keep being served by the
domain mapping. Wait one TTL, then close ingress so only the LB can reach
Cloud Run. Teardown is the mirror, and both directions drain for the same
reason. The script will not move DNS until the LB provably answers, and will
not close ingress until DNS has moved and one TTL has passed.

**Why probe port 80 before 443?**
Because they finish at different times and only one is on the critical path
for the other. Port 80 returns the URL map's redirect to HTTPS, which proves
the forwarding rule, target proxy, URL map, backend service and NEG are all
live, and needs no certificate. 443 additionally needs the certificate map to
have propagated. Probing 443 first means waiting on the slower half before
learning whether the faster half works at all.

**How is the agent kept private if its ingress is `all`?**
Two independent checks: Cloud Run's IAM invoker check requires a Google
identity token for an authorised service account before the container is even
invoked, and the agent code refuses requests without an `Authorization` header
as a second line. The browser never holds such a token. The service has one
invoker, the website's service account.

**Why is the quota claimed before the agent call, not after?**
Because the agent call costs money. Claiming first means a burst of concurrent
requests cannot all pass the check and all bill; the refund path returns the
credit if the call provably didn't spend.

**Cloud Armor let 57 through, not 60. Why?**
Rate-based rules are enforced at each edge proxy with a shared, eventually
consistent counter. The threshold is a target, not a contract. For a
rate-limit, that is the right trade: precise counting would need a
round-trip to a central store on every request.

**How do you follow one user action across Django and the agent?**
Cloud Run stamps a trace id on the inbound request. Django forwards it in
`X-Cloud-Trace-Context`; the agent logs `trace=<id>` on every line for that
request. Searching Cloud Logging for the id shows both services. Turning
that into a structured trace view is the observability layer.

**What did moving DNS to Cloud DNS buy you?**
Automation. Terraform owns the records, so "point the domain at the LB" and
"point it back" are the same apply that creates or destroys the LB. It also
put the zone file under version control, so I know exactly what records exist
and why.
