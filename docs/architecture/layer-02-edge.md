# Layer 2 — Edge (the front door)

Status: audited 2026-09-11; implemented and verified live 2026-09-13. Decision D:
the reference edge (load balancer + Cloud Armor) is defined in Terraform,
was brought up, verified, and is torn down between uses. Gaps 2 and 3 closed.

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

Seven files, 417 lines including the script. State is in a versioned, private
GCS bucket so it survives a laptop (`versions.tf` 14–15).

| File | What it defines |
|---|---|
| `variables.tf` | `edge_enabled` (28), default `false`. Also domain, zone, service names, and the rate-limit numbers. |
| `certificate.tf` | Certificate Manager: DNS authorization for apex and `www` (10, 15), the `_acme-challenge` CNAMEs those need (22, 30), the managed certificate (38), a certificate map (50) and its two entries (54, 61). **Always present** — this is what lets the edge come up in minutes instead of waiting for a new certificate each time. |
| `edge.tf` | Everything that costs money, each with `count = edge_enabled ? 1 : 0` (13): static IP (16); serverless NEG pointing at the Cloud Run service (25); Cloud Armor policy (33) whose one rule is `rate_based_ban` per client IP (43, 53), 60 requests per 60 seconds, exceed → `deny(429)` (52), ban 300 seconds (58); backend service in `EXTERNAL_MANAGED` scheme with the policy attached (78, 82–83); HTTPS proxy using the certificate map (106); forwarding rules on 443 and 80; and an HTTP→HTTPS redirect URL map (123). |
| `dns.tf` | The apex `A` record points at the LB IP when enabled, else at Google's four `ghs` addresses (23). The `AAAA` record exists only when disabled (28) — the LB has no IPv6 address. `www` is a CNAME to the apex when enabled, else to `ghs.googlehosted.com.` (41). All TTL 300 (22, 32, 40). |
| `outputs.tf` | `edge_enabled`, `edge_ip`, `certificate_state`, the live DNS answers. |
| `edge.sh` | `up` / `down` / `status`. Needed because the Cloud Run service is owned by `cloudbuild.yaml`, so its ingress setting is changed here with `gcloud`, and the two changes have to happen in the right order. |

**Why `up` is two phases.** If DNS moved to the LB and ingress closed at the
same moment, every resolver still holding the old address would get a 404
for up to one TTL. So `up` (34) applies with `edge_enabled=true` (38) — LB
built, DNS flipped, ingress still open — waits for the LB to actually serve
(51 aborts if it never does), waits 330 seconds (20) for the TTL, and only
then sets ingress to `internal-and-cloud-load-balancing` (54). `down` (57)
is the mirror: open ingress first (60), then apply `false` (63). Resolvers
still holding the LB address are served by the LB until it is gone. There is
no dark window in either direction.

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

### 7.4 Gap 3 closed — agent invokers

Removed `roles/run.invoker` on `agent` from the default compute service
account and from the personal account. The service now has exactly one
invoker, `website-sa`.

One honest footnote for the interview: the personal account still gets `200`
from the agent, because it holds `roles/owner` on the project and Owner
includes `run.routes.invoke`. The explicit binding was redundant, not the
thing granting access. The default compute service account holds
`roles/editor`, which does **not** include invoke, so removing that binding
did change something. Restricting an Owner is an organisation-policy
question, not a service-IAM one, and belongs to the security layer.

### 7.5 Gap 2 closed — one request id across both services

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

Not yet deployed — the Django and agent changes ship with layer 3, as agreed
for layer 1.

Test status: Django `manage.py test` 84 OK. Agent `pytest tests/agent`
240 passed, 6 skipped, 10 errors — the 10 are the live-Gemini tests, which
fail identically without this change when no cloud credentials are present.

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
Two phases. Build the LB and flip DNS while Cloud Run ingress is still open,
so old resolvers keep hitting the domain mapping and new ones hit the LB.
Wait one TTL. Then close ingress. Teardown is the mirror. The script refuses
to close ingress if the LB is not serving.

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
