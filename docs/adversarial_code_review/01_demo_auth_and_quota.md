# §1 — Demo Auth + Per-User Quota

## Scope

This section covers the demo's backend-for-frontend in `demo/`: the authentication gate on the public console and the per-user daily quota that budgets every paid agent call. It is the crown jewel of the review — Django is the only publicly reachable service in the topology; the agent and MCP services behind it are IAM-private, so this layer is both the identity boundary and the spend-control boundary for the entire system.

Authentication is stock Django (`django.contrib.auth` on the default user model), deliberately minimal: accounts are issued by an operator rather than self-registered, there is no signup route (guarded by a regression test), and every demo view requires login. Production hardening — Secret Manager–sourced signing key, SSL redirect, secure cookies — is gated on an environment flag.

The quota is a per-user, per-day counter enforced atomically in the database. Consumption is a single conditional UPDATE rather than a read-modify-write, so two concurrent requests cannot both pass on the last credit; day rollover happens lazily on first use (no cron to fail silently). The views claim a credit *before* dispatching to the agent and refund it on failure — including the "graceful" case where MCP tools return an error payload inside an HTTP 200 — so a credit is never spent on an answer that never materialized.

The agent client mints a Google ID token per request, with the private Cloud Run service URL as the audience: the metadata server in production, a gcloud-subprocess fallback for local development. Fixture mode answers starter chips from captured real payloads (honestly labeled `source: 'fixture'`, no quota consumed) when the paid endpoints are torn down. An A2UI variant mirrors the ask flow with identical quota semantics but composes the canvas as A2UI messages, resolving citations deterministically by section intent because the model's own citation numbering is known to be unreliable.

## Adversarial Review Pass

§1 ran under the **blind protocol**: an independent adversarial model (Claude Fable 5, distinct from the primary reviewer) examined the code with a hostile prompt and no access to the primary pass's conclusions, and the two sets of findings were merged afterward. Automated scanners — pip-audit, bandit, and Django's deployment check — supplemented the reading. Eighteen findings resulted (S1-01…S1-18); the dominant theme is not broken mechanisms but **fail-open configuration**: the individual pieces are well built, yet several of them silently degrade when an environment variable is missing.

### The production gate fails open

- **S1-01 · Major (security/ops).** Production mode is keyed off a single environment variable that defaults to `development`, and the Cloud Build deploy step never sets it. A missing or misspelled variable — or a service recreated from the repo — yields a public deployment running `DEBUG=True` with the committed fallback secret key, non-secure cookies, and no SSL redirect. *Remediate:* fail closed — require an explicit environment, set it in the deploy step, and gate CI on `manage.py check --deploy`.
- **S1-18 · Minor (ops).** The deployment check reports six warnings that independently corroborate this cluster: the HSTS warning confirms S1-12, and the remaining warnings (SSL redirect, secure cookies, debug mode, fallback key) all fire precisely because the run isn't marked production — demonstrating what a misconfigured prod deploy would look like. *Remediate:* covered by S1-01/S1-12; add the check as a CI gate.
- **S1-12 · Minor (security).** No HSTS: transport security stops at the SSL redirect, leaving first-visit and downgrade interception of the session cookie possible. *Remediate:* set HSTS with a ramped max-age, plus subdomain/preload flags as appropriate.

### Fixture mode defaults to on

- **S1-02 · Major (ops/correctness).** Fixture mode defaults to *true* and is silent. A production deploy that omits the flag serves captured fixtures instead of the live agent — exactly what the project's no-shortcuts stance forbids. The payload is honest about its source, but the page never signals mode. *Remediate:* default to false; forbid fixture mode outright when the environment is production, failing loudly at startup.

### The spend cap can be defeated

- **S1-09 · Major (security).** The refund loop defeats the quota's purpose. The admission id in a request is never validated against the demo cohort, so a request guaranteed to produce a downstream tool error burns a full Gemini round trip yet is always refunded — the quota stops being a spend limit for the most expensive component. Unvalidated ids and free text also open id-probing and prompt-injection surface against the agent. *Remediate:* validate the admission id against the cohort before dispatch, treating the cohort as the server-side authorization boundary; refund only failures that provably consumed no model spend, or cap refunds per user per day.
- **S1-11 · Minor (correctness).** The agent's response shape is never validated: a non-dict body or malformed tool entry raises an unhandled exception *after* the credit was consumed, outside both refund paths. *Remediate:* validate the shape immediately after the call and route malformed responses through the existing refund path as agent errors.
- **S1-07 · Minor (correctness).** Two day-boundary defects: rollover is keyed to UTC midnight for every user, and a refund is scoped to *today's* counter rather than the credit it reverses — a consume-before-midnight, refund-after-midnight sequence silently loses the credit or decrements the wrong day. *Remediate:* have consume return the period it debited (or a claim token) for refund to target; document the UTC boundary.
- **S1-08 · Minor (ops).** Per-user caps only; no global daily budget or kill switch. Low risk while accounts are issued to a small N. *Remediate:* tracked on the deployment roadmap; revisit before any public window.

### No throttling on the auth surface

- **S1-05 · Major (security/ops).** Neither the login endpoint nor the admin — sitting at Django's default path — has throttling or lockout. The quota caps agent spend, not credential guessing, and the admin controls both quotas and content. *Remediate:* add lockout (django-axes or equivalent); move or shield the admin (non-default path, allowlist, or IAP). Admin hardening is also tracked under §6.
- **S1-06 · Minor (ops).** Including all of Django's auth URLs exposes password-change/reset routes for which no templates exist, so those paths 500 rather than 404. *Remediate:* include only the intended login/logout routes.
- **S1-04 · Minor (security).** No session-expiry override: the default two-week session survives browser close, so a stolen demo cookie stays valid for two weeks. *Remediate:* expire at browser close or set a short cookie age.

### Information disclosure

- **S1-03 · Major (security).** The 502 error response echoes the raw exception string to the client, which for network errors embeds the private agent's host and port — contradicting the same module's own sanitization of non-200 responses. The existing test guards the wrong field. *Remediate:* log the exception server-side; return a fixed generic detail.
- **S1-10 · Minor (security).** The real private agent URL is committed as the setting's default, disclosing internal topology — and combined with the S1-01 fail-open, a misconfigured instance silently points at and mints ID tokens for the production agent. *Remediate:* default to empty (the client already raises a clear error) and require the URL via env or secret.
- **S1-13 · Minor (security, cross-cutting — owned by §6).** CKEditor's HTML support allows every element and attribute, the upload route is mounted publicly with no permission pinned, and the storage default ACL makes uploads world-readable. Captured here because it lives in settings; remediation (server-side allowlist, staff-only uploads, ACL review) is tracked with the content section.

### Serving and container posture

- **S1-15 · Minor (architecture).** The agent call is a blocking HTTP POST with a 120-second timeout inside synchronous views under an ASGI server with two workers. Sync views run on a bounded thread pool, so a handful of slow agent calls — or the S1-09 loop — can exhaust it and stall the entire public site, login and blog included. *Remediate:* an async view with httpx, or a per-instance semaphore around agent calls; size Cloud Run concurrency to the pool.
- **S1-14 · Minor (ops).** The container runs as root — no user directive in the Dockerfile. Cloud Run's sandbox limits blast radius, but it drops a standard defense layer. *Remediate:* add a non-root user after the build steps.

### Scanner results

- **S1-16 · Major (dependencies).** pip-audit found 72 known CVEs across four pinned packages: Django, Pillow, sqlparse, and bleach. The bleach finding is the most pointed — it is the HTML sanitizer, and a vulnerable sanitizer undermines the stored-XSS defense behind CKEditor content. *Remediate:* upgrade all four pins to their fixed versions and re-run the audit.
- **S1-17 · Minor (security).** bandit reported ten low-severity items, all triaged: seven hardcoded-password hits are test-setup false positives, and the subprocess findings concern the local-only gcloud token path (no shell, fixed arguments, unreachable in the container by design). *Remediate:* none required beyond awareness; optionally annotate the subprocess as reviewed.

## Remediation Scope

**Summary.** The strategic read: this is a *configuration-posture* problem, not a mechanism problem. The quota's atomic enforcement, claim-before-spend discipline, and honest fixture labeling all survived adversarial scrutiny; what didn't is everything that depends on an environment variable being present. The recommended order is (1) **close the fail-open cluster** — S1-01, S1-02, and S1-10 share one fix shape (no permissive defaults, explicit env in the deploy step, startup validation) and one CI gate, and together they eliminate the scenario where a routine redeploy silently becomes an insecure or fake-live instance; (2) **restore the spend cap's integrity** — S1-09's cohort validation plus S1-11's shape validation, since the quota is the only thing standing between a hostile user and unbounded model spend; (3) **harden the auth surface and stop the leak** — S1-05 throttling and S1-03's generic error detail; (4) **sweep the minors** — session expiry, HSTS, auth routes, day-boundary semantics, non-root container, dependency upgrades.

**Detail.** The fail-open cluster resolves with a fail-closed settings module (raise on missing or unknown environment), explicit env vars in the Cloud Build deploy step, a production-time prohibition on fixture mode, an empty agent-URL default, and `check --deploy` in CI — which also retires S1-18. Spend-cap integrity requires validating the admission id against the demo cohort before any agent dispatch and narrowing refunds to failures that provably spent nothing (or capping refunds per user per day), with response-shape validation feeding malformed payloads into the existing refund path. Auth hardening is django-axes plus an admin move/shield; the disclosure fix is a one-line swap to a fixed generic detail with server-side logging. The remaining minors are each small, independent changes: browser-close session expiry, ramped HSTS, trimming the auth URL include to login/logout, a claim-token refund keyed to the debited period, a Dockerfile user directive, the four dependency upgrades from S1-16, and the async-or-semaphore change for the blocking agent call. S1-13 executes with §6's content remediation; S1-17 needs no action.
