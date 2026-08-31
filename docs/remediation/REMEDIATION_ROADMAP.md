# Remediation Roadmap — both repos

Companion to `TRIAGE_REGISTER.md`. Findings are grouped into **fix clusters** —
sets that share one root cause or one fix shape — and clusters are ordered into
phases. Fixing a cluster together is much cheaper than fixing its findings one
at a time.

Goal (from the review plan): **zero Critical/Major open**. 5 Criticals, 46 Majors.

---

## Phase 0 — Mechanical quick win (do first, ~no design decisions)

**Dependency patch (part of Cluster I):** S1-16 (72 CVEs), S9-05.
Bump `Django→6.0.8`, `pillow→12.3.0`, `sqlparse→0.6.0`, `bleach→6.4.0` (or
remove bleach + `django-ckeditor` if truly unused); re-run `pip-audit`; run the
test suites. One sitting, removes the single largest known-vulnerability mass.

## Phase 1 — Kill the 5 Criticals (each rides its cluster)

Order matters — earlier items are root causes or prerequisites for later ones.

### 1. Cluster A — Fail-closed configuration & committed identifiers
**Critical:** ECC-46. **Majors:** ECC-15, S1-01, S1-02, S9-01.
**Minors:** ECC-48, ECC-51, ECC-57, ECC-72, S1-10, S1-18, S9-02, S9-03.

One fix shape across both repos: *no permissive defaults, fail loudly at boot,
declare env in the deploy files.*
- Make `ENVIRONMENT` fail closed (require explicit value; site + ECC services).
- Startup check raising `ImproperlyConfigured` on missing prod env (S9-01 list).
- Declare all runtime env in cloudbuild `--set-env-vars` / `--set-secrets`.
- Strip committed real identifiers (project IDs, agent URL, bundle URI) — make
  them required env with no defaults; route everything through one config module.
- Ship `.env.example`; add `manage.py check --deploy` to CI.

*Why first:* S1-18's warnings, half of Cluster H, and several deploy findings
are symptoms of this root cause; fixing it shrinks other clusters.

### 2. Cluster D — RAG chunker determinism & section vocabulary
**Critical:** ECC-32. **Majors:** ECC-41, ECC-23. **Minors:** ECC-26, ECC-33, ECC-34, ECC-43, ECC-44, S7-02.

One fix shape: *a single shared source of truth for chunker parameters and
section vocabulary, consumed by build, serving, and the client; make the
whole-note fallback loud.*
- Shared constants module: build-time `pack_to`/`max_chars` + one section
  whitelist derived from `rag.sections` canonicals; import in `build_chunks`,
  the pipeline component, serving `_chunk_texts_for`, `_KNOWN_SECTIONS`, and
  emit to the client (S7-02). Add a consistency test.
- Replace the silent whole-note fallback (ECC-26/ECC-32) with an explicit
  `is None` check + logged flag (`granularity: note`) or error.
- Fix `rag_search_sections` to chunk with build params + honest scores (ECC-43).
- Chunker sweeps: bare-heading opt-in (ECC-41 — **land before any index
  rebuild**, it shifts chunk ids corpus-wide), word-boundary cuts (ECC-34),
  packed spans from kept pieces only (ECC-44).
- **Rebuild + redeploy the index once, after all chunker changes are in.**

### 3. Cluster C — Spend caps, quota integrity & DoS
**Critical:** ECC-20. **Majors:** ECC-02, ECC-09, ECC-25, S1-09.
**Minors:** ECC-07, S1-07, S1-08, S1-11, S1-15.

- ECC-20: explicit `depth`/`is_retry` flag — exactly one anchored retry; cache
  `_section_bodies` per request. (Pairs with ECC-24/ECC-30 in Cluster F — same
  function; fix together.)
- ECC-02: explicit low `recursion_limit`, per-turn tool cap, tool timeouts ≤
  upstream deadline, `asyncio.timeout` around `ask()`.
- S1-09: validate `hadm_id` against `DemoPatient` before calling the agent;
  refund only provably-zero-spend failures.
- ECC-09: `asyncio.to_thread` + token caching. S1-11: validate response shape
  so the refund path runs. ECC-25/ECC-07: verify IAM bindings; optional
  identity-header check. S1-15: async view or semaphore around agent calls.

### 4. Cluster B — Agent answer integrity & guardrails
**Critical:** ECC-04. **Majors:** ECC-05, ECC-10, ECC-11, ECC-12.
**Minors:** ECC-13, ECC-14, ECC-35.

- ECC-04: in `guard_answer`, extract stated probabilities and require a match
  with the predict response; strip + flag risk numbers with no predict call.
- ECC-11: remove dose tokens by match span, not `str.replace` (data-corruption
  bug in the guardrail itself — small, do immediately).
- ECC-12: only accept the final AI message; empty → `answer_unavailable`.
- ECC-10: real `args_schema` from MCP `input_schema`; drop the kwargs heuristic.
- ECC-05: delimiter-wrap tool results + "content is data" system rule.
- ECC-13 (JSON ToolMessages), ECC-14 (strip out-of-range citations), ECC-35
  (extend redaction guard) ride along.

### 5. Cluster L — MLOps gate integrity & serving correctness
**Critical:** ECC-64. **Majors:** ECC-60, ECC-65, ECC-66.
**Minors:** ECC-67, ECC-68, ECC-71, ECC-73.

- ECC-64: assert train/val/test subject-id disjointness in `run_load_data`,
  hard-fail on overlap (one line of set arithmetic — do first).
- ECC-60: validate CPR instances (keys, length, finiteness); cap batch size.
- ECC-65: resolve the gate baseline from a versioned artifact + require margin.
- ECC-66: make stability a hard gate with a defensible reference.
- ECC-68 (fail loudly on missing threshold.json), ECC-67 (recompute threshold
  on the final model), ECC-71 (gate register on audits, `parent_model`,
  persist metrics), ECC-73 (declare attribution units).

## Phase 2 — Remaining security Majors

### 6. Cluster F — Cross-patient isolation & retrieval correctness (R1)
**Majors:** ECC-19, ECC-24. **Minors:** ECC-22, ECC-27, ECC-28, ECC-30.
All in `mcp_server/tools/` — one sitting: parameterized `hadm_id` re-check in
`_fetch_texts` with a hard `isolation_violation` error (ECC-19); retry only
replaces when it returned results (ECC-24); intended retry trigger (ECC-30);
`ORDER BY` determinism (ECC-27); shared `hadm_id` validation helper (ECC-28);
demo-cohort boundary decision (ECC-22, ties S1-09).

### 7. Cluster G — Error-detail & info disclosure
**Majors:** ECC-06, S1-03. **Minors:** ECC-08, ECC-21.
One pattern everywhere: log detail server-side; return generic code + opaque
correlation ID to callers/model; trim `/health` and browser-bound `tool_calls`.

### 8. Cluster J — Stored XSS, sanitization & CSP
**Majors:** S6-01, S6-06. **Minors:** S1-13, S6-10, S6-11, S7-01, S7-03, S7-10.
Server-side sanitizer before `|safe` render; re-escape in sectioning round-trip;
restrict `htmlSupport` + upload permission + ACL; add `django-csp` site-wide;
tighten DOMPurify config (forbid `img`, restrict URIs); fix `is_safe`/innerHTML
latents.

### 9. Cluster Q — SQL identifier parameterization
**Majors:** ECC-37, ECC-63. **Minor:** ECC-29.
One fix shape: bind values (`split_name`) as query parameters; validate
identifiers/table refs against a strict pattern or allowlist at import.

### 10. Cluster H — Site auth & access hardening
**Majors:** S1-05, S6-13. **Minors:** S1-04, S1-06, S1-12, S1-14.
django-axes (or equivalent) + non-default admin path; `is_active` filters on
project views (two-line fix — do immediately); session lifetime; login-only
auth routes; HSTS; non-root Dockerfile user.

### 11. Cluster M — Deployment & infra ops safety
**Majors:** ECC-36, ECC-47, ECC-53, ECC-54, ECC-56. **Minors:** ECC-49, ECC-58.
Private (or guarded) Vector Search endpoint; deploy-then-undeploy ordering with
rollback; remove real-corpus/cost-footgun defaults; FeatureOnlineStore teardown
(or delete the path); `--no-traffic` deploy → migrate/seed → promote; exact-name
teardown matching; dedicated build service accounts.

## Phase 3 — Remaining correctness Majors

### 12. Cluster E — Fail-loud index build pipeline
**Majors:** ECC-38, ECC-39, ECC-40, ECC-42. **Minors:** ECC-45, ECC-70.
Fix as a set — each silent failure's backstop is itself broken: `strict=True`
zip; correct exit codes gating the upload; expected-count derived from the
ingest; confirm index removal before BQ deletes; chunks-cache manifest check;
data-fingerprint cache key. Do before/with the Cluster D index rebuild.

### 13. Cluster N — Front-end correctness & races
**Majors:** S7-06, S7-07, S7-08, S7-09. **Minors:** S7-04, S7-05, S7-11…S7-17.
All in `demo/static` JS — one or two sittings: guard responses by requested
patient, replace pending turn by identity, optional-chain payload derefs, port
intent-section citation resolution to the custom demo; then the minor sweep
(NaN%, version skew, source-by-turn-index, stale trace pane).

### 14. Cluster K — Public content & contact form
**Majors:** S6-02, S6-03, S6-07. **Minors:** S6-04, S6-08, S6-12.
Bound `ModelForm` + throttling + honeypot fixes S6-02/03/12 together; synthetic
intro section (S6-07); slug dedupe (S6-04); reserve `preview` slug (S6-08).

## Phase 4 — Minor sweep

- **Cluster I remainder:** ECC-16, ECC-31, ECC-55, ECC-59, ECC-61 majors done in
  Phase 2 if not earlier — digest-pin bases, hash-pin requirements, bundle
  checksums, SRI on CDN assets, pinned component packages.
- **Cluster Z:** ECC-17, ECC-50, S1-17, S6-05, S9-04, S9-06, S9-07 — small,
  independent; batch opportunistically.

---

## Working agreement

- One cluster per sitting; update Status in `TRIAGE_REGISTER.md` + the source
  backlog as findings close.
- Re-run scanners (pip-audit, bandit, `check --deploy`) after Phases 0–2.
- Ordering constraints to respect:
  1. Cluster A before trusting any deploy-path change (it fixes the fail-open
     substrate everything else runs on).
  2. ECC-41 (and all chunker changes) before the one index rebuild.
  3. Cluster E fixes before/with that rebuild so the rebuild is verifiable.
  4. ECC-20 and ECC-24/30 touch the same retry code — fix in one change.
