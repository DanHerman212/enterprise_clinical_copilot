# Post-Demo Code Review Plan

**Status:** planned — start when the ECC software demo is launched.
**Scope:** both repos — `enterprise_clinical_copilot` (ML / agent / RAG) and
`danielmherman` (Django site + demo console).

## Why this review exists
We shipped a lot of code quickly. Two goals:
1. **Understand** the codebase — exactly what was built and how it works, so Dan
   can troubleshoot and explain it.
2. **Find and fix issues** — vulnerabilities, poor architecture, wrong
   principles, correctness and operational risks.

## The core idea: one loop, two outputs per section
For each section of the codebase, produce two artifacts:
- **Understand doc** — how it works: entry points, key files, data flow, config,
  failure modes. (This doubles as the "learn the codebase" deliverable.)
- **Review doc** — findings, each tagged:
  - Severity: `Critical` / `Major` / `Minor`
  - Category: `security` / `correctness` / `architecture` / `ops`
  - Location: file + line
  - Suggested remediation

Review in **risk order**, not codebase order — the surfaces that could actually
bite get reviewed first. Understand docs still cover everything.

## Phase 0 — Map first (before deep review)
Produce a one-page codebase map: repos → projects → entry points → the
data/auth flow (browser → Django → agent → MCP → Vertex; and the
training → serving path). Then name the **crown jewels** so effort is
prioritized:
- Demo auth + per-user quota
- Cross-patient isolation in RAG retrieval (R1)
- IAM / service accounts / secrets
- Public Django admin + content
- Dependencies (CVEs)

Dan reviews the map in ~10 min to confirm the shared mental model before the
by-section loop starts.

## Phase 1 — Section cut (risk-ordered)
**`enterprise_clinical_copilot` (project repo):**
1. Agent harness — `agent/` (graph, prompts, server, mcp_client)
2. MCP tools — `mcp_server/` (`predict_readmission`, `rag_search`,
   `rag_search_sections`) + cross-patient isolation
3. RAG — index build, retrieval, citation groundedness
4. MLOps — training / HPO / serving / feature store / deploy scripts
5. Deployment — Cloud Run, IAM, secrets, endpoint config
6. Dependencies — CVE scan

**`danielmherman` (site repo):**
1. Demo app — views, quota, A2UI canvas composition, auth
2. Content / blog / admin
3. Front-end JS — demo flow, A2UI renderer, splitpane
4. Django config / settings / deployment

## Phase 2 — Adversarial pass (independent second set of eyes)
The trap: the *same* model reviewing twice repeats itself. The second pass must
be genuinely independent:
- Run a **subagent with a different model** and a hostile prompt: "assume
  everything is broken; find the failure; rank by severity; don't excuse what a
  normal reviewer would wave through."
- Optionally cross-check with an **external tool/model** on the same files.
- Run **automated scanners** so the pass isn't reading-only:
  - Dependency CVE scan (pip-audit / GitHub advisories)
  - CWE rule scan on the workspace
  - `bandit` (Python security lint)
  - Django security checks (`manage.py check --deploy`)

## Phase 3 — Triage & remediation
- Consolidate findings into a **severity-ranked backlog** (Critical first).
- Fix in the same by-section order; re-run scanners after each fix to confirm.
- Track remaining items in the backlog until resolved.

## Continuous guardrails (so it doesn't rot)
- Enable **Dependabot** on both repos.
- Add CI lint/security steps: `bandit`, `pip-audit`, `manage.py check --deploy`.
- Ensure new code can't silently reintroduce the fixed classes of issues.

## How to resume
1. Open this file.
2. Kick off **Phase 0** — generate the codebase map + crown jewels (use the
   Explore agent / primary assistant; read-only, fast).
3. Dan confirms the map, then we start the by-section loop at the highest-risk
   surface (auth/quota and the agent→tool chain).

## References
- Repos: `DanHerman212/enterprise_clinical_copilot`, `DanHerman212/danielmherman`
- Relevant tooling: dependency CVE assessment, CWE-rules scan, adversarial
  subagent pass, Explore agent for mapping.
