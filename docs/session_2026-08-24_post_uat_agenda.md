# Post-UAT Agenda — Session 2026-08-24

**Context:** UAT of the production demo (`www.danielmherman.com/demo/a2ui/`) is
in progress. Both serving endpoints were redeployed in parallel via
`scripts/launch_endpoints.sh` (prediction + RAG index). This doc captures
everything queued **after UAT passes** — per the user: "summarize in an agenda
doc all the stuff after UAT."

Hard constraint: **Capital One interview at 16:30 today.** Interview prep is
timeboxed before it.

---

## 1. Adversarial code review — logical next step

**Strategy doc to review together:** `docs/code_review_plan.md` (Phases 0–3)
— this is the "informal approach" document referenced today. Cross-ref:
`danielmherman/docs/hybrid_demo_test_agenda.md` §6 "Sprint D".

**Strategy discussion the user wants:**
- How to create a **subagent with a different, more powerful model** to find
  issues in our code and **document remediation**.
- The plan's Phase 2 already sketches it: run a subagent on a *different
  model* with a hostile prompt — *"assume everything is broken; find the
  failure; rank by severity; don't excuse what a normal reviewer would wave
  through."* Optionally cross-check the same files with an external tool/model
  so the second set of eyes is genuinely independent.
- Add automated scanners so the pass isn't reading-only: dependency CVE scan,
  CWE-rules scan, `bandit`, `manage.py check --deploy`.

**Execution shape (from the plan, to confirm with the user):**
- Phase 0 — one-page codebase map + crown jewels (Dan confirms in ~10 min).
- Phase 1 — by-section review in **risk order**, two artifacts per section:
  an *Understand* doc and a *Review* doc (severity / category / location /
  suggested remediation).
- Phase 2 — independent adversarial subagent pass + scanners.
- Phase 3 — triage into a severity-ranked backlog; fix in risk order; re-run
  scanners after each fix.
- Continuous guardrails — Dependabot on both repos, CI lint/security steps.

**Open items to decide with the user:**
- Which model runs the adversarial subagent, and which sections it gets first.
- Format/location of the remediation documentation (per-section review docs +
  a backlog tracker).

## 2. Architecture review

- The user flagged this as a definite topic today: a full architecture review
  of both repos — `enterprise_clinical_copilot` (ML pipelines / MLOps / agent
  harness / MCP / RAG) and `danielmherman` (Django site + demo console).
- Natural fit: this is Phase 0 of the code review plan (codebase map + crown
  jewels) — do the map first, then the architecture review can consume it.
- Optional: produce structured architecture artifacts (project structure,
  tech stack, data model) from the map.

## 3. Capital One interview prep (before 16:30)

**Goal (user):** capture the essence of this project as it relates to their
skills and what they can offer a future employer — refresh the mental model.

**Material to assemble:**
- **MLOps / enterprise ML serving** — training + HPO pipeline, Vertex Model
  Registry, Custom Prediction Routine serving with native TreeSHAP
  attributions, BigQuery-backed Feature Store, cost-conscious endpoint
  lifecycle (teardown/stand-up scripts).
- **Agentic RAG** — MCP server exposing prediction + retrieval tools, explicit
  agent graph (not create_react_agent), Gemini grounding, citation-grounded
  answers, cross-patient isolation.
- **Eval rigor** — deterministic rails, tiered acceptance suites, LLM-judged
  eval with Langfuse observability (self-hosted).
- **Full-stack product** — production Django site with issued-account auth +
  per-user quota, A2UI (agent-composed UI) demo console.
- **Discipline signals** — synthetic-data compliance posture (no real MIMIC
  data in public demo), reproducible deploy docs, CI/CD (Cloud Build +
  deploy-on-push), failure-mode documentation.

**Format (suggested):** a one-pager "project essence" briefing — what was
built, the hardest problems solved, and the skills each demonstrates — that
Doubles as interview talking points.

## 4. Carried-over demo issue (from last night's session)

- **Citation-link mismatch on the canvas** — passages shown don't match the
  citation link clicked. If UAT confirms it is still broken, it joins the
  code review backlog (correctness, demo surface) rather than being fixed
  ad hoc.

---

## Today's run order

1. Finish demo UAT (endpoints up + user signs in) — exit criterion: all §3.3 /
   §3.4 checks from `hybrid_demo_test_agenda.md` pass on production.
2. Strategy discussion: adversarial subagent setup (Item 1).
3. Review `docs/code_review_plan.md` together — confirm Phase 0 kickoff.
4. Architecture review discussion (Item 2).
5. Interview prep one-pager (Item 3) — **hard stop before 16:30**.
6. After the interview (or next session): kick off code review Phase 0/1.
