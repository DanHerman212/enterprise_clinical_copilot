# Phase 6 Wrap-Up — 2026-08-19

_Short, plain-language summary of what happened today and what's next._

## What we did today

1. **Fixed the guardrail regression** (from yesterday). Three root causes:
   the frequency checker double-counting "twice daily" (dropping correct
   "daily"), a newline-collapse bug, and a dose-normalization gap (couldn't
   match "2,000 mcg" to "2000 mcg"). The dry-run went 6 → 3 modified, and all
   3 remaining are *justified* (real med-frequency errors the judge missed).
   Committed as `5991197`; 20 guardrail tests pass.

2. **Completed the Phase 6 live validation.**
   - Deployed both endpoints in parallel (~38 min); `validate_rag` +
     `integration_test_live` all PASS.
   - Drove the full live screen guide: risk / meds / summarize chips, citations
     + source cards, trace view, episodic memory, quota countdown, honest-empty
     paths — all working on the deployed site.
   - **Caught a real bug:** the deployed agent's environment was mangled (a bad
     `^@^` delimiter collapsed all its settings into one variable). Redeployed
     the agent (`agent-00017-s7m`) with correct `MCP_URL` / `LANGFUSE_HOST` etc.
   - Error/refund paths verified live (all rejected before quota); groundedness
     spot-check passed (fully cited summaries).

3. **Two demo improvements (pushed `c8c5583`).**
   - Free-text questions now carry the selected patient, so the agent answers
     instead of asking "what's the hadm_id?" — zero impact on the eval.
   - Login page now has a contact line (mailto) for demo account requests.

4. **Built + verified Langfuse rubric-score attachment** (the last Phase 6 item).
   - `collect` records each trace's Langfuse ID; `judge` attaches 6 scores per
     trace (5 dimensions + verdict) with the judge's comments.
   - Verified on a 9-trace pilot (54 scores landed in Langfuse, values correct).
   - **Full 300-trace eval completed with Langfuse** — every trace scored in
     Langfuse (1,854 scores total across the run + pilot).

5. **Completed the full golden eval (reproducible).** Collect 300/300 (zero
   errors), judge 300/300 → **95% pass (285/300), 3 safety failures, 0 agent
   errors — identical to the 08-18 run**, confirming the eval is stable.

6. **Hardened the eval pipeline.** Both `collect` and `judge` could hang
   forever on a stuck Vertex/Gemini call (collect stalled 55 min at 210/300;
   judge stalled 18 min at 117/300). Both now have a per-call timeout and
   resume support, so a hang retries/flags instead of freezing the run.

7. **Infra / housekeeping.**
   - Mapped `observability.danielmherman.com` → Langfuse (DNS record added).
   - Teardown: both billable endpoints deleted — billing stopped.
   - Discussed eval-pipeline automation (recommend Cloud Run Jobs or Prefect)
     and Terraform IaC for the cleanup pass.

## Where the demo stands

- The eval gate has passed and is **reproducible**: **95% pass (285/300)**, 3
  safety failures, 0 agent errors — identical across two independent full runs.
- **Every trace is scored in Langfuse** (per-dimension scores + judge comments),
  ready for review and the fix-and-retest loop.
- The full live journey works on deployed endpoints; endpoints are now torn
  down for the night (billing stopped).
- Phase 6 is essentially complete; remaining wrap-up: commit the Langfuse build
  (`graph.py` / `collect.py` / `judge.py`), tick `go_live_plan.md` boxes,
  Block B agent-down live verify (quick check now that endpoints are down),
  and a confirmed quota-refund gap fix (see tomorrow's agenda).

## Tomorrow's plan (repurposed priorities)

1. **Synthetic cohort** (swap real data for fabricated data — the final gate
   before anything public) + a **self-service user-guide page** for the demo.
2. **Google Analytics integration** + demo launch strategy (budget and tracking
   to connect a launch campaign to new traffic / login requests).
3. **Preview launch** — friends & family try it, collect feedback, adjust.
4. **Public launch** (LinkedIn / HackerNews) — only after the UX is blessed on
   the synthetic cohort.
5. **Post-launch** — observe and iterate.

**Also on the list:** adversarial code review (`docs/code_review_plan.md` —
must do before launch), GitHub cleanup, and public website project page cleanup.

## Housekeeping

- Endpoints torn down EOD (`teardown.py --yes`) — billing stopped. Cloud Run
  services (agent / mcp-server / Langfuse) left running; they scale to zero.
- **Found a real bug tonight:** quota is NOT refunded when the agent surfaces a
  downstream (endpoint) failure as HTTP 200 — the BFF only refunds on 502. The
  fix (check for errored tool responses, refund + 502) is designed and on
  tomorrow's agenda.
- The Langfuse build (`graph.py` / `collect.py` / `judge.py`) still needs
  committing.
