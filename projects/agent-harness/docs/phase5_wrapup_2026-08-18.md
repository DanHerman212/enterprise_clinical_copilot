# Phase 5 Wrap-Up — 2026-08-18

_Short, plain-language summary of what happened today and what's next._

## What we did today

1. **Deployed the AI endpoints.** The prediction endpoint and the RAG search
   endpoint are now live in the cloud.
2. **Rebuilt and redeployed the agent** with the new safety guardrail and a
   stricter medication prompt.
3. **Re-ran the full evaluation.** 300 test questions, all scored fresh — we did
   not reuse any old results.

## Where the agent stands

The agent is getting better:

- **Pass rate: 95%** (285 of 300) — up from 92.7% yesterday and 88.6% at the start.
- **Safety failures: 3** — down from 16.
- **Agent errors: 0.**

The 3 remaining safety issues are all medication-related:

- Contradictory Enoxaparin instructions
- An invented Albuterol dosage ("2 Puffs")
- A wrong furosemide frequency

## One thing to fix before we're done

We built a "guardrail" that checks the agent's answers and removes mistakes
before they reach the user. When we dry-ran it over today's results, it
**modified 6 answers that were actually correct** — it wrongly deleted the word
"daily" from phrases like "twice daily."

Root cause: the guardrail's frequency checker counts "twice daily" twice (once
as "BID" and once as "daily"), so it thinks "daily" is an error and deletes it.

**Bottom line:** today's evaluation results are valid, but the guardrail is not
safe to ship until this bug is fixed.

## Also done today

- The website's **Agent Evaluation page** was redesigned and pushed to
  production: sticky table of contents, scroll highlighting, and a clearer
  text-size ladder.

## Teed up for tomorrow

1. **Fix the guardrail bug** — stop it from double-counting "twice daily" /
   "once daily".
2. **Re-run the guardrail check** — confirm it no longer touches correct answers.
3. **Re-run the evaluation if answers change** — otherwise today's 95% stands.
4. **Commit the code** — guardrail, prompts, tests, and this wrap-up.
5. **Update the phase 5 result docs** with today's numbers.

## Housekeeping

- The billable cloud endpoints were **shut down for the night**.
  Rebuild with:
  - `projects/mlops/scripts/deploy_cpr.py` (prediction endpoint)
  - `projects/agent-harness/scripts/deploy_rag_endpoint.py` (RAG index endpoint)
