# Platform-Owned Evaluation — Decision Plan

**Status:** proposal, awaiting review
**Date:** 2026-10-20
**Audience:** observability team + engineering
**Decision needed:** approve the approach, answer the open questions, and authorise the one
capability check in step 0.

---

## 1. Why we are changing approach

Four days of building a bespoke evaluation harness produced four failed runs, each failing
for a different reason. The common cause is structural, not a run of bad luck:

> The harness passes a plain Python dictionary through seven hops — case, HTTP request,
> collected record, judge evidence, judge prompt, scores, report, gate — and **nothing
> declares or checks what each hop must contain.**

Every test verified one hop in isolation. No test ever ran the whole chain, so each cloud
run was the first time the hops met, and each run found one broken join:

| Run | What broke | Cost |
| :--- | :--- | :--- |
| 1 | Turn-2 request dropped `run_tag`, orphaning every follow-up trace | one build + one run |
| 2 | Trace tags never set — the SDK method used does not exist in v4 | one build + one run + live debug |
| 3 | Safety failures summed across two populations, so 30 probes failed a clinical gate | one run |
| 4 | Patient manifest decoded correctly, then dropped by a field-rebuilding function | one build + one run |

The lesson is not "write better code". It is that **an evaluation system without a declared
schema and without an end-to-end test will keep failing at its seams**, and each discovery
costs a twenty-minute cloud cycle.

The secondary problem is readability. The harness wrote report JSON to a bucket. A human
reviewing a run could not answer, from the artifact, "what was asked, what was answered,
what was judged, and why did it fail". That is what made the work feel like a black box.

---

## 2. What we propose instead

Use **Langfuse datasets and experiments** as the evaluation platform. Langfuse is already
deployed and already receives our traces. The SDK in this repo (4.14.4) exposes the full
experiment surface; the following signatures were read directly from the installed SDK on
2026-10-20, not from documentation:

```text
create_dataset(*, name, description=None, metadata=None,
               input_schema=None, expected_output_schema=None)
create_dataset_item(*, dataset_name, input=None, expected_output=None,
                    metadata=None, source_trace_id=None, ...)
run_experiment(*, name, run_name=None, description=None, data, task,
               evaluators=[], composite_evaluator=None, run_evaluators=[],
               max_concurrency=50, metadata=None) -> ExperimentResult
get_dataset_runs(*, dataset_name, page=None, limit=None)
get_dataset_run(*, dataset_name, run_name)
```

Plus, in `langfuse.experiment`: `Evaluation`, `EvaluatorFunction`,
`RunEvaluatorFunction`, `CompositeEvaluator` support, `RegressionError`, and
`create_evaluator_from_autoevals`.

Two of these matter more than the rest:

- **`input_schema` / `expected_output_schema`** — the declared contract that has been
  missing. A malformed dataset item is rejected when the dataset is written, not after a
  run has been paid for.
- **`composite_evaluator` + `run_evaluators`** — the pass rule and the run-level
  aggregates become a single declared function, instead of bespoke aggregation code.

### Shape

```
census (82 admissions x 4 prompts x 2 turns)  +  suite 2 (30 probes)
        |
        v
Langfuse dataset  -- input_schema / expected_output_schema ----------------
        |
        v
task(item)        ->  one HTTP call to the deployed agent          ~40 lines
        |
        v
evaluators        ->  assertions.py (deterministic) + LLM judge (rubric.md)
        |
        v
composite_evaluator -> the pass rule, defined once
        |
        v
run_evaluators    ->  pass rate, safety count, per-dimension rates
        |
        v
Langfuse UI       ->  the readable report, plus run-to-run comparison
```

### What is deleted, and what survives

| Today (bespoke, ours) | Under this plan |
| :--- | :--- |
| `collect_http.py` record/merge plumbing | task function, ~40 lines |
| `judge.py` consensus, aggregation, report assembly | composite evaluator + run evaluators |
| `_merge_for_judging`, resume keys, run-tag propagation | removed — the platform owns it |
| report JSON + GCS artifact copy | dataset runs, compared in the UI |
| ad-hoc pass-rate gate script | run evaluators + regression comparison |

**Survives unchanged in spirit:** `rubric.md` (becomes the judge evaluator's prompt),
`assertions.py` (deterministic checks stay in Python), `prompt_set.py`,
`suite2_cases.json`, and the census builder.

Net: roughly 1,500 lines of harness become ~250 lines plus configuration. More importantly,
the failure class disappears — the platform owns the plumbing between hops.

### What we are explicitly NOT doing

- No new bespoke judge, consensus, aggregation or report code.
- No evaluation orchestration inside Cloud Run jobs.
- No case fixtures delivered through a bucket mount and an environment variable. Fixtures
  live in git; the dataset is seeded from them.

---

## 3. Step 0 — the one thing that must be verified first

**The SDK supports this. The self-hosted server (Langfuse 4.38.0) may not.**

The Python SDK is 4.14.4; the server is 4.38.0. Experiment support is comparatively new,
and a version mismatch would surface *after* we had built step 3 or 4.

**Check:** write one throwaway dataset with a trivial schema, run one experiment with a
no-op task and one evaluator, read it back, delete it.

**Cost:** about ten minutes. **Needs approval**, because it writes to the observability
database.

If the server does not support it, the fallbacks are, in order: upgrade the self-hosted
server; or use the hybrid option (promptfoo/DeepEval driving runs, Langfuse storing traces
and scores).

---

## 4. Work plan

Each step must be green *locally* before the next begins. No cloud execution until step 6
passes.

| # | Step | Done when |
| :--- | :--- | :--- |
| 0 | Verify server support | a throwaway dataset + run round-trips |
| 1 | Declare the dataset schema (input and expected output) | a malformed item is rejected locally |
| 2 | Seed datasets from git fixtures | census + suite 2 load, counts assert exactly 656 and 30 |
| 3 | Task function: one agent call per item | runs against a fake agent locally, no network |
| 4 | Deterministic evaluators (`assertions.py`) | a fixture with a known structural defect fails |
| 5 | LLM judge evaluator (rubric.md) | a fixture answer with a known clinical defect fails |
| 6 | Composite pass rule + run evaluators | a mixed fixture set yields the expected pass rate |
| 7 | **End-to-end local dry run** — fake agent, full chain, seconds | whole path green before any cloud run |
| 8 | One-patient live run | slice reads correctly in the UI |
| 9 | Full census run | run comparison available for the gate |

Step 7 is the piece that has been missing and is the reason for this plan. It is not
optional.

---

## 5. Open questions for review

1. **Server capability (step 0)** — approve the throwaway dataset write?
2. **Judge sampling.** Platform evaluators make one model call per item by default. Our
   adaptive 2-of-3 exists because of `429` exhaustion and judge disagreement. Proposal:
   start with one call per item and add repeat sampling only if measured variance demands
   it. Agree?
3. **Deterministic assertions** — keep them as evaluators alongside the LLM judge? We
   recommend yes: a widget that fails to render should never be sent to an LLM for
   opinion.
4. **Census scope for the first live run** — 1-patient slice first, then all 82? Proposal:
   yes, because 656 judged items is the dominant cost of a run.
5. **Where the release threshold lives.** Proposal: a run-level evaluator in Langfuse, with
   a thin CI check reading the latest run, so the threshold is data and not code.
6. **Rubric ownership.** `rubric.md` becomes the judge prompt and is now a reviewed
   artifact. Who signs off on rubric changes?
7. **Historical comparability.** The rubric is at v4; earlier runs are not comparable. Is
   the previous evaluation evidence to be archived as-is, or formally withdrawn?

---

## 6. Risks

| Risk | Likelihood | Mitigation |
| :--- | :--- | :--- |
| Self-hosted server lacks experiment support | medium | step 0, before any building; hybrid fallback available |
| Clinical rubric not expressible as an evaluator | low | evaluators are plain Python functions; rubric.md is a prompt |
| Judge cost at 656 items | certain, bounded | 1-patient slice first; one call per item to begin |
| Platform becomes a new black box | low | datasets and runs are inspectable in the UI, unlike report JSON |
| Repeating the same architectural failure | low **if** step 7 is enforced | end-to-end local test is a gate on progress, not an afterthought |

---

## 7. Applying the same discipline to upcoming production hardening

The same pattern would produce the same result. Three rules, proposed for that work:

1. **Schema first.** Any boundary between components gets a declared contract before code
   is written, and a test that a malformed payload is rejected.
2. **Local end-to-end before cloud, always.** A single test that runs the whole path
   against fakes, in seconds, existing *before* the first deployment.
3. **One change per verification run.** Never bundle a fix with a hypothesis and then read
   the result as evidence for both.

---

## 8. Repo state at time of writing

- Two files modified, uncommitted: a `manifest` field fix in `collect_http.py` and its test.
  Since `collect_http.py` is a component this plan replaces, the recommendation is to
  discard it and carry the *requirement* — the patient manifest must reach the evaluators —
  into the task function design.
- Nothing is running. The prediction and RAG endpoints remain deployed, costing
  approximately $0.20/hour, until torn down.
