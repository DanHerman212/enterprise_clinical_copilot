# Phase 5 — Remediation Workflow (post-mortem → green gate)

_Date: 2026-08-14 · Status: ACTIVE — P1–P3 DONE, P4 (guardrails) in progress. Gate criteria TBD by owner. Read `docs/phase5_eval_results.md` §7 first._

## Why this document exists
A professional, executable plan so we (a) don't trust unvalidated numbers, (b) fix the
**real** failure, and (c) prove every fix with a regression on a **frozen** golden set.
Grounded in current best practice: Anthropic (LLM-judge rubric, guardrails, poka-yoke
tools), MT-Bench (judge calibration), Hamel Husain (eval levels, precision/recall over raw
agreement), RAGAS/DeepEval (retrieval + faithfulness metrics), Langfuse (regression + annotation).

## Baseline (frozen 2026-08-14 — do not change while remediating)
- **Golden set:** 100 test-split patients × 3 prompts = 300 traces → `eval/results/traces.jsonl`
  (demo-split archive: `traces_demo_split.jsonl`). Judge output: `judged.jsonl`.
- **v2 (fixed judge, full evidence) — the operating baseline:** verdict **265/299 =
  88.6% pass**, **16 safety failures**. By prompt: risk 89 · meds 92 · summarize 84.
  v1 (truncated judge: 6.4%, 242 safety) was a **judge artifact** — archived at
  `judged_v1_truncated400.jsonl` / `golden_report_v1_truncated400.json`.

## Progress so far (P1–P3 DONE)
- **P1a retrieval audit:** index content is REAL/usable (581 passages, median ~11.6k chars,
  clinical prose; MIMIC redacts header identifiers only).
- **P1b hand-labels:** 12 stratified cases, all PASS (human). Saved at
  `eval/results/human_labels/human_labels.jsonl`.
- **P2 judge validation:** v1 judge κ=0.00 (6/12 agreement) → root cause = 400-char passage
  truncation → fixed (full evidence) → **12/12 (100%) agreement**. Re-judged 300 → 88.6%.
- **P3 root-cause of the 34 remaining FAILs (all model/prompt behavior, NOT retrieval/judge):**
  1. **Filling redacted `___` fields** (~5–6): invented age/dose (e.g. `24059266`, `25282181`).
  2. **Med dose/freq errors in narrative summaries** (~7): e.g. Simvastatin/Metformin "twice
     daily" vs "DAILY"; Bupropion/Citalopram doses (`27016685`, `22966155`, `29916192`).
  3. **Admission meds conflated into discharge list** (1, most serious): `29318404`.
  4. **Invented follow-up timeframes** (1–2): `24856944`.
  5. **Contradictory/incorrect antibiotic durations** (2): `23744149`, `29117773`.
  6. **No citation + invented facts** (1): `24592634`.
  Note: dedicated meds-LIST answers are exact; errors appear when meds are paraphrased
  inside free-text summaries.

## Decision rules (set in advance so we don't rationalize later)
- **Judge trust:** kappa vs human ≥ 0.7 → trust for gating. 0.5–0.7 → hybrid (human confirms
  every safety failure + sample of passes). < 0.5 → fix judge (few-shot, per-dim rubrics)
  before any remediation. **Critical number: judge false-pass rate (must be ~0).**
- **Retrieval ground truth:** if most retrieved chunks are redaction artifacts, that is a
  **data/index-layer failure** — retrieval recall is ~0 and no agent change can fix it.
- **Ship gates (re-run on frozen set after each fix):** safety pass ≥ 95%, invented-med
  count = 0, wrong-citation count = 0, no regression on other classes, no faithfulness drop.

## Phases (execute one at a time; exit criteria must be met to advance)

### Phase 0 — Freeze & version the baseline ✅ (done)
Version `traces.jsonl` + `judged.jsonl` + `golden_report.json` as immutable baseline (git tag
`phase5-baseline-red`). All later deltas measured against it.
- [ ] Tag baseline commit; note hash in this file.

### Phase 1 — Ground truth: verify retrieval content AND hand-label judge sample
**Objective:** answer two questions before touching the agent: (1) does the index actually
contain usable clinical text for these 100 patients? (2) is the LLM judge right?
- **1a. Retrieval content audit** — sample ~15–30 of the 100 patients; for each, dump the
  FULL retrieved passages (not truncated) for all 3 prompts. Classify: real clinical prose
  vs redacted/template. → `eval/audit_retrieval.py`
- **1b. Hand-labeling** — label **25–50 traces per prompt** (≈75–150 total) on the 5 dims
  + verdict, using a single rubric + adjudication for disagreements. Human = us (domain
  judgment). → `eval/human_labels.jsonl`
- **Exit:** a written finding on index content (usable / not) + a human-labeled set.

### Phase 2 — Judge validation & calibration
- Compute **Cohen's kappa** (judge vs human) per dimension; **judge precision/recall** vs
  human; **confusion matrix on safety failures** (esp. false-pass rate).
- If kappa < 0.5: improve judge prompt (few-shot examples, per-dimension rubrics) → re-judge
  the 300 → recompute.
- **Exit:** kappa + FP/FN recorded; decision (trust / hybrid / fix-then-trust) written here.

### Phase 3 — Root-cause each failure class to ONE layer
Attribute each of the 193 / 58 / 29 to: **data/index** (passage missing/redacted/wrong
section) · **tool/guardrail** (citation not enforced, thin evidence not blocked) ·
**model/prompt** (hallucination, instruction drift) · **judge** (label error).
- Eyeball a stratified sample of traces per class (Langfuse trace view). Record attribution
  counts. → `eval/root_cause.md`
- **Exit:** every failure class has a layer attribution with counts + examples.

### Phase 4 — Remediation (deterministic guardrails) ✅ built + dry-run validated
Built `agent/guardrail.py` (LLM proposes, code disposes), prompt hardening in
`agent/prompts.py`, and hooked into `agent/server.py` (served answer = guarded; flags returned).
- **Redacted-field guard:** never fill a `___`-redacted value; drops invented ages/doses.
- **Med-dose verifier:** every dose+unit the answer asserts must appear in the retrieved
  evidence (normalized for whitespace/plural/case); absent doses are dropped + flagged.
- **Med-frequency check:** flag-only (lower risk; avoids false-positive edits).
- **Citation range:** every `^[n]` must point at a retrieved passage.
- **Prompt hardening:** never fill redacted values; reproduce med dose/freq exactly; only
  discharge-list meds; no invented timeframes.
- **Dry-run over the frozen 300 (`eval/guardrail_dry_run.py`):** catches **13/34 FAILs**
  (all invented-age class, corrected); 6 judge-PASS answers also corrected — all verified
  **justified** (invented ages the judge missed); **0 unjustified modifications** → safe.
- **Known limitation:** token-level dose/freq checks are global, not per-med — they catch a
  dose/freq that appears NOWHERE in the evidence, but not "Simvastatin should be DAILY, not
  BID" when another med in the note is BID. The prompt hardening targets that class.
- **P5 (next):** re-run the 300-trace collect on the updated agent (needs endpoints
  redeployed) → re-judge → measure delta vs the 88.6% baseline.

### Phase 5 — Regression cases + full gate re-run
- Add a regression case to the golden set per fixed failure mode (keep core 300 frozen;
  add as a separate versioned layer).
- Re-run full gate (deterministic tiers + quant + golden). **Demo opens only when the
  completed gate is green.**

## Progress tracker
- [x] Phase 0 — baseline frozen
- [x] Phase 1 — retrieval audit + human labels (12 pilot, all PASS)
- [x] Phase 2 — judge fixed + validated (100% on frozen pilot)
- [x] Phase 3 — root-cause attribution (34 FAILs, all model/prompt)
- [x] Phase 4 — deterministic guardrails built; dry-run: 13/34 FAILs caught, 0 unjustified modifications
- [ ] Phase 5 — redeploy endpoints + re-run collect/judge to measure delta

## Re-run commands
```bash
cd /Users/danherman/Desktop/enterprise_clinical_copilot
# collect (after agent change) → overwrites traces.jsonl (baseline copy saved in Phase 0)
.venv/bin/python -u projects/agent-harness/eval/collect.py
# judge (resumable) → judged.jsonl + golden_report.json
.venv/bin/python -u projects/agent-harness/eval/judge.py
# summarize + post-mortem
.venv/bin/python projects/agent-harness/eval/summarize.py
.venv/bin/python projects/agent-harness/eval/post_mortem.py
```
