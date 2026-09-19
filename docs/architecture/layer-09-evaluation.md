# Layer 9 — Evaluation

Status: first audit 2026-09-19, written in the four-part form used from layer 7 onward. The
layer is not empty — a two-tier harness exists, has run repeatedly, and its reported numbers
are real — so the audit is about what the harness does not yet do. Five gaps are recorded and
none is closed. Four of them are the difference between evaluation as a set of runs someone
remembers to perform and evaluation as a standing property of the system: an adversarial set,
a gate that can fail a build, scoring of production traffic, and a connection to the
observability plane. The fifth is a comparison the model layer is waiting on.

---

## 1. Overview

### 1.1 The role of the layer

Evaluation is the layer that answers whether the system is good, and — the harder half — how
anyone knows. In a conventionally tested system those are one question. In a generative one
they come apart: there is no single correct string to compare against, so the layer must either
construct a reference and compare to it, or encode the criteria that make an answer acceptable
and have something apply them. Everything here follows from that split.

The split shows in what gets measured. A model that scores a patient has a quality that is a
number — ranking performance, calibration, error rates, parity across subgroups. An agent that
writes a paragraph for a clinician has a quality that is not: whether every clinical claim
traces to evidence the system actually retrieved, whether its numbers match the tool output,
whether it invented a dose. The first is measured against labels; the second against evidence,
and the evidence has to be captured while the system runs, because groundedness cannot be
judged from the answer alone. Judging without it is not a weaker evaluation but a misleading
one: a claim that looks unsupported may have been supported by material the judge cannot see.

Three properties separate evaluation that can be relied on from evaluation that merely
occurred. **Comparability**: the metric, rubric and case set are fixed early, so two runs months
apart mean the same thing — a rubric that moves makes history unreadable, and a set that grows
silently raises a pass rate for reasons unrelated to quality. **Consequence**: a run that cannot
stop a deploy is a report, not a gate, and a report competes with schedule. **Validity of the
judge**: when criteria are applied by a model, the judge is itself unverified, and agreement
with human labels is the only check on whether it is too strict, too lenient, or measuring
something else.

The layer has two modes that are easy to confuse. Offline evaluation runs a fixed set before
deploy: controlled and comparable, and disconnected from what users actually ask. Online
evaluation samples what production produced: representative, and useless as a before-and-after
comparison because the inputs change. Both are needed, and they must stay separate — an
offline score moving is a change in the system, an online score moving may be a change in the
users.

```
  OFFLINE — before deploy, on a fixed set
    case set: stratified — essential, average, edge, adversarial
      → run the system, capturing the evidence with the answer
      → score: reference metrics, or a rubric applied by a judge validated on labels
      → gate: threshold compared, deploy blocked or allowed

  ONLINE — in production, on what users actually sent
    sample traffic → score with the same rubric → store beside the inputs
    capture user feedback → feed it back as labels
    watch for drift: the inputs a fixed set no longer represents
```

### 1.2 Components

**A case.** The unit of evaluation: an input the system will be given, with what counts as a
good outcome for it. Cases are grouped by stratum — essential, average, edge — and a set
without the edge stratum measures the system only where it was expected to work.

**Ground truth.** What an outcome is compared against: a reference answer where one exists, and
otherwise the criteria themselves, which is the normal case for a narrative. Synthetic ground
truth is legitimate when real ground truth is missing, provided it is labelled as such, because
a synthetic reference embeds its author's assumptions.

**A rubric.** The criteria, the scale, and the rule that turns scores into a verdict. Written
down and versioned: a rubric held in someone's head cannot be argued with or compared over
time.

**A judge.** Whatever applies the rubric — usually a model, which makes the judge a component
under test in its own right. A judge has no error rate until one is measured, the measurement
is agreement with human labels on a frozen set, and its instructions are part of its identity:
changing them changes the metric.

**Metrics.** Numbers computed over the scored set, reported with the versions of everything
that produced them — case set, rubric, judge, model, prompt.

**Thresholds and gates.** The value that decides pass, and the point in the delivery path
where failing it stops something. A threshold with no gate is documentation.

**A harness.** The code that runs the system under test, captures the evidence, scores it and
writes a durable result. It must be resumable and append-only: a long run dying two-thirds of
the way through is normal, and re-running from the start is how evaluation becomes too
expensive to do.

**Sampling.** How cases are chosen, and why. Stratification keeps a set representative, a fixed
seed keeps it reproducible, and a recorded procedure is what lets a claim about coverage be
checked.

**Feedback.** Human labels, which validate the judge and refine the criteria, and user signals,
which are the only evidence about the population the system actually serves.

**The reported artifact.** The durable record of a run: what was run, against what, with what
result. A number quoted without it is an anecdote.

### 1.3 Boundaries with adjacent layers

Evaluation measures; it does not own what it measures. The prompt, the chain and the wiring
belong to the orchestrator, which owns them as a versioned unit — evaluation states what a
change did to quality, and the decision to change is taken there. The corpus, its ground truth
and its index belong to the data layer, and an evaluation is only as meaningful as the
representativeness of the corpus it samples.

Two boundaries are worth naming because they are where evaluation is most often quietly wrong.
The first is with the observability plane: the per-request evidence an evaluation needs is the
same evidence that plane records, but the two holdings answer different questions — one exists
to judge quality, the other to debug incidents — so they have different retention, access and
obligations. Evaluation reads from observability; it does not inherit its retention. The second
is with data handling: what may be retained while evaluating is bounded by the same rules as
what may be logged, and a scoring store is not exempt because its purpose is statistical. Where
a requirement asks for prompts and responses to be stored, that instruction and the
data-minimisation posture are in direct tension, and the resolution is a decision rather than
an implementation detail.

Delivery owns where a gate is enforced; evaluation owns what the gate compares. A gate
implemented in a pipeline that evaluation did not agree to is a gate that will be removed the
first time it is inconvenient.

**Terms used throughout this document.**

| Term | Definition |
|---|---|
| Case | One input to the system under test, with the criterion for a good outcome on it. |
| Case set | The collection of cases a run uses, versioned and stratified. |
| Ground truth | What an outcome is compared against: a reference answer, or the criteria when no reference exists. |
| Rubric | The criteria, the scale, and the rule that turns scores into a verdict. |
| Dimension | One criterion within a rubric, scored independently before the verdict is derived. |
| Verdict | The pass or fail decision for a case, derived from its dimension scores. |
| Judge | The component that applies a rubric — usually a model, and therefore itself under evaluation. |
| Human label | A human's verdict on a case, used to validate a judge and refine criteria. |
| Metric | A number computed over a scored set, reported with the versions that produced it. |
| Gate | A threshold at a point in the delivery path where failing it stops a deploy. |
| Offline evaluation | A run on a fixed set, before deploy, for comparability. |
| Online evaluation | Scoring of sampled production traffic, for representativeness. |
| Adversarial case | An input designed to make the system fail, rather than to represent normal use. |
| Evidence | The tool results and retrieved passages an answer was produced from; required to judge groundedness. |
| Regression case | A case kept because a past failure must not return. |

---

## 2. What Google requires

| # | Requirement | Level | Status |
|---|---|---|---|
| D1 | A custom evaluation dataset covering essential, average, and edge cases. Synthetic ground truth is acceptable when real ground truth is missing; refine with human feedback over time. | MUST | Met for the intended and edge bands of the synthetic cohort; no adversarial band, and no coverage of the real note corpus. |
| D2 | Evaluation is automated (LLM-as-judge or rubric; BLEU/ROUGE are insufficient) and metrics + approach are frozen early so runs are comparable. | MUST | Met. A versioned rubric is applied by a model judge, validated against frozen human labels. |
| D3 | Eval set includes adversarial prompts (injection, leakage, fuzzing). | MUST | Unmet. The set contains clinical questions only; adversarial work in this repository is a code review, which is a different activity. |
| D4 | Continuous evaluation in production: sample outputs, score them, store scores with prompts/responses in BigQuery; collect direct user feedback. | MUST | Unmet. Nothing samples or scores production traffic, and the product collects no feedback. Its storage clause also conflicts with data minimisation, and needs a decision. |
| D5 | Evaluation runs as a gate in CI/CD before deploy. | MUST | Unmet. The agent's build runs no tests and no evaluation; it builds, pushes and deploys. |

The group is uniform in one respect worth stating plainly: all five are MUST, which removes the
option of declining one as optional. It also splits in two. D1 and D2 are about the quality of
the apparatus, and this repository has both — a stratified case set, and a judge checked
against human labels. D3, D4 and D5 are about the apparatus having reach: into hostile inputs,
into production, and into delivery. It has none of those, and the pattern is the same in each
case. The apparatus was built for a question someone had at the time; the reach was left for
later.

One requirement carries a tension rather than a task. D4 asks that scores be stored with
prompts and responses in BigQuery, which is the right design for a system whose inputs are not
sensitive — and these inputs are discharge notes. The demonstration's data is synthetic, so the
clause can be honoured here; the risk is that the shape it produces becomes the template for a
deployment whose data is not. That is recorded as a decision to take, not work to do.

Three adjacent requirements govern this layer's subject matter and are audited elsewhere: B1,
which requires ingestion, serving and quality evaluation to be separate subsystems that share a
database layer but not code paths; C4, which requires chains to be evaluated end-to-end rather
than component by component; and F6 and G6, which forbid personal data in logs and require data
minimisation. C4 shapes this layer most: it is why a judge is given the evidence rather than
the answer alone.

---

## 3. Implementation in this repository

The layer is built as two tiers that are deliberately not merged, on the reasoning that
conflating model correctness with answer quality is how an agent project loses the ability to
say which of the two regressed (`docs/evaluation.md`).

**The quantitative tier** measures the readmission model against labels (`mlops/evaluation/`).
Ranking performance is the headline metric because the task is imbalanced at roughly 15%
positive: test AUCPR 0.328, against 0.251 for a reimplementation of the HOSPITAL clinical
baseline over the same cohort, split and metric, and 0.309 for a default-parameter XGBoost
benchmark. The operating threshold is 0.11, chosen to weight recall at β = 2 and checked with
decision-curve analysis, so a missed readmission costs more than a false flag. Calibration is
reported as a Brier score of 0.112, and a fairness audit reports error-rate parity at that
threshold as a diagnostic that flags subgroups for human review rather than as a gate. Feature
selection is recorded as two staged experiments whose result is the 49 features across 23
parent groups the model ships with.

**The qualitative tier** is an LLM-as-judge harness over the agent (`evaluation/agent/`), and
it is the half that matters for a generative layer. Its parts:

*The case set.* `sample.py` scores the demo-split admissions with the deployed endpoint, bins
them into calibrated risk bands (low below 0.10, borderline 0.10–0.20, high at or above 0.20)
and draws a fixed, reproducible, risk-weighted subset of 40 high, 40 borderline and 20 low
patients. Three prompt types are asked of each — risk, discharge medications, discharge
summary — giving 300 cases over the 100-patient golden set and 324 over the 108-patient demo
cohort the deployed artifacts are built from. The bands are the stratum structure D1 asks for:
the high band is the average case, the borderline band is where the model is least decisive,
and the low band is the edge case where a plausible wrong answer is most likely to go
unnoticed.

*Collection.* `collect.py` runs the same local agent graph and MCP toolbox as the Tier 1
integration tests, against the live endpoints over stdio transport, and writes an append-only,
resumable JSONL record per case: the answer, the tool calls with their results, the retrieved
passages, and the Langfuse trace id when the run produced one
(`evaluation/agent/collect.py` 143–146). Capturing the tool results and the passages is what
makes the next step possible, since the judge is given the evidence the agent had rather than
the answer alone (C4).

*Judging.* `judge.py` sends the question, the answer and that evidence to Gemini, which
applies the versioned rubric in `evaluation/agent/rubric.md`: five dimensions — faithfulness,
groundedness, citation accuracy, clinical sensibility, safety — each scored 0–3, with a case
passing only when faithfulness, groundedness and safety all reach 2. The judge returns JSON
with the dimension scores, a verdict, its flags and a reason, and the run aggregates into a
pass-rate report by prompt type. The rubric states its own gate: at least 95% of cases passing,
zero safety failures, and no ungrounded claim anywhere in the sample, any failure cited by
trace id.

*Judge validation.* A judge is a model, so its verdicts were checked against a human. Twelve
cases were hand-labelled (`evaluation/agent/results/human_labels/pilot.md`), and
`validate_judge.py` scores a judge version against those frozen labels — the same twelve cases
rather than a fresh sample, which is what makes the comparison meaningful. This is how D1's
"refine with human feedback over time" is satisfied, and it is the part of the layer easiest to
lose: the labels are the only external check on the judge.

*Regression on the guardrails.* `guardrail_dry_run.py` replays the deterministic post-hoc
guardrails over the frozen traces offline, counting how many failing answers they catch and
whether any *passing* answer is modified. The second number is the important one: a guardrail
that rewrites a good answer is not a safety measure, it is a defect, and it is invisible to any
metric that only counts failures caught.

*Retrieval measurement.* `measure_retrieval.py`, `audit_retrieval.py` and
`analyze_retrieval.py` measure the retrieval path separately from the narrative, with ground
truth parsed from the notes themselves: section recall 100% for the deterministic summary
path, recall@5 100% for the free-text index, and recall@1 of 84.7% for medications and 93.8%
for the hospital course. The distinction between the two k values is the useful part — the
agent always receives the right evidence, and the display layer resolves the exact section
deterministically.

*The runner.* `run_eval.py` chains the three stages as a gate over the pipeline itself:
preflight a small smoke set through the real agent path and exit non-zero if retrieval is
unhealthy, then collect, then judge. It exists because of a specific failure on 2026-08-23: a
broken serving configuration pointed the discharge table at the real MIMIC table while the
deployed index was built from the synthetic notes, and roughly two-thirds of a 324-question run
silently produced zero-passage traces before anyone noticed. The preflight makes that class of
failure die in about thirty seconds instead of after ninety minutes.

**Reported results**, as recorded in `docs/evaluation.md`: 95% pass on the 300-case golden set
with three safety failures and no agent errors, and 97.2% on the 324-case demo cohort with two
safety failures; dimension rates on the golden set are faithfulness 96%, groundedness 98.7%,
citation 99.7%, clinical 99%, safety 99%. The open items recorded there are feature ablation,
bias mitigation, retrieval over the real note corpus, and the migration of score attachment to
the v4 observability path — the last of which is gap 4 below.

---

## 4. Gaps, recommended approach and record of change

### 4.1 Gaps

**Gap 1 — no adversarial set (D3).** Every case in the set is a well-formed clinical question
asked of a cooperative patient record. Nothing tests note text that contains an instruction, a
request for the agent to ignore its constraints, a fabricated citation, or malformed input.
Adversarial cases cannot be extrapolated from ordinary performance: a system can be 97%
faithful on legitimate questions and follow an instruction embedded in a discharge note. The
gap is not that adversarial testing has been overlooked — `docs/adversarial_code_review/`
contains nine reviews — but that a code review exercises an engineer's reading of the source,
while an adversarial evaluation exercises the deployed system's behaviour, and the two find
different things.

**Gap 2 — no gate in the delivery path (D5).** The agent's Cloud Build is three steps: build,
push, deploy (`services/agent/cloudbuild.yaml`). It runs no tests and no evaluation, so a
change that breaks a golden case reaches production as quickly as one that does not. This is
wider than evaluation: the harness's unit and integration suites are not run in CI either. The
consequence for this layer is specific, though — the harness that could fail a build exists,
is run by hand, and therefore competes with whatever else the day holds.

**Gap 3 — nothing evaluates production (D4).** No component samples production traffic,
scores it, or records a score. The raw material is better than it looks: `chain.record_execution`
writes a structured record per request with the model, the code revision, token usage, per-step
latency, the guardrail flags that fired, and the trace id, so *what happened* is already
recorded. Nothing consumes it. The product also collects no user feedback: there is no signal
in the interface for a clinician to mark an answer as wrong, and the conversation store keeps
turns for 24 hours before the retention sweep deletes them, which bounds how late a
post-hoc review of live traffic can happen.

**Gap 4 — the scores do not live with the traces (a condition of D2 and D4).** `judge.py`
already knows how to attach a verdict and its dimension scores to the Langfuse trace that
produced the answer (`evaluation/agent/judge.py` 103–117), keyed by the trace id that
`collect.py` records. That path is inert: it was written against the pre-v4 client API, it
That path is inert: it was written against the pre-v4 client API, it reads its
credentials from a file named `.env.lanfuse` that is not in the repository, and the observability
stack was torn down on 2026-09-12, at which point the orchestrator layer explicitly deferred
both this and the trace-id capture to this layer rather than deleting them
(`layer-03-orchestrator.md`, gap 6). As of 2026-09-19 the preconditions are in place — the agent
returns a `langfuse_trace_id` with every answer, and the v4 stack ingests traces — so what
remains is the migration to the v4 scoring path and its verification. The stale references
also matter on their own: five references across `judge.py` and `run_eval_parallel.py` read
their environment from `.env.lanfuse` — a name missing a letter of the product it
configures — which is the kind of leftover that makes a working path look broken and a
broken one look configured.

**Gap 5 — the comparison the model layer is waiting on.** The model pin moved to
`gemini-3.1-flash-lite` on 2026-09-16, and the comparison the swap owes — the old pin against
the new one, over the question set — has not been run. The model layer recorded it as this
layer's, on the ground that it needs this harness and the retrieval endpoint, and that a
comparison run without tools would measure the wrong thing
(`layer-04-model-runtime.md`, section 5). The same harness is what the model-owner layer needs
before a retrained model can be judged (`layer-06-own-models.md`). Until it is run, the pin is
an assertion rather than a measurement, and the "cheapest model that passes eval" clause has
no eval result behind it.

### 4.2 Recommended approach to fill the gaps

**For gap 1.** Add a fourth prompt family to the case set rather than a separate harness, so
adversarial cases are scored by the same rubric and reported alongside the rest. The cases
worth having are the ones this architecture actually risks: an instruction embedded in note
text, since note text is untrusted input that reaches the model as evidence; a request for a
citation that does not exist; an admission the request is not about, to probe cross-patient
leakage; and malformed input — a truncated note, an empty retrieval result, an out-of-range
admission id. Each needs an explicit pass criterion, because "the answer is unhelpful" is a
pass and "the answer is fluent and wrong" is a failure. Scoring must keep the adversarial band
separate in the report: averaging them into the clinical pass rate would hide both numbers.

**For gap 2.** Two gates at different speeds, both in the agent's build between push and
deploy, where they can stop a rollout without stopping a deploy that is already serving. The
fast one runs the unit and integration suites and the eval preflight smoke set, and must finish
in a couple of minutes because a gate people wait for is a gate people route around. The slow
one runs the full judged set on a schedule rather than on every push, since a 300-case judge
run costs model spend and wall-clock time. Both need the rubric, judge and case-set versions
recorded in the build log, so a run can be compared with the one before it. Canary and rollback
belong to the delivery layer; this layer's contribution is the assertion that canary should
watch.

**For gap 3.** Work with what is already recorded rather than adding an instrument: a scheduled
job that reads recent execution records, re-judges a bounded sample with the same rubric, and
writes the result to BigQuery alongside the model and code revision. Three constraints shape
it. First, the scoring window has to respect the 24-hour conversation retention, so either the
sample is drawn within it or the execution record is the only input the job can use. Second,
the storage clause of D4 needs the decision recorded in section 2: with synthetic demo data it
can be honoured as written, and the design should say in one line that prompts and responses
are stored because the inputs are synthetic, so that the shape is not copied into a deployment
where they are not. Third, user feedback is a product decision rather than an implementation
one: a demo that invites a clinician to mark an answer wrong is collecting an opinion about a
system with synthetic data, and the honest version of that feature is one that says so.

**For gap 4.** Migrate score attachment to the v4 path, verify it end to end by scoring one
trace and confirming the score in ClickHouse and in the UI, and correct the `.env.lanfuse`
references while in the file. One rule carries over from the agent's tracing: attaching a score
is best-effort and must never fail an eval run, for the same reason a trace must never fail an
answer. The value of the work is that a failure becomes browsable — a judge flag and the
evidence behind it read together in the trace, rather than a JSONL line and a trace id joined
by hand.

**For gap 5.** Run the existing harness unchanged over both pins and record the result in the
model layer, which is where the pin is owned. The comparison needs the retrieval endpoint up,
so it belongs to a window where the demo endpoints are running, and it should use the same
judge version as the last baseline or the comparison measures the judge. The output worth
having is not the pass rate alone but the cases that differ between the two pins: that is where
the decision "does the cheaper model cost us anything" is actually settled.

### 4.3 Record of change

**2026-08-14 to 2026-08-19 — the golden set is built and judged repeatedly.** The 100-patient
stratified sample is drawn, 300 traces are collected, and judged runs follow on the 14th, 17th,
18th and 19th. The repeated baselines are the layer's first real artifact: the same set scored
more than once, which is what makes a later change attributable.

**2026-08-23 — a silent configuration failure, and the preflight that came from it.** A run
produced zero-passage traces for roughly two-thirds of 324 cases because the discharge table
and the deployed index were built from different corpora. `run_eval.py` is written as a
consequence: preflight first, fail in thirty seconds rather than after a ninety-minute run.

**2026-08-24 — retrieval evaluation reported.** Section recall and recall@k are measured
separately from the narrative, with ground truth parsed from the notes. The report now lives at
`Archive/docs/agent-root/agent/retrieval_eval_2026-08-24.md`, while `docs/evaluation.md` still
cites its pre-archive path under `services/docs/`, which no longer exists — a small example of
the stale-pointer problem this layer records elsewhere. Later the same day the fixed
108-patient demo cohort run is collected and judged, giving the 97.2% figure at section 3.

**2026-08-25 to 2026-09-16 — the judge is validated and the guardrails are regression-tested.**
Twelve cases are hand-labelled, `validate_judge.py` scores a judge version against those frozen
labels, and `guardrail_dry_run.py` replays the deterministic guardrails over the frozen traces
to check that no passing answer is modified. The model pin moves to `gemini-3.1-flash-lite` on
2026-09-16, and the comparison the swap owes is recorded as owed.

**2026-09-12 — the observability stack is torn down, and this layer inherits the residue.**
Trace ids stop being produced and `judge.py`'s score attachment becomes unreachable. The
orchestrator layer removes the agent-side residue but deliberately keeps both pieces of the
evaluation path, recording them as this layer's rebuild rather than deleting them — so that an
inert path is not later mistaken for a working one.

**2026-09-19 — the preconditions for that rebuild are restored.** The agent emits
`langfuse_trace_id` with every answer, and a self-hosted Langfuse v4 stack ingests traces and
serves the UI at `observability.danielmherman.com`. This document is written the same day: the
layer's apparatus is audited, five gaps are recorded, and none is closed. Gap 4 is the first of
them that can be closed, because it is the only one whose blockers were technical and are now
gone.
