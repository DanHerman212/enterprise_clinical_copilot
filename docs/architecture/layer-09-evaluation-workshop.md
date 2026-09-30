# Layer 9 — Offline evaluation workshop

Companion to `layer-09-evaluation.md`. The workshop has one deliverable: a run that shows how the
agent performs, and names the defects that are worth correcting.

The method is Google's, from *Agent evaluation* in the Gemini Enterprise Agent Platform
documentation, read on 2026-09-30. The vocabulary below is theirs, used in their sense.

---

## 1. The method

| Phase | Activity | Goal |
|---|---|---|
| Design | Define eval cases | Specify the agent's tasks and the expected outcomes. |
| Execution | Run inferences | Generate traces from real or simulated conversations. |
| Scoring | Compute metrics | Grade traces using automated raters. |
| Refinement | Optimize the agent | Propose and verify improvements to instructions or tools. |

The loop has six steps, and each produces one thing.

**Define eval cases.** An *eval case* specifies a task for the agent: the conversation context, one
or more conversation steps, and the expected outcome. Google notes that cases may also specify how a
user's responses are to be simulated, which we do not need, since each of our questions is asked in a
single turn.

**Run inferences.** An *inference* is the execution of one case against the agent.

**Generate traces.** Each inference produces a *trace*: an immutable record of what the agent did,
including the model's inputs, its responses and its tool calls. The trace is the unit that everything
downstream operates on.

**Compute metrics.** A *metric* is a score computed from a trace by a *rater*. Raters may be prebuilt
or custom. A metric that compares the trace against a stored answer is *reference-based*; one that
judges the trace on its own merits is *reference-free*. Google's examples are Exact Match for the
first kind and Helpfulness for the second.

**Conduct analysis.** The metrics, the rubric verdicts and the flagged cases are read together, and
each defect is traced back to the case that exposed it.

**Optimize the agent.** The agent is changed — its instructions or its tools — and the run is
repeated to establish whether the change helped. This step is a loop, not a conclusion.

Two statements in the source bear directly on us. First, automated scoring "lets you score traces
captured from production traffic or external logs, independent of a managed test environment":
evaluation is a property of traces, not of a harness. Second, the Refinement phase assumes the
process is iterated, so the first run is a baseline rather than a verdict.

---

## 2. What already exists here

| Google's term | What it is here |
|---|---|
| Eval case | An admission paired with a prompt type. 89 admissions × 3 prompt types = 267 cases, plus 20 adversarial probes and 89 two-turn follow-up sequences. |
| Inference | A run of the deployed agent over HTTP for one case. A follow-up sequence is two inferences, and the second cannot be issued until the first has answered: it carries that answer back as `turns`. |
| Trace | A JSONL record per case: the answer, every tool call with its result, the retrieved passages, and the trace identifier. Langfuse holds the same run as a browsable trace. |
| Rater | `judge.py`, a custom rater that applies the versioned rubric. |
| Metric | Five dimensions scored 0–3, aggregated to a per-case verdict and a pass rate. |
| Reference-based metric | The adversarial probes: each carries an explicit list of things that must not appear in the answer. Retrieval is also scored against ground truth parsed from the notes. The follow-up set carries exact criteria of its own — the admission on every request, the shape of the replayed turn, a figure repeated identically — which are computed rather than judged. |
| Reference-free metric | The five clinical dimensions. No reference answer exists, so the rubric supplies the criteria instead. |
| Analysis | Scripts that summarise a run, group failures and open a single case for reading. |
| Optimization | Editing the prompt or a tool, then re-running the case set. |

The apparatus is therefore in place. What the workshop adds is a run whose numbers can be trusted,
and a documented reading of what it found.

---

## 3. The steps

Each step states what we do and what it produces. We execute them in order, one at a time.

**Step 1 — Design: define the eval cases.** Review the case set against Google's definition. Our
clinical cases carry a prompt type and a risk band, but no explicit expected outcome; only the
generic rubric applies to them. Decide what "good" means for each class of case, so that a failure
can be described rather than merely scored. *Produces:* the case set, with its expectations, in a
form we can state as a version.

**Step 2 — Execution: run the inferences.** Run the agent over the case set and capture a trace per
case, including the tool results and the retrieved passages. The passages are what make the next step
possible. Two sets are run, because they execute differently: the 267 single-turn cases, whose
requests are independent of one another, and the 89 follow-up sequences, whose second request is
built from the first answer. *Produces:* two append-only trace files for the run, with their dates
and the versions they were produced under.

**Step 3 — Scoring: fix the rater and compute the metrics.** Two decisions are taken here, and they
are the substance of this step.

- **The rater must not be the model under test.** *Taken.* The judge is pinned to `gemini-2.5-pro`
  on the regional endpoint, against the agent's `gemini-3.1-flash-lite`, recorded beside the agent's
  constants so the separation is visible rather than intended. The two differ in generation and in
  tier, which is as much separation as one vendor provides; family independence would need a second
  vendor. The judge's error rate is unmeasured, and no human labels exist for this case set, so that
  check has to begin by producing them.
- **Prefer an exact check to a judged one wherever one is possible.** A judged dimension is an
  opinion with an error rate; an exact comparison is a fact. Where the answer makes a claim that a
  tool result can contradict — a probability, a medication list, an admission identifier — that
  comparison should be computed exactly, and the judge reserved for what cannot be checked that way.
  The guardrail already performs one such exact check; it should be counted as a metric rather than
  left as an internal step. *In force for the follow-up set:* five of its criteria are computed from
  the trace, and only the readings of prose go to the judge.

*Produces:* per-trace scores on every dimension, per-case verdicts, and rates broken down by prompt
type, with the versions of the case set, rubric and rater recorded.

**Step 4 — Analysis: read the failures.** Group the failures by dimension and by prompt type. Open
the individual cases behind each group, and for every defect decide whether its cause lies in the
instructions, a tool, or the evidence the tools returned. *Produces:* a list of defects, each citing
the case that evidences it and the component that appears to be responsible.

**Step 5 — Refinement: change one thing and re-run.** Take the most consequential defect, change the
prompt or the tool that causes it, and re-run the case set. Compare the two runs on the dimensions
concerned, not on the overall rate, which is too coarse to attribute a change. *Produces:* a changed
system, and a comparison showing whether the change helped.

---

### 3.1 The commands, in order

From the harness root:

1. `.venv/bin/python evaluation/agent/collect_followup_http.py` — the 89 sequences. It preflights
   five, one per kind, and stops before the run if the service refuses a replayed turn, because a
   harness that has built the wrong replay shape would fail every sequence at the same step.
   `--dump-cases` prints the plan without issuing anything.
2. `.venv/bin/python evaluation/agent/collect_http.py` — the 267 single-turn cases and the 20
   adversarial probes. Resumable.
3. `.venv/bin/python evaluation/agent/judge.py` — the clinical scores; `--mode adversarial` for the
   probes.
4. `.venv/bin/python evaluation/agent/check_followup_exact.py` — the exact follow-up criteria, which
   are decided by code.
5. `.venv/bin/python evaluation/agent/judge_followup.py` — the follow-up criteria that need reading.

Steps 3 to 5 refuse to score a run whose health gate failed, which is what makes the resulting
numbers worth reading.

One operational obligation stands over the whole window: both Vertex endpoints bill by the hour and
do not scale to zero, so `scripts/agent/teardown.py` closes them when the run is done.

---

## 4. Decisions, and where they stand

1. **The rater model.** *Taken.* `gemini-2.5-pro` on the regional endpoint, pinned beside the agent's
   model constant rather than derived from it. Availability was measured rather than assumed: the 2.5
   models answer on the regional endpoint and are not found on the global one, which is the reverse
   of the agent's arrangement. The judge's error rate is unmeasured. The twelve-case pilot in
   `results/human_labels/` is a labelling worksheet, and the verdicts recorded beside it were written
   into the scripts as a constant rather than given by a person, so it validates nothing and the
   labels have to be produced.
2. **Expected outcomes.** *Open.* Whether the 267 clinical cases gain per-case expectations, and in
   what form. Google's definition of an eval case includes them; ours still do not. The follow-up set
   does not wait on this decision, because each of its criteria is a statement about the sequence
   rather than about clinical content, so it is gradeable without one.
3. **The size of the run.** *Taken for the follow-up set, open for the clinical set.* The follow-up
   set is a census: one sequence per admission, with the five kinds dealt over a cohort sorted by
   probability so that each kind's band mix resembles the cohort's own. `--kinds` widens a single
   kind to all 89 admissions for the runs where one kind's rate is the number being asked for.
   Whether the clinical set runs whole or as a stratified subset is a question about spend rather
   than about method, and the defects are the point either way.

## 5. What the artifact records

A run is only evidence if it can be reproduced and compared. The artifact records the case-set
version, the rubric version, the rater model, the agent's model pin and code revision, the date, the
per-dimension rates, the per-case verdicts, and the defects found. A number quoted without those
versions is not comparable with any other run.

## 6. Out of scope

Production sampling, user feedback and a delivery gate are all requirements of this layer, and none
is part of this workshop. The first has no traffic to sample, the second is roadmap, and the third is
an unresolved decision. They are recorded in the layer document and left there.
