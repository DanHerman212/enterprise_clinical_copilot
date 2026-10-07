# Evaluation analysis — agenda

Working document. Its companion, `docs/evaluation-run-2026-09-30.md`, holds the
numbers; this one holds the review of them: what each section's evidence
supports, why, and what follows.

Run under review: tag `eval-093026`, 2026-09-30. 532 clinical answers and 20
adversarial probes, judged by `gemini-2.5-pro` against rubric v2.

## How each item is worked

Every item is interrogated in the same four steps, in order.

1. **Observation** — what the data shows, stated without interpretation.
2. **Attribution** — which of three causes it supports: instrument error (the
   judge), agent behaviour, or infrastructure. Defects look alike until this
   step separates them.
3. **Examination** — what must be inspected to discriminate between those
   causes, and what evidence would settle it.
4. **Remedy** — engineering fix, architectural change, measurement redesign, or
   accepted with a recorded rationale.

A finding is recorded under its item only after step 2, never at step 1.

## Standing cautions

Two attribution failures were made on 2026-09-30. This review exists partly to
prevent their repetition, and both took the same form: the observation was
correct, the cause was asserted before it was examined.

- A 100% failure rate on the `risk` follow-up was read as an agent defect. It
  was the harness: the rater had been handed an empty evidence set.
- The observability outage was attributed to a recent configuration change. It
  was a host network fault.

## Agenda

### 1. Method and scope

What the run establishes, and what lies outside its reach: two turns per
conversation, 532 clinical answers, 20 probes, one rubric, one deployment
revision, 89 admissions.

**Question:** which claims in the results document does this method support?

### 2. Instrument validity

One judge model at temperature 0, with an error rate that has never been
measured. This precedes every causal claim that follows: until the instrument is
bounded, "the agent fails citation" and "the judge dislikes citation" are
indistinguishable.

**Question:** how much of the measured failure belongs to the agent, and how
much to the instrument?

**Examine:** the rubric's determinism under repeat scoring; a second judge's
agreement rate; whether the flags include judgements a clinician would not make.

### 3. Clinical headline

86.8% pass, 70 failures of 530 scored.

**Question:** is the failure distribution structural or diffuse?

**Examine:** failures by chip, by turn, by dimension and by patient, and whether
a small number of admissions or question shapes carries them.

### 4. Citation, 76.2%

The dominant defect: 126 failures against 34 for fidelity. The flags describe
one recurring error rather than noise — a marker on the wrong passage, or one
marker covering two evidence sources.

**Question:** is this retrieval (the evidence never contained the cited claim),
generation (the marker is placed carelessly), or presentation (the passage is
right but the marker is misnumbered)?

**Examine:** the cited passage against its sentence for the failing rows;
whether failures concentrate on multi-paragraph answers; whether markers are
consistently off by one.

### 5. Adversarial, 16/20

Four failures of differing kinds: tool-schema disclosure, a fabricated error
message, an unrefused empty prompt, and padded precision.

**Question:** do they share a cause, or are they independent defects?

**Examine:** each probe's transcript; whether disclosure and fabrication are the
same failure of instruction-following; whether the empty prompt exposes a
contract gap rather than a model flaw.

### 6. Retrieval

Twenty zero-passage retrievals, clustered in seven patients, all of which
passed, against a 13.2% overall failure rate.

**Question:** does the agent handle an empty evidence set correctly, or do the
retrieval metric and the verdict measure different things?

**Examine:** the twenty answers, for abstention, reliance on the prediction, or
answer from the prior turn; and the seven patients' notes for a corpus cause.

### 7. Harness and infrastructure

Six harness defects found and fixed mid-run, each of which produced wrong numbers
before it was found, and four infrastructure faults still open.

**Question:** which of these recur, and under what load?

**Examine:** the fixed defects for the same class of error elsewhere; the open
faults — DHCP renewal, Redis socket timeouts, the ClickHouse memory ceiling, and
score-deletion throughput. Remedy here is engineering by default.

### 8. Disposition

**Question:** what do we change, in what order, and what re-measurement would
confirm it?

**Output:** a ranked list, each entry carrying its remedy class and the
observation that would falsify the fix.

## Recording

Findings are appended beneath the item they belong to, in step order, as we work
through the agenda. The agenda is not renumbered.
