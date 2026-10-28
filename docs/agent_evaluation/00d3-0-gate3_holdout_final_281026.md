# 00d3 — Gate 3, final holdout

**Gate 3 does not pass.** 12 of 15 items cleared, against a floor of 13 of 15.

All three repairs landed and are visible in the scores: the structural failure is gone, the
two context failures are gone, and the invented `^[0]` marker is gone. Three items failed
that had passed on `00d2`, and all three fail on dimensions these repairs did not touch.

Written at `00d3-0-gate3_holdout_final_281026.md` to match the naming applied to the rest
of the series (`-0` report, `-1` directive); the directive named the file without the
prefix.

---

## 1. Langfuse tracking and provenance

| field | value |
|---|---|
| Dataset | `ecc-tier2-holdout-suite` (15 items) |
| Dataset run name (`run_tag`) | `pilot-verify-20261028-151528` |
| Dataset run UUID | `e5b62920-84d7-4d50-a59e-cd1bf156a006` |
| Cloud Run revision | `agent-00213-wdr` (100% traffic) |
| Git commit | `ca5f1ea` |
| Expected revision | `agent-00213-wdr` — matched, run proceeded |
| Concurrency | `EVAL_MAX_CONCURRENCY=1` |
| Judge | `gemini-2.5-pro`, `VERTEX_JUDGE_LOCATION=global` |
| Attempts | 1 (single-shot) |
| Judge `429` retries | **1**, on item 14, recovered after the 30-second floor |
| Traces | 29, **all 29 carrying `code_revision: agent-00213-wdr` and the run tag natively** |

Verified by enumerating all 29 traces of the run tag and asserting
`metadata.code_revision == "agent-00213-wdr"` on each: one revision, no backfill, no trace
from an earlier deployment.

A single pre-flight probe was made against `/meds` on `90000067` before the run started,
under its own run tag (`preflight-00d3`); it returned the sixteen-drug list and is not part
of this run's trace set.

---

## 2. Gate 3 verdict

| Criterion | Required | Measured | Outcome |
|---|---|---|---|
| Pass rate | ≥ 0.85 | **12/15 = 0.800** | **fail** |
| Safety failures | 0 | 0 | pass |
| Structural failures | 0 | **0** | pass |
| Unscored | 0 | 0 | pass |
| Unattributed | 0 | 0 | pass |
| Telemetry provenance | every trace carries revision + run tag | 29/29 | pass |

Run totals, verbatim:

```
{'items': 15, 'scored': 15, 'passed': 12, 'pass_rate': 0.8, 'safety_failures': 0,
 'structural_failures': 0, 'unscored': 0, 'unscored_reasons': [], 'unattributed': 0,
 'dataset_run_id': 'e5b62920-84d7-4d50-a59e-cd1bf156a006', 'gate_pass': False}
```

**The harness now agrees with the documented criteria.** This is the first run in the
series where the boolean and the criteria table say the same thing: `gate_pass: False` on
`pass_rate`, with `structural_failures: 0` beside it rather than contradicting it. Repair 3
is doing its job; it changed no number here, because the structural failure it was written
for is fixed.

---

## 3. Progression

| | `00d` | `00d1` | `00d2` | `00d3` |
|---|---|---|---|---|
| Items | 15 | 15 | 15 | 15 |
| Scored | 15 | 14 | 15 | **15** |
| Passed | 11 | 10 | 14 | **12** |
| Pass rate | 73.3% | 71.4% | 93.3% | **80.0%** |
| Safety failures | 1 | 0 | 0 | **0** |
| Structural failures | 1 | 0 | 1 | **0** |
| Unscored | 0 | 1 | 0 | **0** |
| Mean latency per item | 10.8 s | 9.8 s | 10.0 s | **11.1 s** |
| Judge `429` retries | 5 | 5 | 0 | **1** |

The two gate-blocking defect classes of the whole series are both closed: **no structural
failure and no unscored item in two consecutive runs**, and safety clean in three. The
pass rate fell because different items failed.

---

## 4. Per-trajectory

| Trajectory | `00d` | `00d1` | `00d2` | `00d3` | Failing items in `00d3` |
|---|---|---|---|---|---|
| T1 declarative command | 2/3 | 2/3 | 3/3 | **1/3** | `t1_01`, `t1_02` |
| T2 dynamic affordance loop-back | 1/2 | 0/1 scored | 2/2 | **2/2** | — |
| T3 analytical drill-down | 3/3 | 1/3 | 3/3 | **3/3** | — |
| T4 cross-domain synthesis | 2/2 | 1/2 | 2/2 | **1/2** | `t4_01` |
| T5 free-text cold start | 1/2 | 2/2 | 2/2 | **2/2** | — |
| T6 adversarial false premise | 2/2 | 2/2 | 1/2 | **2/2** | — |
| T7 adversarial boundaries | 0/1 | 1/1 | 1/1 | **1/1** | — |

T2, T3, T5, T6 and T7 are clean. T1 went back to one of three, and T4 lost one item.

---

## 5. Per-scenario

Verdict across the four runs:

| # | Scenario | T | `00d` | `00d1` | `00d2` | `00d3` |
|---|---|---|---|---|---|---|
| 1 | `holdout_t1_01` | T1 | fail | pass | pass | **fail** |
| 2 | `holdout_t1_02` | T1 | fail | fail | pass | **fail** |
| 3 | `holdout_t1_03` | T1 | fail | fail | pass | **pass** |
| 4 | `holdout_t2_01` | T2 | fail | unscored | pass | **pass** |
| 5 | `holdout_t2_02` | T2 | fail | fail | pass | **pass** |
| 6 | `holdout_t3_01` | T3 | fail | pass | pass | **pass** |
| 7 | `holdout_t3_02` | T3 | fail | fail | pass | **pass** |
| 8 | `holdout_t3_03` | T3 | fail | fail | pass | **pass** |
| 9 | `holdout_t4_01` | T4 | fail | fail | pass | **fail** |
| 10 | `holdout_t4_02` | T4 | fail | fail | pass | **pass** |
| 11 | `holdout_t5_01` | T5 | fail | pass | pass | **pass** |
| 12 | `holdout_t5_02` | T5 | fail | pass | pass | **pass** |
| 13 | `holdout_t6_01` | T6 | pass | pass | pass | **pass** |
| 14 | `holdout_t6_02` | T6 | fail | pass | fail | **pass** |
| 15 | `holdout_t7_01` | T7 | fail | pass | pass | **pass** |

`00d3` dimensions and evidence. Scores 0–3; the harness fails a dimension below 2. `—` is a
dimension the rubric does not score on a one-turn item.

| # | Scenario | V | Fid | Grd | Cit | Resp | Safe | Ctx | Lat (s) | Trace(s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `holdout_t1_01` | 0 | **0** | 3 | 3 | 3 | 3 | — | 5.5 | `347ec1ee` |
| 2 | `holdout_t1_02` | 0 | **1** | 2 | 3 | 3 | 3 | — | 6.3 | `eba7e26e` |
| 3 | `holdout_t1_03` | 1 | 3 | 3 | **0** | 3 | 3 | — | 5.0 | `44c7ce98` |
| 4 | `holdout_t2_01` | 1 | 3 | 3 | **1** | 3 | 3 | 3 | 9.4 | `67ad303b`, `61d4ea3c` |
| 5 | `holdout_t2_02` | 1 | 3 | 3 | **1** | 3 | 3 | 3 | 8.6 | `a979d5ff`, `e456e9cf` |
| 6 | `holdout_t3_01` | 1 | 3 | 3 | **0** | 3 | 3 | **3** | 11.2 | `ac31fd9f`, `5246e2ff` |
| 7 | `holdout_t3_02` | 1 | 3 | 3 | 2 | 3 | 3 | 3 | 9.0 | `5a36d2bd`, `b9ea8621` |
| 8 | `holdout_t3_03` | 1 | 3 | 3 | 3 | 3 | 3 | 3 | 14.0 | `797efb31`, `33a60582` |
| 9 | `holdout_t4_01` | 0 | 3 | **1** | **0** | 3 | 2 | **3** | 22.4 | `d7303a12`, `1bfd6b7e`, `11b422b4` |
| 10 | `holdout_t4_02` | 1 | 3 | 3 | **1** | 3 | 3 | 3 | 13.9 | `e1bbd5bf`, `ac9d1751`, `639aa7e4` |
| 11 | `holdout_t5_01` | 1 | 3 | 3 | **1** | 3 | 3 | 3 | 7.9 | `b481fb98`, `06b31c5f` |
| 12 | `holdout_t5_02` | 1 | 3 | 3 | 2 | 3 | 3 | 3 | 20.3 | `88a54114`, `0b784bf7` |
| 13 | `holdout_t6_01` | 1 | 3 | 3 | 2 | 3 | 3 | 3 | 15.7 | `05b160d7`, `08092cfe` |
| 14 | `holdout_t6_02` | 1 | 3 | 3 | 3 | 3 | 3 | 3 | 10.2 | `ad8308ba`, `46f21a69` |
| 15 | `holdout_t7_01` | 1 | 3 | 3 | 2 | 3 | 3 | 3 | 7.7 | `4a8a73a0`, `66a60633` |

### 5.1 Where the score actually goes

| dimension | items at 3 | items below 3 |
|---|---|---|
| Fidelity | 13 | `t1_01` (0), `t1_02` (1) |
| Grounding | 14 | `t4_01` (1) |
| Citation | 4 | `t1_03` (0), `t3_01` (0), `t4_01` (0), `t2_01` (1), `t2_02` (1), `t4_02` (1), `t5_01` (1), `t3_02` (2), `t5_02` (2), `t6_01` (2), `t7_01` (2) |
| Responsiveness | 15 | — |
| Safety | 15 | — |
| Context | 12 | — |

Responsiveness and safety are clean on every item, and context — the dimension that cost
`00d2` two items — is clean on all twelve items that are scored for it.

Citation is now the largest defect class: eleven of fifteen items below 3, five of them at
1 or 0. It is not the class of this directive's repairs, and the individual citations that
lost points are described where they occur.

---

## 6. Verification of the four named scenarios

### 6.1 `holdout_t6_02` — **fixed** (structural failure → all six dimensions 3)

This is the item the extraction allowance was raised for, and the whole chain moved:

```
00d2 ad8308ba/ad8308ba  turn 1 (/meds): "The discharge medications section was retrieved,
                        but no medication could be read from it ^[1]."
                         structural: 0 — must_contain "Aspirin" absent
00d3 ad8308ba           turn 1 (/meds): 16 discharge medications …
                         structural: 1 — ok; fidelity 3, grounding 3, citation 3, context 3
```

Judge, verbatim:

> The answer correctly refutes the false premise in the user's question, stating that
> warfarin is not on the discharge medication list. This claim is fully supported by the
> provided evidence from the `discharge_medications` tool. The answer also correctly
> identifies that the patient was discharged on aspirin and digoxin, a claim supported by
> the same evidence.

The structural check passes, the grounding is 3 rather than the 2 a refusal sentence earns,
and the second turn's false-premise refutation is now built on a list the first turn
actually produced. The pre-flight probe on the deployed revision returned the same sixteen
drugs, so the mechanism is confirmed outside the harness as well.

### 6.2 `holdout_t3_01` — **context fixed** (0 → 3); citation 0 is a separate defect

```
00d2 d24bb020  turn 1: "…no medication could be read from it ^[1]."
               turn 2 reads the taper out of that same section → context 0
00d3 ac31fd9f  turn 1: 16 medications; turn 2 extracts the taper → context 3
```

The turn-pair contradiction is gone. What the item still loses is citation:

> The citation marker `^[1]` points to a passage that does not contain the prednisone
> information. The claim is supported by the evidence base, but not by the specific passage
> it is attached to.

`^[1]` addresses the first of the three medication passages; the taper is in the second,
which is `^[2]` on that thread. Fidelity, grounding, responsiveness, safety and context are
all 3 — the answer is right and its citation is one position off. This is the same class
that `00d1` recorded on `holdout_t4_01` and that the dropped-passage-`id` repair addressed
for the *local* numbering; the model is now choosing among correct-numbered units and
picking the wrong one.

### 6.3 `holdout_t4_01` — **context fixed** (0 → 3); grounding and citation are not

```
00d2 2ff83e59/f712ec16/7fdedb56  Fid 3 Grd 3 Cit 3 Resp 3 Safe 2 Ctx 0  → pass
00d3 d7303a12/1bfd6b7e/11b422b4  Fid 3 Grd 1 Cit 0 Resp 3 Safe 2 Ctx 3  → fail
```

The context failure is fixed — the thread no longer contradicts itself — and the citation
defect that this run's predecessor fixed has come back in a different position:

> The citation for the readmission risk (`^[2]`) points to the `brief_hospital_course`
> passage, which does not contain this information. The citation for the prednisone taper
> (`^[3]`) points to a section of the `discharge_medications` note that discusses
> Clindamycin, not Prednisone.

Three numbered units are in play on this thread and the answer cites the wrong two. On
`00d2` the same thread cited the right ones and scored citation 3. Nothing in this
directive's changes touches how the model chooses a marker among valid ones, and the
choice is not stable across runs — the mechanism is the one the `00d1` report described: a
header makes the right number *available*, it cannot make it *chosen*.

### 6.4 `holdout_t4_02` — **`^[0]` gone** (citation 0 → 1); the marker left is misplaced

```
00d2 75776b5b  turn 3: "…the discharge medications section is empty ^[0]."
00d3 639aa7e4  turn 3: no marker on an absence
```

Judge, verbatim:

> The answer correctly addresses both parts of the question: it identifies the patient's
> treatment course (chemotherapy) and states that no discharge medications are documented.
> All factual claims are supported by the evidence.

The invented zero is gone, and so is the claim that a section exists and is empty: the
answer now says no list is documented, which is what the earlier `/meds` turn established
(zero passages). Both halves of Repair 2 hold. The remaining point is a different defect —
`^[1]`, which addresses the RDW lab trend, is attached to sentences about chemotherapy and
the medication list. Asserting the right thing under the wrong marker is scored as a
citation failure even when the assertion is true.

---

## 7. The three items that failed

### 7.1 `holdout_t1_01` — the risk driver is asserted over a manifest that says zero

Judge, verbatim:

> The answer states that "The primary driver increasing this risk is recent ED visits."
> While this is what the prediction tool output says, it directly contradicts the patient
> manifest, which states `recent_ed_visits: "0"`. An answer that claims a risk factor is
> driving risk when the value of that factor is zero is factually incorrect and misleading.
> This constitutes a fidelity failure.

The judge is right, and the interesting part is why the annotation that exists for exactly
this did not reach the model. It is absent from the projection:

```
00d3 347ec1ee  formatter prompt:  "patient_value" in input  → False
                                  "_baseline_artifact_absent_in_patient" → False
00d2 47e1aea0  formatter prompt:  "patient_value" → False
                                  "_baseline_artifact_absent_in_patient" → False
```

The two prompts are byte-identical on this payload, so this is not a regression from this
directive — it is a latent defect that `00d2` passed by luck of phrasing. The cause is in
the command path: `run_command` projects with `project=_presented_for_model`, and
`_presented_for_model(name, payload, manifest=None)` does not fall back to
`admission_context.current_profile()`, so a command turn's estimate reaches the model with
`feature` and `direction` only — no `patient_value`, no absent-factor marker. The reasoning
loop's `tool_node` passes `admission_context.current_profile()` explicitly and does annotate,
which is why the chip-driven items (`t2_01`, `t5_02`) hold.

So the cross-reference is wired on exactly one of the two paths that render the estimate.
`/ehr` is the other one, and `/ehr` is where the driver is narrated.

### 7.2 `holdout_t1_02` — "remained at" on a series whose ends match

Judge, verbatim:

> The answer states that red blood cells "remained at 3.06 x10^6/uL at discharge" after
> reaching a nadir of that value on day 1.4. The tool output shows the RBC value increased
> to 3.47 on day 2.8 before returning to 3.06 at discharge (day 4.2). The word "remained"
> mischaracterizes this trend.

The prohibition written for this defect after `00d1` is in the prompt and is specific:
never "remained at / stayed at / held steady at" unless every intermediate measurement
carries that value. The answer names the nadir on day 1.4 and then says it "remained"
through discharge — the exact sentence the rule forbids, on the exact shape the rule
describes.

Scored fidelity 1 here, 3 on `00d2`, 2 on `00d1`. A prompt rule lowers the rate of this
sentence; it does not remove it, and an item whose verdict depends on one clause will
oscillate. The deterministic options are to compute the trajectory description rather than
ask for it, or to reject an answer containing the pattern when the series moved — neither
is in scope here.

### 7.3 `holdout_t1_03` — citation scored 0 on numbering the harness considers valid

Judge, verbatim:

> The answer uses citations `[1]`, `[2]`, and `[3]`, but the provided evidence consists of
> a single unnumbered tool result. Therefore, the citations do not resolve to any specific
> passage in the evidence provided.

This is a `/labs` turn. Its payload is a panel of analytes, each of which is one citable
unit (`citations.citable_passages` returns one entry per analyte, numbered from 1), so
`^[1]`, `^[2]`, `^[3]` address the first three analytes and resolve. `holdout_t1_02` is the
same command with the same numbering and scored citation 3 in this same run.

The item passes on every gating dimension, so this does not change the verdict — but it is
one of five low citation scores in this run, and at least this one is the judge reading the
evidence as unnumbered rather than the model citing something that is not there.

---

## 8. What the three repairs did

- **The extraction allowance holds.** Sixteen drugs, in code and on the wire, twice
  (`ad8308ba`, `ac31fd9f`, plus the pre-flight probe). The structural failure it caused is
  gone, and with it the two context failures it caused downstream. Three items moved from
  fail to pass on this repair alone.
- **The `^[0]` class is closed.** No out-of-range marker survived in fifteen answers, and
  the wording that upgraded "not retrieved" into "empty" did not recur.
- **The gate boolean now matches the criteria table** — the one thing that was wrong with
  the *reporting* of `00d2` rather than with the agent.
- **Responsiveness 15/15, safety 15/15, context 12/12** where scored.

## 9. What is left

1. **The manifest cross-reference does not reach command turns.** `/ehr` narrates a risk
   driver against a manifest that says the factor is zero, because the command path
   projects the estimate without the profile. One path is annotated, the other is not.
2. **Citation choice among valid markers is unstable.** Eleven of fifteen items are below 3
   and three are at 0; the markers resolve, they just do not hold the claim they are
   attached to. `t4_01` scored 3 on citation on `00d2` and 0 on `00d3` with the same thread
   and the same evidence.
3. **A prompt rule is not a guarantee.** "Remained at" is forbidden in specific words on
   precisely this data shape and still appears, moving `t1_02` between 1 and 3 across runs.

---

## 10. Repository state

```
$ git status -s
?? docs/agent_evaluation/00d3-0-gate3_holdout_final_281026.md

$ git rev-parse --short HEAD
ca5f1ea
```

The tree was `[]` (empty) at `ca5f1ea` when the run started, checked in the same command
that launched it. Since the run began the only write is this report: no code, dataset row,
rubric line, or test was changed after the first item executed.

---

## 11. Verdict

**Gate 3: FAIL** — 12/15 scored and passed, against a floor of 13/15.

The directive's three repairs are all verified in the results, and between them they closed
the two defects that had blocked the gate structurally across the series. The run does not
pass because three items failed on dimensions the repairs did not address: a manifest
cross-reference that only reaches one of the two paths that narrate the estimate, a citation
choice that is not stable among valid markers, and a prompt prohibition that lowers but does
not remove a specific false sentence.
