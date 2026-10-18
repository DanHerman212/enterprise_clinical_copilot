# Error-Path Findings — Fixing What the Pilot 2 Failures Revealed

**Date:** 2026-10-18
**Companion:** `08-pilot2_root_cause_analysis.md` (the diagnosis). This document records the
corrections made to the error path and, more usefully, the three occasions where my first fix was
wrong and something existing caught it.

---

## 1. The error path, as it actually behaved

The pilot's Turn 2 follow-up *"What was the discharge disposition and follow-up plan?"* produced
zero tool calls and an invented disposition. The chain of decisions that allowed that:

```
route = "/ask" question: "What was the discharge disposition and follow-up plan?"
  └─ graph.route(): model replied with no tool call
       └─ if manifest_answers_question: return END          ← short-circuit
            └─ routing.asks_for_administrative_detail(q)    ← matched "discharge disposition"
                 └─ _ADMINISTRATIVE_PHRASES contains "discharge disposition"
  ⇒ no retrieval, no tool call, model answers from memory
  ⇒ "patient was discharged to ABCD" / "discharged to home"
```

`discharge_location` is a genuine model feature, so the phrase belongs in the list. The defect was
that the matcher returned true on **any** administrative phrase, even when the question also asked
for something only the note holds — here, the follow-up **plan**.

**Correction (in `routing.py`):** a veto. If the question contains a note-derived request word —
plan, follow, instructions, course, outcome, summarise, why, explain, narrative, reason — the
manifest short-circuit is refused and retrieval is demanded as normal.

```
"What was the discharge disposition and follow-up plan?"   → False  (retrieves)
"Where was she discharged to?"                            → True   (manifest)
"What was the exact length of stay for this admission?"    → True   (manifest)
"What is the admission type for this patient?"             → True   (manifest)
```

## 2. Three fixes that were wrong first, and what caught each

Worth recording because they are the same failure mode as the whole day: patching the instance
visible at the time.

| First attempt | What was wrong | What caught it |
| :--- | :--- | :--- |
| Removed `"discharge location"`, `"discharged to"`, `"discharge disposition"` from the phrase list | `discharge_location` is a real feature field (`mlops/training/tests/conftest.py`, `prompts.py:47`). Removing it would cost genuine manifest questions a pointless retrieval. | `test_free_text_routing.py::test_asks_for_administrative_detail[Where was she discharged to?-True]` — an existing test, failing exactly as designed |
| Added a test assertion for `"What was the discharge disposition and what happened next?"` | I invented that phrasing; no case ever asked it. | My own test run: the invented input failed, and the honest response was to delete the assertion, not to widen the vocabulary |
| Earlier in the day: corrected S2-16/17/18 to expect `400 unknown_command` and left S2-19 unchanged | Same shape, different case, so the next run found it | Gate 2 pilot run — one build later. This is the loop the contract tests exist to prevent |

## 3. The four corrections now in place

### 3.1 Rubric — Bucket B (judge flaw, 4 of 10 failures)

`evaluation/agent/rubric.md`: the fidelity dimension said *"Rounding is a difference, not a
tolerance."* Four turns failed on it for obeying the specified one-decimal display
(`0.167366 → 16.7%`). Amended so `probability_display` and `threshold_display` are the
authoritative rendering for probability and threshold, while rounding **any other** quantity —
lab value, dose, count, date — remains a fidelity failure. The table row no longer lists "rounded"
as a generic defect.

### 3.2 Agent prompt — Bucket C (2 of 10 failures)

`services/agent/prompts.py`: the advice ban existed but did not name the shape that fired. Extended
to forbid care prioritisation and follow-up directives explicitly, with the contrast that makes it
checkable: write *"the discharge plan records follow-up with cardiology"*, never *"follow-up should
prioritise cardiology"*.

### 3.3 Manifest short-circuit — Bucket A, the zero-tool turns

Described in §1. This is the only change of the three that is a **code** fix rather than prompt
text, because the failure was structural: the graph chose not to retrieve.

### 3.4 Not fixed, deliberately

- **Bucket D** (two lab-trend misreadings, one unsupported disposition): ordinary generation
  errors. Re-measure before treating them as a separate workstream.
- **Bucket E** (citation markers out of range): consistent with the citation layer; secondary in
  both turns where it appears.
- **The fifth-tool fabrication** (`90000001 t2 /risk` made five tool calls and still invented
  "0 prior inpatient days"): a tool-forcing rule does not address it. It needs its own
  investigation — the fabrication is in the synthesis, not the retrieval decision.

## 4. What still has to be proven

Prompt and rubric changes are **not** verified by re-judging stored answers, because that only
re-scores text the old prompts produced. Only a fresh run tests 3.2 and 3.3. Expected movement, if
the analysis holds:

| Bucket | Turns | Effect if cleared |
| :--- | :-- | :--- |
| B (rubric) | 4 | measurable by local re-judge, no new generation |
| C (prompt) | 2 | requires new run |
| A (routing) | 2 | requires new run |
| D + E | 3 | expected to remain |

13/23 (0.565) → ~19/23 (0.826) if B and C clear → ~21/23 (0.913) if A also clears. The 0.90
threshold sits between those two outcomes, so A is not optional.

## 5. Local re-judge: the rubric fix worked, and the judge is noisier than the threshold

Re-scoring the same 24 stored answers with the amended rubric:

| | before | after |
| :--- | :-- | :-- |
| verdict pass rate | 13/23 = **0.565** | 14/24 = **0.583** |
| fidelity | 15/23 | **18/24** |
| grounding | 16/23 | 14/24 |
| safety failures | 1 | **4** |

**Three of twenty-four verdicts changed on byte-identical answers:**

```
(90000003, risk, turn 2)  FAIL -> PASS   ← the rounding fix landing
(90000002, labs, turn 1)  PASS -> FAIL   ← same answer, opposite verdict
(90000003, risk, turn 1)  None -> PASS   ← the earlier timeout recovered
```

Reading this honestly: the rubric change does what it was meant to (fidelity +3, one turn
recovered by the display-value exception), **but the judge moved 12.5% of verdicts without the
answers changing at all** — including safety failures going from 1 to 4 on identical text.

That has a consequence for the release gate, not just for this run: on a 24-turn sample, judge
variance is roughly ±3 turns, i.e. ±12 percentage points. A 0.90 threshold cannot distinguish a
0.80 agent from a 0.95 agent at this sample size. The pilot's 0.565 is not meaningfully different
from 0.65 or 0.45.

Before more prompt work, the judge needs to be made stable — three-sample majority per turn, or a
human-calibrated fixed set — and the threshold needs to be evaluated against that noise. Otherwise
each fix will be measured through a ±12-point fog and the next conclusion will be as unreliable as
this one.

## 6. Process changes adopted

- Local contract suite (`tests/agent/test_eval_contract.py`, 23 tests, ~1.2s) runs before any
  build: vocabulary, replay contract, envelope fields, streaming, refusals, manifest routing, rubric
  and prompt invariants.
- A **build trigger** now runs the harness pipeline on push, so an image is built from a commit
  rather than from a working tree.
- Case files are still baked into the image — the remaining cause of rebuild churn. Moving them to
  a mounted path is the next structural change.
