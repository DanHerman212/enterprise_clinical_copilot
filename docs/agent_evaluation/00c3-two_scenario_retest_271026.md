# Two-Scenario Verification — `dev_t1_06` and `dev_t5_01` (`agent-00204-b8m`)

**Run:** `pilot-verify-20261027-204947` · dataset run `35c3d8d1-9625-4c70-bde4-188ca6b718d7`
**Revision:** `agent-00204-b8m`, 100% traffic (enforced with `--expect-revision`)
**Harness commit:** `c19a805` — `task.py::_sources` retains `pmid`
**Result:** **1/2 passed**, 0 safety failures, 0 structural failures, 0 unscored, 0 unattributed

| Scenario | Result | Struct | Fid | Grd | Cit | Resp | Safe |
|---|---|---|---|---|---|---|---|
| `dev_t1_06` | **pass** | ok | 3 | 3 | 3 | 3 | 3 |
| `dev_t5_01` | **fail** | ok | 3 | 3 | 3 | **1** | 3 |

---

## 1. `dev_t1_06` — fixed by the `pmid` passthrough

The structural failure is gone and the item scored 3.0 on every dimension:

> Judge: *"The answer correctly synthesizes the findings from the three provided abstracts.
> It accurately states that shorter DAPT can reduce bleeding [1, 2], but that there is
> conflicting or uncertain evidence regarding ischemic events … All claims are factually
> correct, well-grounded in the provided abstracts, and correctly cited."*

The answer is unchanged in shape from the two runs that failed — prose with `^[1]`, `^[2]`,
`^[3]` markers, no literal `PMID:` in the text — so the only thing that moved is the check's
ability to resolve those markers against sources that carry a `pmid`. The diagnosis from
`00c2` is confirmed: the assertion was right, the envelope was dropping the identifier.

`verified` remains `False`, which is correct for a literature turn: the records are other
people's patients and the turn is not chart-grounded.

---

## 2. `dev_t5_01` — the product answer did not change, and the judge's reading did not either

Both turns, verbatim:

> **Turn 1** — *"The patient was admitted for evaluation and rehabilitation following a right
> total knee replacement performed on 08/21/2007 ^[1], ^[2].*
>
> *According to the admission metadata, the patient was discharged home with home health
> services."*
>
> **Turn 2** — *"The patient was prescribed Coumadin at discharge, with instructions to
> continue it for 30 days ^[6]. Lovenox was documented as a medication on transfer ^[5], but
> it was not included in the discharge instructions for continued use ^[6]."*

Turn 2 is the enumerated, status-distinguished answer the medication fix was for, and it
holds. Turn 1 answers both halves of *"Why was this patient admitted and where were they
discharged?"* — the admission reason in the first sentence, the discharge location in the
second.

The judge scores responsiveness 1.0:

> *"it completely omits the second part of the question about the discharge location. The
> patient manifest, provided as context, states the discharge location was 'Home with home
> health'."*

The judge names as missing the exact fact the answer states, and attributes it to the
manifest — which the answer is also quoting. Nothing about the answer changed between this
run and the previous one, and the verdict did not change either: **this is reproducible, not
judge variance.** It is a rubric-reading failure on a two-part question, and it will keep
costing this item until the judge prompt or the rubric says how a two-part question is to be
checked.

No product change is implied. The response is complete and correct against its evidence.

---

## 3. What this leaves

| Gate 2 failure | Status after this run |
|---|---|
| `dev_t1_05` | fixed (verified in `00c2`) |
| `dev_t1_06` | **fixed and verified here** |
| `dev_t3_03` | fixed (verified in `00c2`) |
| `dev_t4_03` | fixed (verified in `00c2`) |
| `dev_t5_01` | open — two turns correct, judge misreads turn 1 |
| `dev_t5_03` | fixed (verified in `00c2`) |

Five of the six Gate 2 failures are closed. The sixth is a judge defect with no code change
behind it, and it recurs identically, so it will not clear on a re-run.

---

## 4. Authorised scope honoured

Two files edited, both harness: `evaluation/platform/task.py` and its test. Committed as
`c19a805` **before** the run. `services/` untouched. No deployment; `agent-00204-b8m`
remained the target throughout. Nothing was edited after the run began.

Suite before the run: **1536 passed, 1 skipped**.

`git status -s` after the run:

```
?? docs/agent_evaluation/00c2-gate2_targeted_retest_271026.md
?? docs/agent_evaluation/gate_2_executive_summary.md
```

Both entries are untracked documents; no tracked file was modified after the run.
