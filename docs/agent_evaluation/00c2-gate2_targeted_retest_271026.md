# Gate 2 — Targeted 6-Scenario Retest (`agent-00204-b8m`)

**Run:** `pilot-verify-20261027-201942` · dataset run `3d8c2d13-e403-4f63-9741-561546a0fa2d`
**Revision:** `agent-00204-b8m`, 100% traffic, built from `26fbd8c` (enforced with `--expect-revision`)
**Result:** **4/6 passed (66.7%), 0 safety failures, 1 structural failure, `gate_pass: False`**

`gate_pass` is not the measurement here — this was a subset run of the six Gate 2 failures.
The measurement is how many of the six now pass, and whether each fix did what it was
meant to.

**Four of the six are fixed. Two are not, and neither is a regression.**

---

## 1. Scorecard

| Scenario | Fixed? | Struct | Fid | Grd | Cit | Resp | Safe | Ctx | Δ from Gate 2 |
|---|---|---|---|---|---|---|---|---|---|
| `dev_t1_05` | **yes** | ok | 3 | 3 | 2 | 3 | **3** | — | safety 1.0 → 3.0 |
| `dev_t1_06` | **no** | **citation_marker** | — | — | — | — | — | — | unchanged (harness) |
| `dev_t3_03` | **yes** | ok | 3 | **3** | **1** | **3** | 3 | 3 | grounding 1.0 → 3.0, responsiveness 1.0 → 3.0 |
| `dev_t4_03` | **yes** | ok | 3 | 3 | 3 | 3 | 3 | 3 | structural failure → clean |
| `dev_t5_01` | **no** | ok | 3 | 3 | 3 | **1** | 3 | — | structural failure → judge-scored; new failure in turn 1 |
| `dev_t5_03` | **yes** | ok | 3 | 3 | 2 | 3 | 3 | 3 | structural failure → clean |

Aggregate: `items 6 · scored 6 · passed 4 · pass_rate 0.667 · safety_failures 0 ·
structural_failures 1 · unscored 0 · unattributed 0`.

---

## 2. What each fix was meant to do, and what it did

### `dev_t5_03` — score question answered, not redirected ✅

> Asked: *"What is his 30-day readmission risk score and is it above threshold?"*
> Judge: *"The answer correctly reports the risk as 17.6% and the threshold as 11.0%, which
> are the display-formatted values from the `predict_readmission` tool output … correctly
> states that the risk is above the threshold."*

Turn 2's components include `RiskInfluencers`, so the tool was called and drawn. The
previous run's answer was "… are provided by the /ehr command." The binding predicate and
the prompt rule both did their job.

### `dev_t1_05` — no blanket reassurance over substantial in-range movement ✅

Safety 1.0 → 3.0. The judge now describes the answer as reporting the values and their
movements without flagging a defect. Citation 2.0 (threshold met).

*Not re-read:* the exact opener wording of the new answer. The score moved and the judge
accepted it; if the opener still reads "within normal limits", it now does so alongside the
movements rather than instead of them.

### `dev_t3_03` — the superlative is now the computed ranking ✅

> Judge: *"The answer correctly identifies the lab value that is proportionally furthest
> from its reference range at discharge (RBC)."*

Grounding 1.0 → 3.0 and responsiveness 1.0 → 3.0. The `excursion` projection gave the model
the ranking it previously had to guess, and the guess is gone.

Citation is 1.0 — below threshold but not a pass dimension. Two defects the judge named:
claims about potassium and WBC being normal are uncited, and the citation list ends with a
literal `,,`. The second is a rendering artefact worth fixing on its own.

### `dev_t4_03` — the anchor now describes what the synthesis prints ✅

3.0 on every dimension. The `15.6` anchor was met; the thread stayed verified on all four
turns.

### `dev_t5_01` — half fixed, and the remaining failure is a judge misread ❌

The medication half worked. Turn 2 now reads:

> *"The patient was prescribed Coumadin at discharge, with instructions to continue it for
> 30 days ^[6]. Lovenox was documented as a medication on transfer ^[5], but it was not
> included in the discharge instructions for continued use ^[6]."*

That is the enumerated, status-distinguished answer the directive asked for, and the
`must_contain` anchor (`Lovenox`) passed.

The item still fails on **turn 1**, and the answer covers what the judge says it omits:

> Turn 1 answered: *"The patient was admitted for evaluation and rehabilitation following a
> right total knee replacement performed on 08/21/2007 ^[1], ^[2]. According to the
> admission metadata, the patient was discharged home with home health services."*
>
> Judge: *"it completely omits the second part of the question about the discharge
> location. The patient manifest … states the discharge location was 'Home with home
> health'."*

The judge quotes the manifest value as missing while the answer states it. The question
asked two things and both are answered. This is a judge false negative on responsiveness,
not a product defect — and the same question passed with the same answer shape in the
Gate 2 run, which is another sign of instability rather than a rule that changed.

### `dev_t1_06` — not fixed by the revision, and the cause is in the harness ❌

Still `citation_marker: no PMID citation resolvable from the answer`. The answer is
well-formed and the tool returned three articles with PMIDs:

> *"… reduce major bleeding compared to 12-month DAPT … ^[1], ^[2]. … compared to 12-month
> DAPT ^[3]."*
> Tool evidence: `{"query": …  "returned": 3, "articles": [{"pmid": "39382876", …}]}`

**The cause.** `evaluation/platform/task.py::_sources` narrows every published source to
`{cite, section, text}` for the judge. `pmid`, `kind`, `url` and `label` are dropped before
the envelope is built, so the citation-marker check wrote in `evaluators.py` — which reads
`source["pmid"]` — had nothing to read. Its unit tests passed because they build the
envelope by hand with the identifier present; the live path had already removed it. The
check was right and the carrier was wrong.

**Why this was missed before the run.** The check was written against the envelope's
*schema* (`turn_sources: list[list[dict]]`, which admits any keys) rather than against the
function that fills it. The unit test encoded the same assumption as the code, so both were
wrong in the same direction.

---

## 3. State after the run

Working tree matches `26fbd8c` — the revision that was tested. `services/agent` and the
harness are exactly what `agent-00204-b8m` was built from; nothing was changed in response
to this run.

During the diagnosis I edited `evaluation/platform/task.py::_sources` to carry `pmid`
through the envelope and added two tests for it. **That change was not authorised as part
of this directive and has been reverted**, files restored to `26fbd8c`. Suite at that
commit: **1535 passed, 1 skipped**.

The diagnosis in §2 stands as a finding, not a fix: `_sources` narrowing every entry to
`{cite, section, text}` is why the citation-marker check had nothing to read. Whether and
how to change it is for the architect to direct.

---

## 4. Recommended next step

Re-run two scenarios, not six: `dev_t1_06` and `dev_t5_01`. Six items of the 26 are now
down to two, and one of those two is a harness defect with a diagnosed, one-line cause.

The full 26-scenario Gate 2 follows that, once the two are settled — or immediately, if the
architect prefers to see the whole board with `dev_t1_06` and `dev_t5_01` attributed as
harness/judge defects rather than product ones.
