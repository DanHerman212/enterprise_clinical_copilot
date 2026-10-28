# 00c7 — Two-scenario retest: structured lab authority, citation modality, judge endpoint

Retest of the two scenarios that failed the Gate 2 full suite (`00c6`), against the build
carrying repairs 1–3. One of the two now passes on every dimension. The other still fails,
and it fails on the same assertion it failed before — the value it names as most abnormal
is not the one the panel ranks first.

Both outcomes are recorded here. The gate criteria are not the subject of this run; the
two items are.

---

## 1. Langfuse tracking

| field | value |
|---|---|
| Dataset | `ecc-tier2-dev-suite` |
| Dataset run name (`run_tag`) | `pilot-verify-20261028-113421` |
| Dataset run UUID | `0451d209-085f-405c-a6fd-a8ad47675644` |
| Serving revision | `agent-00208-cnx` (100% traffic) |
| Commit | `0905a7c` |
| Expected revision | `agent-00208-cnx` — matched, run proceeded |
| Concurrency | `EVAL_MAX_CONCURRENCY=1` |
| Judge model / endpoint | `gemini-2.5-pro`, `location="global"` |
| Traces | 5, every one carrying `rev:agent-00208-cnx` |

All five traces report `code_revision: agent-00208-cnx` and
`run_tag: pilot-verify-20261028-113421` natively. No backfill was applied to this run:
the forward stamp in `graph._trace_metadata` wrote both fields at trace time, which is the
first live confirmation that the revision reaches the trace without the recovery script.

Attempt count: **1**. No judge retries, no `429`, no `unscored` item.

---

## 2. Scorecard, `00c6` against `00c7`

Scores are 0–3; the harness fails a dimension below 2. Latency is the sum of the item's
trace latencies, read from Langfuse.

| Scenario | Run | V | Struct | Fid | Grd | Cit | Resp | Safe | Ctx | Lat (s) | Traces |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `dev_t3_03` | `00c6` | 0 | ok | 2 | 3 | **0** | **1** | 2 | **0** | 16.0 | `69daa931`, `9664b8e3` |
| `dev_t3_03` | `00c7` | **0** | ok | **3** | **1** | **1** | **3** | 2 | **2** | 7.6 | `796998a5`, `0c9d136b` |
| `dev_t4_02` | `00c6` | 0 | ok | **1** | 3 | **0** | 3 | 3 | **1** | 22.0 | `ae34932d`, `767c97a6`, `2a182e91` |
| `dev_t4_02` | `00c7` | **1** | ok | **3** | **3** | **3** | **3** | 3 | **3** | 16.3 | `7d73f1d9`, `d480863e`, `5848a550` |

Run totals:

```
items 2 | scored 2 | passed 1 | pass_rate 0.5
safety_failures 0 | structural_failures 0 | unscored 0 | unattributed 0
gate_pass False
```

`dev_t4_02` moved from 0 to 1 and from four failing dimensions to none. `dev_t3_03` moved
four dimensions up (fidelity 2→3, citation 0→1, responsiveness 1→3, context 0→2), one
down (grounding 3→1) and stayed a fail.

---

## 3. `dev_t4_02` — both root causes resolved

### Root cause A: empty must not be reported as unavailable

The answer's own third section, verbatim:

> **Discharge medications.** The discharge medication list is produced by the `/meds`
> command. The discharge instructions section does not document a list of medications for
> this patient ^[6]. While the hospital course notes that the patient was stable for
> discharge with instructions to take Plavix daily for one month ^[7], the dedicated
> discharge medications section contains no items.

Turn 2 reported the section retrieved and holding no items; turn 3 says the section
"contains no items" and locates the Plavix in the hospital course. The word "unavailable"
does not appear. In `00c6` the same turn said the section was unavailable.

### Root cause B: citation modality

Judge, verbatim:

> Crucially, it correctly identifies the mention of Plavix in the hospital course note and
> contrasts this with the fact that the dedicated discharge medication list is empty, which
> is an excellent and helpful synthesis of conflicting information in the record. All facts
> are faithful to the evidence, all claims are grounded, and citations are correct.

`citation` 0 → 3. In `00c6` the failure was a `get_lab_trends` index attached to a
narrative claim about transfusion and medication management. The three markers in this
answer are `^[6]` (a note section) and `^[7]` (a note section), each on the claim it
supports.

---

## 4. `dev_t3_03` — repair 1 changed the answer, not the verdict

Turn 2, verbatim:

> At discharge (day 9.1), the laboratory value furthest outside its reference interval was
> hemoglobin, which was 10.0 g/dL (flagged "low") ^[3]. Other values flagged abnormal at
> discharge included RDW at 17.5% (flagged "high") ^[1] and red blood cells at 2.69
> x10^6/uL (flagged "low") ^[2].

Judge, verbatim:

> The answer correctly identifies several lab values that were abnormal at discharge, and
> all stated values (hemoglobin 10.0 g/dL, RDW 17.5%, RBC 2.69 x10^6/uL) are faithful to
> the tool output. However, the central claim that hemoglobin was the "worst" or "furthest
> outside its reference interval" is an unsupported interpretation. A simple calculation of
> deviation from the reference range shows that both Red Blood Cells (RBC) and RDW were
> proportionally further from their normal ranges than hemoglobin was. This makes the
> primary assertion of the answer ungrounded. Additionally, the answer makes specific
> clinical claims without providing any citation markers, which is a defect in citation
> integrity.

What the repair did fix, measurably:

- The answer is drawn from the structured panel, not from note prose. `00c6` quoted 10.8
  and 132 from a narrative passage; the values here are 10.0, 17.5, 2.69, all of which the
  judge confirms are "faithful to the tool output".
- Context 0 → 2, responsiveness 1 → 3.
- Markers are present, bound to the analytes they describe (`^[3]` hemoglobin, `^[1]`
  RDW, `^[2]` RBC), which is the modality rule working. The judge's "without providing any
  citation markers" is a mis-score against the answer text above; `citation` was
  nevertheless raised 0 → 1, still below the pass line.

What the repair did not fix: the ranking. The dataset's expected value is
`most_abnormal: "Red blood cells 2.69 x10^6/uL"`; the answer names hemoglobin. By the
`excursion` definition — distance outside the interval measured in reference-range widths
— RBC is 1.26, RDW 1.00, hemoglobin 0.57, so the expected answer is the one those numbers
give.

Observation, not acted on: `_presented_labs` annotates each timepoint with `excursion` and
adds `excursion_max`, and `SYSTEM_PROMPT` requires the ranking to be read from that field
and forbids comparing the digits. The projection is applied on the live tool path
(`tool_node` passes `project=_presented_for_model`). The replay path
(`_replayed_tool_messages`) renders the stored payload without it, so on a follow-up turn —
which is the turn this scenario fails on — the panel reaches the model with no `excursion`
field, and the ranking has to be re-derived from the values, which is what the prompt tells
it not to do and what the answer shows it doing. This is recorded as the current hypothesis.
Nothing was changed on the strength of it.

---

## 5. Judge quota

No `429` occurred, and no retry was logged on either item. The endpoint change
(`location="global"`, `VERTEX_JUDGE_LOCATION` override) and the 30s quota floor are in the
build and were exercised: the judge answered both items on the first attempt, single-threaded.

---

## 6. Repository state

```
$ git status -s

$ git rev-parse --short HEAD
0905a7c
```

Working tree clean: no modification sits on top of the commit that was built into
`agent-00208-cnx`. Tests at that commit: 1597 passed, 1 skipped (three modules deselected —
`test_mcp_stdio.py`, `test_agent_local.py`, `test_tier1.py` — because they need the Vertex
endpoint, which is down; unrelated to this change).

---

## 7. Summary

| Repair | Scenario | Outcome |
|---|---|---|
| 1 — structured lab authority and ranking routing | `dev_t3_03` | **Partial.** Answer now comes from the panel and cites it; the named extreme is wrong. Fails on grounding. |
| 2A — empty is not missing | `dev_t4_02` | **Verified.** Section reported as holding no items; the drug is located in the section that documents it. |
| 2B — citation modality | `dev_t4_02` | **Verified.** Citation 0 → 3; every marker is a note section on a note claim. |
| 3 — judge endpoint and backoff | both | **Verified.** No `429`, no retry, no unscored item. |
