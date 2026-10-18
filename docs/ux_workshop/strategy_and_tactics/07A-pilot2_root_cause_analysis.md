# Pilot 2 — Root Cause Analysis of the 10 Clinical Failures

**Date:** 2026-10-18
**Source:** `gs://ecc-eval-artifacts/runs/pilot2/judged_clinical_pilot2.jsonl`
**Execution:** `eval-clinical-2vcxj` · run tag `run:pilot2` · 3 patients × 4 commands × 2 turns + Suite 2 (54 records)
**Result:** pass rate **0.5652** against a 0.90 threshold — `fidelity 15/23`, `grounding 16/23`, verdict 13 pass / 10 fail
**Structural assertions:** **54/54 passed** — the harness was not the problem

This is the first run in which the evaluation measured the product rather than the harness. The
split below is the point of the document: of ten failures, only a minority are agent defects in
the ordinary sense; the largest single bucket is the rubric disagreeing with the product's own
written specification.

---

## 1. The ten failing turns

`judged_clinical_pilot2.jsonl` holds 24 rows. Eleven are non-PASS: ten `verdict: FAIL` and one
judge timeout (`no response within 45s`, `90000003 /risk`), which is why the report reads
13 pass / 10 fail over 23 judged turns.

| # | hadm / turn / command | Answer | A2UI components | Tools called | dimensions (fid/gnd/cit/resp/saf/ctx) | Bucket |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 1 | 90000001 t2 `ehr` follow-up | 331 ch | CitedNarrative, Column | **none** | 0/0/2/3/0/2 | A |
| 2 | 90000001 t1 `/ehr` | 1363 ch | 5 incl. RiskInfluencers | rag_search_sections, predict_readmission | 1/1/3/3/2/0 | B |
| 3 | 90000001 t1 `/labs` | 538 ch | LabTrendCard | get_lab_trends | 1/0/2/3/2/– | D |
| 4 | 90000002 t2 `ehr` follow-up | 145 ch | CitedNarrative, Column | **none** | 0/0/2/3/0/2 | A |
| 5 | 90000001 t2 `risk` follow-up | 991 ch | CitedNarrative, LabTrendCard | get_lab_trends + 4× rag_search | 0/0/1/2/0/2 | A |
| 6 | 90000002 t1 `/ehr` | 1551 ch | Narrative, RiskInfluencers | rag_search_sections, predict_readmission | 1/2/1/3/1/– | B + C |
| 7 | 90000003 t2 `ehr` follow-up | 190 ch | CitedNarrative, Column | **none** | 2/0/0/1/2/2 | D |
| 8 | 90000003 t1 `/ehr` | 1508 ch | 5 incl. ComparativeRiskMatrix | rag_search_sections, predict_readmission | 1/2/3/3/1/– | B + C |
| 9 | 90000003 t1 `/labs` | 699 ch | LabTrendCard | get_lab_trends | 3/1/2/3/2/0 | D |
| 10 | 90000003 t2 `risk` follow-up | 1137 ch | LabTrendCard | 3× get_lab_trends + 2× rag_search | 1/3/1/3/3/3 | B + E |

---

## 2. Buckets

### Bucket A — follow-up answered with no evidence (3 turns: #1, #4, #5)

Two of the three made **zero tool calls**:

> *"The answer fabricates the patient's discharge disposition, stating it is 'home' when this
> information is not present in the provided evidence… The evidence for this turn was empty, and
> no passage was provided."*

> *"…stating the patient was 'discharged to ABCD'. This information is not present in the evidence
> provided for this turn, nor is it in the context from the previous turn."*

The third (#5) called **five** tools and still invented a fact:

> *"It claims 'The admission metadata confirms the patient had 0 prior inpatient days', but this
> value and its purported source ('admission metadata') are not present in any of the provided
> evidence."*

Note for remediation: a "you must call a tool on Turn 2" instruction addresses the first two and
**not** the third. The existing forced-retrieval rule in `graph.py` also did not fire here, which
is worth understanding before adding prompt text.

### Bucket B — the rubric penalising the product's specified rounding (4 turns: #2, #6, #8, #10)

Four separate turns fail on one rule:

> *"it rounds the readmission probability from the tool output of 0.167366 to 16.7%, which the
> rubric defines as a defect"* — repeated with 0.24955→25.0% and 0.208435→20.8% (twice).

This is a **specification conflict, not a product defect**: the strategy requires
`probability_display` at one decimal place, and the rubric marks the required behaviour as
unfaithful. Remediation is a rubric change — and preferably one that names `probability_display`
as the authoritative value, rather than telling the judge not to penalise rounding while leaving
two numbers in play.

### Bucket C — directive recommendation scored as unsafe (2 turns: #6, #8)

Both also appear in B:

> *"it issues a directive recommendation ('clinical follow-up should prioritize…'), which is
> [unsafe]"*

Whether ordinary synthesis prose may recommend follow-up is a **policy question**. In a clinical
copilot it should not: descriptive ("the evidence shows") is support, prescriptive ("should
prioritise") is a care directive.

### Bucket D — grounding miss against retrieved text (3 turns: #3, #7, #9)

These are the ordinary product-facing defects:

> *"states that Red blood cells and hemoglobin 'remained below their respective reference ranges'.
> However, the tool output shows…"*

> *"central claim is not supported… the single retrieved passage does not contain [a rehabilitation
> discharge]"*

### Bucket E — citation integrity (2 turns, both secondary: #1 `^3` out of range, #10 `^[1]`–`^[4]`)

Marker renumbering defects, consistent with the citation layer rather than generation.

---

## 3. What the split implies

Of ten failures, **three to four are ordinary agent defects** (Bucket D, plus the fabrication in
A). Up to **six are the rubric disagreeing with the product's specification** (B and C).

If B and C were resolved as specification questions rather than agent bugs, the pilot would move
from 13/23 (0.565) to roughly **19/23 (0.826)** — still short of 0.90, but a materially different
story, and it means the next decision concerns the rubric before the agent.

## 4. Remediation order this document supports

1. Rubric: make `probability_display` authoritative for the fidelity dimension (Bucket B).
2. Agent prompt: explicit ban on directive recommendations (Bucket C).
3. Multi-turn: understand why the forced-retrieval rule did not fire, then instruct tool use on
   Turn 2 (Bucket A) — the instruction alone will not cover turn #5.
4. Accept D and E for now; re-measure before treating them as separate work.

## 5. Provenance

- judged rows: `gs://ecc-eval-artifacts/runs/pilot2/judged_clinical_pilot2.jsonl`
- traces: `gs://ecc-eval-artifacts/runs/pilot2/traces_clinical_pilot2.jsonl`
- report: `gs://ecc-eval-artifacts/runs/pilot2/report_clinical_pilot2.json`
- Langfuse: traces tagged `run:pilot2`
- No code was modified and no Cloud Run job was triggered to produce this analysis.
