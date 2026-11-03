# Stage Gate 1 Report — Live PubMed Retrieval & Structured Comparator Extraction

**Date:** 2026-11-02
**Directive:** `docs/debugging/ehr_comparative_literature_matrix_re_architecture_directive (1).md`
**Status:** Stage Gate 1 complete, backend only. **Nothing wired into `/ehr`.** `presentation/domain.py` still supplies the comparative table; `turn.py` and `evaluation/` are untouched.

---

## 1. What was built

| file | change |
|---|---|
| `services/mcp/tools/literature.py` | `_fetch` split into `_search(query, *, retmax, reldate_days)`; `/pubmed`'s behaviour unchanged (`_fetch` now delegates with `RECENCY_DAYS`, `DEFAULT_MAX_RESULTS = 3`). New: `readmission_query()`, `quantitative_readmission_articles()` (both pure), `search_readmission_benchmarks()` (async, live). |
| `services/agent/engine/deterministic.py` | New `Live literature comparators` section: `ClinicalTarget`, `note_sections()`, `build_target()`, `derive_target()`, `names_target()`, `split_by_clinical_anchor()`, `verify_comparator()`, `verify_comparators()`, `extract_comparators()`. |
| `services/agent/engine/prompts.py` | `LITERATURE_TARGET_PROMPT`, `LITERATURE_COMPARATOR_PROMPT`, `literature_target_input()`, `literature_comparator_input()`. |
| `tests/agent/test_live_literature_extraction.py` | 17 hermetic tests + the 8-admission live report. |

**No new module was created under `services/agent/`.** `tests/agent/test_layer_separation.py` pins an exact frozenset of agent modules and is immutable, so the extraction logic lives in `deterministic.py` — the location the directive names as its alternative.

### Retrieval parameters

| constant | value | reason |
|---|---|---|
| `READMISSION_OUTCOME_CLAUSE` | four `[Title/Abstract]` readmission phrases | field terms, so a MeSH-only hit cannot qualify a record whose abstract never says readmission |
| `READMISSION_RECENCY_YEARS` | 15 | a landmark readmission cohort is not recent |
| `READMISSION_CANDIDATE_POOL` | 12 | measured yield when the table was curated by hand: 4 of 30 records state an absolute 30-day rate |
| `READMISSION_DEFAULT_RESULTS` | 6 | extraction reads at most two rows; the rest is context |
| `READMISSION_MIN_QUANTITATIVE` | 2 | one record is not a comparison |

### The three gates, in the order they run

1. **Abstract pre-filter (deterministic).** `readmission`-term **and** `\d+(?:\.\d+)?%` in the abstract. A record with no number cannot become a row.
2. **Clinical alignment (deterministic).** The abstract must contain one of the patient's own anchor phrases, derived from the chart in the same request. A single-word anchor is accepted only if the chart itself used that word, so `hip`/`knee` cannot be anchors while `COPD` can.
3. **Row verification (deterministic).** PMID ∈ retrieved set; abstract names the condition; the model did not decline; `horizon_days == 30` and the quoted sentence says so; `rate_sentence` appears in the abstract character for character; `reported_rate` is the figure that sentence states. Publication metadata (journal, date, title, authors, url) is taken from the record, never from the model's row.

---

## 2. Gate results

```
pytest tests -m "not live"            1671 passed, 1 skipped
pytest -m live (8 admissions)         1 passed in 59.45s
```

Full live output: `~/Library/Application Support/Code/copilot-terminal-output/copilot-terminal-output-e3820766-97e0-44c5-82e4-da71a56ea1fb.txt`

Change set (`git status --porcelain`): the three source files above plus `tests/agent/test_live_literature_extraction.py`. `evaluation/`, `presentation/`, `turn.py` untouched.

---

## 3. The eight admissions

| hadm | derived condition / procedure | candidates | anchor-pass | verified rows |
|---|---|---|---|---|
| 90000090 | femoral neck fracture / hemiarthroplasty | 6 | 6 | **0** |
| 90000069 | cervical spondylosis / cervical surgery | 6 | 3 | **1** |
| 90000081 | pleural effusion / thoracentesis | 4 | 4 | **1** |
| 90000001 | tibial plateau fracture / ORIF | 1 | 0 | 0 |
| 90000019 | total hip arthroplasty / total hip replacement | 6 | 5 | **2** |
| 90000037 | chest pain / cardiac catheterisation | 2 | 2 | 0 |
| 90000058 | *(none derived)* | — | — | 0 |
| 90000082 | uterine leiomyoma / total abdominal hysterectomy, BSO | 6 | 1 | 0 |

Corpus: 3 diagnostic (`90000090`, `90000069`, `90000081`) + 5 held-out, taken at every Nth sorted entry of the served inpatient cohort minus outpatients.

### 90000090 — the honest empty

Discharge diagnosis: *"Fracture of the right femoral neck, also history of Alzheimer's dementia, and hypothyroidism."*

Query sent: `(Femoral neck fracture Hemiarthroplasty) AND ("30-day readmission"[Title/Abstract] OR "readmission rate"[Title/Abstract] OR "unplanned readmission"[Title/Abstract] OR "hospital readmission"[Title/Abstract])`

Returned 6: `34838925`, `32222266`, `39655937`, `31023515`, `35061652`, `31343596`. All six accepted by the anchor gate. All six rejected at extraction:

| pmid | reason |
|---|---|
| 34838925 | reports readmission within 90 days, not 30 |
| 32222266 | states no numerical 30-day rate |
| 39655937 | 30-day sentence is statistically ambiguous, no cohort-wide rate |
| 31023515 | 90-day rates |
| 35061652 | 90-day rates |
| 31343596 | 90-day rates |

No rows. This is the behaviour the directive asks for: the literature genuinely offers no 30-day rate for this condition in the retrieved window.

### 90000069 — one row, one false rejection

Discharge diagnosis: *"1. Neck pain with right upper extremity radiculopathy. 2. Cervical spondylosis with herniated nucleus pulposus C4-C5, C5-C6, and C6-C7 with stenosis."*
Derived: `Cervical spondylosis` / `cervical surgery`; anchors `cervical spondylosis`, `herniated nucleus pulposus`, `cervical stenosis`, `cervical surgery`.

Returned 6: `27246481`, `38504222`, `29146432`, `28142261`, `35314510`, `27120061`.

- Accepted: `38504222`, `28142261`, `27120061`
- Rejected by anchor: `27246481`, `29146432`, `35314510`
- `28142261` rejected at verification: *"rate_sentence states no 30-day figure"*
- `38504222` declined in favour of the two-comparator limit

Row:

```
pmid 27120061  ·  Spine (2017 Jan 15)
cohort: 57,387 elderly patients (65-84) undergoing primary one-to-two and ≥three-level
        anterior cervical fusion for cervical spondylosis; Medicare 2005-2012
tier:   Retrospective database review (Level 3)
rate:   1.0% to 1.4% at 30 days
quote:  "Readmission rates of 1.0% to 1.4%, 2.7% to 3.6%, and 13.2% to 14.1% were
         observed within 30 days, 90 days, and one year."
```

**Defect.** `29146432` — *"Predictors of Hospital Length of Stay and 30-Day Readmission in Cervical Spondylotic Myelopathy Patients: An Analysis of 3057 Patients Using the ACS-NSQIP Database"* — is the best record returned and was rejected because the chart says *spondylosis* and the paper says *spondylotic myelopathy*.

### 90000081 — one row, three correct refusals

Diagnosis: *"Pleural effusion, atrial fibrillation, anemia, ascites, esophageal reflux, and history of deep venous thrombosis."*
Derived: `Pleural effusion` / `Thoracentesis`; anchors `pleural effusion`, `thoracentesis`.

Returned 4; all four passed the anchor gate; three refused by the model:

- `37197490` — robotic transthoracic diaphragmatic plication cohort
- `31340633` — infected necrotizing pancreatitis cohort
- `41172271` — robotic pulmonary resection cohort

Row:

```
pmid 31441096  ·  Journal of gastroenterology and hepatology (2020 Apr)
cohort: pleural effusion / hydrothorax with cirrhosis, n=2371, Nationwide Readmission Database
tier:   Retrospective analysis
rate:   33% at 30 days
quote:  "Of the 2371 patients included from the Nationwide Readmission Database, 870 (33%)
         were readmitted within 30 days."
```

The anchor gate passes these because they contain the phrase "pleural effusion" somewhere in the abstract; the extraction step is what recognises that the *cohort* is a different population. Both gates are needed.

### 90000001 — retrieval too narrow

Diagnosis: *"Right tibial plateau fracture."* Procedure: ORIF, right Schatzker III.
Query: `(Tibial plateau fracture Open reduction and internal fixation) AND (…)`.
Returned 1 (`35021031`, cirrhosis/MELD and operatively treated tibia fractures); rejected by anchor. No rows.

### 90000019 — two rows from a chart with no diagnosis section

`discharge_diagnosis` was empty; the target was derived from the brief hospital course (*"Total hip arthroplasty"* / *"Total hip replacement"*).

Returned 6: `33422393`, `36326830`, `34838925`, `34293911`, `27783839`, `32742739`. Accepted 5, verified 2, declined 4 (`36326830` 90-day; `34838925` 90-day and a femoral-neck trauma population; `27783839` mentions 30-day readmission without a rate; `34293911` failed the anchor).

```
pmid 33422393  ·  The Journal of arthroplasty (2021 Jul)
cohort: primary THA, n=25,081, PearlDiver
rate:   3.3% at 30 days
quote:  "The 30-day readmission rate for conversion THA was significantly higher compared
         to the primary group (7.3% vs 3.3%) (P < .001)."

pmid 32742739  ·  Geriatric orthopaedic surgery & rehabilitation (2020)
cohort: THA in patients over 80, n=7730, NSQIP
rate:   4.9% at 30 days
quote:  "… 376 (4.9%) patients experienced an unplanned readmission to the hospital within
         30 days."
```

### 90000037 — correct refusal

`discharge_diagnosis` empty; derived `Chest pain` / `Cardiac catheterization`.
Returned 2 (`23754777`, `30045799`), both STEMI cohorts — declined: STEMI is not the patient's condition, and one rate is at one year. No rows.

### 90000058 — no condition derived

Diagnosis list: *"Atypical chest pain. Nausea. Vomiting. Diabetes. Hypokalemia. Diarrhea. Panic and depression. Hypertension."*; hospital course opens *"Serotonin syndrome."*
`derive_target` returned an empty condition, so no query ran. The chart does name conditions, so this is a miss rather than an honest empty.

### 90000082 — anchors too literal

`discharge_diagnosis` empty; derived from the operations block: `Uterine leiomyoma` / `Total abdominal hysterectomy, bilateral salpingo-oophorectomy`.
Returned 6: `39164326`, `33667943`, `37145314`, `26855103`, `36706789`, `40154572`. Five rejected by the anchor gate because the records say *fibroids*, *myomectomy*, *hysterectomy*, *uterine artery embolization* rather than the chart's exact phrases. No rows.

---

## 4. Defects, with proposals

**D1 — anchors are chart-literal, so they reject the literature's name for the same condition.**
Evidence: `90000069`/`29146432` (spondylosis vs spondylotic myelopathy), `90000082` (five records).
Proposal: have the target prompt emit anchors as **synonym sets, one per clinical concept**, and accept a concept when any of its members matches. The single-word rule stays. Still deterministic, still no table.

**D2 — recall is the weak link, and the broaden retry keys on the wrong quantity.**
Evidence: `90000001` (1 candidate), `90000037` (2), and `90000090` (6 candidates, none at 30 days — the retry did not fire because six abstracts had percentages).
Proposal: trigger the broaden on **verified rows < 2**, not on quantitative abstracts < 2, and add a condition-only query in the widened pass.

**D3 — a chart with no single primary diagnosis can yield nothing.**
Evidence: `90000058`.
Proposal: prompt change — a numbered discharge list is still a condition list; return the leading acute item (here: atypical chest pain, with serotonin syndrome as the course's finding). Needs a rule so it cannot force a wrong condition.

**D4 — a range rate passes verification.**
Evidence: `27120061` (`1.0% to 1.4%`), whose quoted sentence also carries 90-day and 1-year figures.
Proposal: keep the quote, and carry `rate_kind` (`point` / `range`) so Stage 2 decides how the cell renders. Not a verification failure; a display decision that should be explicit.

**Not a defect.** `90000081`'s three refusals and `90000037`'s two show the two gates doing different jobs: the anchor gate finds records that mention the condition, and the extraction step recognises that the cohort is somebody else.

---

## 5. Stop

Stage Gate 1 ends here per the directive. `presentation/domain.py` still supplies the table; nothing is wired into `run_command`.

Recommended next step, if approved: **D1 and D2**, both inside the Stage Gate 1 backend and both deterministic, then re-run these eight and compare row counts. On this run D1 accounts for two of the five empty results and D2 for two more; D3 for one.
