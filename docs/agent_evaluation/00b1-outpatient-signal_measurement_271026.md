# Outpatient signal — measurement before implementation

**Question put by the directive:** replace `routing.OUTPATIENT_HADM_IDS` with a
signal-driven `is_outpatient(profile_or_state)`, verified at 100% accuracy over the 5
outpatient and 82 inpatient records.

**Finding: no such signal exists in the current data.** Four candidate sources were
measured against the 87 served records. None separates the two groups.

---

## 1. The feature row — complete overlap

`get_admission_profile` decodes ten fields from `readmission.hybrid_features`. The five
outpatient rows are indistinguishable from the 82 inpatient rows on every one of them:

| Field | Outpatient values | Separable |
|---|---|---|
| `age` | 67, 68, 69, 73, 78 | no |
| `gender` | Female, Male | no |
| `admission_type` | Emergency (EW EMER.), Emergency direct (DIRECT EMER.), Urgent | no — same three levels as inpatient |
| `discharge_location` | Home, Home with home health, LTAC, SNF | no |
| `insurance` | Medicare | no |
| `race` | White | no |
| `prior_admission_count` | 0, 2, 28 | no |
| `prior_inpatient_days` | 0, 13.4167, 272.917 | no |
| `recent_ed_visits` | 0 | no |
| `index_los_days` | 1.54167, 5.875, 7.83333, 7.91667, 24.7917 | no |

Decisive against the `eligible`-flag idea in the directive: the flag is **derived from**
`OUTPATIENT_HADM_IDS` today, so reading it as the signal would be circular. And the rows
carry a length of stay and a discharge location because they were encoded as admissions —
`90000051` reads 7.91667 days, the exact figure the canary found leaking into prose.

## 2. The cohort row — no encounter column

`readmission.demo_cohort` holds 56 columns: demographics, one-hot admission and discharge
levels, counts, thirteen clinical features, `readmission_30d`, `split_name`, `subject_id`.
Nothing names the encounter type. The rendered `demo_cohort.json` summary is worse than
uninformative — it describes all five outpatient records as stays ("3-day stay · discharged
to rehab"), so consuming it would harden the error rather than fix it.

## 3. Note structure — correlates, does not classify

Parsing every note with `services.mcp.retrieval.sections.parse_note` and asking whether any
`SUMMARY_SECTIONS` body is present (`history_of_present_illness`, `brief_hospital_course`,
`discharge_diagnosis`, `discharge_instructions`):

| | Predicted outpatient | Predicted inpatient |
|---|---|---|
| **5 outpatient** | 2 (`90000051`, `90000073`) | 3 |
| **82 inpatient** | **3** (`90000026`, `90000033`, `90000087`) | 79 |

**6 of 87 misclassified.** Two of the five outpatient notes carry a recognisable
`brief_hospital_course` or `history_of_present_illness`; three inpatient records carry none
of the four sections. This rule is worse than the set it would replace: it would suppress the
risk card on three real admissions — `90000087` among them, an inpatient with a discharge
condition, disposition and medication list.

## 4. Note provenance — overlapping ranges

Every note is `MT-*-DS`, one transcription set. The outpatient ids cluster in 1793–1849 and
the inpatient ids span 1106–2792, so a prefix or a range test would be a disguised lookup
with a fabricated justification.

---

## What this means

The outpatient/inpatient distinction for these records is **curation metadata**, not a
derived property. It was introduced by hand when the demo cohort was built, and
`OUTPATIENT_HADM_IDS` is the only place it is recorded. That is why the directive is right
that a runtime `hadm_id` branch is wrong, and also why no predicate over the existing rows can
reproduce it: the fact is not in them.

With 87 records and 5 positives, any rule fitted to separate them would be fitted to those
five rows. That is the reward hacking the rule against `hadm_id` sets exists to prevent,
reached by a longer route.

---

## Recommended path

**Record the fact where it belongs, then read it.** The set stops being a runtime branch and
becomes seed data at the source of truth:

1. Add `encounter_type` to `readmission.hybrid_features` and `readmission.demo_cohort`,
   populated for the 5 (`outpatient`) and the 82 (`inpatient`) at load time, from the
   curation decision that already exists. New records must state it or be refused.
2. Expose it through `decode_profile`, so it arrives in the payload the pre-flight already
   fetches — no extra call, no extra latency.
3. `routing/encounter.py::is_outpatient(profile) -> bool | None` reads that one field and
   returns `None` for absent, which is treated as *unknown* rather than as *inpatient*.
   Unknown keeps today's behaviour and is logged, so a record added without the column is
   visible rather than silently mis-typed.
4. Wire the one predicate into the three places the directive names. The tests then hold:
   a synthetic profile with `hadm_id=99999999` and `encounter_type="outpatient"` classifies
   as outpatient with no id involved, and the 87 real rows classify from the column at 87/87
   by construction rather than by inference.
5. `OUTPATIENT_HADM_IDS` is deleted from `services/agent/routing/routing.py`. The evaluation
   harness keeps its own copy as the oracle, which is what a test fixture is for.

Steps 1 and 2 are a schema change and a backfill against two production tables, so they need
your authorisation — I have not touched BigQuery. Steps 3–5 are the code the directive asked
for and land immediately once the column exists.

**Not implemented, nothing deployed, Gate 2 not run.** The canary remains at 2/3 on
`agent-00202-bnz`: `dev_t7_01` still fails, for the reason this document measures.
