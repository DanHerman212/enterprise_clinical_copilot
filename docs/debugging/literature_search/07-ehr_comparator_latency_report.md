# `/ehr` comparator step — why Pass 1 retries, and what the second extraction cost

2026-11-02. Investigated against the Langfuse trace for `command.summarize` on `hadm:90000069`
(22.08s) and measured on the live stack. One policy change; no change to retrieval.

**Gates:** `pytest tests -m "not live"` → **1699 passed**, 1 skipped.
`python -m evaluation.platform.dry_run` → **STEP 7 PASSED**.

---

## 1. Why Pass 1 does not short-circuit on `/ehr`

**It is not passing different arguments, and it is not a regression from Gate 1E.**

Three measurements on `90000069`, on the production path:

| check | result |
| --- | --- |
| `derive_target`, three consecutive runs | identical: condition `Cervical spondylosis`, procedure `cervical decompression`, the same six anchors |
| Pass 1 query text | `(Cervical spondylosis) AND ("30-day readmission"[Title/Abstract] OR …)` — byte-identical to `readmission_query(target.condition)`, which is what the Gate 1E harness called |
| Pass 1 pool | 25 fetched → **18** with a readmission term and a percentage — the same figure Gate 1D recorded (`18 / 4 / 2`) |

The difference between the harness and the tool is cosmetic: `search_literature` normalises
whitespace in `query` before delegating, and `max_results` is ignored on the readmission
scope (the pool is returned whole). Both were checked by inspection and by the identical pool.

What actually happens is that **Pass 1 aligns four records, not eighteen**:

```
PASS 1: candidates=18 aligned=4 excluded=0
  aligned: 38504222, 40073748, 28142261, 27120061
```

`split_by_clinical_anchor` keeps a record only if its title or abstract names one of the
target's anchors. Four of eighteen do. Gate 1D verified two of those four; today two
different runs verified one and zero. That is the whole of it: the retry fires because the
verdict over a four-record pool is one short of `MIN_COMPARATORS`, and the verdict moves
between runs on identical input and `temperature=0`.

So the premise in the trace reading — "2 searches because Pass 1 regressed" — is wrong, and
Gate 1D's pass-1 result was at the boundary rather than ahead of it. The correction that
matters is to the *policy*, not the query.

## 2. The cost that was worth removing

Measured, `90000069`, before the change:

```
search PASS 1   1.2s   aligned=4
extract #1      ~3.5s  4 candidates
search WIDE     0.5s   aligned=10
extract #2      ~5.5s  10 candidates
```

Two sequential model calls, the first of them over a pool that could not fill the model's
budget of ten. The searches are ~0.5s each; the extraction is 3–8s. The retry was decided
from the **verdict**, and the verdict can only be had by paying for a call.

### The change

`extract_comparators` now decides the retry from the **candidates**, and reads the pool once:

* Pass 1 fills the budget (`MAX_CANDIDATES_FOR_MODEL` records that name this condition):
  extract once. A second extraction runs only if a full budget still yields fewer than
  `MIN_COMPARATORS` rows — the case the retry was built for (90000090: six records, every
  one a 90-day rate).
* Pass 1 cannot fill the budget: the wide search runs immediately, the two aligned pools are
  merged in specificity order, and the model reads the union **once**. A thin pass 1 is
  exactly when the wide pass adds records, so asking the model what it makes of the thin
  pool first spends five seconds to learn what the candidate count already said.

`CandidatePass` carries the alignment work out of the model's way, `ComparatorPass` gained a
`pooled` flag so a report showing two passes and one call does not read as a lost pass, and
`verify_comparators` is untouched — the model still has to produce a verbatim sentence for
every figure.

### Measured after

| hadm | before | after | outcome |
| --- | --- | --- | --- |
| 90000069 (thin Pass 1, aligned 4) | 2 searches, 2 extractions, **16.0s** | 2 searches, **1 extraction**, **13.9s** | 2 rows |
| 90000019 (full Pass 1, aligned 10) | 1 search, 1 extraction, 9.6s | unchanged: 1 search, 1 extraction, **9.6s** | 2 rows, no wide query |
| 90000037 (full Pass 1, verifies nothing) | 2 searches, 2 extractions | 2 searches, 2 extractions, **11.1s** | 0 rows, honest empty |

`90000069`'s saving is smaller than the trace suggested because its Pass 1 pool was small
enough to make the call it removes cheap (3.5s, not 5s). The saving is largest exactly where
the old policy was most wasteful: an admission whose Pass 1 aligns nothing, which previously
paid two extractions to return an empty table.

## 3. What was deliberately not changed

**The retry keeps its own extraction when Pass 1 filled the budget and still starved.**
90000037 is that case: `coronary artery disease` returns ten condition-naming records,
none of them at 30 days, and the wide pass then finds four more. Removing that second call
restores the 90000090 defect — six candidates, all reporting a 90-day rate, no retry — which
is what the retry exists to catch. Filling the budget is a necessary condition for skipping
the wide pass, not a sufficient one.

**The two searches were not made concurrent.** They are ~0.5s each, and two simultaneous
E-utilities requests is how the 429 seen during Gate 2 testing happens.

## 4. Residual costs on a `/ehr` turn

Not addressed here, and each is a separate decision:

1. `derive_target` is a 2.5–4s model call before any search can be issued. It is the
   condition the two queries are built from, so it cannot be moved off the critical path
   without a deterministic target — which Gate 1 measured as materially worse.
2. The narrated prose turn is ~4.4s and starts only after the rows are settled.
3. PubMed's returned set moves between calls, so per-admission row counts are not a trend
   and are not asserted anywhere.

## 5. Tests

Two policy tests stated the old rule and were replaced, not weakened:

| file | change |
| --- | --- |
| `test_a_thin_first_pass_pools_both_queries_into_one_extraction` | both searches run, **one** model call, two passes reported with `pooled=True` |
| `test_a_first_pass_that_fills_the_budget_is_extracted_alone` | ten aligned records: no wide query, one extraction |
| `test_a_full_first_pass_that_verifies_nothing_still_spends_the_retry` | the D2 case, preserved: two calls, the wide pass's row survives |

The 49 other hermetic tests in that file, and the live eight-admission gate, are unchanged.
