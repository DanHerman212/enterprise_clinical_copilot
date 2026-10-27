"""The curation decision, kept as an oracle rather than as a runtime branch.

Why this file is in `evaluation/` and not in `services/`
--------------------------------------------------------
Until 2026-10-27 the product knew which records were outpatient visits by carrying
this frozenset and testing a `hadm_id` against it. That is a branch on an identifier:
the record was classified by recognising its number rather than by anything the record
said. The measurement in
`docs/agent_evaluation/00b1-outpatient-signal_measurement_271026.md` established that
no field on the served rows separated the two groups — the distinction was curation
metadata introduced by hand when the demo cohort was built — so the fact was recorded
where it belongs, as `encounter_type` on `readmission.hybrid_features` and
`readmission.demo_cohort`, and the runtime now reads that column
(`services.agent.routing.encounter`).

The set still has a job: it is the answer key. A test that asks "does the column
classify every served record the same way the cohort builder meant it to?" needs the
builder's intention from somewhere, and this is it. Keeping the copy here is what makes
the migration checkable — the product reading a column and the harness holding the
original decision are two independent statements that can be compared. If both lived in
`services/`, the test would be asserting that a value equals itself.

Nothing in `services/` imports this module.
"""

from __future__ import annotations

#: The five records the demo cohort was built as outpatient visits: speech-therapy and
#: rehab progress notes with no admission to be readmitted after. Frozen deliberately —
#: it records a decision taken in the past and must not track the product.
OUTPATIENT_HADM_IDS: frozenset[int] = frozenset({
    90000028, 90000040, 90000051, 90000054, 90000073,
})

#: The column value the rows carry for each side of the split.
OUTPATIENT = "outpatient"
INPATIENT = "inpatient"
