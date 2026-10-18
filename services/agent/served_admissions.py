"""The admissions this deployment is authorised to answer for.

Provenance: the 87 rows of `readmission.hybrid_features` that resolve to a served
record — 82 inpatient admissions plus 5 outpatient progress notes. The two
unscoreable orphans (`90000039`, `90000107`) are deliberately absent: they are in the
authorization boundary with neither a note nor a feature row, so nothing can be served
for them at all.

This is a literal rather than a live BigQuery read for one reason: it is used to refuse
a request *before* any work is done, and a refusal that depends on a network call is a
refusal that fails open when the network does. It is generated from the same census the
evaluation builds, and `tests/agent/test_eval_contract.py` asserts this set is exactly
the set `evaluation/agent/results/cases_82.json` + `cases_outpatient.json` describe, so
drift between the served cohort and this list is caught locally rather than discovered
by a patient who is told they do not exist.

Regenerate after the cohort changes:

    python -c "import json,pathlib; r=pathlib.Path('evaluation/agent/results'); \
ids=set(); [ids.update(p['hadm_id'] for p in json.loads((r/f).read_text())['patients']) \
for f in ('cases_82.json','cases_outpatient.json')]; print(sorted(ids))"
"""

from __future__ import annotations

SERVED_HADM_IDS: frozenset[int] = frozenset({
    90000001, 90000002, 90000003, 90000004, 90000005, 90000009, 90000010, 90000011,
    90000012, 90000013, 90000014, 90000015, 90000016, 90000017, 90000018, 90000019,
    90000020, 90000022, 90000023, 90000024, 90000025, 90000026, 90000027, 90000028,
    90000029, 90000030, 90000031, 90000033, 90000034, 90000035, 90000036, 90000037,
    90000040, 90000041, 90000043, 90000044, 90000045, 90000046, 90000047, 90000048,
    90000049, 90000050, 90000051, 90000052, 90000053, 90000054, 90000055, 90000056,
    90000057, 90000058, 90000059, 90000060, 90000063, 90000064, 90000065, 90000067,
    90000068, 90000069, 90000070, 90000071, 90000073, 90000074, 90000076, 90000077,
    90000078, 90000079, 90000081, 90000082, 90000084, 90000086, 90000087, 90000088,
    90000090, 90000091, 90000092, 90000094, 90000095, 90000097, 90000099, 90000100,
    90000101, 90000102, 90000103, 90000104, 90000105, 90000106, 90000108,
})


def is_served(hadm_id: object) -> bool:
    """Whether this admission may be answered for.

    Non-integers are not served. A caller that sends `hadm_id` as a string gets the
    same refusal as one who sends an unknown id, because both are asking about a
    record this deployment cannot open — and the contract already normalises the type
    upstream, so reaching here with a non-int means something is wrong rather than
    merely unfamiliar.
    """
    return isinstance(hadm_id, int) and not isinstance(hadm_id, bool) \
        and hadm_id in SERVED_HADM_IDS
