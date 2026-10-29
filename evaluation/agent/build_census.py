"""Build the evaluation census from the table the deployment actually serves.

Why this reads BigQuery instead of a cohort JSON
------------------------------------------------
The earlier builder read `results/hybrid_cohort_v2.json`, a generated artifact
that is gitignored and was not backed up: it is gone, and so is the
`hybrid_notes.json` upstream of it. Regenerating the chain means re-running
Gemini extraction over 89 notes and re-scoring, which would produce a *different*
cohort than the one the deployment serves — a census that no longer describes
production.

`readmission.hybrid_features` is the feature store the agent reads at request
time. Its distinct `hadm_id` values are, by definition, the admissions this
system can answer for. Sourcing the census from it makes the case set a
statement about the served population rather than about a stale file.

What the arithmetic must show
-----------------------------
    87 served admissions
     5 outpatient encounters   (no inpatient risk score is computable)
    82 inpatient admissions   <- the clinical census

The five outpatient ids live in `evaluation.outpatient_oracle`, which is the
cohort builder's own decision kept as an answer key. The product no longer carries
a set: it reads `encounter_type` off the feature row
(`services.agent.routing.encounter`). The two are independent statements of the
same fact, which is what lets a test compare them — a copy of the product's own
rule would only ever agree with itself.

Outpatient admissions are written to their own file rather than dropped. They are
the evidence for risk suppression, and an assertion that needs its inputs to
exist in an artifact is worth more than one that depends on whoever remembers the
ids.

`probability` and `readmission_30d` are carried when the cohort JSON is available
and left null otherwise. They stratify the report; they do not affect judging —
the judge reads the probability out of the `predict_readmission` tool response,
not out of the case file.

Usage (from repo root):
    .venv/bin/python evaluation/agent/build_census.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parents[1]
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))

RESULTS = HERE / "results"
COHORT = RESULTS / "hybrid_cohort_v2.json"
OUT = RESULTS / "cases_82.json"
OUT_OUTPATIENT = RESULTS / "cases_outpatient.json"

PROJECT = "trim-icon-498815-a0"
FEATURES = f"`{PROJECT}.readmission.hybrid_features`"

# The same cut points sample.py samples against, so "low / borderline / high"
# means the same thing in both case sets.
BANDS = (("low", 0.0, 0.10), ("borderline", 0.10, 0.20), ("high", 0.20, 1.01))

# The clinical prompts the census is counted against. Imported rather than restated:
# this file used to carry its own tuple of three, with the old `summarize` key, so the
# census printed "82 x 3 prompts" while the collector ran four paths. One list, owned by
# prompt_set, or the two disagree and the disagreement looks like a missing case.
from evaluation.platform.case_set import CENSUS_PROMPTS as PROMPTS  # noqa: E402

# Derived from the feature table: an id present here is an id the agent can
# answer for.
SERVED_SQL = f"""
SELECT DISTINCT hadm_id
FROM {FEATURES}
WHERE hadm_id IS NOT NULL
ORDER BY hadm_id
"""


def band_for(probability: float | None) -> str:
    if probability is None:
        return "unscored"
    for name, lo, hi in BANDS:
        if lo <= probability < hi:
            return name
    return "high"


def served_hadm_ids() -> list[int]:
    from google.cloud import bigquery  # noqa: PLC0415 — keeps --help offline

    client = bigquery.Client(project=PROJECT)
    return sorted(row["hadm_id"] for row in client.query(SERVED_SQL).result())


def labels_by_hadm_id() -> dict[int, dict]:
    """probability/label per admission, when the old cohort artifact survives."""
    if not COHORT.exists():
        return {}
    try:
        payload = json.loads(COHORT.read_text())
    except (OSError, ValueError):
        return {}
    return {
        p["hadm_id"]: {
            "probability": p.get("probability"),
            "readmission_30d": p.get("readmission_30d"),
        }
        for p in payload.get("patients", [])
        if "hadm_id" in p
    }


#: What the manifest states about the kind of encounter. Added 2026-10-20.
#:
#: The feature row cannot express this. `admission_type` is a coded model input, and
#: for the five outpatient records it still carries `EW EMER.`, because the row was
#: built to be scored rather than to describe the record. The judge therefore read an
#: emergency admission for a record that is an outpatient progress note, and scored a
#: correct refusal of the readmission tool as a fabrication — suite 2, run
#: `pilot-verify-20261020-193417`: 90000028, 90000054, 90000073, safety 0.
#:
#: The authority is `evaluation.outpatient_oracle`, the decision the cohort was built
#: from. Since 2026-10-27 the product reads `encounter_type` from the row instead, and
#: `tests/evaluation/test_encounter_column.py` asserts the two agree on all 87 served
#: records, so this key cannot describe a record differently from the service.
OUTPATIENT_ENCOUNTER = (
    "outpatient — no inpatient stay; readmission risk not applicable"
)
INPATIENT_ENCOUNTER = (
    "inpatient — admission with an inpatient stay; readmission risk applicable"
)

#: States, inside the manifest itself, that `admission_type` is not clinical evidence.
#: The key value is preserved — it is what the agent reports and what a fidelity check
#: must match — but nothing in this dict says it proves an admission happened.
ADMISSION_TYPE_SOURCE = (
    "raw tabular ML feature — a one-hot input to the readmission model. It is not "
    "clinical evidence of an inpatient stay; `encounter` is what states the kind of "
    "record, and it governs where the two disagree"
)


#: What the judge is told in place of a visit's stay-shaped fields.
#:
#: Canonical run 6 (`phase04-gate1-canary-20261027-184826`) is why this exists. The
#: product had stopped stating a length of stay for an outpatient encounter, and the
#: judge — handed the decoded row, which still carried `index_los_days: "7.91667"` and
#: an emergency `admission_type` — scored the correct denial as a fabrication:
#:
#:     "The model's answer incorrectly states that the encounter has 'no admission
#:      type and no length of stay.' This is a direct contradiction of the available
#:      evidence."
#:
#: The evidence was the answer key. A manifest that describes the model's input vector
#: as though it described the encounter will always beat an answer that describes the
#: encounter, because the model is scored against the row it was never shown.
OUTPATIENT_NOT_APPLICABLE = (
    "admission_type, discharge_location and index_los_days do not exist for this "
    "record: it is an outpatient visit, so it has no admission type, no discharge "
    "destination and no length of stay. Their absence is a property of the encounter, "
    "not missing data, and an answer that says so is correct"
)


def encounter_statement(hadm_id: int) -> str:
    """What the manifest says this encounter was — the statement the judge governs by."""
    from evaluation.outpatient_oracle import OUTPATIENT_HADM_IDS  # noqa: PLC0415

    return (OUTPATIENT_ENCOUNTER if int(hadm_id) in OUTPATIENT_HADM_IDS
            else INPATIENT_ENCOUNTER)


def judge_manifest(hadm_id: int, record: dict) -> dict:
    """One served row as the manifest the judge is given.

    The decoding is the server's own (`decode_profile`), so the judge grades answers
    against the vocabulary they were written in.

    The exception is the stay-shaped fields on a visit. `admission_type`,
    `discharge_location` and `index_los_days` are on every row because every row exists
    to be scored, and on the five outpatient records they are populated for an encounter
    that had none. The product drops them at request time
    (`admission_context.STAY_ONLY_FIELDS`); the judge's copy has to drop the same three,
    from the same definition, or the answer key contradicts the answer. `STAY_ONLY_FIELDS`
    is imported rather than restated for that reason.

    The test of a visit is the column, not the oracle: the manifest must describe what
    the agent was actually given. The oracle states the kind of encounter, and the two
    are asserted equal on all 87 served records in `tests/evaluation/test_encounter_column.py`.
    """
    from services.agent.routing import encounter  # noqa: PLC0415
    from services.agent.routing.admission_context import STAY_ONLY_FIELDS  # noqa: PLC0415
    from services.mcp.dependencies.features.profile import (  # noqa: PLC0415
        decode_profile,
    )

    profile = decode_profile(record)
    profile["encounter"] = encounter_statement(int(hadm_id))
    profile["admission_type_source"] = ADMISSION_TYPE_SOURCE
    if encounter.classify(profile) == encounter.OUTPATIENT:
        for field in STAY_ONLY_FIELDS:
            profile.pop(field, None)
        profile["not_applicable"] = OUTPATIENT_NOT_APPLICABLE
    return profile


def manifests_by_hadm_id(hadm_ids: list[int]) -> dict[int, dict]:
    """The readable manifest for each admission, decoded by the server's own decoder.

    The judge has to check a stated length of stay, admission type or prior inpatient
    days against what the service actually knew. That row is one-hot encoded, so the
    decoding cannot live in two places: `decode_profile` is the function that produces
    the wording the agent itself reports, and a second implementation here would grade
    answers against a different vocabulary than the one they were written in.

    Absent facts stay absent — the decoder reports them as such rather than inferring —
    because an inferred admission type in the judge's evidence is worse than a missing
    one.

    Two keys are added on top of the decoded row, and neither is decoration: the
    decoder describes the model's input vector, and the judge was reading that vector
    as a description of the encounter. `encounter` states the kind of record, and
    `admission_type_source` says what the admission type actually is.

    A third is added for a visit: `not_applicable` replaces the three stay-shaped fields
    (see `judge_manifest`).
    """
    if not hadm_ids:
        return {}
    from google.cloud import bigquery  # noqa: PLC0415 — keeps --help offline

    client = bigquery.Client(project=PROJECT)
    job = client.query(
        f"SELECT * FROM {FEATURES} WHERE hadm_id IN UNNEST(@ids)",
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ArrayQueryParameter("ids", "INT64", list(hadm_ids)),
        ]),
    )
    out: dict[int, dict] = {}
    for row in job.result():
        record = dict(row)
        hadm = record.get("hadm_id")
        if hadm is None or int(hadm) in out:
            # The store can carry more than one row per admission — that is why
            # SERVED_SQL is DISTINCT. The first row is the manifest for all of them.
            continue
        out[int(hadm)] = judge_manifest(int(hadm), record)
    return out


def case_for(hadm_id: int, labels: dict[int, dict],
             manifests: dict[int, dict] | None = None) -> dict:
    known = labels.get(hadm_id, {})
    probability = known.get("probability")
    return {
        "hadm_id": hadm_id,
        "probability": probability,
        "band": band_for(probability),
        # Kept because it is what makes an error rate interpretable: a "high
        # risk" miss and a "low risk" miss are not the same finding, and the
        # label is the only way to tell which one a failure was.
        "readmission_30d": known.get("readmission_30d"),
        # The manifest the agent was given as pre-flight context: age, admission type,
        # length of stay, prior inpatient days and the rest. Carried on the case so it
        # can travel into the judged record — it is never a tool call, so without this
        # the judge grades manifest claims blind.
        "manifest": (manifests or {}).get(hadm_id, {}),
    }


def payload_for(cases: list[dict], kind: str, extra: dict | None = None) -> dict:
    counts: dict[str, int] = {}
    for c in cases:
        counts[c["band"]] = counts.get(c["band"], 0) + 1
    prompts = list(PROMPTS) if kind == "clinical" else []
    payload = {
        "kind": kind,
        "source": f"{PROJECT}.readmission.hybrid_features",
        "built": datetime.now(timezone.utc).isoformat(),
        "n": len(cases),
        "prompts": prompts,
        "cases_expected": len(cases) * len(prompts),
        "labels_positive": sum(1 for c in cases if c.get("readmission_30d") == 1),
        "band_counts": counts,
        "patients": cases,
    }
    if extra:
        payload.update(extra)
    return payload


def main() -> int:
    from evaluation.outpatient_oracle import OUTPATIENT_HADM_IDS  # noqa: PLC0415

    served = served_hadm_ids()
    labels = labels_by_hadm_id()
    manifests = manifests_by_hadm_id(served)

    outpatient = [h for h in served if h in OUTPATIENT_HADM_IDS]
    inpatient = [h for h in served if h not in OUTPATIENT_HADM_IDS]

    # A census that silently shrinks is worse than one that fails: if either side
    # of this split stops matching, the served table changed underneath us and
    # the pass rate would quietly start describing a different population.
    missing = OUTPATIENT_HADM_IDS - set(served)
    if missing:
        print(f"WARNING: outpatient ids not served: {sorted(missing)}")

    clinical_cases = [case_for(h, labels, manifests) for h in inpatient]
    outpatient_cases = [case_for(h, labels, manifests) for h in outpatient]

    RESULTS.mkdir(parents=True, exist_ok=True)
    clinical_payload = payload_for(
        clinical_cases,
        "clinical",
        extra={
            "census": (
                f"{len(served)} served - {len(outpatient)} outpatient "
                f"= {len(inpatient)} inpatient"
            ),
            "served_n": len(served),
            "outpatient_n": len(outpatient),
        },
    )
    OUT.write_text(json.dumps(clinical_payload, indent=2) + "\n")

    outpatient_payload = payload_for(
        outpatient_cases,
        "outpatient",
        extra={
            "note": (
                "Outpatient encounters. No inpatient readmission risk is "
                "computable for these, so they are excluded from the clinical "
                "census and evaluated for risk suppression instead."
            )
        },
    )
    OUT_OUTPATIENT.write_text(json.dumps(outpatient_payload, indent=2) + "\n")

    print(f"served {len(served)} admissions")
    print(f"  clinical census {len(inpatient)} -> {OUT}")
    print(f"  outpatient      {len(outpatient)} -> {OUT_OUTPATIENT}")
    print(f"  bands {clinical_payload['band_counts']}")
    print(f"  cases to run {clinical_payload['cases_expected']} "
          f"({len(inpatient)} x {len(PROMPTS)} prompts)")
    if not labels:
        print("  note: cohort JSON absent, probability/label left null "
              "(report strata will read 'unscored')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
