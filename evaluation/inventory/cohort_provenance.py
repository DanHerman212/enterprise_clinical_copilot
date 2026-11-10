"""Phase 03, Section 4 — the ground-truth cohort provenance inventory.

Read-only. No model is called: every field comes from the same four patient-scoped tools
the sweep exercises, plus the feature row each admission was scored on.

The phenotype map that used to resolve an admission's condition is gone: Stage Gate 2
deleted it from `presentation/domain.py` together with `bucket_for`, because a
hand-maintained `hadm_id -> condition` table decides a clinical question. The archetype
choice below therefore reads `oncology_flag` from the row the model scored. That is the
same fact the Dev suite asserts on, and it is read, never inferred.

    .venv/bin/python evaluation/inventory/cohort_provenance.py

Writes `evaluation/cohort_provenance_inventory.json` and prints a summary. The JSON is
the artifact; this module decides nothing about which patient is interesting — it records
what each record holds and leaves the archetype choice to the reader, except for the
mechanical criteria the phase names.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from google.cloud import bigquery  # noqa: E402

from services.agent.ingress.contracts import SERVED_HADM_IDS  # noqa: E402
from evaluation.outpatient_oracle import OUTPATIENT_HADM_IDS  # noqa: E402
from services.mcp.config import ENTITY_ID_COLUMN, PROJECT, TABLE  # noqa: E402
from services.mcp.tools import (  # noqa: E402
    discharge_medications,
    get_lab_trends,
    predict_readmission,
    rag_search_sections,
)
from services.mcp.tools.admission_profile import get_admission_profile  # noqa: E402
from services.mcp.retrieval.sections import (  # noqa: E402
    DISCHARGE_INSTRUCTIONS,
    DISCHARGE_MEDICATIONS,
)

#: Same transport settings as the Section 3 sweep, for the same reason: past ~8
#: simultaneous token refreshes `oauth2.googleapis.com` starts refusing, so the sweep
#: runs at 6 with a transport-only retry.
CONCURRENCY = 6
ATTEMPTS = 3
BACKOFF_SECONDS = 2.0

OUT = REPO / "evaluation" / "cohort_provenance_inventory.json"

#: The archetypes the master strategy names, plus the two outpatient records.
DEV_INPATIENT = (90000015, 90000108, 90000106, 90000105, 90000023)
DEV_OUTPATIENT = (90000051, 90000073)
#: The outpatient pool a holdout boundary record may come from.
HOLDOUT_OUTPATIENT_POOL = (90000028, 90000040, 90000054)
#: Notes with no discharge-summary headings (TF-2), which is itself an archetype.
HEADINGLESS = frozenset({90000026, 90000033, 90000087})

#: The headings that exist to carry a discharge medication list. `discharge_medications`
#: serves either one of them when the note has one, and falls back to the narrative
#: (`brief_hospital_course`, the admission list, the transfer list) when it has neither.
#: Recording which of the two happened is what separates "this record documents no
#: discharge medications" from "this record documents them inside the hospital course".
DEDICATED_MED_SECTIONS = frozenset({DISCHARGE_MEDICATIONS, DISCHARGE_INSTRUCTIONS})

ANTICOAGULANT = re.compile(
    r"warfarin|apixaban|rivaroxaban|dabigatran|edoxaban|enoxaparin|heparin", re.I)
TAPER = re.compile(r"taper|tapering|reduce|decrease|wean", re.I)


async def _read(call, hadm_id: int):
    last: Exception | None = None
    for attempt in range(ATTEMPTS):
        try:
            return await call(hadm_id)
        except Exception as exc:  # noqa: BLE001
            last = exc
            await asyncio.sleep(BACKOFF_SECONDS * (attempt + 1))
    raise last  # type: ignore[misc]


def _feature_rows(hadm_ids: tuple[int, ...]) -> dict[int, dict[str, Any]]:
    """The exact row each admission was scored on, in one read of the serving table.

    The row is the only place four facts live: the `encounter_type` column the product
    reads, the `oncology_flag` the archetype choice below needs now that the phenotype
    map is gone, and the raw value of every driver the profile cannot decode
    (`rdw_max`, `sodium_min`, `monocytes_min`). All four are structured facts, and the
    point of this artifact is that each is copied rather than inferred.
    """
    client = bigquery.Client(project=PROJECT)
    query = f"SELECT * FROM {TABLE} WHERE {ENTITY_ID_COLUMN} IN UNNEST(@ids)"
    job = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ArrayQueryParameter("ids", "INT64", list(hadm_ids))])
    return {int(row[ENTITY_ID_COLUMN]): dict(row.items())
            for row in client.query(query, job_config=job).result()}


def _profile_fields(profile: dict[str, Any]) -> dict[str, Any]:
    keep = ("encounter_type", "age", "gender", "discharge_location", "insurance", "race",
            "admission_type", "prior_admission_count", "prior_inpatient_days",
            "recent_ed_visits", "index_los_days")
    return {key: profile.get(key) for key in keep if profile.get(key) is not None}


def _medication_facts(payload: dict[str, Any]) -> dict[str, Any]:
    passages = [passage for passage in payload.get("passages") or []
                if isinstance(passage, dict)]
    text = "\n\n".join(
        str(passage.get("text") or "") for passage in passages
    ).strip()
    return {
        "section_present": bool(passages),
        # `chars` is the sum of the passage lengths, not the length of a joined string:
        # the separator is an artefact of how this module concatenates for a regex, and
        # counting it made every two-passage record read 2 characters long.
        "chars": sum(len(str(passage.get("text") or "")) for passage in passages),
        "dedicated_section_present": any(
            passage.get("section") in DEDICATED_MED_SECTIONS for passage in passages),
        "sections": [passage.get("section") for passage in passages],
        "chunk_ids": [passage.get("id") for passage in passages],
        "snippet": text[:240],
        "mentions_anticoagulant": bool(ANTICOAGULANT.search(text)),
        "mentions_taper": bool(TAPER.search(text)),
        "note": payload.get("note"),
    }


def _lab_facts(payload: dict[str, Any]) -> dict[str, Any]:
    trends = payload.get("trends") or []
    abnormal: list[dict[str, Any]] = []
    normal_count = 0
    for trend in trends:
        series = trend.get("series") or []
        discharge = series[-1] if series else {}
        flag = str(discharge.get("flag") or "")
        if flag and flag != "normal":
            abnormal.append({"label": trend.get("label"), "flag": flag,
                             "value": discharge.get("value"),
                             "unit": trend.get("unit")})
        elif flag:
            normal_count += 1
    return {
        "panels": len(trends),
        "abnormal_at_discharge": abnormal,
        "abnormal_count": len(abnormal),
        "normal_at_discharge": normal_count,
        "analytes_with_decimals": [
            trend.get("label") for trend in trends
            if isinstance(trend.get("decimals"), int) and trend["decimals"] > 0
        ],
        "synthetic": bool(payload.get("synthetic")),
    }


def _note_facts(hadm_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    passages = [passage for passage in payload.get("passages") or []
                if isinstance(passage, dict)]
    return {
        "sections": [passage.get("section") for passage in passages],
        "section_count": len(passages),
        "chunk_ids": [passage.get("id") for passage in passages],
        "headingless": hadm_id in HEADINGLESS,
        "error": payload.get("error"),
        "note": payload.get("note"),
    }


def _risk_facts(payload: dict[str, Any], profile: dict[str, Any],
                row: dict[str, Any]) -> dict[str, Any]:
    drivers = []
    for factor in (payload.get("top_factors") or [])[:5]:
        feature = factor.get("feature")
        drivers.append({
            "name": feature,
            "contribution": factor.get("contribution"),
            "direction": factor.get("direction"),
            # The value the model was scored on: the row first, because the profile
            # decodes a fixed set of administrative columns and a lab-derived or
            # oncology feature has no readable form anywhere else. A driver this
            # cannot state is `None` — an absent value must not read as a measured 0.
            "value": row.get(feature, profile.get(feature)),
        })
    return {
        "probability": payload.get("probability"),
        "threshold": payload.get("threshold"),
        "decision": payload.get("decision"),
        "model_version": payload.get("model_version"),
        "drivers": drivers,
    }


async def _record(hadm_id: int, gate: asyncio.Semaphore,
                  row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    async with gate:
        profile_payload = await _read(get_admission_profile, hadm_id)
        risk_payload = await _read(predict_readmission, hadm_id)
        meds_payload = await _read(discharge_medications, hadm_id)
        labs_payload = await _read(get_lab_trends, hadm_id)
        notes_payload = await _read(rag_search_sections, hadm_id)

    profile = {}
    if isinstance(profile_payload, dict) and not profile_payload.get("error"):
        profile = profile_payload.get("profile") or {}

    return hadm_id, {
        "hadm_id": hadm_id,
        # The oracle's decision and the row's own column are recorded side by side:
        # the product reads `encounter_type`, the oracle is the answer key, and the
        # two disagreeing is one of the things this artifact exists to make visible.
        "outpatient": hadm_id in OUTPATIENT_HADM_IDS,
        "encounter_type": row.get("encounter_type"),
        "oncology_flag": row.get("oncology_flag"),
        "profile": _profile_fields(profile),
        "risk": _risk_facts(risk_payload, profile, row),
        "medications": _medication_facts(meds_payload),
        "labs": _lab_facts(labs_payload),
        "notes": _note_facts(hadm_id, notes_payload),
    }


async def _inventory(hadm_ids: tuple[int, ...],
                     rows: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    gate = asyncio.Semaphore(CONCURRENCY)
    pairs = await asyncio.gather(
        *(_record(hadm_id, gate, rows.get(hadm_id, {})) for hadm_id in hadm_ids))
    return dict(pairs)


def _pick(records: dict[int, dict[str, Any]], exclude: set[int]) -> dict[str, int | None]:
    """The mechanical archetype matches, chosen by the phase's own criteria.

    Mechanical on purpose: every choice names the fact that decided it, so a reader can
    disagree with the choice without having to reconstruct it.
    """
    pool = {h: r for h, r in records.items()
            if h not in exclude and not r["outpatient"]}

    def best(key) -> int | None:
        if not pool:
            return None
        return max(pool, key=key)

    complex_meds = best(
        lambda h: (pool[h]["medications"]["chars"]
                   + 400 * pool[h]["medications"]["mentions_taper"]
                   + 400 * pool[h]["medications"]["mentions_anticoagulant"]))
    oncology_risk = best(
        lambda h: (bool(pool[h]["oncology_flag"]),
                   pool[h]["risk"]["probability"] or 0,
                   pool[h]["labs"]["abnormal_count"]))
    minimal = best(
        lambda h: (pool[h]["medications"]["chars"] == 0,
                   -pool[h]["labs"]["abnormal_count"],
                   -pool[h]["medications"]["chars"]))
    headingless = next(
        (h for h in sorted(pool) if pool[h]["notes"]["headingless"]), None)
    outpatient = next(
        (h for h in HOLDOUT_OUTPATIENT_POOL
         if h in records and records[h]["notes"]["sections"]), None)

    return {
        "archetype_1_complex_meds": complex_meds,
        "archetype_2_oncology_risk": oncology_risk,
        "archetype_3_minimal_meds": minimal,
        "archetype_4_headingless_note": headingless,
        "archetype_5_outpatient": outpatient,
    }


def main() -> int:
    served = tuple(sorted(SERVED_HADM_IDS))
    rows = _feature_rows(served)
    records = asyncio.run(_inventory(served, rows))

    dev = [*DEV_INPATIENT, *DEV_OUTPATIENT]
    recommended = _pick(records, exclude=set(dev))

    artifact = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "services/mcp tools + the scoring feature row, read-only",
        "llm_calls": 0,
        "n_records": len(records),
        "dev": {str(h): records[h] for h in dev if h in records},
        "recommended_holdout": {
            role: {"hadm_id": hadm_id,
                   "record": records[hadm_id] if hadm_id in records else None}
            for role, hadm_id in recommended.items()
        },
        "all_records": {str(h): r for h, r in sorted(records.items())},
    }
    OUT.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")

    print(f"wrote {OUT.relative_to(REPO)} ({len(records)} records, 0 LLM calls)")
    for role, hadm_id in recommended.items():
        print(f"  {role}: {hadm_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
