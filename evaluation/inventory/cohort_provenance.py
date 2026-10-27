"""Phase 03, Section 4 — the ground-truth cohort provenance inventory.

Read-only. No model is called: every field comes from the same four patient-scoped tools
the sweep exercises, plus the phenotype map that already resolves each admission.

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

from services.agent import admission_phenotypes
from services.agent import routing  # noqa: E402
from services.agent.served_admissions import SERVED_HADM_IDS  # noqa: E402
from services.mcp.tools import (  # noqa: E402
    discharge_medications,
    get_lab_trends,
    predict_readmission,
    rag_search_sections,
)
from services.mcp.tools.admission_profile import get_admission_profile  # noqa: E402

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


def _profile_fields(profile: dict[str, Any]) -> dict[str, Any]:
    keep = ("age", "gender", "discharge_location", "insurance", "race",
            "admission_type", "prior_admission_count", "prior_inpatient_days",
            "recent_ed_visits", "index_los_days")
    return {key: profile.get(key) for key in keep if profile.get(key) is not None}


def _medication_facts(payload: dict[str, Any]) -> dict[str, Any]:
    passages = payload.get("passages") or []
    text = "\n\n".join(
        str(passage.get("text") or "") for passage in passages
        if isinstance(passage, dict)
    ).strip()
    return {
        "section_present": bool(passages),
        "chars": len(text),
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
    return {
        "sections": [passage.get("section") for passage in payload.get("passages") or []],
        "headingless": hadm_id in HEADINGLESS,
        "error": payload.get("error"),
        "note": payload.get("note"),
    }


def _risk_facts(payload: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    drivers = []
    for factor in (payload.get("top_factors") or [])[:5]:
        feature = factor.get("feature")
        drivers.append({
            "name": feature,
            "contribution": factor.get("contribution"),
            "direction": factor.get("direction"),
            # The value the model was scored on, when the row carries that feature.
            "value": profile.get(feature),
        })
    return {
        "probability": payload.get("probability"),
        "threshold": payload.get("threshold"),
        "decision": payload.get("decision"),
        "model_version": payload.get("model_version"),
        "drivers": drivers,
    }


async def _record(hadm_id: int, gate: asyncio.Semaphore) -> tuple[int, dict[str, Any]]:
    async with gate:
        profile_payload = await _read(get_admission_profile, hadm_id)
        risk_payload = await _read(predict_readmission, hadm_id)
        meds_payload = await _read(discharge_medications, hadm_id)
        labs_payload = await _read(get_lab_trends, hadm_id)
        notes_payload = await _read(rag_search_sections, hadm_id)

    profile = {}
    if isinstance(profile_payload, dict) and not profile_payload.get("error"):
        profile = profile_payload.get("profile") or {}
    bucket = admission_phenotypes.bucket_for(hadm_id)

    return hadm_id, {
        "hadm_id": hadm_id,
        "outpatient": hadm_id in routing.OUTPATIENT_HADM_IDS,
        "phenotype": bucket.key if bucket else None,
        "profile": _profile_fields(profile),
        "risk": _risk_facts(risk_payload, profile),
        "medications": _medication_facts(meds_payload),
        "labs": _lab_facts(labs_payload),
        "notes": _note_facts(hadm_id, notes_payload),
    }


async def _inventory(hadm_ids: tuple[int, ...]) -> dict[int, dict[str, Any]]:
    gate = asyncio.Semaphore(CONCURRENCY)
    pairs = await asyncio.gather(*(_record(h, gate) for h in hadm_ids))
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
        lambda h: (pool[h]["phenotype"] == "oncology",
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
    records = asyncio.run(_inventory(served))

    dev = [*DEV_INPATIENT, *DEV_OUTPATIENT]
    recommended = _pick(records, exclude=set(dev))

    artifact = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "services/mcp tools + admission_phenotypes, read-only",
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
