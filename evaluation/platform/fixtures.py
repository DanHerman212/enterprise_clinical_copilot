"""Dataset fixtures, built from files in the repo — never from a bucket mount.

Two populations, deliberately separate (spec §1A):

    clinical-census-pilot-v1   1 patient   x 4 workflows = 4 two-turn items
    clinical-census-full-v1    82 patients x 4 workflows = 328 two-turn items
    suite2-adversarial-v1      30 single probes

A census item is a two-turn *script*, not two items: Turn 2 needs Turn 1's session, so
flattening them would run the follow-up out of order under concurrency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
#: Dataset fixtures built from committed files. `DATA` is where the legacy collector kept
#: its case JSON, and it stays a data directory: no module is imported from it (`00d8`
#: Row 9).
DATA = REPO / "evaluation" / "agent"

from evaluation.platform import case_set  # noqa: E402
from evaluation.platform.schemas import DatasetItemInput  # noqa: E402

CASES = DATA / "results" / "cases_82.json"
SUITE2 = DATA / "suite2_cases.json"

#: How many items each dataset must contain. Asserted at seed time: a seeded dataset
#: that is quietly short is a pass rate over an unknown subset.
#:
#: The pilot is `v2` because `clinical-census-pilot-v1` was created on 2026-10-20 with
#: `TaskOutput` as its `expected_output_schema`, then refused every item: the server
#: demands an expected output shaped like a finished answer, which a census item does
#: not have. Datasets cannot be deleted in this Langfuse version, so the shell stays and
#: the corrected dataset takes the next version.
EXPECTED_COUNTS = {
    "clinical-census-pilot-v2": 1 * len(case_set.CENSUS_PROMPTS),
    "clinical-census-full-v1": 82 * len(case_set.CENSUS_PROMPTS),
    "suite2-adversarial-v1": 30,
}

ManifestProvider = Callable[[list[int]], dict[int, dict]]


def manifest_from_bigquery(hadm_ids: list[int]) -> dict[int, dict]:
    """The decoded manifest, from the same table and the same decoder the service uses."""
    from evaluation.build_census import manifests_by_hadm_id  # noqa: PLC0415

    return manifests_by_hadm_id(hadm_ids)


def _census_patients() -> list[dict]:
    payload = json.loads(CASES.read_text())
    patients = payload.get("patients") or []
    if not patients:
        raise RuntimeError(f"no patients in {CASES} — run build_census.py first")
    return patients


def _question_for(prompt: str) -> str:
    """The question a census command turn actually puts on the wire."""
    try:
        return case_set.question_for(prompt)
    except Exception:  # noqa: BLE001 — fall back to the documented slash form
        return case_set.SLASH_COMMANDS.get(prompt, f"/{prompt}")


def census_items(
    *,
    hadm_ids: list[int] | None = None,
    manifest_provider: ManifestProvider | None = None,
) -> list[DatasetItemInput]:
    """One two-turn item per (patient, workflow)."""
    patients = _census_patients()
    if hadm_ids is not None:
        wanted = {int(h) for h in hadm_ids}
        patients = [p for p in patients if int(p["hadm_id"]) in wanted]
    ids = [int(p["hadm_id"]) for p in patients]
    provider = manifest_provider or manifest_from_bigquery
    manifests = provider(ids)

    items: list[DatasetItemInput] = []
    for patient in patients:
        hadm = int(patient["hadm_id"])
        manifest = manifests.get(hadm) or {}
        for prompt in case_set.CENSUS_PROMPTS:
            items.append(DatasetItemInput(
                patient_id=str(hadm),
                workflow_type="command_plus_followup",
                turns=[_question_for(prompt), case_set.follow_up_for(prompt)],
                patient_manifest=manifest,
                command=prompt,
            ))
    return items


def suite2_items(*, manifest_provider: ManifestProvider | None = None) -> list[DatasetItemInput]:
    """One single-probe item per targeted modality case."""
    payload = json.loads(SUITE2.read_text())
    cases = payload.get("cases") or []
    ids = sorted({int(c["hadm_id"]) for c in cases if c.get("hadm_id")})
    provider = manifest_provider or manifest_from_bigquery
    manifests = provider(ids)

    items: list[DatasetItemInput] = []
    for case in cases:
        hadm = int(case["hadm_id"])
        expect = dict(case.get("expect") or {})
        items.append(DatasetItemInput(
            patient_id=str(hadm),
            workflow_type="single_probe",
            turns=[case["question"]],
            patient_manifest=manifests.get(hadm) or {},
            command=expect.get("command"),
            expectations=expect,
        ))
    return items


def all_datasets(*, manifest_provider: ManifestProvider | None = None,
                  pilot_hadm_ids: list[int] | None = None) -> dict[str, list[DatasetItemInput]]:
    """The three datasets, with their expected counts asserted here."""
    pilot_ids = pilot_hadm_ids or [int(_census_patients()[0]["hadm_id"])]
    built = {
        "clinical-census-pilot-v2": census_items(
            hadm_ids=pilot_ids, manifest_provider=manifest_provider),
        "clinical-census-full-v1": census_items(manifest_provider=manifest_provider),
        "suite2-adversarial-v1": suite2_items(manifest_provider=manifest_provider),
    }
    for name, items in built.items():
        expected = EXPECTED_COUNTS[name]
        if len(items) != expected:
            raise RuntimeError(f"{name}: {len(items)} items, expected {expected}")
        missing = [i.patient_id for i in items if not i.patient_manifest]
        if missing:
            raise RuntimeError(f"{name}: {len(missing)} items without a manifest")
    return built
