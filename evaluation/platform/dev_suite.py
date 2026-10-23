"""The Tier 2 dev suite, read from `evaluation/datasets/dev_suite.jsonl`.

Twenty-six scenarios over seven dev records and seven trajectories. The file is the
record: every prompt and every expected value in it was copied from
`evaluation/dev_cohort_fact_sheet.md`, and nothing here recomputes a fact.

Two rows do not carry their second prompt. Trajectory 2 tests the affordance the
product actually renders — the chips the service proposes under an answer — so its
second turn is the placeholder `{{turn_1.follow_ups[0]}}`, resolved at run time by
`task.make_task`. A typed-in chip would be a test of a prompt nobody sees.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from evaluation.platform.schemas import DatasetItemInput, MAX_TURNS

REPO = Path(__file__).resolve().parents[2]
DEV_SUITE_PATH = REPO / "evaluation" / "datasets" / "dev_suite.jsonl"

#: The seven records Phase 04 may run against.
DEV_HADM_IDS = ("90000015", "90000108", "90000106", "90000105",
                "90000023", "90000051", "90000073")

#: Reserved for Phase 05. Nothing in this suite may name one of these, and a row that
#: did would invalidate the holdout before it was ever run.
HOLDOUT_HADM_IDS = ("90000087", "90000067", "90000101", "90000050",
                    "90000026", "90000028")

TRAJECTORIES = {
    "t1_declarative_command": 1,
    "t2_dynamic_affordance_loopback": 2,
    "t3_analytical_drilldown": 3,
    "t4_cross_domain_synthesis": 4,
    "t5_freetext_cold_start": 5,
    "t6_adversarial_false_premise": 6,
    "t7_adversarial_boundaries": 7,
}

ManifestProvider = Callable[[list[int]], dict[int, dict]]


class DevSuiteError(ValueError):
    """A row the suite cannot run. Raised at load, not at scoring time."""


def read_rows(path: Path | None = None) -> list[dict[str, Any]]:
    """Every row, parsed and shape-checked. No model, no network."""
    source = path or DEV_SUITE_PATH
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(source.read_text().splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            rows.append(json.loads(text))
        except json.JSONDecodeError as exc:
            raise DevSuiteError(f"{source.name} line {number}: {exc}") from exc
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> None:
    """The invariants that make the suite safe to run, checked before it is run.

    Kept here rather than in the test so the sync script refuses a bad file for the
    same reasons a run would.
    """
    seen: set[str] = set()
    for row in rows:
        scenario = str(row.get("id") or "").strip()
        if not scenario:
            raise DevSuiteError("every scenario needs an id")
        if scenario in seen:
            raise DevSuiteError(f"duplicate scenario id {scenario}")
        seen.add(scenario)

        hadm = str(row.get("hadm_id") or "")
        if hadm in HOLDOUT_HADM_IDS:
            raise DevSuiteError(f"{scenario}: {hadm} is a holdout record")
        if hadm not in DEV_HADM_IDS:
            raise DevSuiteError(f"{scenario}: {hadm} is not a dev record")

        trajectory = str(row.get("trajectory") or "")
        if trajectory not in TRAJECTORIES:
            raise DevSuiteError(f"{scenario}: unknown trajectory {trajectory!r}")

        turns = row.get("turns")
        if not isinstance(turns, list) or not turns:
            raise DevSuiteError(f"{scenario}: turns must be a non-empty list")
        if len(turns) > MAX_TURNS:
            raise DevSuiteError(f"{scenario}: {len(turns)} turns exceeds {MAX_TURNS}")

        expected = row.get("expected_output")
        if not isinstance(expected, dict) or not expected:
            raise DevSuiteError(f"{scenario}: expected_output is mandatory")

        workflow = row.get("workflow_type")
        wanted = {1: {"single_probe"}, 2: {"command_plus_followup"},
                  3: {"multi_turn"}, 4: {"multi_turn"}}[len(turns)]
        if workflow not in wanted:
            raise DevSuiteError(
                f"{scenario}: {len(turns)} turn(s) need workflow_type in {wanted}, "
                f"got {workflow!r}"
            )


def item_payload(row: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """One row as the dataset item's `input`.

    `expectations` carries the deterministic subset (`verified`, components, the
    literal tokens the answer must contain). The prose expectations stay in the item
    for the judge, which reads `expected_output` alongside the answer.
    """
    expected = dict(row.get("expected_output") or {})
    expectations: dict[str, Any] = {}
    if "verified" in expected:
        expectations["verified"] = expected["verified"]
    for key in ("components", "forbidden_components", "must_contain",
                "must_not_disclose", "literature_nonempty"):
        if key in expected:
            expectations[key] = expected[key]

    return {
        "scenario_id": row["id"],
        "patient_id": str(row["hadm_id"]),
        "workflow_type": row["workflow_type"],
        "turns": list(row["turns"]),
        "patient_manifest": manifest,
        "command": row.get("command"),
        "expectations": expectations,
        "trajectory": row["trajectory"],
        "expected_output": expected,
        "metadata": dict(row.get("metadata") or {}),
    }


def dev_suite_items(*, manifest_provider: ManifestProvider) -> list[DatasetItemInput]:
    """The suite as harness items. One manifest lookup per distinct record."""
    rows = read_rows()
    validate_rows(rows)
    hadm_ids = sorted({int(row["hadm_id"]) for row in rows})
    manifests = manifest_provider(hadm_ids)
    return [
        DatasetItemInput.model_validate(
            item_payload(row, manifests.get(int(row["hadm_id"])) or {})
        )
        for row in rows
    ]
