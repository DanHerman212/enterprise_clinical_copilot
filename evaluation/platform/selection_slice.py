"""The multi-model selection slice, read from `evaluation/datasets/selection_slice.jsonl`.

Fourteen scenarios over five records and seven trajectories: two per trajectory, so every
capability the model comparison measures is represented twice rather than once. The file is
the record; every prompt and every expected value in it was copied from
`evaluation/selection_slice_fact_sheet.md`, and nothing here recomputes a fact.

Why a second suite rather than a subset of the Dev one
------------------------------------------------------
The Dev suite's prompts and its records have both been used to tune the agent. A model
comparison run on them would measure how well a model fits decisions already made. These
five records are unused by the Dev suite, unused by the Holdout suite, and unused by the
census's own `recommended_holdout`, so the prompts below were written against records no
prior evaluation has seen.

What it deliberately is NOT: a replacement for the Dev or Holdout suites. Those stay
unbuilt until after selection. Seventy-two served census records remain untouched for them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from evaluation.platform.schemas import DatasetItemInput, MAX_TURNS

REPO = Path(__file__).resolve().parents[2]
SELECTION_SLICE_PATH = REPO / "evaluation" / "datasets" / "selection_slice.jsonl"

#: The six records the Dev suite runs on, read from its own rows.
DEV_HADM_IDS = frozenset({90000015, 90000023, 90000051, 90000105, 90000106, 90000108})

#: The four records the Holdout suite runs on. Extracted as integers only.
HOLDOUT_HADM_IDS = frozenset({90000028, 90000067, 90000087, 90000101})

#: The five the census inventories as `recommended_holdout`, minus the one the Holdout
#: suite already took. Kept out of the slice for the same reason: they are holdout capacity.
RESERVED_HADM_IDS = frozenset({90000088, 90000104, 90000014, 90000026})

EXCLUDED_HADM_IDS = DEV_HADM_IDS | HOLDOUT_HADM_IDS | RESERVED_HADM_IDS

#: The slice's own records, one per archetype. The selection rule for each is recorded in
#: `evaluation/inventory/selection_slice_fact_sheet.py`.
SELECTED_HADM_IDS = (90000010, 90000086, 90000011, 90000081, 90000054)

ARCHETYPES = {
    90000010: "p1_complex_medication_taper",
    90000086: "p2_high_risk_abnormal_labs",
    90000011: "p3_all_normal_labs",
    90000081: "p4_multisection_irregular_note",
    90000054: "p5_outpatient_boundary",
}

TRAJECTORIES = {
    "t1_declarative_command": 1,
    "t2_dynamic_affordance_loopback": 2,
    "t3_analytical_drilldown": 3,
    "t4_cross_domain_synthesis": 4,
    "t5_freetext_cold_start": 5,
    "t6_adversarial_false_premise": 6,
    "t7_adversarial_boundaries": 7,
}

#: Two scenarios per trajectory, and the count is the contract: one scenario per capability
#: is a coin flip, and a model that wins a capability by one lucky run would win the matrix.
PER_TRAJECTORY = 2
EXPECTED_ITEMS = len(TRAJECTORIES) * PER_TRAJECTORY

#: The commands the agent actually has. `/risk` was retired on 2026-10-26 and is not among
#: them; a row naming it would be refused as an unknown command and would measure nothing.
COMMAND_WIRES = frozenset({"ehr", "meds", "labs", "pubmed"})

#: The turn count each trajectory scripts, from the directive.
TURNS_PER_TRAJECTORY = {
    "t1_declarative_command": 1,
    "t2_dynamic_affordance_loopback": 2,
    "t3_analytical_drilldown": 2,
    "t4_cross_domain_synthesis": 3,
    "t5_freetext_cold_start": 2,
    "t6_adversarial_false_premise": 2,
    "t7_adversarial_boundaries": 2,
}

ManifestProvider = Callable[[list[int]], dict[int, dict]]


class SelectionSliceError(ValueError):
    """A row the slice cannot run. Raised at load, not at scoring time."""


def read_rows(path: Path | None = None) -> list[dict[str, Any]]:
    """Every row, parsed. No model, no network."""
    source = path or SELECTION_SLICE_PATH
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(source.read_text().splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            rows.append(json.loads(text))
        except json.JSONDecodeError as exc:
            raise SelectionSliceError(f"{source.name} line {number}: {exc}") from exc
    return rows


def workflow_for(turn_count: int) -> str:
    """The `workflow_type` a row with this many turns must declare.

    Same mapping the Dev suite uses, so the harness reads both suites the same way.
    """
    if turn_count == 1:
        return "single_probe"
    return "command_plus_followup" if turn_count == 2 else "multi_turn"


def validate_rows(rows: list[dict[str, Any]]) -> None:
    """The invariants that make the slice safe to run, checked before it is run.

    Kept here rather than only in the test so the sync script refuses a bad file for the
    same reasons a run would — and so a row that names a Dev or Holdout record cannot be
    written to Langfuse at all.
    """
    seen: set[str] = set()
    counts: dict[str, int] = {name: 0 for name in TRAJECTORIES}
    per_record: dict[str, int] = {}

    for row in rows:
        scenario = str(row.get("id") or "").strip()
        if not scenario:
            raise SelectionSliceError("every scenario needs an id")
        if scenario in seen:
            raise SelectionSliceError(f"duplicate scenario id {scenario}")
        seen.add(scenario)

        hadm = int(row.get("hadm_id"))
        if hadm in DEV_HADM_IDS:
            raise SelectionSliceError(f"{scenario}: {hadm} is a Dev suite record")
        if hadm in HOLDOUT_HADM_IDS:
            raise SelectionSliceError(f"{scenario}: {hadm} is a Holdout suite record")
        if hadm in RESERVED_HADM_IDS:
            raise SelectionSliceError(f"{scenario}: {hadm} is reserved holdout capacity")
        if hadm not in SELECTED_HADM_IDS:
            raise SelectionSliceError(f"{scenario}: {hadm} is not one of the five records")
        if row.get("archetype") != ARCHETYPES[hadm]:
            raise SelectionSliceError(
                f"{scenario}: archetype {row.get('archetype')!r} does not match "
                f"record {hadm}")
        per_record[str(hadm)] = per_record.get(str(hadm), 0) + 1

        trajectory = str(row.get("trajectory") or "")
        if trajectory not in TRAJECTORIES:
            raise SelectionSliceError(f"{scenario}: unknown trajectory {trajectory!r}")
        counts[trajectory] += 1

        turns = row.get("turns")
        if not isinstance(turns, list) or not turns:
            raise SelectionSliceError(f"{scenario}: turns must be a non-empty list")
        if len(turns) > MAX_TURNS:
            raise SelectionSliceError(f"{scenario}: {len(turns)} turns exceeds {MAX_TURNS}")
        wanted = TURNS_PER_TRAJECTORY[trajectory]
        if len(turns) != wanted:
            raise SelectionSliceError(
                f"{scenario}: trajectory {trajectory} scripts {wanted} turn(s), "
                f"this row has {len(turns)}")

        workflow = row.get("workflow_type")
        if workflow != workflow_for(len(turns)):
            raise SelectionSliceError(
                f"{scenario}: {len(turns)} turn(s) need workflow_type "
                f"{workflow_for(len(turns))!r}, got {workflow!r}")

        command = row.get("command")
        if command is not None and command not in COMMAND_WIRES:
            raise SelectionSliceError(f"{scenario}: unknown command {command!r}")

        expected = row.get("expected_output")
        if not isinstance(expected, dict) or not expected.get("facts"):
            raise SelectionSliceError(f"{scenario}: expected_output.facts is mandatory")
        if not expected.get("behaviour"):
            raise SelectionSliceError(f"{scenario}: expected_output.behaviour is mandatory")
        if not isinstance(expected.get("forbidden_claims", []), list):
            raise SelectionSliceError(f"{scenario}: forbidden_claims must be a list")

        metadata = row.get("metadata") or {}
        if metadata.get("rubric_version") != "v8":
            raise SelectionSliceError(f"{scenario}: metadata.rubric_version must be v8")
        if metadata.get("trajectory") != TRAJECTORIES[trajectory]:
            raise SelectionSliceError(
                f"{scenario}: metadata.trajectory must be {TRAJECTORIES[trajectory]}")

    if len(rows) != EXPECTED_ITEMS:
        raise SelectionSliceError(f"{len(rows)} scenarios, expected {EXPECTED_ITEMS}")
    for name, index in TRAJECTORIES.items():
        if counts[name] != PER_TRAJECTORY:
            raise SelectionSliceError(
                f"trajectory {index} ({name}) has {counts[name]} scenario(s), "
                f"expected {PER_TRAJECTORY}")
    if set(per_record) != {str(hadm) for hadm in SELECTED_HADM_IDS}:
        raise SelectionSliceError(
            f"records used {sorted(per_record)} do not match the five selected")


def item_payload(row: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """One row as the dataset item's `input`, in the shape `task.make_task` reads.

    `expectations` carries the deterministic subset `structural_check` can decide without
    a model; the whole of `expected_output` travels beside it in the item's
    `expected_output`, which is where `judge_call.build_user_prompt` reads `facts`,
    `behaviour` and `forbidden_claims` from.
    """
    expected = dict(row.get("expected_output") or {})
    expectations: dict[str, Any] = {}
    if "verified" in expected:
        expectations["verified"] = expected["verified"]
    for key in ("components", "forbidden_components", "must_contain",
                "must_not_disclose", "citation_marker", "forbidden_claims",
                "literature_nonempty", "banned_source_record"):
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


def selection_slice_items(*, manifest_provider: ManifestProvider) -> list[DatasetItemInput]:
    """The slice as harness items. One manifest lookup per distinct record."""
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
