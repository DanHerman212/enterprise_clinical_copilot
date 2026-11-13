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

#: The seven records the Dev suite runs on, read from its own rows.
DEV_HADM_IDS = frozenset(
    {90000015, 90000023, 90000051, 90000073, 90000105, 90000106, 90000108})

#: The six the Holdout suite runs on plus the two the census reserved for it. Extracted as
#: integers only: no holdout prompt or expected output is read here.
HOLDOUT_HADM_IDS = frozenset(
    {90000026, 90000028, 90000050, 90000067, 90000087, 90000088, 90000101})

#: The two the census inventories as holdout capacity and no suite has taken.
RESERVED_HADM_IDS = frozenset({90000014, 90000104})

#: The five records `ecc-model-selection-slice-v1` used. v2 is a clean-slate rebuild, so
#: reusing one of them would carry the measurements v1 was retired for into the new matrix.
SLICE_V1_HADM_IDS = frozenset({90000010, 90000011, 90000054, 90000081, 90000086})

EXCLUDED_HADM_IDS = (DEV_HADM_IDS | HOLDOUT_HADM_IDS | RESERVED_HADM_IDS
                     | SLICE_V1_HADM_IDS)

#: The slice's own records. Selection reasons are in
#: `evaluation/inventory/selection_slice_fact_sheet.py`.
SELECTED_HADM_IDS = (90000020, 90000040, 90000041, 90000059, 90000069, 90000090,
                     90000099)

#: `evaluation/outpatient_oracle.py` is the independent answer key on encounter type. The
#: matrix needs at least one outpatient record, because the missing-section trap is only a
#: correct answer where the absence really is the record.
OUTPATIENT_HADM_IDS = frozenset({90000040})

#: The two axes of the matrix, and the cell allocation the directive fixes. A cell holding
#: zero is stated rather than omitted: "no chip items at depth 1" is a fact about chips,
#: not a gap in the data.
DEPTH_TIERS: dict[str, int] = {"1_turn": 1, "2_turn": 2, "3_turn": 3}

MODALITY_AXES = ("slash_command", "followup_chip", "free_form",
                 "adversarial_perturbation")

MATRIX: dict[tuple[str, str], int] = {
    ("1_turn", "slash_command"): 2,
    ("1_turn", "followup_chip"): 0,
    ("1_turn", "free_form"): 1,
    ("1_turn", "adversarial_perturbation"): 2,
    ("2_turn", "slash_command"): 2,
    ("2_turn", "followup_chip"): 2,
    ("2_turn", "free_form"): 1,
    ("2_turn", "adversarial_perturbation"): 1,
    ("3_turn", "slash_command"): 1,
    ("3_turn", "followup_chip"): 2,
    ("3_turn", "free_form"): 1,
    ("3_turn", "adversarial_perturbation"): 1,
}

EXPECTED_ITEMS = sum(MATRIX.values())
EXPECTED_TURNS = sum(count * DEPTH_TIERS[depth] for (depth, _), count in MATRIX.items())

#: The commands the agent actually has. `/risk` was retired on 2026-10-26 and is not among
#: them; a row naming it would be refused as an unknown command and would measure nothing.
COMMAND_WIRES = frozenset({"ehr", "meds", "labs", "pubmed"})

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
    cells: dict[tuple[str, str], int] = {cell: 0 for cell in MATRIX}
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
        if hadm in SLICE_V1_HADM_IDS:
            raise SelectionSliceError(
                f"{scenario}: {hadm} was a slice v1 record and v2 is a clean rebuild")
        if hadm not in SELECTED_HADM_IDS:
            raise SelectionSliceError(
                f"{scenario}: {hadm} is not one of the seven records this slice names")
        if not str(row.get("archetype") or "").strip():
            raise SelectionSliceError(f"{scenario}: archetype is mandatory")
        per_record[str(hadm)] = per_record.get(str(hadm), 0) + 1

        turns = row.get("turns")
        if not isinstance(turns, list) or not turns:
            raise SelectionSliceError(f"{scenario}: turns must be a non-empty list")
        if len(turns) > MAX_TURNS:
            raise SelectionSliceError(f"{scenario}: {len(turns)} turns exceeds {MAX_TURNS}")
        for index, turn in enumerate(turns, start=1):
            if isinstance(turn, str):
                text = turn.strip()
            elif isinstance(turn, dict):
                text = str(turn.get("prompt") or "").strip()
            else:
                raise SelectionSliceError(
                    f"{scenario}: turn {index} is neither a prompt nor a turn object")
            if not text:
                raise SelectionSliceError(f"{scenario}: turn {index} is blank")
            word = text.split()[0] if text.split() else ""
            if word.startswith("/") and word.lstrip("/").lower() not in COMMAND_WIRES:
                raise SelectionSliceError(
                    f"{scenario}: turn {index} names an unknown command {word!r}")

        metadata = row.get("metadata") or {}
        depth = str(metadata.get("depth_tier") or "")
        axis = str(metadata.get("modality_axis") or "")
        if depth not in DEPTH_TIERS:
            raise SelectionSliceError(f"{scenario}: unknown depth_tier {depth!r}")
        if axis not in MODALITY_AXES:
            raise SelectionSliceError(f"{scenario}: unknown modality_axis {axis!r}")
        if DEPTH_TIERS[depth] != len(turns):
            raise SelectionSliceError(
                f"{scenario}: depth_tier {depth} needs {DEPTH_TIERS[depth]} turn(s), "
                f"this row has {len(turns)}")
        cells[(depth, axis)] += 1

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
        for token in expected.get("must_contain") or []:
            # A dosage string or a unit in `must_contain` turns a paraphrase into a
            # structural failure, so the contract refuses it before the row can be run.
            if any(character.isdigit() for character in str(token)):
                raise SelectionSliceError(
                    f"{scenario}: must_contain {token!r} carries a number; "
                    "entity names only")
            if len(str(token).split()) > 3:
                raise SelectionSliceError(
                    f"{scenario}: must_contain {token!r} is a phrase, not an entity")

        if metadata.get("rubric_version") != "v8":
            raise SelectionSliceError(f"{scenario}: metadata.rubric_version must be v8")

    if len(rows) != EXPECTED_ITEMS:
        raise SelectionSliceError(f"{len(rows)} scenarios, expected {EXPECTED_ITEMS}")
    turns_total = sum(len(row.get("turns") or []) for row in rows)
    if turns_total != EXPECTED_TURNS:
        raise SelectionSliceError(f"{turns_total} turns, expected {EXPECTED_TURNS}")
    for cell, wanted in MATRIX.items():
        if cells[cell] != wanted:
            raise SelectionSliceError(
                f"cell {cell[0]}/{cell[1]} holds {cells[cell]} scenario(s), "
                f"expected {wanted}")
    if set(per_record) != {str(hadm) for hadm in SELECTED_HADM_IDS}:
        raise SelectionSliceError(
            f"records used {sorted(per_record)} do not match the seven selected")
    if not (set(per_record) & {str(hadm) for hadm in OUTPATIENT_HADM_IDS}):
        raise SelectionSliceError("the matrix needs its outpatient record")


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
