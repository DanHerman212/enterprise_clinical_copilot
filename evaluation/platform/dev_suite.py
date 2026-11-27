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
import re
from pathlib import Path
from typing import Any, Callable

from evaluation.platform.schemas import DatasetItemInput, MAX_TURNS

REPO = Path(__file__).resolve().parents[2]
DEV_SUITE_PATH = REPO / "evaluation" / "datasets" / "dev_suite.jsonl"

#: The twelve records the Tier 2 dev suite may run against.
#:
#: The seven original records stay: they carry the taper (90000015), the critical
#: haemoglobin reading (90000108), the narrative-only medication section (90000106) and
#: both outpatient encounters (90000051, 90000073). Five were added on 2026-11-22, chosen
#: from the 62-record clean pool by structured inventory flags only, so the twelve span
#: the phenotypes the suite has to exercise:
#:
#:   90000045  critical haemoglobin + a dedicated medication section, above threshold
#:   90000046  an Urgent (medical) admission, dedicated section, above threshold
#:   90000103  a documented TAPER beside a dedicated section, below threshold
#:   90000065  a taper and NO dedicated section — the narrative-fallback record
#:   90000035  no dedicated medication section at all, below threshold
#:
#: Every one of them is outside both selection slices, so no dev item can re-measure a
#: record the model-selection arm already scored.
DEV_HADM_IDS = ("90000015", "90000108", "90000106", "90000105",
                "90000023", "90000051", "90000073",
                "90000045", "90000046", "90000103", "90000065", "90000035")

#: Reserved for Phase 05 — ten records, none of them a dev record.
#:
#: The six originals plus four added on 2026-11-22, picked from the remaining clean pool by
#: the same structured flags and nothing else: no holdout note text was read, and none was
#: generated into a fact sheet. 90000088 and 90000104 were already reserved as holdout
#: capacity by `selection_slice.RESERVED_HADM_IDS`; this is the suite taking them up.
#:
#:   90000077  critical haemoglobin, dedicated section, above threshold
#:   90000088  critical haemoglobin, no dedicated section
#:   90000104  critical haemoglobin, dedicated section, above threshold
#:   90000102  an Urgent admission at 0.4237, no dedicated section
#:
#: Nothing in this suite may name one of these, and a row that did would invalidate the
#: holdout before it was ever run.
HOLDOUT_HADM_IDS = ("90000087", "90000067", "90000101", "90000050",
                    "90000026", "90000028",
                    "90000077", "90000088", "90000104", "90000102")

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


# --- The Dev 2D stratification -------------------------------------------------------
#
# The suite is a matrix, not a list. A row states which cell it fills by declaring its
# conversational depth and its modality axis in metadata, and the allocation below is the
# design the audit holds it to. Two axes, five depths by five modalities, forty cells.
#
# It lives here rather than in the audit because `dev_suite.py` owns the file's contract:
# the sync script validates before it uploads, and an audit that held a second copy of the
# allocation would be the copy that drifts.
DEPTH_TIERS: dict[str, int] = {
    "1_turn": 1,
    "2_turn": 2,
    "3_turn": 3,
    "4_turn": 4,
    "5_turn": 5,
}

MODALITY_AXES: tuple[str, ...] = (
    "slash_command",
    "risk_chip",
    "followup_chip",
    "free_form",
    "adversarial_perturbation",
)

#: The six ways an adversarial item may twist a correct question. One row each, because a
#: class that appears twice and another that never appears is a suite that cannot say what
#: it covered.
PERTURBATIONS: tuple[str, ...] = (
    "missing_data_or_outpatient_trap",
    "false_clinical_premise",
    "input_perturbation",
    "scope_or_safety_boundary",
    "prompt_injection_or_override",
    "cross_patient_or_identifier",
)

#: The commands the agent actually has, as the harness names them. `/risk` was retired on
#: 2026-10-26 and is not among them: a row naming it measures nothing.
COMMAND_WIRES = frozenset({"ehr", "meds", "labs", "pubmed"})

#: Zeroes are stated rather than omitted — "no risk chip at depth 1" is a fact about chips,
#: not a gap in the file.
DEV_MATRIX: dict[tuple[str, str], int] = {
    ("1_turn", "slash_command"): 3,
    ("1_turn", "risk_chip"): 0,
    ("1_turn", "followup_chip"): 0,
    ("1_turn", "free_form"): 2,
    ("1_turn", "adversarial_perturbation"): 3,
    ("2_turn", "slash_command"): 1,
    ("2_turn", "risk_chip"): 2,
    ("2_turn", "followup_chip"): 2,
    ("2_turn", "free_form"): 1,
    ("2_turn", "adversarial_perturbation"): 2,
    ("3_turn", "slash_command"): 1,
    ("3_turn", "risk_chip"): 1,
    ("3_turn", "followup_chip"): 2,
    ("3_turn", "free_form"): 2,
    ("3_turn", "adversarial_perturbation"): 2,
    ("4_turn", "slash_command"): 1,
    ("4_turn", "risk_chip"): 2,
    ("4_turn", "followup_chip"): 1,
    ("4_turn", "free_form"): 2,
    ("4_turn", "adversarial_perturbation"): 2,
    ("5_turn", "slash_command"): 1,
    ("5_turn", "risk_chip"): 2,
    ("5_turn", "followup_chip"): 2,
    ("5_turn", "free_form"): 1,
    ("5_turn", "adversarial_perturbation"): 2,
}

EXPECTED_ITEMS = sum(DEV_MATRIX.values())
EXPECTED_TURNS = sum(count * DEPTH_TIERS[depth]
                     for (depth, _axis), count in DEV_MATRIX.items())

#: The records the matrix must reach: every dev record, and both outpatient encounters,
#: because the missing-section trap is only a correct answer where the absence is real.
REQUIRED_HADM_IDS = frozenset(DEV_HADM_IDS)
OUTPATIENT_HADM_IDS = frozenset({"90000051", "90000073"})

#: A driver chip's wording is composed by `shared.labels.driver_question`, so a declared
#: risk-driver turn that does not match this shape is a prompt no control in the product
#: sends.
DRIVER_QUESTION = re.compile(
    r"^How does .+ affect this patient's readmission risk\?$")


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
        # A two-turn thread is a command and its follow-up OR a multi-turn item that
        # happens to be short, so both are legal there. Three turns and above are always
        # `multi_turn`. Computed rather than looked up in a dict keyed by turn count: the
        # dict raised KeyError on any count it did not name, which is how a five-turn row
        # would have failed at load rather than being refused with its scenario id.
        wanted = {"single_probe"} if len(turns) == 1 else (
            {"command_plus_followup", "multi_turn"} if len(turns) == 2
            else {"multi_turn"})
        if workflow not in wanted:
            raise DevSuiteError(
                f"{scenario}: {len(turns)} turn(s) need workflow_type in {wanted}, "
                f"got {workflow!r}"
            )


def _turn_text(turn: Any) -> str:
    """What a turn actually says, whether it is a string or a declared-intent object."""
    if isinstance(turn, str):
        return turn
    return str((turn or {}).get("prompt") or "")


def _leading_command(turn: Any) -> str | None:
    """The command a turn opens with, or `None` when it opens with prose."""
    text = _turn_text(turn).strip()
    if not text.startswith("/"):
        return None
    return text.split()[0].lstrip("/").lower()


def followup_slots(turn: Any) -> list[tuple[int, int]]:
    """Every `{{turn_k.follow_ups[i]}}` a turn reads, as (turn number, slot)."""
    return [(int(match.group(1)), int(match.group(2)))
            for match in re.finditer(
                r"\{\{turn_(\d+)\.follow_ups\[(\d+)\]\}\}", _turn_text(turn))]


def validate_dev_matrix(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Every way the set can claim a shape it does not have. Pure: no file, no model.

    `validate_rows` asks whether a row is runnable. This asks whether the set is the one
    that was designed: each row declares its depth and its modality in metadata, and the
    allocation is compared cell by cell. Returns defects keyed by scenario id, plus
    `"__matrix__"` for the set-level ones, so a caller can print both without having to
    merge two shapes.

    Nothing here raises. A file that is a hundred rows out of shape is a report to read,
    not an exception that hides the other ninety-nine findings.
    """
    found: dict[str, list[str]] = {}
    cells = {cell: 0 for cell in DEV_MATRIX}
    turns_total = 0
    records: set[str] = set()
    commands_seen: set[str] = set()
    perturbations = {name: 0 for name in PERTURBATIONS}

    def add(item: str, problem: str) -> None:
        found.setdefault(item, []).append(problem)

    for row in rows:
        item = str(row.get("id") or "?")
        metadata = row.get("metadata") or {}
        turns = list(row.get("turns") or [])
        turns_total += len(turns)
        records.add(str(row.get("hadm_id") or ""))

        depth = str(metadata.get("depth_tier") or "")
        axis = str(metadata.get("modality_axis") or "")
        if depth not in DEPTH_TIERS:
            add(item, f"unknown depth_tier {depth!r}")
        elif DEPTH_TIERS[depth] != len(turns):
            add(item, f"depth_tier {depth} needs {DEPTH_TIERS[depth]} turn(s), "
                      f"has {len(turns)}")
        if axis not in MODALITY_AXES:
            add(item, f"unknown modality_axis {axis!r}")
        if depth in DEPTH_TIERS and axis in MODALITY_AXES:
            cells[(depth, axis)] += 1

        if str(metadata.get("rubric_version")) != "v9":
            add(item, f"rubric_version {metadata.get('rubric_version')!r}, "
                      "expected 'v9'")

        # The declared command is what the harness binds; Turn 1 is what the reader sees.
        # A row whose two disagree measures a command it does not send.
        declared = row.get("command")
        first = _leading_command(turns[0]) if turns else None
        if first is None:
            if declared is not None:
                add(item, f"turn 1 is free form but command is {declared!r}")
            if "command" in metadata and metadata.get("command") is not None:
                add(item, f"turn 1 is free form but metadata.command is "
                          f"{metadata.get('command')!r}")
        else:
            if str(declared or "").lower() != first:
                add(item, f"command {declared!r} does not match turn 1 '/{first}'")
            if "command" in metadata and str(metadata.get("command") or "").lower() \
                    != first:
                add(item, f"metadata.command {metadata.get('command')!r} does not "
                          f"match turn 1 '/{first}'")

        for turn in turns:
            text = _turn_text(turn).strip()
            if not text.startswith("/"):
                continue
            head = text.split()[0].lstrip("/").lower()
            if head not in COMMAND_WIRES:
                add(item, f"turn names /{head}, which is not a command the agent has")
                continue
            commands_seen.add(head)
            # An argument on a command that takes none reaches routing as ambiguous
            # intent and comes back as a 400: the item would score a transport fault.
            if head in ("ehr", "meds", "labs") and text.lower() != f"/{head}":
                add(item, f"/{head} carries an argument ({text!r}); the route is "
                          "ambiguous with one and the call fails")

        expected = row.get("expected_output") or {}
        for token in expected.get("must_contain") or []:
            text = str(token)
            if any(character.isdigit() for character in text):
                add(item, f"must_contain {text!r} carries a digit; a value belongs in "
                          "facts, not in a literal assertion")
            elif len(text.split()) > 3:
                add(item, f"must_contain {text!r} is longer than three words")

        if axis == "risk_chip":
            drivers = [turn for turn in turns
                       if isinstance(turn, dict)
                       and str(turn.get("intent")) == "risk_driver"]
            if not drivers:
                add(item, "risk_chip item declares no turn with intent 'risk_driver'")
            for turn in drivers:
                prompt = _turn_text(turn).strip()
                if not DRIVER_QUESTION.match(prompt):
                    add(item, "risk_driver prompt is not the question the product "
                              f"asks: {prompt!r}")

        if axis == "followup_chip":
            slots = [slot for turn in turns for slot in followup_slots(turn)]
            if not slots:
                add(item, "followup_chip item declares no {{turn_k.follow_ups[i]}} "
                          "placeholder")
            for number, _slot in slots:
                if number == 1 and not metadata.get("turn1_follow_ups"):
                    add(item, "reads turn 1's chips but metadata.turn1_follow_ups is "
                              "not set")

        if axis == "adversarial_perturbation":
            kind = str(metadata.get("perturbation_type") or "")
            if kind not in PERTURBATIONS:
                add(item, f"unknown perturbation_type {kind!r}")
            else:
                perturbations[kind] += 1

    matrix: list[str] = []
    if len(rows) != EXPECTED_ITEMS:
        matrix.append(f"{len(rows)} scenario(s), expected {EXPECTED_ITEMS}")
    if turns_total != EXPECTED_TURNS:
        matrix.append(f"{turns_total} turn(s), expected {EXPECTED_TURNS}")
    for (depth, axis), wanted in DEV_MATRIX.items():
        if cells[(depth, axis)] != wanted:
            matrix.append(f"{depth}/{axis}: {cells[(depth, axis)]} item(s), "
                          f"expected {wanted}")
    missing_records = sorted(REQUIRED_HADM_IDS - records)
    if missing_records:
        matrix.append(f"dev records never exercised: {missing_records}")
    missing_outpatients = sorted(OUTPATIENT_HADM_IDS - records)
    if missing_outpatients:
        matrix.append(f"outpatient records never exercised: {missing_outpatients}")
    for name, count in perturbations.items():
        if count < 1:
            matrix.append(f"perturbation {name} is never exercised")
    missing_commands = sorted(COMMAND_WIRES - commands_seen)
    if missing_commands:
        matrix.append(f"commands never exercised: {missing_commands}")
    if matrix:
        found.setdefault("__matrix__", []).extend(matrix)
    return found


def item_payload(row: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """One row as the dataset item's `input`.

    `expectations` carries the deterministic subset — the flags, components and literal
    tokens `structural_check` can decide without a model — and the whole of
    `expected_output` travels beside it for the judge, which reads `facts`, `behaviour`
    and `forbidden_claims` from it.

    That second half used to be the ONLY half: six keys were copied into `expectations`,
    so the ground truth of all 41 scenarios reached no evaluator at all. A scenario's
    `facts`, `forbidden_claims`, `turn_2` and `behaviour` asserted nothing, and the judge
    was never shown them (audit, 2026-11-04). `judge_call.build_user_prompt` now renders
    them, and the three that a deterministic rule can decide are copied here.
    """
    expected = dict(row.get("expected_output") or {})
    expectations: dict[str, Any] = {}
    if "verified" in expected:
        expectations["verified"] = expected["verified"]
    for key in ("components", "forbidden_components", "must_contain",
                "must_not_disclose", "citation_marker", "forbidden_claims",
                "literature_nonempty", "banned_source_record",
                "turn_expectations"):
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
