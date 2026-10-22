"""Every turn's trace id, at full length, and the run's honesty about missing ones.

The failure these hold shut: a two-turn item recorded turn 1's id and printed turn 2's
as an eight-character prefix, so the second half of the thread could not be fetched by
any script. Root-cause analysis then had the asked question and none of the second
turn's spans — which is where the interesting defect usually is, because the second turn
is the one carrying a replay.

Both directions matter. An id that is captured but never reported is invisible; an id
that is reported but never counted when it is MISSING is worse, because the run then
looks complete.
"""

from __future__ import annotations

from typing import Any

from evaluation.platform import experiment
from evaluation.platform.schemas import DatasetItemInput
from evaluation.platform.task import make_task

TURN_ONE = "a" * 32
TURN_TWO = "b" * 32


def _item(turns: int = 2) -> dict[str, Any]:
    workflow = "command_plus_followup" if turns == 2 else "single_probe"
    return {
        "id": "item-1",
        "input": DatasetItemInput(
            patient_id="90000015",
            workflow_type=workflow,
            turns=["What is her readmission risk?", "Why is that?"][:turns],
            patient_manifest={"age": "70"},
        ).model_dump(),
        "expected_output": {},
        "metadata": {},
    }


def _ask_returning(*trace_ids: str):
    """An `ask` stand-in that answers each turn with the next id in the list."""
    remaining = list(trace_ids)

    def ask(*, question: str, hadm_id: int, run_tag: str, session_id: str,
            turns: list[dict] | None = None, **_: Any) -> dict[str, Any]:
        return {
            "answer": f"Answered: {question}",
            "langfuse_trace_id": remaining.pop(0),
            "status": 200,
        }

    return ask


def test_a_two_turn_item_records_both_ids_in_order():
    """The list is the item's handle on its own evidence: turn order, full ids."""
    task = make_task(_ask_returning(TURN_ONE, TURN_TWO), run_tag="run",
                     new_session_id=lambda: "session")

    output = task(item=_item())

    assert output["turn_trace_ids"] == [TURN_ONE, TURN_TWO]
    # Named views of the same values, for callers that read them by turn. `turn1_trace`
    # repeats `langfuse_trace_id` because that is the id the scores are written to.
    assert output["turn1_trace"] == TURN_ONE
    assert output["turn2_trace"] == TURN_TWO
    assert output["langfuse_trace_id"] == TURN_ONE
    assert output["turn_2_trace_id"] == TURN_TWO


def test_a_single_turn_item_records_one_id_and_no_turn_two():
    task = make_task(_ask_returning(TURN_ONE), run_tag="run",
                     new_session_id=lambda: "session")

    output = task(item=_item(turns=1))

    assert output["turn_trace_ids"] == [TURN_ONE]
    assert output["turn2_trace"] == ""
    assert output["turn_2_trace_id"] == ""


class _ScoreClient:
    """The two client calls `run_linked` makes, recorded."""

    def __init__(self):
        self.run_items: list[str] = []
        self.scores: list[dict] = []
        self.api = self

    # `client.api.dataset_run_items.create(...)`
    @property
    def dataset_run_items(self):
        return self

    def create(self, **fields):
        self.run_items.append(fields.get("trace_id"))

        class _Row:
            dataset_run_id = "run-id"

        return _Row()

    def create_score(self, **fields):
        self.scores.append(fields)


class _NoEvaluators:
    """`_evaluate` accepts a list; an empty one measures nothing and still runs."""

    def __iter__(self):
        return iter(())


def _run(output: dict[str, Any], capsys) -> tuple[experiment.DatasetRun, str]:
    client = _ScoreClient()
    run = experiment.run_linked(
        client,
        dataset_name="dataset",
        run_name="run",
        items=[{"id": "item-1", "input": {}, "expected_output": {}, "metadata": {}}],
        task=lambda **_: output,
        evaluators=[],  # assertions only; the composite is the judge's, not this test's
        max_concurrency=1,
    )
    return run, capsys.readouterr().out


def test_a_full_turn_two_id_is_printed_rather_than_a_prefix(capsys):
    """An operator reads the second half of a thread out of the log, not out of Langfuse.

    A prefix is not a handle: it resolves to nothing, which is why the 2026-10-21 dump
    could not be opened turn by turn.
    """
    output = {"langfuse_trace_id": TURN_ONE, "turn_trace_ids": [TURN_ONE, TURN_TWO]}

    _, printed = _run(output, capsys)

    assert f"t2={TURN_TWO}" in printed


def test_a_missing_second_turn_id_makes_the_item_unattributed(capsys):
    """The scores still land on turn 1. Half the thread is still unopenable."""
    output = {"langfuse_trace_id": TURN_ONE, "turn_trace_ids": [TURN_ONE, ""]}

    run, printed = _run(output, capsys)

    assert run.stats["unattributed"] == 1
    assert "turn(s) [2]" in printed


def test_a_fully_attributed_item_is_not_counted_unattributed(capsys):
    output = {"langfuse_trace_id": TURN_ONE, "turn_trace_ids": [TURN_ONE, TURN_TWO]}

    run, _ = _run(output, capsys)

    assert run.stats["unattributed"] == 0


def test_an_item_with_no_first_turn_id_is_still_unattributed(capsys):
    """The pre-existing case, kept: no turn 1 id means no scores anywhere."""
    output = {"langfuse_trace_id": "", "turn_trace_ids": [""]}

    run, _ = _run(output, capsys)

    assert run.stats["unattributed"] == 1
