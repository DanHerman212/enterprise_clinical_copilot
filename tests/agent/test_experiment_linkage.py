"""The experiment's trace linkage: a score belongs on the trace that answered.

Langfuse's own `run_experiment` opens a wrapper trace per item (`experiment-item-run`)
and writes the scores there. Our agent opens its own trace, tagged with the run. On
2026-10-21 that produced 814 scores, none of them on a tagged trace, so the Scores
view's trace-tag filter — which is the only way to isolate one run from four — matched
nothing.

These tests pin the replacement loop: the item and every score are created against the
`langfuse_trace_id` the service returns, nothing is written against a wrapper, and an
answer that carries no id is recorded as unscored rather than scored against nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from langfuse import Evaluation  # noqa: E402

from evaluation.platform import evaluators as ev  # noqa: E402
from evaluation.platform.experiment import (  # noqa: E402
    record_run_scores,
    retry_unscored,
    run_linked,
)

MANIFEST = {"age": "70", "admission_type": "Urgent (URGENT)",
            "prior_inpatient_days": "2", "index_los_days": "4"}


class _RunItems:
    """Stands in for `client.api.dataset_run_items`."""

    def __init__(self, log):
        self.log = log

    def create(self, **kwargs):
        self.log.append(kwargs)
        return SimpleNamespace(dataset_run_id="run-1")


class _Client:
    """Records the two writes that decide attribution: run items and scores."""

    def __init__(self):
        self.run_items: list[dict] = []
        self.scores: list[dict] = []
        self.api = SimpleNamespace(dataset_run_items=_RunItems(self.run_items))

    def create_score(self, **kwargs):
        self.scores.append(kwargs)


def _item(item_id: str, **overrides):
    base = dict(id=item_id, dataset_id="ds-1", input={"turns": ["q"], "expectations": {}},
                expected_output=None, metadata={"workflow_type": "single_probe"})
    base.update(overrides)
    return SimpleNamespace(**base)


def _assertions(*, input, output, expected_output=None, metadata=None, **_):
    return [Evaluation(name=ev.STRUCTURAL, value=1.0, data_type="NUMERIC")]


def _composite(*, input=None, output=None, expected_output=None, metadata=None,
               evaluations=None, **_):
    assert evaluations, "the composite must be handed the assertion scores"
    return Evaluation(name=ev.VERDICT, value=1.0, comment="pass", data_type="NUMERIC")


def _task_for(ids: dict[str, str]):
    def task(*, item, **_):
        return {"turn_1_response": f"answer for {item.id}",
                "langfuse_trace_id": ids.get(item.id, "")}
    return task


def test_scores_are_written_onto_the_trace_that_answered():
    client = _Client()
    items = [_item("item-a"), _item("item-b")]

    run_linked(client, dataset_name="suite2-adversarial-v1", run_name="r1", items=items,
               task=_task_for({"item-a": "trace-a", "item-b": "trace-b"}),
               evaluators=[_assertions], composite=_composite, max_concurrency=1)

    assert [row["trace_id"] for row in client.run_items] == ["trace-a", "trace-b"]
    assert {row["trace_id"] for row in client.scores} == {"trace-a", "trace-b"}
    # Two scores per item: the assertion and the verdict.
    assert len(client.scores) == 4
    assert all(row["name"] in (ev.STRUCTURAL, ev.VERDICT) for row in client.scores)


def test_no_wrapper_trace_is_created():
    """The trace the scores land on must be the agent's, never a harness container."""
    client = _Client()
    run_linked(client, dataset_name="suite2-adversarial-v1", run_name="r1",
               items=[_item("item-a")], task=_task_for({"item-a": "trace-a"}),
               evaluators=[_assertions], composite=_composite, max_concurrency=1)

    assert client.run_items[0]["trace_id"] == "trace-a"
    assert "experiment-item-run" not in str(client.run_items + client.scores)


def test_an_answer_without_a_trace_id_keeps_its_verdict_but_is_not_written():
    """A score nobody can trace back to an answer is not written, and the run says so."""
    client = _Client()
    run = run_linked(client, dataset_name="suite2-adversarial-v1", run_name="r1",
                     items=[_item("item-a")], task=_task_for({}),
                     evaluators=[_assertions], composite=_composite, max_concurrency=1)

    assert client.scores == [], "nothing may be written without a trace to write it to"
    # Nothing was sent to the dataset-run API either: `trace_id=None` is a 400 from
    # Langfuse, and four of those errors buried the real message on 2026-10-21.
    assert client.run_items == [], "no run item without a trace id"
    # The measurement happened, so it still counts in the rate...
    assert run.stats["scored"] == 1 and run.stats["pass_rate"] == 1.0
    # ...and the run records that its evidence could not be opened.
    assert run.stats["unattributed"] == 1


def test_the_run_gate_is_recorded_against_the_dataset_run():
    """Run-level scores are the Datasets view's; they hang off the run, not a trace."""
    client = _Client()

    def gate(*, item_results, **_):
        return [Evaluation(name="gate_pass", value=1.0, comment="ok",
                           data_type="NUMERIC")]

    run = run_linked(client, dataset_name="suite2-adversarial-v1", run_name="r1",
                     items=[_item("item-a")], task=_task_for({"item-a": "trace-a"}),
                     evaluators=[_assertions], composite=_composite,
                     max_concurrency=1)
    record_run_scores(client, dataset_run_id=run.dataset_run_id,
                      outcomes=run.item_results, run_evaluators=[gate])

    run_scores = [row for row in client.scores if row.get("dataset_run_id")]
    assert [row["name"] for row in run_scores] == ["gate_pass"]
    assert run_scores[0]["dataset_run_id"] == "run-1"
    assert "trace_id" not in run_scores[0] or run_scores[0]["trace_id"] is None


def test_a_failed_task_does_not_end_the_run():
    client = _Client()

    def task(*, item, **_):
        if item.id == "item-a":
            raise RuntimeError("503 from the service")
        return {"turn_1_response": "fine", "langfuse_trace_id": "trace-b"}

    run = run_linked(client, dataset_name="suite2-adversarial-v1", run_name="r1",
                     items=[_item("item-a"), _item("item-b")], task=task,
                     evaluators=[_assertions], composite=_composite, max_concurrency=1)

    assert run.stats["scored"] == 1
    assert run.stats["unscored"] == 1
    assert run.stats["unattributed"] == 0, "a task failure is not an attribution gap"
    assert {row["trace_id"] for row in client.scores} == {"trace-b"}


def test_the_task_carries_the_services_trace_id_into_the_output():
    """The wiring inside `make_task`, which is where the id could silently be dropped."""
    from evaluation.platform.task import make_task

    def ask(*, question, hadm_id, run_tag, session_id=None, turns=None):
        return {"status": 200, "answer": "readmission risk 33.3%",
                "langfuse_trace_id": "trace-from-service"}

    task = make_task(ask, run_tag="r1", new_session_id=lambda: "sess-1")
    output = task(item=_item("item-a", input={
        "patient_id": "90000023", "workflow_type": "single_probe",
        "turns": ["/risk"], "patient_manifest": MANIFEST}))

    assert output["langfuse_trace_id"] == "trace-from-service"


def test_the_retry_pass_recovers_an_item_the_judge_missed():
    """`8053ca5c`: one slow judge call must not cost a multi-hour census."""
    client = _Client()
    calls = {"n": 0}

    def flaky_composite(*, input=None, output=None, expected_output=None, metadata=None,
                        evaluations=None, **_):
        calls["n"] += 1
        if calls["n"] == 1:
            return Evaluation(name=ev.JUDGE_ERROR, value=1.0, data_type="NUMERIC",
                              comment="unscored — no response within 150s")
        return Evaluation(name=ev.VERDICT, value=1.0, comment="pass",
                          data_type="NUMERIC")

    items = [_item("item-a")]
    run = run_linked(client, dataset_name="suite2-adversarial-v1", run_name="r1",
                     items=items, task=_task_for({"item-a": "trace-a"}),
                     evaluators=[_assertions], composite=flaky_composite,
                     max_concurrency=1)
    assert run.stats["unscored"] == 1
    written_before = len(client.scores)

    stats = retry_unscored(client, items=items, outcomes=run.item_results,
                           evaluators=[_assertions], composite=flaky_composite)

    assert stats["unscored"] == 0 and stats["scored"] == 1
    # Only the verdict is new. The assertion score is not rewritten, and the judge_error
    # stays on the trace as the record of the attempt that failed.
    new = client.scores[written_before:]
    assert [row["name"] for row in new] == [ev.VERDICT]
    assert new[0]["trace_id"] == "trace-a"


def test_a_retry_that_fails_again_leaves_the_item_unscored():
    client = _Client()

    def always_errors(*, input=None, output=None, expected_output=None, metadata=None,
                      evaluations=None, **_):
        return Evaluation(name=ev.JUDGE_ERROR, value=1.0, data_type="NUMERIC",
                          comment="unscored — no response within 240s")

    items = [_item("item-a")]
    run = run_linked(client, dataset_name="suite2-adversarial-v1", run_name="r1",
                     items=items, task=_task_for({"item-a": "trace-a"}),
                     evaluators=[_assertions], composite=always_errors,
                     max_concurrency=1)
    before = len(client.scores)

    stats = retry_unscored(client, items=items, outcomes=run.item_results,
                           evaluators=[_assertions], composite=always_errors)

    assert stats["unscored"] == 1
    assert len(client.scores) == before, "a second failure writes nothing new"


def test_a_composite_that_returns_a_list_is_flattened():
    """`make_judge_composite` returns the dimensions AND the verdict, not one Evaluation.

    The 22:30 run on 2026-10-21 died on its first item inside this loop: the composite's
    list was appended as a single object and then read for `.name`. Every test here used a
    single-Evaluation composite, so none of them could see it. Both shapes are real — the
    SDK's own `composite_evaluator` accepts either — so both are asserted.
    """
    client = _Client()

    def list_composite(*, input=None, output=None, expected_output=None, metadata=None,
                       evaluations=None, **_):
        return [Evaluation(name="fidelity", value=3.0, data_type="NUMERIC"),
                Evaluation(name=ev.VERDICT, value=1.0, comment="pass",
                           data_type="NUMERIC")]

    run = run_linked(client, dataset_name="suite2-adversarial-v1", run_name="r1",
                     items=[_item("item-a")], task=_task_for({"item-a": "trace-a"}),
                     evaluators=[_assertions], composite=list_composite,
                     max_concurrency=1)

    names = [row["name"] for row in client.scores]
    assert names == [ev.STRUCTURAL, "fidelity", ev.VERDICT]
    assert run.stats["scored"] == 1 and run.stats["pass_rate"] == 1.0


def test_a_two_turn_item_reports_both_trace_ids(capsys):
    """The census is multi-turn; an audit of the first items must see both halves.

    Scores belong to turn 1's trace (the one the dataset run item points at), and turn 2
    is a separate request with its own trace. Printing its id is what makes the audit
    possible without a Langfuse query, and a missing id then shows as a missing field.
    """
    client = _Client()

    def task(*, item, **_):
        return {"turn_1_response": "a", "langfuse_trace_id": "trace-a",
                "turn_2_response": "b", "turn_2_trace_id": "trace-bb"}

    run_linked(client, dataset_name="suite2-adversarial-v1", run_name="r1",
               items=[_item("item-a")], task=task, evaluators=[_assertions],
               composite=_composite, max_concurrency=1)

    out = capsys.readouterr().out
    assert "t2=trace-bb" in out, out
