"""The evaluation harness must report what the safety nets did, not only what passed.

Three facts about a settled answer were published and then dropped on the way into the
report: which guardrails fired on the turn, how many medication rows the grid withheld,
and whether a drawn comparison came from an extraction pass that did not finish. A run
could therefore score sixteen answers without being able to say whether any of them was
the answer the model wrote.

These tests pin the three reads (from the response the product publishes and from the
surface it draws), their per-turn carriers on `TaskOutput`, their place in the turn-by-turn
record, and the run-level totals a report quotes.
"""

from __future__ import annotations

from types import SimpleNamespace

from evaluation.platform import diagnostics, run_evals, task, trace_export
from evaluation.platform.schemas import TaskOutput


def _response(*, flags=None, a2ui=None) -> dict:
    return {"guardrail_flags": flags or [], "a2ui": a2ui}


def _surface(*components: dict) -> dict:
    return {"updateComponents": {"components": list(components)}}


def _observation(metadata: dict) -> SimpleNamespace:
    return SimpleNamespace(metadata=metadata)


def _trace(observations: list) -> SimpleNamespace:
    return SimpleNamespace(observations=observations)


# --- reading the three facts off one turn -------------------------------------------


def test_the_flags_are_read_rather_than_recomputed():
    assert task._guardrail_flags(_response(flags=["dose_rewritten"])) == [
        "dose_rewritten"]


def test_a_turn_that_published_no_flags_reports_none():
    """An absent field and a clean turn read the same, which is what the contract means."""
    assert task._guardrail_flags({}) == []


def test_the_withheld_rows_are_counted_off_the_grid():
    """The grid carries the count because the withheld rows are not drawn.

    A list with a hole in it reads as a complete list unless something on the surface says
    otherwise, and the count is that something.
    """
    response = _response(a2ui=_surface({
        "component": "MedicationGrid", "dropped_count": 2,
        "medications": [{"medication": "metformin"}],
    }))

    assert task._meds_dropped(response) == 2


def test_a_turn_with_no_table_withholds_nothing():
    response = _response(a2ui=_surface({"component": "RiskSummaryCard"}))

    assert task._meds_dropped(response) == 0


def test_a_degraded_comparison_is_read_from_the_turn_s_trace():
    """How the comparison pass ended exists on the trace and nowhere on the wire.

    The surface carries the sentence the reader sees, and the sentence is not a value a
    harness can assert on. The trace carries the engine's own word for it.
    """
    trace = _trace([
        _observation({"guardrail_flags": [],
                      "comparator_outcome": "extraction_degraded",
                      "meds_dropped_count": 2}),
    ])

    assert trace_export.signal_metadata(trace) == {
        "guardrail_flags": [],
        "comparator_outcome": "extraction_degraded",
        "meds_dropped_count": 2,
    }


def test_a_comparison_that_read_its_abstracts_is_not_degraded():
    trace = _trace([_observation({"comparator_outcome": "verified_empty"})])

    assert trace_export.signal_metadata(trace)["comparator_outcome"] == "verified_empty"


def test_a_turn_that_drew_no_table_is_not_degraded():
    """The question is asked of the trace, and a trace with no signal says nothing.

    `signal_metadata` returns an empty mapping rather than a dict of zeros, so "this turn
    recorded nothing" stays distinguishable from "this turn recorded a clean outcome".
    """
    trace = _trace([_observation({"presentation": {}})])

    assert trace_export.signal_metadata(trace) == {}


# --- the carriers on the item -------------------------------------------------------


def test_the_task_output_declares_the_two_wire_borne_carriers_per_turn():
    output = TaskOutput(
        turn_1_response="Answer.",
        turn_guardrail_flags=[["dose_rewritten"], []],
        turn_meds_dropped=[2, 0],
    )

    assert output.turn_guardrail_flags == [["dose_rewritten"], []]
    assert output.turn_meds_dropped == [2, 0]


def test_the_recorded_turns_carry_the_signals_and_do_not_shift_numbering():
    """The turn-by-turn record is what a post-mortem reads, so the facts sit on the turn."""
    turns = diagnostics._turns(
        {"turns": ["How is she?", "And her meds?"]},
        {
            "turn_responses": ["First.", "Second."],
            "turn_guardrail_flags": [[], ["dose_rewritten"]],
            "turn_meds_dropped": [0, 3],
        },
    )

    assert [entry["turn"] for entry in turns] == [1, 2]
    assert turns[0]["guardrail_flags"] == []
    assert turns[1]["guardrail_flags"] == ["dose_rewritten"]
    assert turns[1]["meds_dropped"] == 3


def test_a_run_that_pre_dates_the_carriers_still_records_its_turns():
    """Older runs and replayed items have none of these lists."""
    turns = diagnostics._turns({"turns": ["How is she?"]},
                               {"turn_responses": ["First."]})

    (entry,) = turns
    assert entry["guardrail_flags"] == []
    assert entry["meds_dropped"] == 0


# --- the run totals -----------------------------------------------------------------


def _outcome(output: dict) -> SimpleNamespace:
    return SimpleNamespace(output=output)


def test_the_totals_count_turns_corrected_rather_than_items_affected():
    """Two corrected turns in one item is two, because the fix is per answer."""
    totals = run_evals._signal_totals([
        _outcome({"turn_guardrail_flags": [["dose_rewritten"], ["chart_synthesis_without"
                                                               "_citations"]]}),
        _outcome({"turn_guardrail_flags": [[]]}),
    ])

    assert totals["guardrail_rescued_turns"] == 2
    assert totals["guardrail_flags"] == 2


def test_the_distinct_flag_names_are_reported_once_each():
    """"One flag on fourteen turns" and "four flags on one turn" are different findings."""
    totals = run_evals._signal_totals([
        _outcome({"turn_guardrail_flags": [["dose_rewritten"], ["dose_rewritten"]]}),
    ])

    assert totals["guardrail_flags"] == 2
    assert totals["guardrail_flag_names"] == ["dose_rewritten"]


def test_the_totals_sum_the_withheld_rows_and_count_degraded_tables_from_the_traces():
    totals = run_evals._signal_totals(
        [
            _outcome({"turn_meds_dropped": [2, 0]}),
            _outcome({"turn_meds_dropped": [1]}),
        ],
        [
            [{"comparator_outcome": "extraction_degraded"},
             {"comparator_outcome": "verified"}],
            [{"comparator_outcome": "verified_empty"}],
        ],
    )

    assert totals["meds_dropped_rows"] == 3
    assert totals["comparisons_degraded"] == 1
    assert totals["turns_with_trace_signals"] == 3


def test_a_clean_run_reports_zeros_and_no_names():
    totals = run_evals._signal_totals(
        [_outcome({"turn_guardrail_flags": [[]], "turn_meds_dropped": [0]})],
        [[{"comparator_outcome": "verified"}]],
    )

    assert totals == {
        "guardrail_rescued_turns": 0,
        "guardrail_flags": 0,
        "guardrail_flag_names": [],
        "meds_dropped_rows": 0,
        "comparisons_degraded": 0,
        "turns_with_trace_signals": 1,
    }


def test_a_run_whose_traces_said_nothing_is_not_reported_as_a_clean_run():
    """Rule: an unmeasured turn is not a turn that behaved.

    A trace that carried no signal metadata — an older revision, or a failed write —
    counts as zero verified turns, so the summary can say "nobody checked" rather than
    printing a clean sheet over evidence it never had.
    """
    totals = run_evals._signal_totals([_outcome({})], [[], []])

    assert totals["turns_with_trace_signals"] == 0
    assert totals["comparisons_degraded"] == 0
