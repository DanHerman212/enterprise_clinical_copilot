"""TF-1, kept after the command it was written for was retired: an outpatient record is
never given a readmission score.

The rule was enforced by `routing.refuses_readmission_score`, which refused to run
`/risk` at all on the five outpatient ids. `/risk` was retired on 2026-10-26, and the
rule survives in the place it now belongs: `supplementary_arguments` drops
`predict_readmission` from a command's second call on an outpatient record, and no
command names the predictor as its PRIMARY tool any more.

So this file asserts the guarantee rather than the mechanism it used to have — no
estimate is requested for an outpatient, whatever command is run — plus the one thing
the retirement must not break: `/ehr` on an outpatient still runs, because a visit with
no readmission still has a note to summarise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from services.agent.routing import encounter, routing

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluation.outpatient_oracle import OUTPATIENT_HADM_IDS  # noqa: E402

OUTPATIENT = sorted(OUTPATIENT_HADM_IDS)
INPATIENT = [90000015, 90000023, 90000106]


def _classify(hadm_id: int) -> None:
    """Record the encounter type the way `admission_context.hydrate` does.

    The runtime reads `encounter_type` off the feature row; these tests are about what
    the command gate does once that fact is known, so they set it directly rather than
    standing up BigQuery. The column itself is verified against this oracle, for all 87
    served records, in `tests/evaluation/test_encounter_column.py`.
    """
    encounter.set_current(
        encounter.OUTPATIENT if hadm_id in OUTPATIENT_HADM_IDS else encounter.INPATIENT
    )


def test_no_command_asks_for_the_estimate_as_its_primary_call():
    """The estimate is a supplementary call, so there is no command whose whole answer is
    a number drawn on an outpatient record. This is what let the refusal predicate go."""
    assert [command.wire for command in routing.COMMANDS
            if command.tool == "predict_readmission"] == []


@pytest.mark.parametrize("hadm_id", OUTPATIENT)
@pytest.mark.parametrize("wire", ["summarize", "literature"])
def test_no_estimate_is_requested_for_an_outpatient_record(wire, hadm_id):
    _classify(hadm_id)
    command = routing.command_for(wire)

    requested = {tool for tool, _args in
                 routing.supplementary_arguments(command, hadm_id, "heart failure")}

    assert "predict_readmission" not in requested


@pytest.mark.parametrize("hadm_id", INPATIENT)
def test_the_estimate_is_still_requested_for_an_inpatient_record(hadm_id):
    """The other direction, so the rule above cannot pass by requesting nothing at all."""
    _classify(hadm_id)
    command = routing.command_for("summarize")

    requested = {tool for tool, _args in
                 routing.supplementary_arguments(command, hadm_id)}

    assert "predict_readmission" in requested


@pytest.mark.parametrize("hadm_id", OUTPATIENT)
def test_ehr_still_runs_on_an_outpatient_record(hadm_id):
    """A visit with no readmission still has a note. Dropping the estimate must not turn
    the whole command into a refusal — that was the failure TF-6 fixed on the other
    side."""
    _classify(hadm_id)
    command = routing.command_for("summarize")

    calls = [(command.tool, routing.tool_arguments(command, hadm_id))]
    calls += routing.supplementary_arguments(command, hadm_id)

    assert command.tool == "rag_search_sections"
    assert [name for name, _args in calls] == ["rag_search_sections"]
