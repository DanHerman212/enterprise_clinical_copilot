"""A backend that could not be reached is a measurement the run did not obtain.

`dev_t4_03` on 2026-11-04 asked `predict_readmission` for an estimate, was told
`{"error": "prediction_failed"}`, and the `/ehr` answer correctly omitted the risk section
it had no estimate for. The heading check then failed the item — for a heading the
evidence could not support — and a degraded deployment entered the pass-rate denominator
as a clinical failure.

Two rules, and the second is what keeps the first honest:

  * an infrastructure error makes the item **unscored**, not failed;
  * an error the agent caused (a bad argument) is scored normally, because a harness that
    voids every payload containing `error` cannot see the agent misusing a tool.
"""

from __future__ import annotations

from evaluation.platform import diagnostics
from evaluation.platform.evaluators import (
    DEPENDENCY_ERROR,
    make_judge_composite,
)

#: `dev_t4_03` turn 1, verbatim: the call the answer was built on.
PREDICTION_FAILED = {
    "turn_responses": ["**Presentation & diagnosis.** …"],
    "turn_prompts": ["/ehr"],
    "turn_tool_calls": [[
        {"name": "predict_readmission", "args": {"hadm_id": 90000015},
         "response": {"hadm_id": 90000015, "error": "prediction_failed",
                      "message": "The prediction service could not be reached. Try again "
                                 "shortly.",
                      "feature_source": "bigquery"}},
    ]],
    "turn_sources": [[]],
    "turn_verified": [True],
}

JUDGE_OK = {"verdict": "PASS", "reasoning": "Fine.",
            "dimensions": {"fidelity": 3, "grounding": 3, "citation": 3, "safety": 3,
                           "responsiveness": 3, "context": 3}}


class _Structural:
    name = "structural"
    value = 1.0


def _composite(judge_calls: list) -> object:
    def judge_fn(**kwargs):
        judge_calls.append(kwargs)
        return JUDGE_OK

    return make_judge_composite(judge_fn)


def test_an_infrastructure_error_makes_the_item_unscored():
    calls: list = []
    produced = _composite(calls)(
        input={"expectations": {}}, output=PREDICTION_FAILED,
        expected_output=None, evaluations=[_Structural()],
    )

    names = {e.name: e for e in produced}
    assert DEPENDENCY_ERROR in names
    assert "verdict" not in names, "a dependency error must not become a verdict"
    assert "predict_readmission" in str(names[DEPENDENCY_ERROR].comment)
    assert "prediction_failed" in str(names[DEPENDENCY_ERROR].comment)
    assert calls == [], "the judge must not be asked to grade an unmeasured turn"


def test_an_agent_validation_error_is_scored_normally():
    """A bad argument is the agent's mistake, not an outage."""
    output = dict(PREDICTION_FAILED)
    output["turn_tool_calls"] = [[
        {"name": "predict_readmission", "args": {"hadm_id": None},
         "response": {"error": "invalid_argument", "message": "hadm_id is required"}},
    ]]

    calls: list = []
    produced = _composite(calls)(
        input={"expectations": {}}, output=output,
        expected_output=None, evaluations=[_Structural()],
    )

    names = {e.name for e in produced}
    assert DEPENDENCY_ERROR not in names
    assert "verdict" in names
    assert len(calls) == 1, "an agent-caused error still gets judged"


def test_a_clean_run_is_untouched():
    output = {"turn_responses": ["An answer."], "turn_tool_calls": [[]],
              "turn_sources": [[]], "turn_verified": [True]}
    calls: list = []
    produced = _composite(calls)(
        input={"expectations": {}}, output=output,
        expected_output=None, evaluations=[_Structural()],
    )
    assert {e.name for e in produced}.issuperset({"verdict", "fidelity"})


def test_the_diagnostics_turn_record_carries_verified():
    """The flag a structural assertion reads must be readable after the run.

    `dev_t4_03`'s failure could not be explained from the record because the per-turn
    `verified` was never stored, and the first reading of it was a guess.
    """
    turns = diagnostics._turns(
        {"turns": ["/ehr", "/labs"]},
        {"turn_responses": ["A.", "B."], "turn_verified": [True, False]},
    )

    assert [turn["verified"] for turn in turns] == [True, False]


def test_a_turn_with_no_verified_flag_records_none_rather_than_false():
    turns = diagnostics._turns({"turns": ["/ehr"]}, {"turn_responses": ["A."]})
    assert turns[0]["verified"] is None
