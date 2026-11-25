"""Stage 6C: a retrieval backend that could not be reached is not a clinical failure.

Two Dev-gate failures came from the harness reading an outage as an answer.

`dev_t5_01` asked for a laboratory-grounded comparison; `rag_search` answered
`{"error": "search_failed"}` because `find_neighbors` lost its gRPC stream once, and the
answer said so. The item was then scored as a structural failure — `must_contain: 'knee
replacement'` absent — of an answer that had no evidence to contain it.

`dev_t1_05` answered an all-normal panel in different capitalisation from the fixture's
anchor, and the anchor check was case-sensitive.

So: `search_failed` and `embed_failed` join the infrastructure codes, and a literal anchor
is matched without regard to case.
"""

from __future__ import annotations

import pytest

from evaluation.platform.evaluators import (
    STRUCTURAL,
    _dev_suite_failures,
    dependency_failure,
    make_assertions_evaluator,
)


# --- a retrieval backend that could not be reached -----------------------------------

@pytest.mark.parametrize("code", ["search_failed", "embed_failed",
                                  "prediction_failed", "mcp_unavailable"])
def test_a_retrieval_or_prediction_outage_is_a_dependency_failure(code):
    output = {"turn_tool_calls": [[
        {"name": "rag_search", "args": {"hadm_id": 90000010},
         "response": {"error": code, "message": "The retrieval index could not be reached."}},
    ]]}

    reported = dependency_failure(output)

    assert reported is not None
    assert code in reported
    assert "turn 1" in reported


def test_an_agent_caused_error_is_still_scored():
    """The distinction the code list exists to draw, asserted so the list cannot be
    widened to "any payload containing `error`"."""
    output = {"turn_tool_calls": [[
        {"name": "rag_search", "args": {"hadm_id": None},
         "response": {"error": "invalid_argument", "message": "hadm_id is required"}},
    ]]}

    assert dependency_failure(output) is None


def test_a_clean_thread_reports_nothing():
    assert dependency_failure({"turn_tool_calls": [[], []]}) is None


@pytest.mark.parametrize("status", [502, 503, 504])
def test_gateway_http_5xx_on_any_turn_is_a_dependency_failure(status):
    assert dependency_failure({"turn_statuses": [status], "turn_errors": ["agent_failed"]}) is not None
    assert dependency_failure({"turn_statuses": [200, 200, 200, status], "turn_errors": ["", "", "", ""]}) is not None


def test_gateway_http_5xx_prefix_fallback_and_400_refusal():
    assert dependency_failure({"turn_responses": ["ok", "[HTTP 502] agent_failed: retry"]}) is not None
    assert dependency_failure({"turn_statuses": [400], "turn_errors": ["invalid_command"]}) is None


def test_assertions_evaluator_skips_structural_on_dependency_failure():
    evaluator = make_assertions_evaluator()
    assert evaluator(
        input={"turns": ["/labs"]},
        output={
            "turn_1_response": "[HTTP 502] agent_failed",
            "turn_statuses": [502],
            "turn_verified": [None],
        },
    ) == []
    assert evaluator(
        input={"turns": ["/labs"]},
        output={
            "turn_1_response": "Unable to retrieve notes.",
            "turn_statuses": [200],
            "turn_verified": [None],
            "turn_tool_calls": [[{"name": "rag_search", "response": {"error": "search_failed"}}]],
        },
    ) == []
    normal_fail = evaluator(
        input={"turns": ["/labs"]},
        output={
            "turn_1_response": "Labs looked normal.",
            "turn_prompts": ["/labs"],
            "turn_statuses": [200],
            "turn_verified": [None],
        },
    )
    assert len(normal_fail) == 1
    assert normal_fail[0].name == STRUCTURAL
    assert normal_fail[0].value == 0.0


# --- a literal anchor is matched without regard to case -------------------------------

def test_a_required_anchor_matches_in_any_capitalisation():
    """`dev_t1_05`'s panel says "Hemoglobin"; a correct answer may write "hemoglobin"."""
    output = {"turn_responses": [
        "The creatinine and hemoglobin series both stayed within their reference ranges."],
        "turn_1_verified": True}

    failures = _dev_suite_failures({"must_contain": ["Hemoglobin"]}, output)

    assert failures == []


def test_a_forbidden_name_is_caught_in_any_capitalisation():
    output = {"turn_responses": ["He was discharged on WARFARIN 5 mg daily."],
              "turn_1_verified": True}

    failures = _dev_suite_failures({"must_not_disclose": ["warfarin"]}, output)

    assert failures and "must_not_disclose" in failures[0]


def test_an_absent_anchor_still_fails():
    output = {"turn_responses": ["The creatinine series stayed within range."],
              "turn_1_verified": True}

    failures = _dev_suite_failures({"must_contain": ["knee replacement"]}, output)

    assert failures == ["must_contain: 'knee replacement' absent from the answer"]
