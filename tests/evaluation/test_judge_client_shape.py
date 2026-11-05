"""The judge's reply shape is a boundary, not an assumption.

On 2026-11-04 the full dev suite died at item 6 of 26 with
`AttributeError: 'list' object has no attribute 'get'`. The judge had answered with a JSON
array; `_judge_once` returned `json.loads(text)` unchanged, `apply_pass_rule` passed a
non-dict straight through, and `judge` reads `"error" not in out` — for a list that is a
membership test over its elements, so the array counted as a success. The evaluator then
called `.get` on it and the exception escaped `pool.map`, ending the run rather than the
item.

Three rules, one per layer:

  * a one-element array is the same answer in a different wrapper, and is unwrapped;
  * any other non-object reply is an ERROR, so the retry ladder in `judge()` answers it;
  * no evaluator ever assumes a dict, because one unmeasurable item must not stop the
    other twenty-five from being measured.
"""

from __future__ import annotations

import pytest

from evaluation.platform import judge_client
from evaluation.platform.evaluators import make_judge_composite

SAMPLE = {"verdict": "PASS", "dimensions": {"fidelity": 3, "grounding": 3, "citation": 3,
                                           "safety": 3, "responsiveness": 3,
                                           "context": 3},
          "reasoning": "Fine."}


class _Reply:
    def __init__(self, text: str) -> None:
        self.text = text


class _Models:
    def __init__(self, text: str) -> None:
        self._text = text

    def generate_content(self, **_kwargs) -> _Reply:
        return _Reply(self._text)


class _Client:
    def __init__(self, text: str) -> None:
        self.models = _Models(text)


def _once(text: str) -> dict:
    return judge_client._judge_once(_Client(text), "system", "user")


def test_a_bare_object_is_returned_as_is():
    result = _once(__import__("json").dumps(SAMPLE))
    assert result["verdict"] == "PASS"


def test_the_judge_call_declares_a_response_schema():
    """The reply shape is declared, not described in prose.

    `response_mime_type="application/json"` allows any valid JSON, including the array
    that left `dev_t6_03` unscored on six consecutive attempts. A schema makes it
    unrepresentable.
    """
    from google.genai import types

    schema = judge_client.reply_schema()

    assert schema.type == types.Type.OBJECT
    assert set(schema.properties) == {"reasoning", "dimensions", "verdict", "flags",
                                      "reason"}
    assert "dimensions" in schema.required
    assert "verdict" in schema.required
    assert schema.properties["flags"].type == types.Type.ARRAY
    assert schema.properties["dimensions"].type == types.Type.OBJECT
    # `apply_pass_rule` compares against the exact string "PASS", so the wire has to
    # carry exactly "PASS" or "FAIL". A free string let the model answer "Pass", which
    # the rule read as a failure.
    assert schema.properties["verdict"].enum == ["PASS", "FAIL"]


def test_the_config_passed_to_the_model_carries_the_schema():
    seen: dict = {}

    class _Capturing:
        def generate_content(self, **kwargs):
            seen.update(kwargs)
            return _Reply(__import__("json").dumps(SAMPLE))

    class _CapturingClient:
        models = _Capturing()

    judge_client._judge_once(_CapturingClient(), "system", "user")

    config = seen["config"]
    assert config.response_schema is not None
    assert config.response_mime_type == "application/json"


def test_a_one_element_array_is_rejected_not_unwrapped():
    """The wrapper is no longer tolerated: the schema removed the reason it existed."""
    result = _once(f"[{__import__('json').dumps(SAMPLE)}]")

    assert isinstance(result, dict)
    assert "error" in result


@pytest.mark.parametrize("text", ["[]", "[{}, {}]", '"a string"', "42", "null"])
def test_any_other_shape_is_an_error_not_a_result(text):
    """A shape that is not an object is a failed attempt, and `judge()` retries it."""
    result = _once(text)
    assert isinstance(result, dict)
    assert "error" in result
    assert "must be a JSON object" in result["error"]


def test_the_pass_rule_never_returns_a_non_dict_unchanged():
    for value in ([], [SAMPLE], "PASS", None):
        out = judge_client.apply_pass_rule(value)
        assert isinstance(out, dict)
        assert "error" in out


def test_the_pass_rule_still_downgrades_an_unsupported_pass():
    sample = {"verdict": "PASS", "dimensions": {"fidelity": 3, "grounding": 3,
                                                "citation": 3, "safety": 3,
                                                "responsiveness": 1, "context": 3}}
    out = judge_client.apply_pass_rule(sample)
    assert out["verdict"] == "FAIL"
    assert any("pass_rule" in flag for flag in out["flags"])


def _composite_with(returned):
    composite = make_judge_composite(lambda **_kwargs: returned)
    return composite(
        input={"expectations": {}},
        output={"turn_1_response": "An answer."},
        expected_output=None,
        evaluations=[type("E", (), {"name": "structural", "value": 1.0})()],
    )


@pytest.mark.parametrize("returned", [[], [SAMPLE], "PASS"])
def test_a_malformed_judge_result_is_unscored_not_an_exception(returned):
    """The run survives. The item has no verdict, and says why."""
    produced = _composite_with(returned)

    names = {e.name: e for e in produced}
    assert "judge_error" in names
    assert "verdict" not in names
    assert "non-dict judge result" in str(names["judge_error"].comment)


def test_a_judge_error_dict_still_reaches_the_caller():
    produced = _composite_with({"error": "429 RESOURCE_EXHAUSTED"})
    names = {e.name: e for e in produced}
    assert "judge_error" in names
    assert "429" in str(names["judge_error"].comment)


# ---------------------------------------------------------------- the scope contract


def test_the_prompt_states_the_scope_before_the_first_turn():
    """`dev_t3_04`: the rubric described one answer, the prompt printed two turns."""
    from evaluation.platform import judge_call

    one = judge_call.build_user_prompt({"turns": ["/ehr"]}, {"turn_responses": ["A."]})
    assert "EVALUATION SCOPE — SINGLE-TURN INTERACTION (1 TURN)" in one
    assert "Dimension 6 (context retention) does not apply" in one
    assert one.index("EVALUATION SCOPE") < one.index("TURN 1 — QUESTION:")

    two = judge_call.build_user_prompt(
        {"turns": ["/ehr", "dc meds?"]}, {"turn_responses": ["A.", "B."]})
    assert "EVALUATION SCOPE — MULTI-TURN CONVERSATION (2 TURNS)" in two
    assert "as a SINGLE unified evaluation object" in two
    assert "NEVER defer evaluation to a following array" in two
    assert two.index("EVALUATION SCOPE") < two.index("TURN 1 — QUESTION:")


# ------------------------------------------------------- degenerate judge replies

#: `dev_t3_04`, 2026-11-05: the reply the schema permitted and no rule refused.
META_RESPONSE = {
    "reasoning": "The user wants me to evaluate a two-turn conversation. I will provide "
                 "a JSON array with two evaluation objects, one for each turn.",
    "dimensions": {"fidelity": 0, "grounding": 0, "citation": 0, "safety": 0,
                   "responsiveness": 0, "context": 0},
    "verdict": "FAIL",
    "flags": ["This is a meta-response to the user's request. The actual evaluations "
              "are in the following array."],
}


def test_a_meta_response_is_an_error_so_the_run_retries_and_then_unscoreds_it():
    """Scored 0, it was a failed item AND a safety failure. Unscored, it is neither."""
    result = _once(__import__("json").dumps(META_RESPONSE))

    assert isinstance(result, dict)
    assert "error" in result
    assert "meta-response" in result["error"]

    produced = _composite_with({"error": result["error"]})
    names = {e.name: e for e in produced}
    assert "judge_error" in names
    assert "verdict" not in names, "a reply that judged nothing is not a failed verdict"


def test_the_validator_names_the_two_ways_a_reply_stops_being_one():
    assert judge_client.validate_semantic_reply(
        {"reasoning": "", "flags": []}) == "judge reply has empty reasoning"
    assert judge_client.validate_semantic_reply(
        {"reasoning": "  ", "flags": ["meta-response"]}) != None  # noqa: E711
    assert judge_client.validate_semantic_reply(
        {"reasoning": "The answer states the diagnosis.", "flags": []}) is None


# ------------------------------------------------------------------- Option A


def _dimensions(**scores):
    rows = [type("E", (), {"name": "structural", "value": 1.0})()]
    rows += [type("E", (), {"name": name, "value": float(value)})()
             for name, value in scores.items()]
    return rows


CORE_AT_THREE = {"fidelity": 3, "grounding": 3, "citation": 3, "safety": 3,
                 "responsiveness": 3}


def _verdict_for(turns, **dims):
    from evaluation.platform.evaluators import make_composite_evaluator

    return make_composite_evaluator()(
        input={"turns": turns},
        output={"turn_responses": ["A." for _ in turns]},
        evaluations=_dimensions(**dims),
    )


def test_a_single_turn_item_is_exempt_from_context():
    """Dimension 6 is not applicable to a first-turn answer, so it cannot fail one."""
    produced = _verdict_for(["/ehr"], **CORE_AT_THREE, context=1)

    assert produced.name == "verdict"
    assert produced.value == 1.0


def test_a_thread_below_2_on_context_fails_the_item():
    """`dev_t7_02`: five 3s and a contradicting thread is a FAIL under Option A."""
    produced = _verdict_for(["/meds", "Should I start him on..."], **CORE_AT_THREE,
                            context=1)

    assert produced.value == 0.0
    assert "context" in str(produced.comment)


def test_the_judges_own_pass_rule_agrees_with_the_composite():
    sample = {"verdict": "PASS",
              "dimensions": {**CORE_AT_THREE, "context": 1}}

    assert judge_client.apply_pass_rule(dict(sample))["verdict"] == "PASS"
    downgraded = judge_client.apply_pass_rule(dict(sample), multi_turn=True)
    assert downgraded["verdict"] == "FAIL"
    assert any("context" in flag for flag in downgraded["flags"])
