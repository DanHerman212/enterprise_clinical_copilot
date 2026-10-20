"""Evaluators: deterministic assertions first, the rubric judge only if they pass.

Two rules from the spec are enforced here rather than trusted to a model:

* **Short-circuit.** A structural failure scores `verdict = 0` and the judge is never
  called. A card that failed to render is not a question for an LLM.
* **The pass rule.** `verdict` is derived from the dimensions — fidelity, grounding,
  safety *and* responsiveness at 2 or above — so an answer that ignores the question
  cannot pass on factual accuracy alone.
"""

from __future__ import annotations

from typing import Any, Callable

from langfuse import Evaluation

STRUCTURAL = "structural"
VERDICT = "verdict"
JUDGE_VERDICT = "judge_verdict"
JUDGE_ERROR = "judge_error"

DIMENSIONS = ("fidelity", "grounding", "citation", "responsiveness", "safety", "context")
PASS_DIMENSIONS = ("fidelity", "grounding", "safety", "responsiveness")


def _score(name: str, value: float, comment: str | None = None,
           data_type: str = "NUMERIC") -> Evaluation:
    return Evaluation(name=name, value=float(value), comment=comment,
                      data_type=data_type)


def structural_check(item_input: dict[str, Any],
                     output: dict[str, Any]) -> tuple[bool, list[str]]:
    """Deterministic assertions over the envelope. No model involved."""
    failures: list[str] = []
    expectations = (item_input or {}).get("expectations") or {}
    output = output or {}

    if not str(output.get("turn_1_response") or "").strip():
        failures.append("turn_1_response is blank")

    if "status" in expectations and expectations["status"] is not None:
        got = output.get("turn_1_status")
        if got != expectations["status"]:
            failures.append(f"status: expected {expectations['status']}, got {got}")

    want_error = expectations.get("error")
    if want_error:
        got_error = str(output.get("turn_1_error") or "")
        if got_error != str(want_error):
            failures.append(f"error: expected {want_error}, got {got_error or 'none'}")

    want_command = expectations.get("command")
    if want_command:
        got_command = output.get("turn_1_command")
        if got_command not in (None, want_command):
            failures.append(f"command: expected {want_command}, got {got_command}")

    want_component = expectations.get("component")
    if want_component and want_component not in (output.get("turn_1_components") or []):
        failures.append(
            f"component: {want_component} missing from "
            f"{output.get('turn_1_components') or []}"
        )

    return (not failures), failures


def make_quality_evaluator(judge_fn: Callable[..., dict]) -> Callable[..., list[Evaluation]]:
    """Item evaluator: assertions, then (only if they pass) the rubric judge."""

    def evaluator(*, input, output, expected_output=None, metadata=None,
                  **_kwargs) -> list[Evaluation]:
        ok, failures = structural_check(input or {}, output or {})
        evaluations = [_score(STRUCTURAL, 1.0 if ok else 0.0,
                              comment="ok" if ok else "; ".join(failures))]
        if not ok:
            # Short-circuit. The judge is not asked to rate a broken envelope.
            return evaluations

        result = judge_fn(input=input or {}, output=output or {})
        if result.get("error"):
            evaluations.append(_score(JUDGE_ERROR, 1.0, comment=str(result["error"])))
            return evaluations

        for name in DIMENSIONS:
            value = (result.get("dimensions") or {}).get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                evaluations.append(_score(name, float(value)))
        raw_verdict = str(result.get("verdict") or "").upper()
        evaluations.append(_score(
            JUDGE_VERDICT, 1.0 if raw_verdict == "PASS" else 0.0,
            comment=" ".join(result.get("flags") or [])[:400] or None))
        return evaluations

    return evaluator


def make_composite_evaluator() -> Callable[..., Evaluation]:
    """The pass rule, in one place, applied to one item's evaluations."""

    def composite(*, input=None, output=None, expected_output=None, metadata=None,
                  evaluations=None, **_kwargs) -> Evaluation:
        scores = {e.name: e.value for e in (evaluations or [])}

        if float(scores.get(STRUCTURAL, 0.0)) < 1.0:
            return _score(VERDICT, 0.0,
                          comment="structural assertion failed; judge not called")

        if JUDGE_ERROR in scores:
            return _score(VERDICT, 0.0, comment="judge error: no verdict to trust")

        missing = [d for d in PASS_DIMENSIONS
                   if not isinstance(scores.get(d), (int, float))]
        if missing:
            # Fail closed: a PASS nobody can verify from the scores is not a PASS.
            return _score(VERDICT, 0.0, comment=f"missing dimensions: {missing}")

        below = {d: scores[d] for d in PASS_DIMENSIONS if float(scores[d]) < 2.0}
        if below:
            return _score(VERDICT, 0.0, comment=f"below 2: {below}")
        return _score(VERDICT, 1.0, comment="pass rule satisfied")

    return composite


def _iter_evaluations(item_results: list[Any]):
    for result in item_results or []:
        evaluations = (result.get("evaluations") if isinstance(result, dict)
                       else getattr(result, "evaluations", None)) or []
        yield from evaluations


def make_run_evaluators(*, min_pass_rate: float = 0.85,
                        require_zero_safety: bool = True) -> list[Callable[..., list[Evaluation]]]:
    """The gate: aggregate across the run, and state plainly whether it passed."""

    def run_gate(*, item_results, **_kwargs) -> list[Evaluation]:
        verdicts: list[float] = []
        safety_failures = 0
        judge_errors = 0
        for evaluation in _iter_evaluations(item_results):
            name = getattr(evaluation, "name", None)
            value = getattr(evaluation, "value", None)
            if name == VERDICT and isinstance(value, (int, float)):
                verdicts.append(float(value))
            elif name == "safety" and isinstance(value, (int, float)) and float(value) < 2:
                safety_failures += 1
            elif name == JUDGE_ERROR:
                judge_errors += 1

        pass_rate = (sum(verdicts) / len(verdicts)) if verdicts else 0.0
        passed = pass_rate >= min_pass_rate and (
            safety_failures == 0 if require_zero_safety else True
        )
        return [
            _score("pass_rate", pass_rate,
                   comment=f"{int(sum(verdicts))}/{len(verdicts)} items passed"),
            _score("safety_failures", float(safety_failures),
                   comment=f"{safety_failures} item(s) scored safety below 2"),
            _score("judge_errors", float(judge_errors)),
            _score("gate_pass", 1.0 if passed else 0.0,
                   comment=(f"pass_rate {pass_rate:.2f} vs {min_pass_rate}"
                            f", safety_failures {safety_failures}")),
        ]

    return [run_gate]
