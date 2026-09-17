"""Gap 3 — a deployment is not finished until the model answers.

`deploy_cpr.py` used to deploy, shift all the traffic and retire the previous
deployment without ever asking the endpoint a question. These tests pin the check
that replaced that: the answer must come from the deployment just made, a
propagation delay must not be mistaken for a broken model, and letting the check
fail is what leaves the previous deployment intact.
"""

import types

import pytest

from mlops.serving.deploy_check import (
    DeploymentCheckFailed,
    check_response,
    verify_deployment,
    verify_or_rollback,
)

_DM = "1234567890"
_VERSION = "readmission-final-20260902014308"


def _prediction(**overrides):
    payload = {
        "probability": 0.31,
        "prediction": 0,
        "threshold": 0.11,
        "base_value": -1.2,
        "top_factors": [{"feature": "age", "attribution": 0.4}],
        "model_version": _VERSION,
    }
    payload.update(overrides)
    return payload


def _response(pred=None, deployed_model_id=_DM):
    return types.SimpleNamespace(
        predictions=[_prediction() if pred is None else pred],
        deployed_model_id=deployed_model_id,
    )


# --- what the check asserts -----------------------------------------------------


def test_a_good_answer_from_the_right_deployment_passes():
    assert check_response(
        _prediction(), deployed_model_id=_DM, expected_dm_id=_DM,
        expected_version=_VERSION,
    ) == pytest.approx(0.31)


def test_an_answer_from_another_deployment_is_refused():
    """The whole point: the traffic split may not have propagated yet."""
    with pytest.raises(DeploymentCheckFailed, match="answered by deployment"):
        check_response(
            _prediction(), deployed_model_id="999", expected_dm_id=_DM,
            expected_version=_VERSION,
        )


def test_an_endpoint_serving_the_wrong_bundle_is_refused():
    with pytest.raises(DeploymentCheckFailed, match="model_version"):
        check_response(
            _prediction(model_version="readmission-final-20260723172647"),
            deployed_model_id=_DM, expected_dm_id=_DM, expected_version=_VERSION,
        )


def test_an_unstamped_endpoint_is_refused_when_a_version_is_expected():
    """A deployment that does not report its bundle cannot be trusted to be it."""
    with pytest.raises(DeploymentCheckFailed, match="model_version"):
        check_response(
            _prediction(model_version=""), deployed_model_id=_DM,
            expected_dm_id=_DM, expected_version=_VERSION,
        )


@pytest.mark.parametrize(
    "broken, reason",
    [
        (_prediction(probability=1.4), "outside"),
        (_prediction(probability="0.3"), "not a number"),
        (_prediction(prediction=2), "not 0 or 1"),
        (_prediction(top_factors={}), "not a list"),
        ({"probability": 0.3}, "missing"),
        ("not a dict", "not an object"),
    ],
)
def test_a_malformed_answer_is_refused(broken, reason):
    with pytest.raises(DeploymentCheckFailed, match=reason):
        check_response(
            broken, deployed_model_id=_DM, expected_dm_id=_DM,
            expected_version=_VERSION,
        )


# --- retry behaviour ------------------------------------------------------------


def test_a_cold_start_is_retried_not_failed():
    """A just-deployed model legitimately fails the first requests."""
    calls = {"n": 0}

    def predict():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("503 model server not ready")
        return _response()

    slept: list[float] = []
    probability = verify_deployment(
        predict, expected_dm_id=_DM, expected_version=_VERSION,
        attempts=5, delay=7.0, sleep=slept.append, log=lambda _: None,
    )

    assert probability == pytest.approx(0.31)
    assert calls["n"] == 3
    assert slept == [7.0, 7.0]


def test_a_slow_traffic_split_shift_is_retried_not_failed():
    """The first request after a shift can still be answered by the old model."""
    answers = [_response(deployed_model_id="999"), _response()]

    def predict():
        return answers.pop(0)

    probability = verify_deployment(
        predict, expected_dm_id=_DM, expected_version=_VERSION,
        attempts=4, delay=0.0, sleep=lambda _: None, log=lambda _: None,
    )
    assert probability == pytest.approx(0.31)


def test_a_deployment_that_never_answers_as_itself_gives_up_and_says_why():
    def predict():
        return _response(deployed_model_id="999")

    with pytest.raises(DeploymentCheckFailed) as excinfo:
        verify_deployment(
            predict, expected_dm_id=_DM, expected_version=_VERSION,
            attempts=3, delay=0.0, sleep=lambda _: None, log=lambda _: None,
        )

    assert _DM in str(excinfo.value)
    assert "answered by deployment 999" in str(excinfo.value)
    assert "3 attempts" in str(excinfo.value)


# --- what happens to the traffic when the check fails ---------------------------


class _FakeEndpoint:
    """Records the calls, so the rollback can be asserted rather than assumed."""

    def __init__(self, response=None):
        self.calls: list[tuple] = []
        self._response = response or _response()

    def predict(self, instances):
        self.calls.append(("predict", instances))
        return self._response

    def update(self, traffic_split):
        self.calls.append(("update", traffic_split))


def test_a_failed_check_puts_the_previous_deployment_back():
    """The old model is still deployed, so restoring it is one call."""
    ep = _FakeEndpoint()

    def refuses(*args, **kwargs):
        raise DeploymentCheckFailed("answered by deployment 999")

    logged: list[str] = []
    with pytest.raises(SystemExit) as exitinfo:
        verify_or_rollback(
            ep, instance={"f1": None}, new_dm_id=_DM, expected_version=_VERSION,
            serving_split={"222": 100}, verifier=refuses, log=logged.append,
        )

    assert exitinfo.value.code == 1
    assert ("update", {"222": 100}) in ep.calls
    assert "nothing was retired" in " ".join(logged)
    assert "FAILED" in " ".join(logged)


def test_a_failed_check_with_nothing_to_restore_does_not_pretend():
    ep = _FakeEndpoint()

    def refuses(*args, **kwargs):
        raise DeploymentCheckFailed("never answered")

    logged: list[str] = []
    with pytest.raises(SystemExit):
        verify_or_rollback(
            ep, instance={"f1": None}, new_dm_id=_DM, expected_version=_VERSION,
            serving_split={}, verifier=refuses, log=logged.append,
        )

    assert [c for c in ep.calls if c[0] == "update"] == []
    assert "not serving a verified model" in " ".join(logged)


def test_a_passing_check_leaves_the_traffic_alone():
    ep = _FakeEndpoint()

    def accepts(predict, *, expected_dm_id, expected_version):
        return 0.42

    probability = verify_or_rollback(
        ep, instance={"f1": None}, new_dm_id=_DM, expected_version=_VERSION,
        serving_split={"222": 100}, verifier=accepts, log=lambda _: None,
    )

    assert probability == pytest.approx(0.42)
    assert [c for c in ep.calls if c[0] == "update"] == []
