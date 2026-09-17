"""Post-deploy verification: does the endpoint answer, as the model we deployed?

`deploy_cpr.py` shipped without this. The 0% traffic stage proved only that the
container becomes ready, and the script's last act was to shift all the traffic
and retire the previous deployment — so a deployment that did not serve was
indistinguishable from one that worked until somebody used the demo.

The check is deliberately narrow, because a narrow check can be trusted: it
proves that what we just deployed loads, runs, and answers **as itself**. It does
not judge whether the numbers are clinically sensible — that is what the demo
itself tests — and it does not need a real patient, because the CPR treats a null
value as a missing measurement by design.

The names it sends come from the code-owned feature contract
(``mlops.data.encoding.feature_order``). That is also a check: if the deployed
bundle's manifest disagrees with the contract, the endpoint refuses the request
as an unknown vocabulary, and the deploy fails here rather than during a demo.
"""

from __future__ import annotations

import time
from typing import Any, Callable

# The fields the CPR promises on every prediction. Missing any of them means the
# response is not the contract we think we are serving.
PROMISED_KEYS = ("probability", "prediction", "threshold", "base_value", "top_factors")


class DeploymentCheckFailed(RuntimeError):
    """The endpoint did not answer as the deployment it was handed."""


def check_response(
    pred: Any,
    *,
    deployed_model_id: str,
    expected_dm_id: str,
    expected_version: str,
) -> float:
    """Return the probability, or raise if this answer is not the deployment's.

    Both identities are checked, because they are different authorities on
    different questions: ``deployed_model_id`` is Vertex's record of which
    deployment served the request, and ``model_version`` is the endpoint's own
    report of the provenance record whose bundle it loaded.
    """
    if not isinstance(pred, dict):
        raise DeploymentCheckFailed(f"response is {type(pred).__name__}, not an object")

    missing = [k for k in PROMISED_KEYS if k not in pred]
    if missing:
        raise DeploymentCheckFailed(f"response is missing {missing}")

    probability = pred["probability"]
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise DeploymentCheckFailed(
            f"probability is {type(probability).__name__}, not a number"
        )
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise DeploymentCheckFailed(f"probability {probability} is outside [0, 1]")

    if pred["prediction"] not in (0, 1):
        raise DeploymentCheckFailed(f"prediction is {pred['prediction']!r}, not 0 or 1")
    if not isinstance(pred["top_factors"], list):
        raise DeploymentCheckFailed("top_factors is not a list")

    if deployed_model_id and str(deployed_model_id) != str(expected_dm_id):
        raise DeploymentCheckFailed(
            f"answered by deployment {deployed_model_id}, expected {expected_dm_id}"
            " (a traffic split takes a moment to propagate)"
        )

    reported = pred.get("model_version")
    if expected_version and reported != expected_version:
        raise DeploymentCheckFailed(
            f"the endpoint reported model_version {reported!r}, expected "
            f"{expected_version!r} (stamped at upload; an older deployment "
            "reports nothing)"
        )

    return probability


def verify_deployment(
    predict: Callable[[], Any],
    *,
    expected_dm_id: str,
    expected_version: str,
    attempts: int = 5,
    delay: float = 15.0,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> float:
    """Call ``predict`` until the answer comes from this deployment, or give up.

    Everything is retried, including a wrong identity: a freshly deployed model
    is legitimately slow to serve, and a traffic split takes a moment to
    propagate, so the first request after the shift can still be answered by the
    previous deployment. Retrying the *identity* — rather than only the transport
    — is what keeps this from rolling back a perfectly good deploy.

    Raises ``DeploymentCheckFailed`` with the last reason, so the caller can both
    report why and decide what to restore.
    """
    last_reason = "no attempt was made"
    for attempt in range(1, attempts + 1):
        try:
            response = predict()
            probability = check_response(
                response.predictions[0] if getattr(response, "predictions", None) else None,
                deployed_model_id=getattr(response, "deployed_model_id", ""),
                expected_dm_id=expected_dm_id,
                expected_version=expected_version,
            )
        except DeploymentCheckFailed as exc:
            last_reason = str(exc)
        except Exception as exc:  # cold start, 429, transport, permissions
            last_reason = f"{type(exc).__name__}: {exc}"

        else:
            log(f"    verified on attempt {attempt}/{attempts}: probability {probability:.4f}")
            return probability

        if attempt < attempts:
            log(
                f"    attempt {attempt}/{attempts} not yet right — {last_reason} "
                f"(retrying in {delay:g}s)"
            )
            sleep(delay)

    raise DeploymentCheckFailed(
        f"the endpoint did not answer as deployment {expected_dm_id} after "
        f"{attempts} attempts: {last_reason}"
    )


def verify_or_rollback(
    ep: Any,
    *,
    instance: dict[str, None],
    new_dm_id: str,
    expected_version: str,
    serving_split: dict[str, int],
    verifier: Callable[..., float] = verify_deployment,
    log: Callable[[str], None] = print,
) -> float:
    """Ask the new deployment to answer; if it will not, put the old one back.

    Returns the probability on success. On failure, restores the traffic split
    the endpoint had before this deploy — which is still valid because the
    previous deployment is deliberately not retired until after this returns —
    and exits non-zero, so a broken deploy is a failed command rather than a
    broken endpoint nobody noticed.

    With nothing to restore (a fresh endpoint), it says so rather than pretending
    the endpoint is healthy.
    """
    log("Verifying the new deployment answers as itself …")
    try:
        probability = verifier(
            lambda: ep.predict(instances=[instance]),
            expected_dm_id=new_dm_id,
            expected_version=expected_version,
        )
    except DeploymentCheckFailed as exc:
        log(f"  FAILED: {exc}")
        if serving_split:
            ep.update(traffic_split=serving_split)
            log(f"  Traffic restored to {sorted(serving_split)}; nothing was retired.")
        else:
            log(
                "  There was no previous deployment to restore — the endpoint is "
                "not serving a verified model."
            )
        raise SystemExit(1)

    log(f"  Verified: probability {probability:.4f}")
    return probability
