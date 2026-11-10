"""Cached handle on the deployed Vertex endpoint.

`smoke_test.py` lists endpoints on every run, which is right for a CLI. In a
long-lived server that would be an API round-trip per tool call, so the lookup
is cached here for the process lifetime.
"""

from functools import lru_cache
import time
from typing import Any

from google.api_core import exceptions as google_exceptions
from google.cloud import aiplatform

from ..config import ENDPOINT_NAME, LOCATION, PROJECT

#: The failures a second call can fix. Each one is the transport or the server ask
#: gone wrong rather than the request: a stream the server dropped, a deadline the
#: backend missed, an internal error on the serving host. A `InvalidArgument` or a
#: `PermissionDenied` is not here, because retrying it fails identically.
TRANSIENT_PREDICT_ERRORS: tuple[type[BaseException], ...] = (
    google_exceptions.ServiceUnavailable,
    google_exceptions.DeadlineExceeded,
    google_exceptions.InternalServerError,
)


@lru_cache(maxsize=1)
def endpoint() -> aiplatform.Endpoint:
    """The readmission endpoint, looked up once per process."""
    aiplatform.init(project=PROJECT, location=LOCATION)
    for ep in aiplatform.Endpoint.list(order_by="create_time desc"):
        if ep.display_name == ENDPOINT_NAME:
            return ep
    raise RuntimeError(
        f"No endpoint with display_name={ENDPOINT_NAME!r} in {PROJECT}/{LOCATION}. "
        "It may have been torn down; redeploy with mlops/serving/deploy_cpr.py."
    )


def predict_one(instance: dict[str, float | None]) -> tuple[dict[str, Any], str]:
    """One named instance in; the prediction, plus who served it, out.

    Vertex returns ``deployed_model_id`` on every prediction and this function
    used to keep only ``predictions[0]``, which is why the caller had to infer
    the model from the registry. That id is the authoritative answer to "which
    model produced this number", and the endpoint's own response is the only
    place it exists.

    Raises rather than returning a partial result if the endpoint answers with
    an empty prediction list — an empty 200 is the kind of silent success that
    is far more expensive to debug than an exception.

    A transient failure is retried twice before it becomes the caller's error. The
    measured case is a broken gRPC stream: on 2026-11-10 an evaluation turn recorded
    ``ServiceUnavailable: 503 Stream removed (recvmsg:Connection reset by peer (104))``
    from this call, and the whole item was left unscored. The stream belongs to the
    cached client channel, so ``endpoint.cache_clear()`` runs before each retry — a
    retry on the same broken channel fails the same way, which is a retry in name only.
    """
    last: BaseException | None = None
    for pause in (0.0, 0.5, 1.5):
        if pause:
            time.sleep(pause)
            endpoint.cache_clear()
        try:
            response = endpoint().predict(instances=[instance])
        except TRANSIENT_PREDICT_ERRORS as exc:
            last = exc
            continue
        if not response.predictions:
            raise RuntimeError("Endpoint returned a response with no predictions.")
        return response.predictions[0], getattr(response, "deployed_model_id", "") or ""
    raise last if last is not None else RuntimeError("Endpoint prediction failed.")
