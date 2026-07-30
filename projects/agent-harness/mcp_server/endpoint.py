"""Cached handle on the deployed Vertex endpoint.

`smoke_test.py` lists endpoints on every run, which is right for a CLI. In a
long-lived server that would be an API round-trip per tool call, so the lookup
is cached here for the process lifetime.
"""

from functools import lru_cache
from typing import Any

from google.cloud import aiplatform

from .config import ENDPOINT_NAME, LOCATION, PROJECT


@lru_cache(maxsize=1)
def endpoint() -> aiplatform.Endpoint:
    """The readmission endpoint, looked up once per process."""
    aiplatform.init(project=PROJECT, location=LOCATION)
    for ep in aiplatform.Endpoint.list(order_by="create_time desc"):
        if ep.display_name == ENDPOINT_NAME:
            return ep
    raise RuntimeError(
        f"No endpoint with display_name={ENDPOINT_NAME!r} in {PROJECT}/{LOCATION}. "
        "It may have been torn down; redeploy with projects/mlops/scripts/deploy_cpr.py."
    )


def predict_one(features: list[float | None]) -> dict[str, Any]:
    """One instance in, one prediction out.

    Raises rather than returning a partial result if the endpoint answers with
    an empty prediction list — an empty 200 is the kind of silent success that
    is far more expensive to debug than an exception.
    """
    response = endpoint().predict(instances=[features])
    if not response.predictions:
        raise RuntimeError("Endpoint returned a response with no predictions.")
    return response.predictions[0]
