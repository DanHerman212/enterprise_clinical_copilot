"""Feature Store online source — low latency, for live demos.

Reads the same encoded dataset, but served from a Feature Store online store
(~50ms instead of BigQuery's ~1-2s). Provision it with
`scripts/setup_feature_store.py`; remove it with `scripts/teardown.py`, since a
provisioned online store bills continuously.
"""

from google.cloud import aiplatform_v1

from ..config import (
    API_ENDPOINT,
    FEATURE_VIEW_ID,
    LOCATION,
    ONLINE_STORE_ID,
    PROJECT,
)
from .base import FeatureRow
from .manifest import feature_order


def feature_view_path() -> str:
    return (
        f"projects/{PROJECT}/locations/{LOCATION}"
        f"/featureOnlineStores/{ONLINE_STORE_ID}/featureViews/{FEATURE_VIEW_ID}"
    )


def _scalar(feature_value) -> float | None:
    """Unwrap a FeatureValue oneof into a float, or None if unset/non-numeric."""
    kind = feature_value._pb.WhichOneof("value")
    if kind is None:
        return None
    raw = getattr(feature_value, kind)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


class FeatureStoreFeatureSource:
    """Online lookup keyed by hadm_id."""

    def __init__(self, feature_view: str | None = None) -> None:
        self._client = aiplatform_v1.FeatureOnlineStoreServiceClient(
            client_options={"api_endpoint": API_ENDPOINT}
        )
        self._feature_view = feature_view or feature_view_path()

    def fetch(self, hadm_id: int) -> FeatureRow:
        request = aiplatform_v1.FetchFeatureValuesRequest(
            feature_view=self._feature_view,
            # The entity key is a string even though hadm_id is INT64 in BigQuery.
            data_key=aiplatform_v1.FeatureViewDataKey(key=str(hadm_id)),
            data_format=aiplatform_v1.FeatureViewDataFormat.KEY_VALUE,
        )
        response = self._client.fetch_feature_values(request=request)
        pairs = response.key_values.features
        if not pairs:
            raise KeyError(f"No online feature values for hadm_id={hadm_id}")

        values = {pair.name: _scalar(pair.value) for pair in pairs}
        # Same restriction as the BigQuery source: model features only, and every
        # one of them present so a missing column reads as null rather than
        # shifting the vector.
        return {col: values.get(col) for col in feature_order()}
