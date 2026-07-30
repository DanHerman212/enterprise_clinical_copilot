"""Feature source selection.

Defaults to BigQuery — the cheap path — so dev, CI and tests never depend on
provisioned online-serving infrastructure.
"""

from ..config import FEATURE_SOURCE
from .base import FeatureRow, FeatureSource, to_vector

__all__ = ["FeatureRow", "FeatureSource", "to_vector", "get_feature_source"]


def get_feature_source(name: str | None = None) -> FeatureSource:
    """Return the configured feature source. `name` overrides FEATURE_SOURCE."""
    choice = (name or FEATURE_SOURCE).lower()
    if choice == "bigquery":
        from .bigquery_source import BigQueryFeatureSource
        return BigQueryFeatureSource()
    if choice == "feature_store":
        from .feature_store import FeatureStoreFeatureSource
        return FeatureStoreFeatureSource()
    raise ValueError(
        f"Unknown FEATURE_SOURCE {choice!r}; expected 'bigquery' or 'feature_store'"
    )
