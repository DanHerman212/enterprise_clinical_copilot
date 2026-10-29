"""Feature source.

BigQuery is the only source. Vertex AI Feature Store was removed on 2026-08-03:
its online store bills ~$0.94/node-hour whether or not anyone is querying it,
which is not defensible for a demo that is idle most of the time. BigQuery reads
the same table with no standing infrastructure.
"""

from .base import FeatureRow, FeatureSource, to_instance, to_vector
from .profile import decode_profile

__all__ = [
    "FeatureRow",
    "FeatureSource",
    "decode_profile",
    "get_feature_source",
    "to_instance",
    "to_vector",
]


def get_feature_source() -> FeatureSource:
    """Return the feature source."""
    from .bigquery_source import BigQueryFeatureSource
    return BigQueryFeatureSource()
