"""The feature-source seam.

The MCP tool takes only an `hadm_id`. Everything about *where* features come
from lives behind this Protocol, so the tool code is identical whether it is
running against BigQuery in CI or Feature Store in a live demo.
"""

from typing import Protocol, runtime_checkable

# One admission's features: column name -> value, with missing values as None.
FeatureRow = dict[str, float | None]


@runtime_checkable
class FeatureSource(Protocol):
    """Fetches one admission's model features.

    Implementations MUST return a dict keyed by the manifest's `feature_order`
    columns — every one of them, with missing values as `None` (which becomes
    JSON null, which the predictor reads as NaN).

    Returning a dict rather than an ordered array is deliberate. Ordering
    happens once, in the caller, using `feature_order` from the manifest. If
    each source ordered its own array, a column mismatch would produce a
    silently wrong prediction instead of an error.
    """

    def fetch(self, hadm_id: int) -> FeatureRow:
        ...


def to_vector(row: FeatureRow, feature_order: list[str]) -> list[float | None]:
    """Order a feature dict into the array the endpoint expects."""
    return [row.get(col) for col in feature_order]
