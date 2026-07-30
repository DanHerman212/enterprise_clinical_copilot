"""BigQuery feature source — the default for dev, CI and tests.

Lifted from `projects/mlops/scripts/smoke_test.py`, which is proven against the
live endpoint. Point lookups run ~1-2s and cost effectively nothing.
"""

from google.cloud import bigquery

from ..config import ENTITY_ID_COLUMN, PROJECT, TABLE
from .base import FeatureRow
from .manifest import feature_order


class BigQueryFeatureSource:
    """Reads one admission's features straight from the encoded dataset."""

    def __init__(self, project: str = PROJECT, table: str = TABLE) -> None:
        self._client = bigquery.Client(project=project)
        self._table = table

    def fetch(self, hadm_id: int) -> FeatureRow:
        query = f"SELECT * FROM {self._table} WHERE {ENTITY_ID_COLUMN} = @hid LIMIT 1"
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("hid", "INT64", hadm_id)]
        )
        rows = list(self._client.query(query, job_config=job_config).result())
        if not rows:
            raise KeyError(f"No row found for {ENTITY_ID_COLUMN}={hadm_id}")

        row = dict(rows[0].items())
        # Restrict to the model's features. SELECT * also returns the label and
        # bookkeeping columns; letting those reach the model input would be a
        # silent correctness bug rather than an error.
        return {
            col: (None if row.get(col) is None else float(row[col]))
            for col in feature_order()
        }
