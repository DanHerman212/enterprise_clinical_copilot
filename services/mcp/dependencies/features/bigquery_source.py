"""BigQuery feature source — the default for dev, CI and tests.

Lifted from `mlops/scripts/smoke_test.py`, which is proven against the
live endpoint. Point lookups run ~1-2s and cost effectively nothing.
"""

from google.cloud import bigquery

# Two levels below services/mcp/ (dependencies/features/), so the config import
# needs three dots: `..config` resolved to services.mcp.dependencies.config, which
# does not exist, and that made the whole predict path unimportable.
from ...config import ENTITY_ID_COLUMN, PROJECT, TABLE
from .base import FeatureRow
from .manifest import feature_order


class BigQueryFeatureSource:
    """Reads one admission's features straight from the encoded dataset."""

    def __init__(self, project: str = PROJECT, table: str = TABLE) -> None:
        self._client = bigquery.Client(project=project)
        self._table = table

    def exists(self, hadm_id: int) -> bool:
        """One row, one column: is this admission in the dataset we serve?"""
        query = (
            f"SELECT 1 FROM {self._table} WHERE {ENTITY_ID_COLUMN} = @hid LIMIT 1"
        )
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("hid", "INT64", hadm_id)]
        )
        return bool(list(self._client.query(query, job_config=job_config).result()))

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
        #
        # A column the table does not carry is left out rather than filled in with
        # None. The two are not the same thing, and treating them as one is what
        # made the tool's completeness check unreachable: a null VALUE is a
        # measurement that was not taken, which the booster reads as NaN by design;
        # an ABSENT COLUMN is a table that does not speak the model's vocabulary,
        # and scoring it produced a confident answer to a question the model was
        # never asked. Left out, the check in `tools/prediction.py` refuses the row.
        expected = set(feature_order())
        return {
            col: (None if value is None else float(value))
            for col, value in row.items()
            if col in expected
        }
