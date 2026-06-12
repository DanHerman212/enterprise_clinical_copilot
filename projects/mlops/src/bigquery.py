"""
BigQuery helper — thin wrapper around google.cloud.bigquery.Client.

Centralises project resolution and query execution so notebooks and
scripts don't each reimplement the .env-reading logic.
"""

from __future__ import annotations

import pandas as pd
from google.cloud import bigquery

from src.config import PROJECT_ID


def get_client() -> bigquery.Client:
    """Return a BigQuery client authorised against the configured project."""
    return bigquery.Client(project=PROJECT_ID)


def run_query(sql: str, *, use_bqstorage: bool = True) -> pd.DataFrame:
    """Execute a BigQuery SQL statement and return results as a DataFrame.

    Parameters
    ----------
    sql : str
        The SQL query to execute.
    use_bqstorage : bool
        When True, use the BigQuery Storage API for faster downloads (requires
        the bigquery-storage extra). Falls back to the standard REST API if the
        library is not installed.
    """
    client = get_client()

    if use_bqstorage:
        try:
            return client.query(sql).result().to_dataframe(
                create_bqstorage_client=True
            )
        except Exception:
            pass  # Fall back to REST API below

    return client.query(sql).result().to_dataframe()


def read_table(
    table_ref: str | None = None,
    *,
    split: str | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read the analytics dataset (or a slice of it) into a DataFrame.

    Parameters
    ----------
    table_ref : str, optional
        Fully-qualified BigQuery table reference. Defaults to the configured
        analytics dataset.
    split : str, optional
        If provided, filter to a single split_name (e.g. 'train').
    columns : list of str, optional
        If provided, SELECT only these columns.
    """
    from src.config import FULL_TABLE_REF, SPLIT_COLUMN

    source = table_ref or FULL_TABLE_REF
    col_list = ", ".join(columns) if columns else "*"

    sql = f"SELECT {col_list} FROM `{source}`"
    if split:
        sql += f" WHERE {SPLIT_COLUMN} = '{split}'"

    return run_query(sql)
