"""
fit_imputer — fit the missingness imputer on the TRAIN split and persist it.

Making the fitted imputer a first-class pipeline artifact (rather than a
pre-fitted blob loaded from GCS) removes hidden coupling and guarantees the
exact same imputation is reproducible at training and inference time.

Pure fit logic lives in ``data.fit_imputer`` and is unit-tested there; this
module only adds BigQuery IO + artifact persistence.
"""

import joblib
from google.cloud import bigquery
from kfp import dsl

from ._image import TRAINING_IMAGE, component
from .data import fit_imputer


def run_fit_imputer(
    *,
    project_id: str,
    full_table_ref: str,
    split_col: str,
    train_split: str,
    imputer_output_path: str,
) -> None:
    """Query the TRAIN split, fit the imputer, write it to ``imputer_output_path``."""
    client = bigquery.Client(project=project_id)
    sql = f"""
        SELECT *
        FROM `{full_table_ref}`
        WHERE {split_col} = '{train_split}'
    """
    train_df = client.query(sql).result().to_dataframe()

    imputer = fit_imputer(train_df)
    joblib.dump(imputer, imputer_output_path)
    print(f"  Fit imputer on {len(train_df):,} train rows -> {imputer_output_path}")


@component(
    base_image=TRAINING_IMAGE,
    packages_to_install=["google-cloud-bigquery", "pandas", "pyarrow", "joblib"],
)
def fit_imputer_op(
    project_id: str,
    full_table_ref: str,
    split_col: str,
    train_split: str,
    imputer: dsl.Output[dsl.Artifact],
):
    """KFP component: fit the missingness imputer on train, output the artifact."""
    from pipelines.components.fit_imputer import run_fit_imputer

    run_fit_imputer(
        project_id=project_id,
        full_table_ref=full_table_ref,
        split_col=split_col,
        train_split=train_split,
        imputer_output_path=imputer.path,
    )
