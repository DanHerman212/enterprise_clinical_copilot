"""
load_data — query BigQuery, impute (via the pipeline-fit imputer), encode
categoricals with TRAIN-only categories, and write model-ready parquet splits.

The data-preparation logic lives in ``data.prepare_splits`` and is unit-tested
there; this module only adds BigQuery IO, the fitted-imputer artifact input,
and parquet output.
"""

import json

import joblib
import pandas as pd
from google.cloud import bigquery
from kfp import dsl

from ._image import TRAINING_IMAGE, component
from .data import prepare_splits


def run_load_data(
    *,
    project_id: str,
    full_table_ref: str,
    label_col: str,
    split_col: str,
    id_col: str,
    train_split: str,
    val_split: str,
    test_split: str,
    selected_features: list[str],
    cat_features: list[str],
    imputer_path: str,
    x_train_path: str,
    y_train_path: str,
    x_val_path: str,
    y_val_path: str,
    x_test_path: str,
    y_test_path: str,
    groups_train_path: str,
    groups_val_path: str,
    schema_path: str,
) -> None:
    """Load splits, impute + encode via ``prepare_splits``, write parquet.

    Also emits ``groups_train`` (the train-split ``id_col``, e.g. subject_id) so
    HPO can run patient-grouped cross-validation without leaking a patient's
    admissions across folds.
    """
    client = bigquery.Client(project=project_id)

    cols = ", ".join(selected_features)
    sql = f"""
        SELECT {id_col}, {cols}, {label_col}, {split_col}
        FROM `{full_table_ref}`
        WHERE {split_col} IN ('{train_split}', '{val_split}', '{test_split}')
    """
    df = client.query(sql).result().to_dataframe()

    def _split(name: str) -> pd.DataFrame:
        return (
            df[df[split_col] == name]
            .drop(columns=[split_col])
            .reset_index(drop=True)
        )

    train_df, val_df, test_df = _split(train_split), _split(val_split), _split(test_split)

    # Capture the train group key before features are selected out.
    groups_train = train_df[[id_col]].copy()
    groups_val = val_df[[id_col]].copy()

    imputer = joblib.load(imputer_path)

    out = prepare_splits(
        train_df, val_df, test_df,
        imputer=imputer,
        selected_features=selected_features,
        cat_features=cat_features,
        label_col=label_col,
    )

    out["X_train"].to_parquet(x_train_path, index=False)
    pd.DataFrame(out["y_train"]).to_parquet(y_train_path, index=False)
    out["X_val"].to_parquet(x_val_path, index=False)
    pd.DataFrame(out["y_val"]).to_parquet(y_val_path, index=False)
    out["X_test"].to_parquet(x_test_path, index=False)
    pd.DataFrame(out["y_test"]).to_parquet(y_test_path, index=False)
    groups_train.to_parquet(groups_train_path, index=False)
    groups_val.to_parquet(groups_val_path, index=False)

    # Persist the serving schema so training and online inference share the
    # exact same feature order and category encoding.
    with open(schema_path, "w") as f:
        json.dump(
            {
                "feature_order": out["feature_order"],
                "cat_categories": out["cat_categories"],
            },
            f,
            indent=2,
        )
    print(
        f"  Train: {out['X_train'].shape}, "
        f"Val: {out['X_val'].shape}, Test: {out['X_test'].shape}"
    )


@component(
    base_image=TRAINING_IMAGE,
    packages_to_install=["google-cloud-bigquery", "pandas", "pyarrow", "joblib"],
)
def load_data(
    project_id: str,
    full_table_ref: str,
    label_col: str,
    split_col: str,
    train_split: str,
    val_split: str,
    test_split: str,
    selected_features: list,
    cat_features: list,
    imputer: dsl.Input[dsl.Artifact],
    x_train: dsl.Output[dsl.Dataset],
    y_train: dsl.Output[dsl.Dataset],
    x_val: dsl.Output[dsl.Dataset],
    y_val: dsl.Output[dsl.Dataset],
    x_test: dsl.Output[dsl.Dataset],
    y_test: dsl.Output[dsl.Dataset],
    groups_train: dsl.Output[dsl.Dataset],
    groups_val: dsl.Output[dsl.Dataset],
    schema: dsl.Output[dsl.Artifact],
    id_col: str = "subject_id",
):
    """KFP component: load, impute, encode, and write parquet splits."""
    from pipelines.components.load_data import run_load_data

    run_load_data(
        project_id=project_id, full_table_ref=full_table_ref,
        label_col=label_col, split_col=split_col, id_col=id_col,
        train_split=train_split, val_split=val_split, test_split=test_split,
        selected_features=selected_features, cat_features=cat_features,
        imputer_path=imputer.path,
        x_train_path=x_train.path, y_train_path=y_train.path,
        x_val_path=x_val.path, y_val_path=y_val.path,
        x_test_path=x_test.path, y_test_path=y_test.path,
        groups_train_path=groups_train.path,
        groups_val_path=groups_val.path,
        schema_path=schema.path,
    )
