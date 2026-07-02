"""
load_data — Query BigQuery, impute, encode categoricals.
Mirrors feature_selection_v2.ipynb Cell 5.
"""

from __future__ import annotations

import tempfile
from typing import NamedTuple

import joblib
import pandas as pd
from google.cloud import bigquery, storage
from kfp import dsl
from ._image import TRAINING_IMAGE


def run_load_data(
    *,
    project_id: str,
    full_table_ref: str,
    label_col: str,
    split_col: str,
    train_split: str,
    val_split: str,
    test_split: str,
    selected_features: list[str],
    cat_features: list[str],
    imputer_gcs_path: str,
    x_train_path: str,
    y_train_path: str,
    x_val_path: str,
    y_val_path: str,
    x_test_path: str,
    y_test_path: str,
) -> None:
    """Load train/val/test splits, impute, encode, write parquet."""
    client = bigquery.Client(project=project_id)
    storage_client = storage.Client(project=project_id)

    cols = ", ".join(selected_features)
    sql = f"""
        SELECT {cols}, {label_col}, {split_col}
        FROM `{full_table_ref}`
        WHERE {split_col} IN ('{train_split}', '{val_split}', '{test_split}')
    """
    df = client.query(sql).result().to_dataframe()

    train_df = df[df[split_col] == train_split].reset_index(drop=True)
    val_df = df[df[split_col] == val_split].reset_index(drop=True)
    test_df = df[df[split_col] == test_split].reset_index(drop=True)
    for d in (train_df, val_df, test_df):
        d.drop(columns=[split_col], inplace=True)

    # Load fitted imputer from GCS.
    bucket_name, *parts = imputer_gcs_path.replace("gs://", "").split("/", 1)
    blob_path = parts[0] if parts else ""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    tmpdir = tempfile.mkdtemp()
    local_imputer = f"{tmpdir}/imputer.joblib"
    blob.download_to_filename(local_imputer)
    imputer = joblib.load(local_imputer)

    # Impute all three splits.
    train_imp = imputer.transform(train_df)
    val_imp = imputer.transform(val_df)
    test_imp = imputer.transform(test_df)

    y_train = train_imp[label_col].astype(int)
    y_val = val_imp[label_col].astype(int)
    y_test = test_imp[label_col].astype(int)
    X_train = train_imp[selected_features].copy()
    X_val = val_imp[selected_features].copy()
    X_test = test_imp[selected_features].copy()

    # Category dtype for XGBoost native handling.
    for col in cat_features:
        if col in X_train.columns:
            all_cats = pd.concat([
                X_train[col].astype(str), X_val[col].astype(str),
                X_test[col].astype(str),
            ])
            dtype = pd.CategoricalDtype(categories=all_cats.unique())
            X_train[col] = X_train[col].astype(str).astype(dtype)
            X_val[col] = X_val[col].astype(str).astype(dtype)
            X_test[col] = X_test[col].astype(str).astype(dtype)

    X_train.to_parquet(x_train_path, index=False)
    pd.DataFrame(y_train).to_parquet(y_train_path, index=False)
    X_val.to_parquet(x_val_path, index=False)
    pd.DataFrame(y_val).to_parquet(y_val_path, index=False)
    X_test.to_parquet(x_test_path, index=False)
    pd.DataFrame(y_test).to_parquet(y_test_path, index=False)
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")


@dsl.component(
    base_image=TRAINING_IMAGE,
    packages_to_install=["google-cloud-storage", "google-cloud-bigquery", "pandas", "pyarrow", "joblib"],
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
    imputer_gcs_path: str,
    x_train: dsl.OutputPath("Dataset"),
    y_train: dsl.OutputPath("Dataset"),
    x_val: dsl.OutputPath("Dataset"),
    y_val: dsl.OutputPath("Dataset"),
    x_test: dsl.OutputPath("Dataset"),
    y_test: dsl.OutputPath("Dataset"),
):
    """KFP component: load, impute, encode, return parquet paths."""
    run_load_data(
        project_id=project_id, full_table_ref=full_table_ref,
        label_col=label_col, split_col=split_col,
        train_split=train_split, val_split=val_split,
        test_split=test_split,
        selected_features=selected_features, cat_features=cat_features,
        imputer_gcs_path=imputer_gcs_path,
        x_train_path=x_train, y_train_path=y_train,
        x_val_path=x_val, y_val_path=y_val,
        x_test_path=x_test, y_test_path=y_test,
    )
