"""
load_data — query the ONE-HOT ENCODED BigQuery view and write model-ready
parquet splits, plus the serving manifest.

The heavy lifting (categorical encoding, missingness policy) now lives in a
static, leakage-free BigQuery view (``analytics_dataset_encoded``, generated
from :mod:`src.encoding`). This component is therefore a plain projection: it
selects the numeric ``feature_order`` columns, splits them into train/val/test
parquet, and emits the ``manifest.json`` serving contract (feature order +
one-hot -> parent group map). No imputer, no in-code encoding — so there is no
train/serve skew.
"""

import json

import pandas as pd
from google.cloud import bigquery
from kfp import dsl

from src import encoding
from ._image import TRAINING_IMAGE, component


def assert_patient_disjoint(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_col: str,
) -> None:
    """Hard-fail if any patient's admissions straddle two splits (ECC-64).

    Everything downstream — the test AUCPR gate, the stability check, the
    fairness audit — assumes patient-level disjointness but only the upstream
    `split_name` column enforces it. If it ever regresses, every gate is
    leakage-contaminated while reporting PASS. One set intersection per pair
    is cheap insurance against that silent failure.
    """
    ids = {
        "train": set(train_df[id_col]),
        "val": set(val_df[id_col]),
        "test": set(test_df[id_col]),
    }
    leaked = {
        f"{a}/{b}": ids[a] & ids[b]
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
        if ids[a] & ids[b]
    }
    if leaked:
        detail = "; ".join(
            f"{pair}: {len(overlap)} shared {id_col}s (e.g. {sorted(overlap)[:3]})"
            for pair, overlap in leaked.items()
        )
        raise ValueError(
            f"Patient leakage across splits — {detail}. The upstream "
            f"'{id_col}' split assignment is broken; every downstream gate "
            "would be contaminated. Refusing to emit training data."
        )


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
    x_train_path: str,
    y_train_path: str,
    x_val_path: str,
    y_val_path: str,
    x_test_path: str,
    y_test_path: str,
    groups_train_path: str,
    groups_val_path: str,
    manifest_path: str,
) -> None:
    """Load the encoded splits, write parquet, and emit the serving manifest.

    ``full_table_ref`` must point at the ONE-HOT ENCODED view. Also emits
    ``groups_train`` (the train-split ``id_col``, e.g. subject_id) so HPO can run
    patient-grouped cross-validation without leaking a patient's admissions
    across folds.
    """
    feature_order = encoding.feature_order()
    client = bigquery.Client(project=project_id)

    cols = ", ".join(feature_order)
    sql = f"""
        SELECT {id_col}, {cols}, {label_col}, {split_col}
        FROM `{full_table_ref}`
        WHERE {split_col} IN ('{train_split}', '{val_split}', '{test_split}')
    """
    df = client.query(sql).result().to_dataframe()

    def _split(name: str) -> pd.DataFrame:
        return df[df[split_col] == name].reset_index(drop=True)

    train_df, val_df, test_df = _split(train_split), _split(val_split), _split(test_split)
    assert_patient_disjoint(train_df, val_df, test_df, id_col)

    def _xy(frame: pd.DataFrame):
        # All feature columns are already numeric; coerce to float64 so NULLs
        # arrive as NaN for XGBoost native missing handling (never nullable Int64).
        X = frame[feature_order].astype("float64")
        y = frame[label_col].astype(int)
        return X, y

    X_train, y_train = _xy(train_df)
    X_val, y_val = _xy(val_df)
    X_test, y_test = _xy(test_df)

    X_train.to_parquet(x_train_path, index=False)
    pd.DataFrame(y_train).to_parquet(y_train_path, index=False)
    X_val.to_parquet(x_val_path, index=False)
    pd.DataFrame(y_val).to_parquet(y_val_path, index=False)
    X_test.to_parquet(x_test_path, index=False)
    pd.DataFrame(y_test).to_parquet(y_test_path, index=False)

    train_df[[id_col]].to_parquet(groups_train_path, index=False)
    val_df[[id_col]].to_parquet(groups_val_path, index=False)

    # Serving contract: feature order (array layout) + one-hot -> parent map for
    # aggregating Sampled Shapley attributions. Single source of truth.
    with open(manifest_path, "w") as f:
        json.dump(encoding.manifest(), f, indent=2)

    print(
        f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape} "
        f"({len(feature_order)} numeric features)"
    )


@component(
    base_image=TRAINING_IMAGE,
    packages_to_install=["google-cloud-bigquery", "pandas", "pyarrow"],
)
def load_data(
    project_id: str,
    full_table_ref: str,
    label_col: str,
    split_col: str,
    train_split: str,
    val_split: str,
    test_split: str,
    x_train: dsl.Output[dsl.Dataset],
    y_train: dsl.Output[dsl.Dataset],
    x_val: dsl.Output[dsl.Dataset],
    y_val: dsl.Output[dsl.Dataset],
    x_test: dsl.Output[dsl.Dataset],
    y_test: dsl.Output[dsl.Dataset],
    groups_train: dsl.Output[dsl.Dataset],
    groups_val: dsl.Output[dsl.Dataset],
    manifest: dsl.Output[dsl.Artifact],
    id_col: str = "subject_id",
):
    """KFP component: load the encoded splits and emit the serving manifest."""
    from pipelines.components.load_data import run_load_data

    run_load_data(
        project_id=project_id, full_table_ref=full_table_ref,
        label_col=label_col, split_col=split_col, id_col=id_col,
        train_split=train_split, val_split=val_split, test_split=test_split,
        x_train_path=x_train.path, y_train_path=y_train.path,
        x_val_path=x_val.path, y_val_path=y_val.path,
        x_test_path=x_test.path, y_test_path=y_test.path,
        groups_train_path=groups_train.path,
        groups_val_path=groups_val.path,
        manifest_path=manifest.path,
    )
