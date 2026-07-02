"""
data — pure, KFP-free data-preparation logic for the training pipeline.

This module deliberately contains **no** ``kfp`` or Google Cloud imports so it
can be unit-tested hermetically. The thin ``@dsl.component`` wrappers live in
``fit_imputer.py`` and ``load_data.py`` and delegate here.

Contract (module 1):
  * The missingness imputer is fit on the TRAIN split only.
  * Lab NULLs are NOT imputed — they pass through to XGBoost, which learns a
    default split direction (see ``artifacts/missingness_policy.csv``: labs are
    ``no_missing``). Only categorical demographics are filled.
  * Categorical features are encoded with TRAIN-only categories, so any level
    seen only in val/test becomes NaN instead of leaking a new level into the
    encoding.
  * Numeric features are coerced to plain ``float64`` (NaN preserved) so the
    pandas nullable ``Int64`` dtype never reaches XGBoost.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.imputer import MissingnessImputer


def fit_imputer(
    train_df: pd.DataFrame,
    *,
    policy_path: str | Path | None = None,
) -> MissingnessImputer:
    """Fit the missingness imputer on the TRAIN split only.

    Returns the fitted imputer so it can be persisted as a reproducible
    pipeline artifact and reused unchanged at inference time.
    """
    imputer = MissingnessImputer(policy_path)
    imputer.fit(train_df)
    return imputer


def train_categories(df: pd.DataFrame, cat_features: list[str]) -> dict[str, list[str]]:
    """Ordered category levels per categorical feature (NaN excluded).

    Order matters: pandas category *codes* depend on the order of the category
    list, so serving must reuse this exact order to reproduce training codes.
    """
    categories: dict[str, list[str]] = {}
    for col in cat_features:
        if col in df.columns:
            categories[col] = [
                level for level in pd.unique(df[col].astype(str)) if level != "nan"
            ]
    return categories


def encode_frame(
    df: pd.DataFrame,
    *,
    feature_order: list[str],
    cat_categories: dict[str, list[str]],
) -> pd.DataFrame:
    """Select features, coerce numerics to float64, encode categoricals.

    This is the single encoding function shared by training (``prepare_splits``)
    and serving (the predictor), so the two can never drift. A categorical
    value absent from ``cat_categories[col]`` becomes NaN (leakage-free), and
    numeric columns become plain float64 with NaN preserved.
    """
    # KeyError here is intentional: a missing feature is a contract violation.
    X = df[feature_order].copy().reset_index(drop=True)

    for col in feature_order:
        if col in cat_categories:
            dtype = pd.CategoricalDtype(categories=cat_categories[col])
            X[col] = X[col].astype(str).astype(dtype)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype("float64")

    return X


def prepare_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    imputer: MissingnessImputer,
    selected_features: list[str],
    cat_features: list[str],
    label_col: str,
) -> dict[str, object]:
    """Impute, select, encode, and return model-ready splits + the schema.

    Parameters
    ----------
    train_df, val_df, test_df
        Raw split frames (as returned by BigQuery): object-dtype categoricals,
        float labs with NaN, and the ``label_col``.
    imputer
        A ``MissingnessImputer`` already fit on TRAIN.
    selected_features
        The exact feature columns to keep, in order.
    cat_features
        Which of ``selected_features`` are categorical.
    label_col
        The binary target column name.

    Returns
    -------
    dict with keys ``X_train, y_train, X_val, y_val, X_test, y_test`` plus
    ``feature_order`` and ``cat_categories`` (the serving schema).
    """
    splits = {"train": train_df, "val": val_df, "test": test_df}
    imputed = {name: imputer.transform(df) for name, df in splits.items()}

    y = {
        name: frame[label_col].astype(int).reset_index(drop=True)
        for name, frame in imputed.items()
    }

    # Categories are fixed by TRAIN only (leakage-free).
    cat_categories = train_categories(imputed["train"], cat_features)

    X = {
        name: encode_frame(
            frame, feature_order=selected_features, cat_categories=cat_categories,
        )
        for name, frame in imputed.items()
    }

    return {
        "X_train": X["train"], "y_train": y["train"],
        "X_val": X["val"], "y_val": y["val"],
        "X_test": X["test"], "y_test": y["test"],
        "feature_order": list(selected_features),
        "cat_categories": cat_categories,
    }
