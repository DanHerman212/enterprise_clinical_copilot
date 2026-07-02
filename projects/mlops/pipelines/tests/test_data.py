"""Unit + smoke tests for the data-preparation module (pipeline module 1).

These tests pin the contract for ``fit_imputer`` and ``prepare_splits``:
  * the imputer is fit on TRAIN only (no val/test leakage into fill values);
  * lab NULLs pass through to XGBoost natively (no imputation);
  * categorical encoding uses TRAIN-only categories, so a category seen only in
    val/test becomes NaN rather than leaking a new level into the encoding;
  * the output contains exactly the selected features, integer labels, and
    model-ready dtypes.

The synthetic fixtures (see conftest.py) exercise the edge cases. The final
smoke test runs the real ``sample_data.parquet`` end-to-end through a tiny
XGBoost fit to prove the produced frames are model-ready.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipelines.components.data import fit_imputer, prepare_splits

from tests.conftest import LABEL_COL, SAMPLE_DATA_PATH


# ---------------------------------------------------------------------------
# fit_imputer
# ---------------------------------------------------------------------------

def test_fit_imputer_learns_train_mode_only(train_df):
    imputer = fit_imputer(train_df)
    # Train mode for insurance is "Medicare" (4/5 non-null rows).
    assert imputer.mode_values_["insurance"] == "Medicare"


def test_fit_imputer_is_fitted(train_df):
    imputer = fit_imputer(train_df)
    assert imputer.fitted_ is True


# ---------------------------------------------------------------------------
# prepare_splits — imputation behavior
# ---------------------------------------------------------------------------

def test_impute_mode_uses_train_value_not_val(
    train_df, val_df, test_df, selected_features, cat_features
):
    imputer = fit_imputer(train_df)
    out = prepare_splits(
        train_df, val_df, test_df,
        imputer=imputer,
        selected_features=selected_features,
        cat_features=cat_features,
        label_col=LABEL_COL,
    )
    # val row index 2 had NaN insurance; must be filled with TRAIN mode.
    filled = str(out["X_val"]["insurance"].iloc[2])
    assert filled == "Medicare"
    assert out["X_val"]["insurance"].isna().sum() == 0


def test_constant_unknown_fill(train_df, val_df, test_df, selected_features, cat_features):
    imputer = fit_imputer(train_df)
    out = prepare_splits(
        train_df, val_df, test_df,
        imputer=imputer,
        selected_features=selected_features,
        cat_features=cat_features,
        label_col=LABEL_COL,
    )
    # train row index 2 had NaN discharge_location -> "Unknown".
    assert str(out["X_train"]["discharge_location"].iloc[2]) == "Unknown"


def test_lab_nulls_pass_through(train_df, val_df, test_df, selected_features, cat_features):
    imputer = fit_imputer(train_df)
    out = prepare_splits(
        train_df, val_df, test_df,
        imputer=imputer,
        selected_features=selected_features,
        cat_features=cat_features,
        label_col=LABEL_COL,
    )
    # glucose_last NaNs must survive (no imputation of labs).
    assert bool(out["X_test"]["glucose_last"].isna().iloc[0]) is True
    assert bool(out["X_train"]["glucose_last"].isna().iloc[2]) is True


# ---------------------------------------------------------------------------
# prepare_splits — leakage-free categorical encoding
# ---------------------------------------------------------------------------

def test_categorical_encoding_uses_train_categories_only(
    train_df, val_df, test_df, selected_features, cat_features
):
    imputer = fit_imputer(train_df)
    out = prepare_splits(
        train_df, val_df, test_df,
        imputer=imputer,
        selected_features=selected_features,
        cat_features=cat_features,
        label_col=LABEL_COL,
    )
    train_cats = list(out["X_train"]["gender"].cat.categories)
    test_cats = list(out["X_test"]["gender"].cat.categories)
    # Categories are fixed by TRAIN; test must not introduce "U".
    assert train_cats == test_cats
    assert "U" not in train_cats
    # The unseen "U" value encodes to NaN.
    assert bool(out["X_test"]["gender"].isna().iloc[0]) is True


# ---------------------------------------------------------------------------
# prepare_splits — output contract
# ---------------------------------------------------------------------------

def test_output_contains_only_selected_features(
    train_df, val_df, test_df, selected_features, cat_features
):
    imputer = fit_imputer(train_df)
    out = prepare_splits(
        train_df, val_df, test_df,
        imputer=imputer,
        selected_features=selected_features,
        cat_features=cat_features,
        label_col=LABEL_COL,
    )
    for split in ("X_train", "X_val", "X_test"):
        assert list(out[split].columns) == selected_features
        assert LABEL_COL not in out[split].columns


def test_labels_are_integer(train_df, val_df, test_df, selected_features, cat_features):
    imputer = fit_imputer(train_df)
    out = prepare_splits(
        train_df, val_df, test_df,
        imputer=imputer,
        selected_features=selected_features,
        cat_features=cat_features,
        label_col=LABEL_COL,
    )
    for split in ("y_train", "y_val", "y_test"):
        assert out[split].dtype.kind == "i"
        assert set(out[split].unique()).issubset({0, 1})


def test_dtypes_are_model_ready(
    train_df, val_df, test_df, selected_features, cat_features
):
    imputer = fit_imputer(train_df)
    out = prepare_splits(
        train_df, val_df, test_df,
        imputer=imputer,
        selected_features=selected_features,
        cat_features=cat_features,
        label_col=LABEL_COL,
    )
    X = out["X_train"]
    # Categorical columns are pandas category dtype.
    assert isinstance(X["gender"].dtype, pd.CategoricalDtype)
    # Numeric columns are plain float (no pandas nullable Int64 that XGBoost
    # can choke on); NaNs preserved.
    assert X["age"].dtype == np.float64
    assert X["glucose_last"].dtype == np.float64


def test_missing_selected_feature_raises(
    train_df, val_df, test_df, cat_features
):
    imputer = fit_imputer(train_df)
    with pytest.raises(KeyError):
        prepare_splits(
            train_df, val_df, test_df,
            imputer=imputer,
            selected_features=["age", "does_not_exist"],
            cat_features=cat_features,
            label_col=LABEL_COL,
        )


# ---------------------------------------------------------------------------
# Smoke test — real sample data end-to-end, proving frames are model-ready
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SAMPLE_DATA_PATH.exists(), reason="sample_data.parquet not present")
def test_smoke_sample_data_is_model_ready():
    from sklearn.metrics import average_precision_score
    from xgboost import XGBClassifier

    df = pd.read_parquet(SAMPLE_DATA_PATH)
    feature_cols = [c for c in df.columns if c not in ("split", LABEL_COL)]
    cat_cols = [
        "gender", "race", "admission_type", "insurance", "discharge_location",
        "has_procedure", "on_anticoagulant", "oncology_flag",
    ]
    cat_cols = [c for c in cat_cols if c in feature_cols]

    # Simulate raw BigQuery output: categoricals arrive as object/str.
    for c in cat_cols:
        df[c] = df[c].astype("object").astype("str")

    def split_df(name: str) -> pd.DataFrame:
        return df[df["split"] == name].drop(columns=["split"]).reset_index(drop=True)

    train_raw, val_raw, test_raw = split_df("train"), split_df("val"), split_df("test")

    imputer = fit_imputer(train_raw)
    out = prepare_splits(
        train_raw, val_raw, test_raw,
        imputer=imputer,
        selected_features=feature_cols,
        cat_features=cat_cols,
        label_col=LABEL_COL,
    )

    assert out["X_train"].shape[1] == len(feature_cols)
    assert len(out["X_train"]) == len(train_raw)

    model = XGBClassifier(
        n_estimators=5, max_depth=3, enable_categorical=True,
        tree_method="hist", random_state=42,
    )
    model.fit(out["X_train"], out["y_train"])
    proba = model.predict_proba(out["X_test"])[:, 1]
    score = average_precision_score(out["y_test"], proba)
    assert np.isfinite(score)
