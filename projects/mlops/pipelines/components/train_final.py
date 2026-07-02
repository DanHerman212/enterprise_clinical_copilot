"""
train_final — Train the final XGBoost on the COMBINED train+val set.

Uses the hyperparameters selected during HPO to refit on train+val (so the
production model sees all pre-test data). The returned metric is the
combined-set FIT metric — useful only for logging / overfit sanity, NOT an
unbiased estimate. The honest generalization estimate comes from HPO
(validation), and the final unbiased metric is computed on the untouched
hold-out test set by evaluate_test.
"""

import json
from typing import NamedTuple

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier
from kfp import dsl
from ._image import TRAINING_IMAGE


def run_train_final(
    *,
    x_train_path: str,
    y_train_path: str,
    x_val_path: str,
    y_val_path: str,
    best_params_path: str,
    cat_features: list[str],
    model_artifact_path: str,
) -> float:
    """Refit on train+val with best params, save model, return train-fit AUCPR."""
    X_train = pd.read_parquet(x_train_path)
    y_train = pd.read_parquet(y_train_path).iloc[:, 0]
    X_val = pd.read_parquet(x_val_path)
    y_val = pd.read_parquet(y_val_path).iloc[:, 0]

    X_all = pd.concat([X_train, X_val], ignore_index=True)
    y_all = pd.concat([y_train, y_val], ignore_index=True)

    # Categorical encoding is already applied upstream by load_data and preserved
    # through parquet (category dtype), so we do NOT re-encode here — re-deriving
    # categories could drift from the training/serving schema. cat_features is
    # accepted for interface stability but intentionally not used for encoding.
    _ = cat_features

    with open(best_params_path) as f:
        best_params = json.load(f)

    model = XGBClassifier(**best_params)
    model.fit(X_all, y_all, verbose=False)

    # Fit metric on the combined training set. This is NOT an unbiased estimate
    # (the model trained on these rows); it is logged only as an overfit sanity
    # signal. Unbiased performance is measured on the hold-out test set.
    train_aucpr = float(average_precision_score(y_all, model.predict_proba(X_all)[:, 1]))
    print(f"  Final model trained on {len(X_all):,} combined train+val rows.")
    print(f"  Combined-set fit AUCPR (not unbiased): {train_aucpr:.4f}")
    print(f"  Params: {json.dumps(best_params)}")

    joblib.dump(model, model_artifact_path)
    return train_aucpr


@dsl.component(
    base_image=TRAINING_IMAGE,
    packages_to_install=["xgboost", "scikit-learn", "pandas", "pyarrow", "joblib"],
)
def train_final(
    x_train: dsl.Input[dsl.Dataset],
    y_train: dsl.Input[dsl.Dataset],
    x_val: dsl.Input[dsl.Dataset],
    y_val: dsl.Input[dsl.Dataset],
    best_params: dsl.Input[dsl.Artifact],
    cat_features: list,
    model_artifact: dsl.Output[dsl.Model],
) -> float:
    """KFP component: train final XGBoost with best HPO params."""
    return run_train_final(
        x_train_path=x_train.path, y_train_path=y_train.path,
        x_val_path=x_val.path, y_val_path=y_val.path,
        best_params_path=best_params.path, cat_features=cat_features,
        model_artifact_path=model_artifact.path,
    )
