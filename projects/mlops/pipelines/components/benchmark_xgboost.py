"""
benchmark_xgboost — Train XGBoost with default parameters.
"""

from typing import NamedTuple

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier
from kfp import dsl
from ._image import TRAINING_IMAGE, component


def run_benchmark_xgboost(
    *,
    x_train_path: str,
    y_train_path: str,
    x_val_path: str,
    y_val_path: str,
    xgb_params: dict,
    cat_features: list[str],
    model_artifact_path: str,
) -> float:
    """Train XGBoost, save model, return val AUCPR."""
    X_train = pd.read_parquet(x_train_path)
    y_train = pd.read_parquet(y_train_path).iloc[:, 0]
    X_val = pd.read_parquet(x_val_path)
    y_val = pd.read_parquet(y_val_path).iloc[:, 0]

    for col in cat_features:
        if col in X_train.columns:
            all_cats = pd.concat([
                X_train[col].astype(str), X_val[col].astype(str),
            ])
            dtype = pd.CategoricalDtype(categories=all_cats.unique())
            X_train[col] = X_train[col].astype(str).astype(dtype)
            X_val[col] = X_val[col].astype(str).astype(dtype)

    params = dict(xgb_params)
    params.setdefault("n_jobs", -1)
    params.setdefault("enable_categorical", True)

    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    aucpr = float(average_precision_score(
        y_val, model.predict_proba(X_val)[:, 1],
    ))
    print(f"  Benchmark XGBoost AUCPR: {aucpr:.4f}")
    joblib.dump(model, model_artifact_path)
    return aucpr


@component(
    base_image=TRAINING_IMAGE,
    packages_to_install=["xgboost", "scikit-learn", "pandas", "pyarrow", "joblib"],
)
def benchmark_xgboost(
    x_train: dsl.Input[dsl.Dataset],
    y_train: dsl.Input[dsl.Dataset],
    x_val: dsl.Input[dsl.Dataset],
    y_val: dsl.Input[dsl.Dataset],
    xgb_params: dict,
    cat_features: list,
    model_artifact: dsl.Output[dsl.Model],
) -> float:
    """KFP component: train benchmark XGBoost."""
    from pipelines.components.benchmark_xgboost import run_benchmark_xgboost

    return run_benchmark_xgboost(
        x_train_path=x_train.path, y_train_path=y_train.path,
        x_val_path=x_val.path, y_val_path=y_val.path,
        xgb_params=xgb_params, cat_features=cat_features,
        model_artifact_path=model_artifact.path,
    )
