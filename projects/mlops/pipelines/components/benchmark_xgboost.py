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
from ._artifact_integrity import dump as _model_dump


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

    # Features arrive fully numeric (one-hot encoded in BigQuery); cat_features
    # is retained for signature stability but is unused.
    _ = cat_features

    params = dict(xgb_params)
    params.setdefault("n_jobs", -1)

    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    aucpr = float(average_precision_score(
        y_val, model.predict_proba(X_val)[:, 1],
    ))
    print(f"  Benchmark XGBoost AUCPR: {aucpr:.4f}")
    _model_dump(model, model_artifact_path)
    return aucpr


@component(
    base_image=TRAINING_IMAGE,
    packages_to_install=[
        "xgboost>=2.1,<2.2", "scikit-learn>=1.5,<2", "pandas>=2,<3",
        "pyarrow>=14,<25", "joblib>=1.3,<2",
    ],
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
