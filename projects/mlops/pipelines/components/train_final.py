"""
train_final — Train XGBoost with best HPO params on combined train+val.
"""

from __future__ import annotations

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
    """Train final model with best params, save, return val AUCPR."""
    X_train = pd.read_parquet(x_train_path)
    y_train = pd.read_parquet(y_train_path).iloc[:, 0]
    X_val = pd.read_parquet(x_val_path)
    y_val = pd.read_parquet(y_val_path).iloc[:, 0]

    X_all = pd.concat([X_train, X_val])
    y_all = pd.concat([y_train, y_val])

    for col in cat_features:
        if col in X_all.columns:
            dtype = pd.CategoricalDtype(categories=X_all[col].astype(str).unique())
            X_all[col] = X_all[col].astype(str).astype(dtype)

    with open(best_params_path) as f:
        best_params = json.load(f)

    model = XGBClassifier(**best_params)
    model.fit(X_all, y_all, verbose=False)

    aucpr = float(average_precision_score(y_val, model.predict_proba(X_val)[:, 1]))
    print(f"  Final XGBoost AUCPR: {aucpr:.4f}")
    print(f"  Params: {json.dumps(best_params)}")

    joblib.dump(model, model_artifact_path)
    return aucpr


@dsl.component(
    base_image=TRAINING_IMAGE,
    packages_to_install=[],
)
def train_final(
    x_train_path: dsl.Input[dsl.Dataset],
    y_train_path: dsl.Input[dsl.Dataset],
    x_val_path: dsl.Input[dsl.Dataset],
    y_val_path: dsl.Input[dsl.Dataset],
    best_params_path: dsl.Input[dsl.Artifact],
    cat_features: list,
) -> NamedTuple(
    "FinalOutputs",
    [("final_aucpr", float), ("model_artifact_path", str)],
):
    """KFP component: train final XGBoost with best HPO params."""
    aucpr = run_train_final(
        x_train_path=x_train_path, y_train_path=y_train_path,
        x_val_path=x_val_path, y_val_path=y_val_path,
        best_params_path=best_params_path, cat_features=cat_features,
        model_artifact_path=model_artifact_path,
    )
    return (aucpr, model_artifact_path)
