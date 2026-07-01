"""
optuna_hpo — Hyperparameter optimization with Optuna + XGBoost.

Uses TPE sampler and median pruner to maximize AUCPR.
"""

from __future__ import annotations

import json
from typing import NamedTuple

import optuna
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from kfp import dsl
from ._image import TRAINING_IMAGE


def run_optuna_hpo(
    *,
    x_train_path: str,
    y_train_path: str,
    cat_features: list[str],
    n_trials: int,
    best_params_path: str,
) -> float:
    """Run Optuna study, save best params, return best CV AUCPR."""
    X_train = pd.read_parquet(x_train_path)
    y_train = pd.read_parquet(y_train_path).iloc[:, 0]

    for col in cat_features:
        if col in X_train.columns:
            dtype = pd.CategoricalDtype(
                categories=X_train[col].astype(str).unique(),
            )
            X_train[col] = X_train[col].astype(str).astype(dtype)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss",
            "enable_categorical": True,
        }
        X_a, X_b, y_a, y_b = train_test_split(
            X_train, y_train, test_size=0.25, random_state=42, stratify=y_train,
        )
        model = XGBClassifier(**params)
        model.fit(X_a, y_a, eval_set=[(X_b, y_b)], verbose=False)
        return float(average_precision_score(y_b, model.predict_proba(X_b)[:, 1]))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1
    best_params["eval_metric"] = "logloss"
    best_params["enable_categorical"] = True

    print(f"  Best trial: {study.best_trial.number}")
    print(f"  Best AUCPR: {study.best_value:.4f}")
    print(f"  Best params: {json.dumps(best_params, indent=2)}")

    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    return float(study.best_value)


@dsl.component(
    base_image=TRAINING_IMAGE,
    packages_to_install=[],
)
def optuna_hpo(
    x_train_path: dsl.Input[dsl.Dataset],
    y_train_path: dsl.Input[dsl.Dataset],
    cat_features: list,
    n_trials: int,
) -> NamedTuple(
    "HPOOutputs",
    [("best_aucpr", float), ("best_params_path", str)],
):
    """KFP component: Optuna hyperparameter optimization."""
    aucpr = run_optuna_hpo(
        x_train_path=x_train_path, y_train_path=y_train_path,
        cat_features=cat_features, n_trials=n_trials,
        best_params_path=best_params_path,
    )
    return (aucpr, best_params_path)
