"""
optuna_hpo — Hyperparameter optimization with Optuna + XGBoost.

Cross-validation is patient-grouped (StratifiedGroupKFold on subject_id): no
patient appears in both the train and validation side of a fold, matching the
leakage-controlled outer split. Optimizes mean cross-validated AUCPR with the
TPE sampler (no pruner — a coarse 5-fold objective gives too few, too-noisy
intermediate steps for median pruning to help without risking good trials).
"""

import json
from typing import NamedTuple

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier
from kfp import dsl
from ._image import TRAINING_IMAGE, component


def _scale_pos_weight(y: pd.Series) -> float:
    """Empirical class-imbalance ratio (neg/pos); 1.0 if no positives."""
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return float(neg / pos) if pos > 0 else 1.0


def _grouped_folds(X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int):
    """Patient-grouped, class-stratified CV folds (list of (train_idx, val_idx))."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return list(sgkf.split(X, y, groups))


def run_optuna_hpo(
    *,
    x_train_path: str,
    y_train_path: str,
    groups_path: str,
    cat_features: list[str],
    n_trials: int,
    best_params_path: str,
    n_splits: int = 5,
) -> float:
    """Run grouped-CV Optuna study, save best params, return best CV AUCPR."""
    X_train = pd.read_parquet(x_train_path)
    y_train = pd.read_parquet(y_train_path).iloc[:, 0]
    groups = pd.read_parquet(groups_path).iloc[:, 0]

    for col in cat_features:
        if col in X_train.columns:
            dtype = pd.CategoricalDtype(
                categories=X_train[col].astype(str).unique(),
            )
            X_train[col] = X_train[col].astype(str).astype(dtype)

    spw = _scale_pos_weight(y_train)
    spw_high = max(spw, 1.0001)  # guard degenerate [1, 1] band
    folds = _grouped_folds(X_train, y_train, groups, n_splits)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, spw_high),
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "aucpr",
            "enable_categorical": True,
            "tree_method": "hist",
        }
        scores = []
        for train_idx, val_idx in folds:
            model = XGBClassifier(**params)
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx], verbose=False)
            proba = model.predict_proba(X_train.iloc[val_idx])[:, 1]
            scores.append(average_precision_score(y_train.iloc[val_idx], proba))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.NopPruner(),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = dict(study.best_params)
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1
    best_params["eval_metric"] = "aucpr"
    best_params["enable_categorical"] = True
    best_params["tree_method"] = "hist"

    print(f"  Best trial: {study.best_trial.number}")
    print(f"  Best grouped-CV AUCPR: {study.best_value:.4f}  ({n_splits}-fold)")
    print(f"  scale_pos_weight band: [1.0, {spw_high:.3f}]")
    print(f"  Best params: {json.dumps(best_params, indent=2)}")

    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    return float(study.best_value)


@component(
    base_image=TRAINING_IMAGE,
    packages_to_install=["optuna", "xgboost", "scikit-learn", "pandas", "pyarrow"],
)
def optuna_hpo(
    x_train: dsl.Input[dsl.Dataset],
    y_train: dsl.Input[dsl.Dataset],
    groups: dsl.Input[dsl.Dataset],
    cat_features: list,
    n_trials: int,
    best_params: dsl.Output[dsl.Artifact],
) -> float:
    """KFP component: patient-grouped-CV Optuna hyperparameter optimization."""
    from pipelines.components.optuna_hpo import run_optuna_hpo

    return run_optuna_hpo(
        x_train_path=x_train.path, y_train_path=y_train.path,
        groups_path=groups.path,
        cat_features=cat_features, n_trials=n_trials,
        best_params_path=best_params.path,
    )
