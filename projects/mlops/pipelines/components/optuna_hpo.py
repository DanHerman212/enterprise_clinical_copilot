"""
optuna_hpo — Hyperparameter optimization with Optuna + XGBoost.

Design (tuned to the readmission goals: rank the minority well, stay
probability-first / well-calibrated, leakage-safe, reproducible):

  * **Objective** — mean AUCPR over patient-grouped, class-stratified CV
    (StratifiedGroupKFold on subject_id); no patient straddles a fold.
  * **scale_pos_weight is FIXED at 1.0**, not tuned. AUCPR is invariant to
    monotonic probability scaling, so tuning ``scale_pos_weight`` barely moves
    the objective while badly inflating predicted probabilities — which would
    wreck the Brier score / calibration the pipeline depends on. Class imbalance
    is handled downstream by the F-beta operating threshold, not by distorting
    probabilities.
  * **Early stopping instead of tuning n_estimators.** Each fold fits up to a
    high ceiling and stops on a patient-grouped inner holdout, so the tree count
    is *learned*. The persisted ``n_estimators`` is the median best iteration of
    the best trial — a principled count for the no-early-stopping refit in
    train_final / calibrate.
  * **Multivariate TPE** (models parameter interactions) with 20 startup trials.
  * **MedianPruner** on per-fold intermediate AUCPR kills clearly-losing trials
    after a couple of folds, so the trial budget goes to promising regions.
"""

import json
from typing import NamedTuple

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from xgboost import XGBClassifier
from kfp import dsl
from ._image import TRAINING_IMAGE, component

# Early stopping replaces tuning n_estimators: fit up to the ceiling and stop
# when the inner-holdout AUCPR hasn't improved for this many rounds.
_N_ESTIMATORS_CEILING = 2000
_EARLY_STOPPING_ROUNDS = 50
# Fixed to protect probability calibration (see module docstring).
_FIXED_SCALE_POS_WEIGHT = 1.0


def _grouped_folds(X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int):
    """Patient-grouped, class-stratified CV folds (list of (train_idx, val_idx))."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return list(sgkf.split(X, y, groups))


def _inner_holdout(groups_sub: np.ndarray, *, seed: int = 42, test_size: float = 0.2):
    """Patient-grouped inner holdout for early stopping -> (fit_pos, val_pos).

    Returns positional indices *into the subset* so that no patient straddles the
    fit / early-stopping-holdout boundary.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    fit_pos, val_pos = next(gss.split(np.zeros(len(groups_sub)), groups=groups_sub))
    return fit_pos, val_pos


# Search-space hyperparameters (the tuned knobs); logged as *params*, never metrics.
_HP_KEYS = (
    "n_estimators", "max_depth", "learning_rate", "min_child_weight", "gamma",
    "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "scale_pos_weight",
)


def _log_experiment(
    study, best_params, n_splits, *, project_id, location, experiment, run_name
):
    """Best-effort: log the HPO result to a companion Vertex Experiment run.

    Three destinations, each in its proper UI:
      * **Parameters** — the tuned hyperparameters + HPO config;
      * **Metrics**     — the best grouped-CV AUCPR;
      * **Charts**      — the per-trial CV AUCPR curve (time-series).

    Uses a dedicated ``ExperimentRun`` (see :mod:`._experiment`) because a
    pipeline's auto-created ``system.PipelineRun`` cannot receive ``log_params``
    or time-series. Wrapped so telemetry can never abort a training run.
    """
    from ._experiment import companion_run, safe_log_metrics, safe_log_params

    with companion_run(
        project_id=project_id, location=location,
        experiment=experiment, pipeline_job_name=run_name,
    ) as ap:
        if ap is None:
            return
        safe_log_params(
            ap,
            {f"hp_{k}": best_params[k] for k in _HP_KEYS if k in best_params}
            | {"hpo_n_trials": len(study.trials), "hpo_cv_folds": n_splits},
        )
        safe_log_metrics(ap, {"hpo_cv_aucpr": float(study.best_value)})
        best = float("-inf")
        n = 0
        for t in study.trials:
            if t.value is None:
                continue
            best = max(best, float(t.value))
            try:
                ap.log_time_series_metrics(
                    {"cv_aucpr": float(t.value), "cv_aucpr_best": best},
                    step=int(t.number),
                )
                n += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] time-series step {t.number} skipped: {exc}")
        print(f"  Streamed {n} trial metrics to experiment '{experiment}'.")


def run_optuna_hpo(
    *,
    x_train_path: str,
    y_train_path: str,
    groups_path: str,
    cat_features: list[str],
    n_trials: int,
    best_params_path: str,
    n_splits: int = 5,
    project_id: str = "",
    location: str = "us-east1",
    experiment: str = "",
    run_name: str = "",
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

    folds = _grouped_folds(X_train, y_train, groups, n_splits)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": _N_ESTIMATORS_CEILING,  # capped; early stopping picks the count
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": _FIXED_SCALE_POS_WEIGHT,  # fixed: protect calibration
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "aucpr",
            "enable_categorical": True,
            "tree_method": "hist",
            "early_stopping_rounds": _EARLY_STOPPING_ROUNDS,
        }
        scores: list[float] = []
        best_iters: list[int] = []
        for step, (train_idx, val_idx) in enumerate(folds):
            # Patient-grouped inner holdout (within this fold's train) for early
            # stopping — the outer val fold stays untouched for honest scoring.
            g_sub = groups.iloc[train_idx].to_numpy()
            fit_pos, ival_pos = _inner_holdout(g_sub, seed=42)
            fit_idx, ival_idx = train_idx[fit_pos], train_idx[ival_pos]

            model = XGBClassifier(**params)
            model.fit(
                X_train.iloc[fit_idx], y_train.iloc[fit_idx],
                eval_set=[(X_train.iloc[ival_idx], y_train.iloc[ival_idx])],
                verbose=False,
            )
            proba = model.predict_proba(X_train.iloc[val_idx])[:, 1]
            scores.append(float(average_precision_score(y_train.iloc[val_idx], proba)))
            bi = getattr(model, "best_iteration", None)
            best_iters.append(int(bi) + 1 if bi is not None else _N_ESTIMATORS_CEILING)

            # Report the running mean so MedianPruner can drop losing trials early.
            trial.report(float(np.mean(scores)), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr("best_iterations", best_iters)
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            multivariate=True, n_startup_trials=20, seed=42,
        ),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Tuned knobs come from the study; n_estimators is the median early-stopping
    # iteration of the best trial (the tree count for the no-early-stopping refit
    # in train_final / calibrate); the rest are fixed reproducible settings.
    best_iters = study.best_trial.user_attrs.get(
        "best_iterations", [_N_ESTIMATORS_CEILING]
    )
    best_params = dict(study.best_params)
    best_params["n_estimators"] = int(np.median(best_iters))
    best_params["scale_pos_weight"] = _FIXED_SCALE_POS_WEIGHT
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1
    best_params["eval_metric"] = "aucpr"
    best_params["enable_categorical"] = True
    best_params["tree_method"] = "hist"

    n_pruned = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    )
    print(f"  Best trial: {study.best_trial.number}")
    print(f"  Best grouped-CV AUCPR: {study.best_value:.4f}  ({n_splits}-fold)")
    print(f"  n_estimators (median early-stop iters): {best_params['n_estimators']}")
    print(f"  Pruned trials: {n_pruned}/{len(study.trials)}")
    print(f"  Best params: {json.dumps(best_params, indent=2)}")

    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    _log_experiment(
        study, best_params, n_splits, project_id=project_id, location=location,
        experiment=experiment, run_name=run_name,
    )

    return float(study.best_value)


@component(
    base_image=TRAINING_IMAGE,
    packages_to_install=[
        "optuna", "xgboost", "scikit-learn", "pandas", "pyarrow",
        "google-cloud-aiplatform",
    ],
)
def optuna_hpo(
    x_train: dsl.Input[dsl.Dataset],
    y_train: dsl.Input[dsl.Dataset],
    groups: dsl.Input[dsl.Dataset],
    cat_features: list,
    n_trials: int,
    best_params: dsl.Output[dsl.Artifact],
    project_id: str = "",
    location: str = "us-east1",
    experiment_name: str = "",
    pipeline_job_name: str = "",
) -> float:
    """KFP component: patient-grouped-CV Optuna hyperparameter optimization."""
    from pipelines.components.optuna_hpo import run_optuna_hpo

    return run_optuna_hpo(
        x_train_path=x_train.path, y_train_path=y_train.path,
        groups_path=groups.path,
        cat_features=cat_features, n_trials=n_trials,
        best_params_path=best_params.path,
        project_id=project_id, location=location,
        experiment=experiment_name, run_name=pipeline_job_name,
    )
