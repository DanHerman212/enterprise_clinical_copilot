"""
calibrate_threshold — choose the operating decision threshold (OOF F-beta).

Runs after HPO: retrains the best-params model in patient-grouped CV to produce
out-of-fold probabilities on TRAIN, then selects the single threshold that
maximizes F-beta. The scalar threshold is the only *operating* output; a
companion artifact records the full precision/recall/F-beta curve for audit.

The model stays probability-first — this threshold is metadata for the decision
layer (dashboards / alerts), never baked into the model, so it can change
without retraining.
"""

import json

import pandas as pd
from kfp import dsl

from ._image import TRAINING_IMAGE, component


def run_calibrate_threshold(
    *,
    x_train_path: str,
    y_train_path: str,
    groups_path: str,
    best_params_path: str,
    cat_features: list[str],
    beta: float,
    threshold_output_path: str,
    n_splits: int = 5,
) -> float:
    """Compute OOF probabilities on train, select F-beta threshold, persist curve."""
    from src.thresholds import oof_probabilities, select_threshold_fbeta

    X_train = pd.read_parquet(x_train_path)
    y_train = pd.read_parquet(y_train_path).iloc[:, 0]
    groups = pd.read_parquet(groups_path).iloc[:, 0]

    with open(best_params_path) as f:
        params = json.load(f)

    # Restore native categorical dtype (parquet + train-only categories).
    for col in cat_features:
        if col in X_train.columns:
            dtype = pd.CategoricalDtype(categories=X_train[col].astype(str).unique())
            X_train[col] = X_train[col].astype(str).astype(dtype)

    oof = oof_probabilities(X_train, y_train, groups, params=params, n_splits=n_splits)
    threshold, fbeta, curve = select_threshold_fbeta(y_train, oof, beta=beta)

    record = {
        "threshold": float(threshold),
        "beta": float(beta),
        "fbeta": float(fbeta),
        "objective": "fbeta",
        "selection": "out_of_fold_grouped_cv",
        "n_splits": n_splits,
        "n_train": int(len(y_train)),
        "prevalence": float(y_train.mean()),
        "curve": curve,
    }
    with open(threshold_output_path, "w") as f:
        json.dump(record, f, indent=2)

    print(f"  Tuned threshold (F{beta:g}, OOF grouped-CV): {threshold:.4f}")
    print(f"  F{beta:g} at threshold:                     {fbeta:.4f}")
    print(f"  Train rows / prevalence:                 {len(y_train):,} / {y_train.mean():.3f}")
    return float(threshold)


@component(
    base_image=TRAINING_IMAGE,
    packages_to_install=[
        "xgboost", "scikit-learn", "pandas", "pyarrow",
        "google-cloud-aiplatform",
    ],
)
def calibrate_threshold(
    x_train: dsl.Input[dsl.Dataset],
    y_train: dsl.Input[dsl.Dataset],
    groups: dsl.Input[dsl.Dataset],
    best_params: dsl.Input[dsl.Artifact],
    cat_features: list,
    beta: float,
    threshold_curve: dsl.Output[dsl.Artifact],
    metrics: dsl.Output[dsl.Metrics],
    project_id: str = "",
    location: str = "us-east1",
    experiment_name: str = "",
    pipeline_job_name: str = "",
) -> float:
    """KFP component: select the operating threshold via OOF F-beta."""
    import json

    from pipelines.components.calibrate_threshold import run_calibrate_threshold
    from pipelines.components._experiment import (
        companion_run, safe_log_metrics, safe_log_params,
    )

    threshold = run_calibrate_threshold(
        x_train_path=x_train.path,
        y_train_path=y_train.path,
        groups_path=groups.path,
        best_params_path=best_params.path,
        cat_features=cat_features,
        beta=beta,
        threshold_output_path=threshold_curve.path,
    )

    with open(threshold_curve.path) as f:
        record = json.load(f)

    # dsl.Metrics on the PipelineRun: genuine measured quantities only (config
    # and hyperparameters go to the companion run's Parameters UI instead, so
    # this tab stays a clean list of *metrics*).
    metrics.log_metric("tuned_threshold", record["threshold"])
    metrics.log_metric("fbeta_at_threshold", record["fbeta"])
    metrics.log_metric("train_prevalence", record["prevalence"])

    # Companion Experiment run: clean params-vs-metrics separation + Charts.
    with companion_run(
        project_id=project_id, location=location,
        experiment=experiment_name, pipeline_job_name=pipeline_job_name,
    ) as ap:
        safe_log_params(ap, {
            "fbeta_beta": record["beta"],
            "threshold_selection": record["selection"],
            "train_rows": record["n_train"],
            "train_prevalence": record["prevalence"],
        })
        safe_log_metrics(ap, {
            "tuned_threshold": record["threshold"],
            "fbeta_at_threshold": record["fbeta"],
        })

    return threshold
