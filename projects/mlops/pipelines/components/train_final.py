"""
train_final — Train the final XGBoost on the COMBINED train+val set.

Uses the hyperparameters selected during HPO to refit on train+val (so the
production model sees all pre-test data). A small patient-grouped *monitor*
split (20% of validation) is held out during training to produce per-round
AUCPR curves — these are persisted as an artifact and streamed to the
companion experiment run's Charts tab for overfit inspection. The honest
generalization estimate comes from the untouched hold-out test set by
evaluate_test.
"""

import json
from typing import NamedTuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier
from kfp import dsl
from ._image import TRAINING_IMAGE, component

_MONITOR_VAL_FRAC = 0.8  # use 80% of val for training, 20% for per-round monitor


def _patient_split_mask(groups: np.ndarray, test_size: float, seed: int = 42):
    """Boolean mask: True for training rows, False for monitor/holdout rows."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_pos, _ = next(gss.split(np.zeros(len(groups)), groups=groups))
    mask = np.zeros(len(groups), dtype=bool)
    mask[train_pos] = True
    return mask


def run_train_final(
    *,
    x_train_path: str,
    y_train_path: str,
    x_val_path: str,
    y_val_path: str,
    best_params_path: str,
    cat_features: list[str],
    model_artifact_path: str,
    training_curve_path: str,
    groups_val_path: str,
) -> float:
    """Refit on train+val with best params, save model, return combined-fit AUCPR.

    A small patient-grouped *monitor* split (20% of val rows) is held out during
    training solely for the per-round AUCPR curves; the final model still trains
    on 80% of val plus all of train — the honest generalization estimate is test.
    """
    X_train = pd.read_parquet(x_train_path)
    y_train = pd.read_parquet(y_train_path).iloc[:, 0]
    X_val = pd.read_parquet(x_val_path)
    y_val = pd.read_parquet(y_val_path).iloc[:, 0]
    g_val = pd.read_parquet(groups_val_path).iloc[:, 0].to_numpy()

    # Categorical encoding is already applied upstream (category dtype).
    _ = cat_features

    with open(best_params_path) as f:
        best_params = json.load(f)

    # Patient-grouped monitor split — no patient straddles train/monitor.
    train_mask = _patient_split_mask(g_val, test_size=1 - _MONITOR_VAL_FRAC)
    X_val_train, y_val_train = X_val.iloc[train_mask], y_val.iloc[train_mask]
    X_monitor, y_monitor = X_val.iloc[~train_mask], y_val.iloc[~train_mask]

    X_all = pd.concat([X_train, X_val_train], ignore_index=True)
    y_all = pd.concat([y_train, y_val_train], ignore_index=True)

    n_estimators = best_params.get("n_estimators", 300)
    # For the final fit we drop early-stopping (it was a HPO selection tool) so
    # the full tree budget is used; but we still need eval_set for the curves.
    fit_params = {**best_params}
    fit_params.pop("early_stopping_rounds", None)

    model = XGBClassifier(**fit_params)
    model.fit(
        X_all, y_all,
        eval_set=[(X_all, y_all), (X_monitor, y_monitor)],
        verbose=False,
    )

    # Per-round AUCPR curves (train on the combined fit, monitor on the held-out
    # val subsample). Streamed to the companion run's Charts tab for overfit
    # inspection — the gap between "train" and "monitor" evals_result entries per
    # round is the most sensitive early-warning signal.
    evals = model.evals_result()
    curve = {}
    for dataset_idx, label in enumerate(["train", "monitor"]):
        key = f"validation_{dataset_idx}"
        if key in evals:
            metrics = evals[key]
            # Read the first available metric (usually "aucpr" from best_params,
            # or "logloss" if eval_metric was not explicitly set).
            for metric_name, values in metrics.items():
                curve[f"{label}_{metric_name}"] = [float(v) for v in values]

    with open(training_curve_path, "w") as f:
        json.dump(curve, f, indent=2)

    # Combined-set fit AUCPR. NOT unbiased — logged only as an overfit sanity
    # signal. Unbiased performance is on the hold-out test set.
    train_aucpr = float(average_precision_score(y_all, model.predict_proba(X_all)[:, 1]))
    print(f"  Final model trained on {len(X_all):,} rows (train + {_MONITOR_VAL_FRAC:.0%} val).")
    print(f"  Monitor rows held out for per-round curve: {len(X_monitor):,}")
    print(f"  Combined-set fit AUCPR (not unbiased): {train_aucpr:.4f}")
    print(f"  n_estimators: {n_estimators}  Params: {json.dumps(best_params)}")

    joblib.dump(model, model_artifact_path)
    return train_aucpr


def _build_training_curve_html(curve: dict) -> str:
    """Render the per-round train-vs-monitor AUCPR curves as one HTML report.

    The gap between the two curves per boosting round is the most sensitive
    overfit signal — divergence means the model is memorising training noise
    without improving its out-of-sample score.
    """
    import base64
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_keys = [k for k in curve if k.startswith("train_")]
    monitor_keys = [k for k in curve if k.startswith("monitor_")]
    if not train_keys or not monitor_keys:
        return "<p>No per-round training curve data available.</p>"

    train_label = train_keys[0].replace("train_", "")
    train_vals = curve[train_keys[0]]
    monitor_vals = curve[monitor_keys[0]]

    fig, ax = plt.subplots(figsize=(7, 4))
    rounds = range(1, len(train_vals) + 1)
    ax.plot(rounds, train_vals, color="#1f77b4", lw=1.8, label="Training set (combined fit)")
    ax.plot(rounds[:len(monitor_vals)], monitor_vals, color="#d62728", lw=1.8,
            label="Monitor set (held-out 20% val)")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel(train_label.upper())
    ax.set_title("Per-Round Training Curve (overfit check)")
    ax.legend(fontsize=8)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    png_b64 = base64.b64encode(buf.getvalue()).decode()

    gap = round(train_vals[-1] - monitor_vals[-1], 4) if monitor_vals else None
    return f"""
<h2>Training Curve — Per-Round {train_label.upper()}</h2>
<p><b>Train–monitor gap at final round:</b> {gap}</p>
<p>The red curve is scored on a small held-out patient-grouped portion of the
validation split that the model never trained on. If the blue training curve
keeps climbing while red plateaus or drops, the model is overfitting — the
final rounds are memorising noise without improving generalization.</p>
<img src="data:image/png;base64,{png_b64}" width="700"/>
"""


@component(
    base_image=TRAINING_IMAGE,
    packages_to_install=[
        "xgboost", "scikit-learn", "pandas", "pyarrow", "joblib",
        "google-cloud-aiplatform",
    ],
)
def train_final(
    x_train: dsl.Input[dsl.Dataset],
    y_train: dsl.Input[dsl.Dataset],
    x_val: dsl.Input[dsl.Dataset],
    y_val: dsl.Input[dsl.Dataset],
    best_params: dsl.Input[dsl.Artifact],
    cat_features: list,
    groups_val: dsl.Input[dsl.Dataset],
    model_artifact: dsl.Output[dsl.Model],
    training_curve: dsl.Output[dsl.Artifact],
    training_curve_plot: dsl.Output[dsl.HTML],
    project_id: str = "",
    location: str = "us-east1",
    experiment_name: str = "",
    pipeline_job_name: str = "",
) -> float:
    """KFP component: train final XGBoost with best HPO params."""
    import json

    from pipelines.components.train_final import _build_training_curve_html, run_train_final
    from pipelines.components._experiment import companion_run

    score = run_train_final(
        x_train_path=x_train.path, y_train_path=y_train.path,
        x_val_path=x_val.path, y_val_path=y_val.path,
        best_params_path=best_params.path, cat_features=cat_features,
        model_artifact_path=model_artifact.path,
        training_curve_path=training_curve.path,
        groups_val_path=groups_val.path,
    )

    # Stream the per-round AUCPR curves to the companion experiment run so the
    # train-vs-monitor gap is visible directly in the Charts tab.
    with open(training_curve.path) as f:
        curve = json.load(f)
    with companion_run(
        project_id=project_id, location=location,
        experiment=experiment_name, pipeline_job_name=pipeline_job_name,
    ) as ap:
        if ap is not None:
            for step in range(max(len(v) for v in curve.values())):
                try:
                    ap.log_time_series_metrics(
                        {k: v[step] for k, v in curve.items() if step < len(v)},
                        step=step,
                    )
                except Exception:  # noqa: BLE001
                    pass

    # Render the training curve as an HTML plot on the pipeline DAG node.
    with open(training_curve_plot.path, "w") as f:
        f.write(_build_training_curve_html(curve))

    return score
