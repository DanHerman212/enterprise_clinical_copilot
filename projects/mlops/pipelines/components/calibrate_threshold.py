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

    # Features arrive fully numeric (one-hot encoded in BigQuery); cat_features
    # is retained for signature stability but is unused.
    _ = cat_features

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


# Thresholds within this *relative* tolerance of the peak F-beta form the
# "sweet-spot" plateau band — communicates how forgiving the choice is.
_PLATEAU_TOL = 0.01  # 1% of the best F-beta


def _build_threshold_html(record: dict) -> str:
    """Render the threshold-selection sweep as an HTML report.

    Precision / recall / F-beta vs. threshold on one axis, the selected
    threshold marked, and the near-optimal *plateau band* (F-beta within
    ``_PLATEAU_TOL`` of the peak) shaded — so the operating "sweet spot" and how
    wide it is are both visible. HTML (base64 PNG) so it renders on the node.
    """
    import base64
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = record["curve"]
    beta = record["beta"]
    selected = record["threshold"]
    best_fbeta = record["fbeta"]

    thr = [r["threshold"] for r in curve]
    prec = [r["precision"] for r in curve]
    rec = [r["recall"] for r in curve]
    fbeta = [r["fbeta"] for r in curve]

    # Plateau band: contiguous-enough range of thresholds within tolerance of peak.
    band = [r["threshold"] for r in curve if r["fbeta"] >= best_fbeta * (1 - _PLATEAU_TOL)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if band:
        ax.axvspan(
            min(band), max(band), color="gold", alpha=0.20,
            label=f"Plateau (within {_PLATEAU_TOL:.0%} of peak F{beta:g})",
        )
    ax.plot(thr, fbeta, color="#1f77b4", lw=2.2, label=f"F{beta:g}")
    ax.plot(thr, prec, color="#2ca02c", lw=1.4, ls="--", label="Precision (PPV)")
    ax.plot(thr, rec, color="#d62728", lw=1.4, ls=":", label="Recall (Sensitivity)")
    ax.axvline(selected, color="black", lw=1.2)
    ax.plot([selected], [best_fbeta], "o", color="black", zorder=5)
    ax.annotate(
        f"selected = {selected:.2f}\nF{beta:g} = {best_fbeta:.3f}",
        xy=(selected, best_fbeta),
        xytext=(min(selected + 0.12, 0.7), min(best_fbeta + 0.18, 0.95)),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "lw": 0.8},
    )
    ax.set_xlabel("Operating threshold (probability)")
    ax.set_ylabel("Score")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Threshold Calibration — F{beta:g}-optimal operating point")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    png_b64 = base64.b64encode(buf.getvalue()).decode()

    band_txt = (
        f"{min(band):.2f} – {max(band):.2f}" if band else f"{selected:.2f}"
    )
    return f"""
<h2>Threshold Calibration (F{beta:g}, out-of-fold grouped-CV on train)</h2>
<p>The operating threshold maximizes F{beta:g} on out-of-fold, patient-grouped
train predictions. The shaded band marks thresholds whose F{beta:g} is within
{_PLATEAU_TOL:.0%} of the peak — the near-equivalent "sweet spot".</p>
<ul>
  <li><b>Selected threshold:</b> {selected:.4f}</li>
  <li><b>F{beta:g} at selection:</b> {best_fbeta:.4f}</li>
  <li><b>Sweet-spot band:</b> {band_txt}</li>
  <li><b>Train rows / prevalence:</b> {record['n_train']:,} / {record['prevalence']:.3f}</li>
</ul>
<img src="data:image/png;base64,{png_b64}" width="700"/>
"""


@component(
    base_image=TRAINING_IMAGE,
    packages_to_install=[
        "xgboost>=2.1,<2.2", "scikit-learn>=1.5,<2", "pandas>=2,<3",
        "pyarrow>=14,<25", "google-cloud-aiplatform",
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
    threshold_plot: dsl.Output[dsl.HTML],
    metrics: dsl.Output[dsl.Metrics],
    project_id: str = "",
    location: str = "us-east1",
    experiment_name: str = "",
    pipeline_job_name: str = "",
) -> float:
    """KFP component: select the operating threshold via OOF F-beta."""
    import json

    from pipelines.components.calibrate_threshold import (
        _build_threshold_html, run_calibrate_threshold,
    )
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

    # Threshold-selection sweep plot (precision/recall/F-beta vs threshold with
    # the selected point + sweet-spot plateau band) as a rendered HTML report.
    with open(threshold_plot.path, "w") as f:
        f.write(_build_threshold_html(record))

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
