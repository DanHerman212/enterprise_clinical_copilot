"""
evaluate_test — Score the trained model against the held-out test set.

Gates registration: test AUCPR must beat the HOSPITAL baseline (passed in;
single source of truth). Also reports a stability flag comparing the honest
HPO validation AUCPR (``hpo_val_aucpr``) to the unbiased test AUCPR.
"""

from typing import NamedTuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from kfp import dsl
from ._image import TRAINING_IMAGE, component
from src.thresholds import net_benefit, net_benefit_curve, point_metrics

MAX_VAL_TEST_DEGRADATION = 0.02  # max absolute AUCPR drop val → test
_ROC_MAX_POINTS = 300  # downsample ROC points for a responsive UI chart


def _downsample(arr, max_points: int = _ROC_MAX_POINTS):
    """Evenly subsample a 1-D array to at most ``max_points`` (keeps endpoints)."""
    arr = np.asarray(arr)
    if len(arr) <= max_points:
        return arr
    idx = np.linspace(0, len(arr) - 1, max_points).astype(int)
    return arr[idx]


def run_evaluate_test(
    *,
    x_test_path: str,
    y_test_path: str,
    model_artifact_path: str,
    tuned_threshold: float,
    hpo_val_aucpr: float,
    benchmark_aucpr: float,
    hospital_aucpr: float,
    beta: float = 2.0,
) -> dict:
    """Score model on the hold-out test set from PROBABILITIES (never labels).

    Returns a dict of threshold-free metrics + gate flags + ROC-curve points:
    ``test_aucpr, test_auroc, brier_score, beat_hospital, stable, roc``.
    Threshold-dependent metrics (confusion matrix, precision/recall) are handled
    downstream at the tuned operating threshold — kept out of here so the model
    stays probability-first and the evaluation stays flexible.
    """
    X_test = pd.read_parquet(x_test_path)
    y_test = pd.read_parquet(y_test_path).iloc[:, 0]

    model = joblib.load(model_artifact_path)
    proba = model.predict_proba(X_test)[:, 1]

    test_aucpr = float(average_precision_score(y_test, proba))
    test_auroc = float(roc_auc_score(y_test, proba))
    brier = float(brier_score_loss(y_test, proba))

    beat_hospital = test_aucpr > hospital_aucpr
    degradation = hpo_val_aucpr - test_aucpr
    stable = degradation <= MAX_VAL_TEST_DEGRADATION

    # ROC-curve points (threshold-free). sklearn sets thresholds[0] = inf; make
    # it finite for the UI, and downsample for a responsive chart.
    fpr, tpr, thr = roc_curve(y_test, proba)
    thr = np.where(np.isinf(thr), 1.0, thr)
    if len(fpr) > _ROC_MAX_POINTS:
        idx = np.linspace(0, len(fpr) - 1, _ROC_MAX_POINTS).astype(int)
        fpr, tpr, thr = fpr[idx], tpr[idx], thr[idx]

    print(f"  Test AUCPR:      {test_aucpr:.4f}")
    print(f"  Test AUROC:      {test_auroc:.4f}")
    print(f"  Brier score:     {brier:.4f}")
    print(f"  HOSPITAL:        {hospital_aucpr:.4f}")
    print(f"  Beat HOSPITAL:   {'PASS' if beat_hospital else 'FAIL'}")
    print(f"  HPO val AUCPR:   {hpo_val_aucpr:.4f}")
    print(f"  Val→test Δ:      {degradation:+.4f}  (threshold: {MAX_VAL_TEST_DEGRADATION})")
    print(f"  Stability:       {'PASS' if stable else 'FAIL'}")

    if not beat_hospital:
        raise ValueError(
            f"Test AUCPR ({test_aucpr:.4f}) did not beat "
            f"HOSPITAL baseline ({hospital_aucpr:.4f})."
        )
    if not stable:
        print(
            f"  WARNING: val→test degradation ({degradation:+.4f}) exceeds "
            f"threshold ({MAX_VAL_TEST_DEGRADATION}). Model may be overfit."
        )

    # --- Threshold-dependent diagnostics at the tuned operating threshold ----
    pm = point_metrics(y_test, proba, tuned_threshold, beta=beta)
    nb_at_threshold = net_benefit(y_test, proba, tuned_threshold)

    # Precision-Recall curve (downsampled, kept paired via shared indices).
    prec, rec, _ = precision_recall_curve(y_test, proba)
    prec, rec = _downsample(prec), _downsample(rec)

    # Calibration (reliability) curve — quantile bins for stable support.
    frac_pos, mean_pred = calibration_curve(
        y_test, proba, n_bins=10, strategy="quantile"
    )

    # Decision Curve Analysis (net benefit) over pt 0.01..0.50.
    dca = net_benefit_curve(y_test, proba)

    print(f"  Tuned threshold: {tuned_threshold:.4f}  (F{beta:g})")
    print(f"  Precision/Recall: {pm['precision']:.3f} / {pm['recall']:.3f}")
    print(f"  Specificity/NPV:  {pm['specificity']:.3f} / {pm['npv']:.3f}")
    print(f"  Net benefit @ thr: {nb_at_threshold:.4f}")

    return {
        "test_aucpr": test_aucpr,
        "test_auroc": test_auroc,
        "brier_score": brier,
        "beat_hospital": beat_hospital,
        "stable": stable,
        "tuned_threshold": float(tuned_threshold),
        "beta": float(beta),
        "point_metrics": pm,
        "net_benefit_at_threshold": float(nb_at_threshold),
        "roc": {
            "fpr": [float(v) for v in fpr],
            "tpr": [float(v) for v in tpr],
            "thresholds": [float(v) for v in thr],
        },
        "pr": {
            "precision": [float(v) for v in prec],
            "recall": [float(v) for v in rec],
        },
        "calibration": {
            "prob_true": [float(v) for v in frac_pos],
            "prob_pred": [float(v) for v in mean_pred],
        },
        "dca": dca,
    }


def _build_eval_html(result: dict) -> str:
    """Render PR / calibration / DCA plots + a metrics table as one HTML report.

    HTML (not Markdown) so the base64 <img> tags render on the Vertex node.
    """
    import base64
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _png(fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()

    pm = result["point_metrics"]
    beta = result["beta"]

    fig1, ax = plt.subplots(figsize=(5, 4))
    ax.plot(result["pr"]["recall"], result["pr"]["precision"])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall (AUCPR={result['test_aucpr']:.3f})")
    pr_b64 = _png(fig1)

    fig2, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
    ax.plot(result["calibration"]["prob_pred"], result["calibration"]["prob_true"], marker="o")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Calibration (Brier={result['brier_score']:.3f})")
    cal_b64 = _png(fig2)

    fig3, ax = plt.subplots(figsize=(5, 4))
    pts = [r["pt"] for r in result["dca"]]
    ax.plot(pts, [r["model"] for r in result["dca"]], label="Model")
    ax.plot(pts, [r["treat_all"] for r in result["dca"]], "--", label="Treat all")
    ax.plot(pts, [r["treat_none"] for r in result["dca"]], ":", color="gray", label="Treat none")
    ax.axvline(result["tuned_threshold"], color="red", lw=0.8, label="Tuned threshold")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_ylim(bottom=-0.02)
    ax.legend(fontsize=8)
    ax.set_title("Decision Curve Analysis")
    dca_b64 = _png(fig3)

    rows = [
        (f"Operating threshold (F{beta:g})", f"{result['tuned_threshold']:.4f}"),
        ("Precision (PPV)", f"{pm['precision']:.4f}"),
        ("Recall (Sensitivity)", f"{pm['recall']:.4f}"),
        ("Specificity", f"{pm['specificity']:.4f}"),
        ("NPV", f"{pm['npv']:.4f}"),
        (f"F{beta:g}", f"{pm['fbeta']:.4f}"),
        ("Net benefit @ threshold", f"{result['net_benefit_at_threshold']:.4f}"),
        ("Test AUCPR / AUROC", f"{result['test_aucpr']:.4f} / {result['test_auroc']:.4f}"),
        ("Brier score", f"{result['brier_score']:.4f}"),
    ]
    table = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
    return f"""
<h2>Test-Set Evaluation @ Tuned Threshold</h2>
<table border="1" cellpadding="6" style="border-collapse:collapse">
<tr><th>Metric</th><th>Value</th></tr>{table}</table>
<h3>Precision–Recall</h3><img src="data:image/png;base64,{pr_b64}" width="500"/>
<h3>Calibration</h3><img src="data:image/png;base64,{cal_b64}" width="500"/>
<h3>Decision Curve Analysis</h3><img src="data:image/png;base64,{dca_b64}" width="500"/>
"""


@component(
    base_image=TRAINING_IMAGE,
    packages_to_install=["google-cloud-aiplatform"],
)
def evaluate_test(
    x_test: dsl.Input[dsl.Dataset],
    y_test: dsl.Input[dsl.Dataset],
    model_artifact: dsl.Input[dsl.Model],
    tuned_threshold: float,
    hpo_val_aucpr: float,
    benchmark_aucpr: float,
    hospital_aucpr: float,
    metrics: dsl.Output[dsl.Metrics],
    classification_metrics: dsl.Output[dsl.ClassificationMetrics],
    eval_report: dsl.Output[dsl.HTML],
    beta: float = 2.0,
    project_id: str = "",
    location: str = "us-east1",
    experiment_name: str = "",
    pipeline_job_name: str = "",
) -> NamedTuple(
    "TestOutputs",
    [("test_aucpr", float), ("beat_hospital", bool), ("stable", bool)],
):
    """KFP component: evaluate model on held-out test set at the tuned threshold."""
    from pipelines.components.evaluate_test import _build_eval_html, run_evaluate_test
    from pipelines.components._experiment import (
        companion_run, safe_log_metrics, safe_log_params,
    )

    result = run_evaluate_test(
        x_test_path=x_test.path, y_test_path=y_test.path,
        model_artifact_path=model_artifact.path,
        tuned_threshold=tuned_threshold,
        hpo_val_aucpr=hpo_val_aucpr,
        benchmark_aucpr=benchmark_aucpr,
        hospital_aucpr=hospital_aucpr,
        beta=beta,
    )

    pm = result["point_metrics"]

    # Scalar metrics -> Vertex UI metrics tab + auto-logged to the experiment.
    metrics.log_metric("test_aucpr", result["test_aucpr"])
    metrics.log_metric("test_auroc", result["test_auroc"])
    metrics.log_metric("brier_score", result["brier_score"])
    metrics.log_metric("hpo_val_aucpr", hpo_val_aucpr)
    metrics.log_metric("benchmark_aucpr", benchmark_aucpr)
    metrics.log_metric("hospital_aucpr", hospital_aucpr)
    metrics.log_metric("precision", pm["precision"])
    metrics.log_metric("recall", pm["recall"])
    metrics.log_metric("specificity", pm["specificity"])
    metrics.log_metric("npv", pm["npv"])
    metrics.log_metric("fbeta", pm["fbeta"])
    metrics.log_metric("net_benefit_at_threshold", result["net_benefit_at_threshold"])

    # Companion Experiment run: mirror the performance metrics into the Metrics
    # UI and file the F-beta config under Parameters (tuned_threshold itself is
    # owned by calibrate-threshold, so it is not re-logged here).
    with companion_run(
        project_id=project_id, location=location,
        experiment=experiment_name, pipeline_job_name=pipeline_job_name,
    ) as ap:
        safe_log_params(ap, {"fbeta_beta": float(beta)})
        safe_log_metrics(ap, {
            "test_aucpr": result["test_aucpr"],
            "test_auroc": result["test_auroc"],
            "brier_score": result["brier_score"],
            "precision": pm["precision"],
            "recall": pm["recall"],
            "specificity": pm["specificity"],
            "npv": pm["npv"],
            "fbeta": pm["fbeta"],
            "net_benefit_at_threshold": result["net_benefit_at_threshold"],
            "hpo_val_aucpr": float(hpo_val_aucpr),
            "benchmark_aucpr": float(benchmark_aucpr),
            "hospital_aucpr": float(hospital_aucpr),
        })

    # ROC curve + confusion matrix (at the tuned threshold) on the node.
    roc = result["roc"]
    classification_metrics.log_roc_curve(roc["fpr"], roc["tpr"], roc["thresholds"])
    classification_metrics.log_confusion_matrix(
        ["No readmit", "Readmit"],
        [[pm["tn"], pm["fp"]], [pm["fn"], pm["tp"]]],
    )

    # PR / calibration / DCA plots + metrics table as a rendered HTML report.
    with open(eval_report.path, "w") as f:
        f.write(_build_eval_html(result))

    return (result["test_aucpr"], result["beat_hospital"], result["stable"])
