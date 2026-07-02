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
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from kfp import dsl
from ._image import TRAINING_IMAGE

MAX_VAL_TEST_DEGRADATION = 0.02  # max absolute AUCPR drop val → test
_ROC_MAX_POINTS = 300  # downsample ROC points for a responsive UI chart


def run_evaluate_test(
    *,
    x_test_path: str,
    y_test_path: str,
    model_artifact_path: str,
    hpo_val_aucpr: float,
    benchmark_aucpr: float,
    hospital_aucpr: float,
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

    return {
        "test_aucpr": test_aucpr,
        "test_auroc": test_auroc,
        "brier_score": brier,
        "beat_hospital": beat_hospital,
        "stable": stable,
        "roc": {
            "fpr": [float(v) for v in fpr],
            "tpr": [float(v) for v in tpr],
            "thresholds": [float(v) for v in thr],
        },
    }


@dsl.component(
    base_image=TRAINING_IMAGE,
    packages_to_install=[],
)
def evaluate_test(
    x_test: dsl.Input[dsl.Dataset],
    y_test: dsl.Input[dsl.Dataset],
    model_artifact: dsl.Input[dsl.Model],
    hpo_val_aucpr: float,
    benchmark_aucpr: float,
    hospital_aucpr: float,
    metrics: dsl.Output[dsl.Metrics],
    roc_curve_plot: dsl.Output[dsl.ClassificationMetrics],
) -> NamedTuple(
    "TestOutputs",
    [("test_aucpr", float), ("beat_hospital", bool), ("stable", bool)],
):
    """KFP component: evaluate model on held-out test set."""
    from pipelines.components.evaluate_test import run_evaluate_test

    result = run_evaluate_test(
        x_test_path=x_test.path, y_test_path=y_test.path,
        model_artifact_path=model_artifact.path,
        hpo_val_aucpr=hpo_val_aucpr,
        benchmark_aucpr=benchmark_aucpr,
        hospital_aucpr=hospital_aucpr,
    )

    # Scalar metrics -> Vertex UI metrics tab + auto-logged to the experiment.
    metrics.log_metric("test_aucpr", result["test_aucpr"])
    metrics.log_metric("test_auroc", result["test_auroc"])
    metrics.log_metric("brier_score", result["brier_score"])
    metrics.log_metric("hpo_val_aucpr", hpo_val_aucpr)
    metrics.log_metric("benchmark_aucpr", benchmark_aucpr)
    metrics.log_metric("hospital_aucpr", hospital_aucpr)

    # Interactive ROC curve rendered on the node.
    roc = result["roc"]
    roc_curve_plot.log_roc_curve(roc["fpr"], roc["tpr"], roc["thresholds"])

    return (result["test_aucpr"], result["beat_hospital"], result["stable"])
