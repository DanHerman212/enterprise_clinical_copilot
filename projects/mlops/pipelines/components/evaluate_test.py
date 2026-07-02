"""
evaluate_test — Score the trained model against the held-out test set.

Gates registration: test AUCPR must beat the HOSPITAL baseline (passed in;
single source of truth). Also reports a stability flag comparing the honest
HPO validation AUCPR (``hpo_val_aucpr``) to the unbiased test AUCPR.
"""

from typing import NamedTuple

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score
from kfp import dsl
from ._image import TRAINING_IMAGE

MAX_VAL_TEST_DEGRADATION = 0.02  # max absolute AUCPR drop val → test


def run_evaluate_test(
    *,
    x_test_path: str,
    y_test_path: str,
    model_artifact_path: str,
    hpo_val_aucpr: float,
    benchmark_aucpr: float,
    hospital_aucpr: float,
) -> tuple[float, bool, bool]:
    """Score model on test set.  Returns (test_aucpr, beat_hospital, stable)."""
    X_test = pd.read_parquet(x_test_path)
    y_test = pd.read_parquet(y_test_path).iloc[:, 0]

    model = joblib.load(model_artifact_path)

    test_aucpr = float(average_precision_score(
        y_test, model.predict_proba(X_test)[:, 1],
    ))

    beat_hospital = test_aucpr > hospital_aucpr
    degradation = hpo_val_aucpr - test_aucpr
    stable = degradation <= MAX_VAL_TEST_DEGRADATION

    print(f"  Test AUCPR:      {test_aucpr:.4f}")
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

    return (test_aucpr, beat_hospital, stable)


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
) -> NamedTuple(
    "TestOutputs",
    [("test_aucpr", float), ("beat_hospital", bool), ("stable", bool)],
):
    """KFP component: evaluate model on held-out test set."""
    test_aucpr, beat_hospital, stable = run_evaluate_test(
        x_test_path=x_test.path, y_test_path=y_test.path,
        model_artifact_path=model_artifact.path,
        hpo_val_aucpr=hpo_val_aucpr,
        benchmark_aucpr=benchmark_aucpr,
        hospital_aucpr=hospital_aucpr,
    )

    # Expose exactly as KFP Metrics so it populates the Vertex UI metrics tab.
    metrics.log_metric("test_aucpr", test_aucpr)
    metrics.log_metric("hpo_val_aucpr", hpo_val_aucpr)
    metrics.log_metric("benchmark_aucpr", benchmark_aucpr)
    metrics.log_metric("hospital_aucpr", hospital_aucpr)

    return (test_aucpr, beat_hospital, stable)
