"""
evaluate_test — Score the trained model against the held-out test set.

Gates registration: test AUCPR must beat the HOSPITAL baseline.
"""

from __future__ import annotations

from typing import NamedTuple

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score
from kfp import dsl
from ._image import TRAINING_IMAGE

HOSPITAL_AUCPR = 0.3325
MAX_VAL_TEST_DEGRADATION = 0.02  # max absolute AUCPR drop val → test


def run_evaluate_test(
    *,
    x_test_path: str,
    y_test_path: str,
    model_artifact_path: str,
    final_val_aucpr: float,
    benchmark_aucpr: float,
) -> tuple[float, bool, bool]:
    """Score model on test set.  Returns (test_aucpr, beat_hospital, stable)."""
    X_test = pd.read_parquet(x_test_path)
    y_test = pd.read_parquet(y_test_path).iloc[:, 0]

    model = joblib.load(model_artifact_path)

    test_aucpr = float(average_precision_score(
        y_test, model.predict_proba(X_test)[:, 1],
    ))

    beat_hospital = test_aucpr > HOSPITAL_AUCPR
    degradation = final_val_aucpr - test_aucpr
    stable = degradation <= MAX_VAL_TEST_DEGRADATION

    print(f"  Test AUCPR:      {test_aucpr:.4f}")
    print(f"  HOSPITAL:        {HOSPITAL_AUCPR:.4f}")
    print(f"  Beat HOSPITAL:   {'PASS' if beat_hospital else 'FAIL'}")
    print(f"  Final val AUCPR: {final_val_aucpr:.4f}")
    print(f"  Val→test Δ:      {degradation:+.4f}  (threshold: {MAX_VAL_TEST_DEGRADATION})")
    print(f"  Stability:       {'PASS' if stable else 'FAIL'}")

    if not beat_hospital:
        raise ValueError(
            f"Test AUCPR ({test_aucpr:.4f}) did not beat "
            f"HOSPITAL baseline ({HOSPITAL_AUCPR:.4f})."
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
    final_val_aucpr: float,
    benchmark_aucpr: float,
    metrics: dsl.Output[dsl.Metrics],
) -> NamedTuple(
    "TestOutputs",
    [("test_aucpr", float), ("beat_hospital", bool), ("stable", bool)],
):
    """KFP component: evaluate model on held-out test set."""
    test_aucpr, beat_hospital, stable = run_evaluate_test(
        x_test_path=x_test.path, y_test_path=y_test.path,
        model_artifact_path=model_artifact.path,
        final_val_aucpr=final_val_aucpr,
        benchmark_aucpr=benchmark_aucpr,
    )
    
    # Expose exactly as KFP Metrics so it populates the Vertex UI metrics tab.
    metrics.log_metric("test_aucpr", test_aucpr)
    metrics.log_metric("val_aucpr", final_val_aucpr)
    metrics.log_metric("benchmark_aucpr", benchmark_aucpr)

    return (test_aucpr, beat_hospital, stable)
