"""Tests for the benchmark_gate and evaluate_test gates (pipeline module 6).

Pins:
  * HOSPITAL_AUCPR is a passed-in value (single source of truth), not a
    hardcoded module constant duplicated across components;
  * benchmark_gate hard-fails when the benchmark does not beat the baseline;
  * evaluate_test scores the hold-out test set, hard-fails below the baseline,
    and reports a stability flag comparing the honest HPO validation AUCPR
    (renamed ``hpo_val_aucpr``) to the test AUCPR.
"""

import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

from pipelines.components.benchmark_gate import run_benchmark_gate
from pipelines.components.evaluate_test import run_evaluate_test


# ---------------------------------------------------------------------------
# benchmark_gate
# ---------------------------------------------------------------------------

def test_benchmark_gate_passes_when_beats():
    assert run_benchmark_gate(benchmark_aucpr=0.50, hospital_aucpr=0.3325) is True


def test_benchmark_gate_hard_fails_when_below():
    with pytest.raises(ValueError):
        run_benchmark_gate(benchmark_aucpr=0.30, hospital_aucpr=0.3325)


def test_benchmark_gate_uses_passed_baseline():
    # A high passed baseline must cause failure, proving no hardcoded 0.3325.
    with pytest.raises(ValueError):
        run_benchmark_gate(benchmark_aucpr=0.50, hospital_aucpr=0.90)


# ---------------------------------------------------------------------------
# evaluate_test
# ---------------------------------------------------------------------------

def _trained_model(tmp_path):
    rng = np.random.RandomState(0)
    n = 300
    X = pd.DataFrame(
        {
            "age": rng.normal(60, 10, n),
            "glucose_last": rng.normal(100, 15, n),
        }
    )
    # Signal so AUCPR is meaningfully above prevalence.
    y = pd.Series((X["glucose_last"] + rng.normal(0, 5, n) > 100).astype(int))
    model = XGBClassifier(n_estimators=20, max_depth=3, random_state=42)
    model.fit(X, y)
    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    x_path = tmp_path / "x_test.parquet"
    y_path = tmp_path / "y_test.parquet"
    X.to_parquet(x_path, index=False)
    y.to_frame("readmission_30d").to_parquet(y_path, index=False)

    test_aucpr = float(average_precision_score(y, model.predict_proba(X)[:, 1]))
    return str(model_path), str(x_path), str(y_path), test_aucpr


def test_evaluate_beats_baseline_and_stable(tmp_path):
    model_path, x_path, y_path, test_aucpr = _trained_model(tmp_path)
    result = run_evaluate_test(
        x_test_path=x_path, y_test_path=y_path,
        model_artifact_path=model_path,
        tuned_threshold=0.5,
        hpo_val_aucpr=test_aucpr,          # equal -> stable
        benchmark_aucpr=0.40,
        hospital_aucpr=0.10,               # low -> beaten
        beta=2.0,
    )
    assert result["beat_hospital"] is True
    assert result["stable"] is True
    assert abs(result["test_aucpr"] - test_aucpr) < 1e-9
    # Threshold-free extras for experiment tracking.
    assert 0.0 <= result["test_auroc"] <= 1.0
    assert 0.0 <= result["brier_score"] <= 1.0
    assert len(result["roc"]["fpr"]) == len(result["roc"]["tpr"]) == len(result["roc"]["thresholds"])
    assert all(np.isfinite(t) for t in result["roc"]["thresholds"])
    # Threshold-dependent diagnostics.
    pm = result["point_metrics"]
    assert pm["tp"] + pm["fp"] + pm["tn"] + pm["fn"] == 300
    assert 0.0 <= pm["precision"] <= 1.0 and 0.0 <= pm["recall"] <= 1.0
    assert result["tuned_threshold"] == 0.5
    assert len(result["pr"]["precision"]) == len(result["pr"]["recall"])
    assert len(result["calibration"]["prob_true"]) == len(result["calibration"]["prob_pred"])
    assert result["dca"][0]["treat_none"] == 0.0
    assert all("model" in r and "treat_all" in r for r in result["dca"])


def test_evaluate_hard_fails_below_baseline(tmp_path):
    model_path, x_path, y_path, test_aucpr = _trained_model(tmp_path)
    with pytest.raises(ValueError):
        run_evaluate_test(
            x_test_path=x_path, y_test_path=y_path,
            model_artifact_path=model_path,
            tuned_threshold=0.5,
            hpo_val_aucpr=test_aucpr,
            benchmark_aucpr=0.40,
            hospital_aucpr=0.999,          # impossibly high -> must fail
        )


def test_evaluate_flags_instability(tmp_path):
    model_path, x_path, y_path, test_aucpr = _trained_model(tmp_path)
    result = run_evaluate_test(
        x_test_path=x_path, y_test_path=y_path,
        model_artifact_path=model_path,
        tuned_threshold=0.5,
        hpo_val_aucpr=test_aucpr + 0.30,   # big val->test drop
        benchmark_aucpr=0.40,
        hospital_aucpr=0.10,
    )
    assert result["beat_hospital"] is True
    assert result["stable"] is False
