"""Tests for calibrate_threshold + the evaluate_test HTML report builder."""

import json

import numpy as np
import pandas as pd

from pipelines.components.calibrate_threshold import run_calibrate_threshold
from pipelines.components.evaluate_test import _build_eval_html


def _write_train(tmp_path):
    """Synthetic patient-grouped train split written as parquet (like load_data)."""
    rng = np.random.RandomState(3)
    n_patients = 60
    rows, groups, labels = [], [], []
    for pid in range(n_patients):
        y = int(pid % 2)
        for _ in range(2):
            rows.append({"f1": rng.normal(1.0 if y else -1.0), "f2": rng.normal()})
            groups.append(pid)
            labels.append(y)
    X = pd.DataFrame(rows)
    x_path = tmp_path / "x_train.parquet"
    y_path = tmp_path / "y_train.parquet"
    g_path = tmp_path / "groups.parquet"
    X.to_parquet(x_path, index=False)
    pd.Series(labels, name="readmission_30d").to_frame().to_parquet(y_path, index=False)
    pd.Series(groups, name="subject_id").to_frame().to_parquet(g_path, index=False)
    params = {
        "n_estimators": 15,
        "max_depth": 2,
        "enable_categorical": True,
        "tree_method": "hist",
        "random_state": 42,
    }
    p_path = tmp_path / "best_params.json"
    p_path.write_text(json.dumps(params))
    return str(x_path), str(y_path), str(g_path), str(p_path)


def test_run_calibrate_threshold_writes_curve_and_returns_threshold(tmp_path):
    x_path, y_path, g_path, p_path = _write_train(tmp_path)
    out = tmp_path / "threshold.json"
    threshold = run_calibrate_threshold(
        x_train_path=x_path,
        y_train_path=y_path,
        groups_path=g_path,
        best_params_path=p_path,
        cat_features=[],
        beta=2.0,
        threshold_output_path=str(out),
        n_splits=3,
    )
    assert 0.0 < threshold < 1.0
    record = json.loads(out.read_text())
    assert record["threshold"] == threshold
    assert record["beta"] == 2.0
    assert record["objective"] == "fbeta"
    assert record["n_train"] == 120
    assert 0.0 < record["prevalence"] < 1.0
    assert len(record["curve"]) > 0
    assert all({"threshold", "precision", "recall", "fbeta"} <= set(r) for r in record["curve"])


def test_build_eval_html_renders_images():
    result = {
        "beta": 2.0,
        "tuned_threshold": 0.30,
        "test_aucpr": 0.42,
        "test_auroc": 0.71,
        "brier_score": 0.18,
        "net_benefit_at_threshold": 0.05,
        "point_metrics": {
            "precision": 0.4, "recall": 0.7, "specificity": 0.6, "npv": 0.85,
            "fbeta": 0.6, "tp": 70, "fp": 105, "tn": 600, "fn": 30,
        },
        "pr": {"recall": [0.0, 0.5, 1.0], "precision": [1.0, 0.5, 0.2]},
        "calibration": {"prob_pred": [0.1, 0.5, 0.9], "prob_true": [0.08, 0.52, 0.88]},
        "dca": [
            {"pt": 0.1, "model": 0.10, "treat_all": 0.08, "treat_none": 0.0},
            {"pt": 0.3, "model": 0.05, "treat_all": -0.02, "treat_none": 0.0},
        ],
    }
    html = _build_eval_html(result)
    assert html.count("data:image/png;base64,") == 3  # PR, calibration, DCA
    assert "Decision Curve Analysis" in html
    assert "Operating threshold" in html
