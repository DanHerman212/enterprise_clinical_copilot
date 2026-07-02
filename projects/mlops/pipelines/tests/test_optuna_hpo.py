"""Tests for optuna_hpo (pipeline module 5).

Pins the corrected HPO contract:
  * cross-validation is patient-grouped (StratifiedGroupKFold on subject_id) so
    no patient appears in both the train and validation side of a fold — the
    within-train leakage guard that the cross-split hash does NOT provide;
  * scale_pos_weight is derived from the empirical class imbalance;
  * the study runs (TPE sampler, no pruner) and writes best params carrying the
    new search space plus the fixed model settings.
"""

import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from pipelines.components.optuna_hpo import (
    _grouped_folds,
    _scale_pos_weight,
    run_optuna_hpo,
)

CAT = ["gender"]


def _grouped_frame(n_patients=40, rows_per=2, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    groups = []
    labels = []
    for pid in range(n_patients):
        # Label assigned per patient so grouped stratification is well-defined.
        y = int(pid % 2 == 0)
        for _ in range(rows_per):
            rows.append(
                {
                    "age": rng.normal(60 + 5 * y, 8),
                    "glucose_last": rng.normal(100 + 15 * y, 12),
                    "gender": rng.choice(["M", "F"]),
                }
            )
            groups.append(pid)
            labels.append(y)
    X = pd.DataFrame(rows)
    y = pd.Series(labels, name="readmission_30d")
    g = pd.Series(groups, name="subject_id")
    return X, y, g


def test_grouped_folds_are_patient_disjoint():
    X, y, g = _grouped_frame()
    folds = _grouped_folds(X, y, g, n_splits=3)
    assert len(folds) == 3
    for train_idx, val_idx in folds:
        train_patients = set(g.iloc[train_idx])
        val_patients = set(g.iloc[val_idx])
        assert train_patients.isdisjoint(val_patients)


def test_scale_pos_weight_is_neg_over_pos():
    y = pd.Series([1, 1, 0, 0, 0, 0, 0, 0])  # 2 pos, 6 neg -> 3.0
    assert _scale_pos_weight(y) == 3.0


def test_scale_pos_weight_handles_no_positives():
    y = pd.Series([0, 0, 0])
    assert _scale_pos_weight(y) == 1.0


def test_run_optuna_writes_best_params_and_returns_score(tmp_path):
    X, y, g = _grouped_frame()
    xp = tmp_path / "x.parquet"
    yp = tmp_path / "y.parquet"
    gp = tmp_path / "g.parquet"
    X.to_parquet(xp, index=False)
    pd.DataFrame(y).to_parquet(yp, index=False)
    pd.DataFrame(g).to_parquet(gp, index=False)
    bp = tmp_path / "best_params.json"

    score = run_optuna_hpo(
        x_train_path=str(xp),
        y_train_path=str(yp),
        groups_path=str(gp),
        cat_features=CAT,
        n_trials=2,
        best_params_path=str(bp),
        n_splits=3,
    )
    assert 0.0 <= score <= 1.0

    params = json.loads(bp.read_text())
    # Search-space params tuned by Optuna.
    for key in (
        "n_estimators", "learning_rate", "max_depth", "min_child_weight",
        "gamma", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda",
        "scale_pos_weight",
    ):
        assert key in params, f"missing tuned param: {key}"
    # Fixed model settings needed for reproducible refit with categoricals.
    assert params["enable_categorical"] is True
    assert params["random_state"] == 42


def test_search_space_bounds_are_respected(tmp_path):
    X, y, g = _grouped_frame()
    xp = tmp_path / "x.parquet"
    yp = tmp_path / "y.parquet"
    gp = tmp_path / "g.parquet"
    X.to_parquet(xp, index=False)
    pd.DataFrame(y).to_parquet(yp, index=False)
    pd.DataFrame(g).to_parquet(gp, index=False)
    bp = tmp_path / "best_params.json"

    run_optuna_hpo(
        x_train_path=str(xp), y_train_path=str(yp), groups_path=str(gp),
        cat_features=CAT, n_trials=3, best_params_path=str(bp), n_splits=3,
    )
    p = json.loads(bp.read_text())
    assert 200 <= p["n_estimators"] <= 800
    assert 3 <= p["max_depth"] <= 8
    assert 0.01 <= p["learning_rate"] <= 0.2
    assert 0.0 <= p["gamma"] <= 5.0
    assert 0.6 <= p["subsample"] <= 1.0
