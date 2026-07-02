"""Tests for src/feature_selection.py (leakage-controlled feature selection).

Pins the properties that were flawed in feature_selection_v2.ipynb:
  * CV folds are patient-grouped (no subject_id across a fold);
  * scale_pos_weight reflects class imbalance;
  * recursive elimination preserves NATIVE categorical dtype (no ordinal codes)
    and actually drops non-informative features while keeping the signal.
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.feature_selection import (
    cv_aucpr,
    default_params,
    gain_ranking,
    grouped_cv_rfe,
    grouped_folds,
    scale_pos_weight,
    select_one_se,
)


def _grouped_data(n_patients=60, rows_per=2, n_noise=6, seed=0):
    rng = np.random.default_rng(seed)
    rows, groups, labels = [], [], []
    for pid in range(n_patients):
        signal = rng.normal()
        y = int(signal > 0)
        for _ in range(rows_per):
            row = {"signal": signal + rng.normal(0, 0.1)}
            for j in range(n_noise):
                row[f"noise_{j}"] = rng.normal()
            row["cat"] = rng.choice(["A", "B", "C"])
            rows.append(row)
            groups.append(pid)
            labels.append(y)
    X = pd.DataFrame(rows)
    X["cat"] = X["cat"].astype("category")
    return X, pd.Series(labels, name="y"), pd.Series(groups, name="subject_id")


def test_scale_pos_weight():
    assert scale_pos_weight(pd.Series([1, 1, 0, 0, 0, 0])) == 2.0
    assert scale_pos_weight(pd.Series([0, 0])) == 1.0


def test_grouped_folds_are_patient_disjoint():
    X, y, g = _grouped_data()
    for train_idx, val_idx in grouped_folds(X, y, g, n_splits=3):
        assert set(g.iloc[train_idx]).isdisjoint(set(g.iloc[val_idx]))


def test_cv_aucpr_returns_valid_score():
    X, y, g = _grouped_data()
    params = default_params(scale_pos_weight(y))
    params["n_estimators"] = 10
    mean, std = cv_aucpr(X, y, g, params=params, n_splits=3)
    assert 0.0 <= mean <= 1.0 and std >= 0.0


def test_gain_ranking_handles_native_categorical():
    X, y, g = _grouped_data()
    params = default_params(scale_pos_weight(y))
    params["n_estimators"] = 10
    ranking = gain_ranking(X, y, params=params)
    assert set(ranking["feature"]) == set(X.columns)
    # "cat" is category dtype and must be handled without ordinal conversion.
    assert "cat" in ranking["feature"].values


def test_grouped_cv_rfe_reduces_and_keeps_signal():
    X, y, g = _grouped_data(n_noise=6)
    params = default_params(scale_pos_weight(y))
    params["n_estimators"] = 20
    selected, curve = grouped_cv_rfe(
        X, y, g, params=params, n_splits=3, min_features=2,
    )
    # Curve goes from all features down to min_features.
    assert curve[0]["n_features"] == X.shape[1]
    assert curve[-1]["n_features"] == 2
    # The dominant signal feature is never the weakest, so it survives to the
    # very end of elimination (and is therefore in the selected set).
    assert "signal" in curve[-1]["features"]
    assert "signal" in selected
    # Selection rule: selected is exactly the peak mean-AUCPR subset.
    peak = max(curve, key=lambda row: row["mean_aucpr"])
    assert selected == peak["features"]


def test_grouped_cv_rfe_respects_min_features():
    X, y, g = _grouped_data()
    params = default_params(scale_pos_weight(y))
    params["n_estimators"] = 10
    _, curve = grouped_cv_rfe(X, y, g, params=params, n_splits=3, min_features=3)
    assert min(row["n_features"] for row in curve) == 3


# ---------------------------------------------------------------------------
# one-standard-error rule
# ---------------------------------------------------------------------------

def _curve(entries):
    # entries: list of (n_features, mean, std) -> curve dicts with dummy features
    return [
        {"n_features": n, "mean_aucpr": m, "std_aucpr": s,
         "features": [f"f{i}" for i in range(n)]}
        for (n, m, s) in entries
    ]


def test_select_one_se_picks_smallest_within_band():
    # Peak is 0.505 at 8 features (std 0.02 -> SE = 0.02/sqrt(5) ≈ 0.0089;
    # threshold ≈ 0.4961). The 5-feature set (0.500) is within the band; the
    # 3-feature set (0.40) is not. So the 5-feature set must be chosen.
    curve = _curve([(10, 0.500, 0.02), (8, 0.505, 0.02), (5, 0.500, 0.02), (3, 0.40, 0.02)])
    chosen = select_one_se(curve, n_splits=5)
    assert chosen["n_features"] == 5


def test_select_one_se_matches_argmax_when_no_smaller_within_band():
    # Only the peak is within its own SE band; nothing smaller qualifies.
    curve = _curve([(10, 0.50, 0.001), (6, 0.40, 0.001), (3, 0.30, 0.001)])
    chosen = select_one_se(curve, n_splits=5)
    assert chosen["n_features"] == 10


def test_grouped_cv_rfe_one_se_is_not_larger_than_argmax():
    X, y, g = _grouped_data(n_noise=6)
    params = default_params(scale_pos_weight(y))
    params["n_estimators"] = 20
    argmax_sel, curve = grouped_cv_rfe(
        X, y, g, params=params, n_splits=3, min_features=2, one_se=False,
    )
    one_se_sel, _ = grouped_cv_rfe(
        X, y, g, params=params, n_splits=3, min_features=2, one_se=True,
    )
    # 1-SE never selects a larger set than argmax, and keeps the real signal.
    assert len(one_se_sel) <= len(argmax_sel)
    assert "signal" in one_se_sel
    # And it equals select_one_se applied to the same curve.
    assert set(one_se_sel) == set(select_one_se(curve, n_splits=3)["features"])
