"""Unit tests for src/thresholds.py — threshold calibration & decision analytics."""

import numpy as np
import pandas as pd
import pytest

from src.thresholds import (
    confusion_counts,
    fbeta_from_counts,
    net_benefit,
    net_benefit_curve,
    oof_probabilities,
    point_metrics,
    select_threshold_fbeta,
    threshold_curve,
)


def test_fbeta_from_counts_matches_definition():
    # tp=8, fp=2, fn=2 -> precision=0.8, recall=0.8
    # F1 = 0.8 ; F2 weights recall == precision here so also 0.8
    assert fbeta_from_counts(8, 2, 2, beta=1.0) == pytest.approx(0.8)
    assert fbeta_from_counts(8, 2, 2, beta=2.0) == pytest.approx(0.8)
    # Recall-heavy asymmetry: many FP, few FN -> F2 > F1 (recall favored)
    assert fbeta_from_counts(9, 9, 1, beta=2.0) > fbeta_from_counts(9, 9, 1, beta=1.0)
    # Undefined -> 0.0
    assert fbeta_from_counts(0, 0, 0, beta=2.0) == 0.0


def test_confusion_counts_and_point_metrics():
    y = [1, 1, 0, 0]
    p = [0.9, 0.4, 0.6, 0.1]
    # threshold 0.5: preds = [1,0,1,0] -> tp=1, fp=1, tn=1, fn=1
    c = confusion_counts(y, p, 0.5)
    assert c == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    m = point_metrics(y, p, 0.5, beta=2.0)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["specificity"] == pytest.approx(0.5)
    assert m["npv"] == pytest.approx(0.5)
    assert m["ppv"] == m["precision"]


def test_threshold_curve_recall_monotone_nonincreasing():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, size=200)
    p = rng.rand(200)
    curve = threshold_curve(y, p, beta=2.0)
    recalls = [row["recall"] for row in curve]
    # Raising the threshold can only drop (never raise) recall.
    assert all(recalls[i] >= recalls[i + 1] - 1e-9 for i in range(len(recalls) - 1))


def test_select_threshold_fbeta_picks_separating_threshold():
    # Perfectly separable at 0.5: positives >=0.7, negatives <=0.3
    y = [1] * 20 + [0] * 20
    p = [0.8] * 20 + [0.2] * 20
    thr, fbeta, curve = select_threshold_fbeta(y, p, beta=2.0)
    assert 0.2 < thr <= 0.8
    assert fbeta == pytest.approx(1.0)


def test_select_threshold_fbeta_tie_breaks_low():
    # Flat perfect region -> tie broken toward the LOWER threshold (higher recall).
    y = [1, 1, 0, 0]
    p = [0.9, 0.85, 0.1, 0.05]
    thr, fbeta, _ = select_threshold_fbeta(y, p, beta=2.0, grid=[0.2, 0.5, 0.87])
    assert fbeta == pytest.approx(1.0)
    assert thr == pytest.approx(0.2)  # lowest threshold among the perfect ties


def test_net_benefit_formula():
    # y = [1,1,0,0], at pt=0.5 with preds threshold 0.5
    y = [1, 1, 0, 0]
    p = [0.9, 0.4, 0.6, 0.1]  # preds=[1,0,1,0] -> tp=1, fp=1, n=4, w=1
    # NB = 1/4 - 1/4 * 1 = 0.0
    assert net_benefit(y, p, 0.5) == pytest.approx(0.0)
    # Out of range -> nan
    assert np.isnan(net_benefit(y, p, 0.0))
    assert np.isnan(net_benefit(y, p, 1.0))


def test_net_benefit_curve_references():
    y = np.array([1] * 30 + [0] * 70)  # prevalence 0.30
    p = np.concatenate([np.full(30, 0.8), np.full(70, 0.2)])
    curve = net_benefit_curve(y, p, pts=[0.1, 0.3, 0.5])
    for row in curve:
        assert row["treat_none"] == 0.0
    # treat_all net benefit is non-increasing in pt and equals prevalence-ish at small pt
    treat_all = [row["treat_all"] for row in curve]
    assert treat_all[0] > treat_all[-1]
    # A well-separated model should beat treat-none somewhere in range
    assert any(row["model"] > 0 for row in curve)


def test_oof_probabilities_every_row_scored_once_grouped():
    # 60 patients x 2 admissions; label is patient-level and feature-correlated.
    rng = np.random.RandomState(1)
    n_patients = 60
    rows = []
    groups = []
    labels = []
    for pid in range(n_patients):
        y = int(pid % 2)  # balanced classes across patients
        for _ in range(2):
            signal = rng.normal(loc=1.0 if y else -1.0)
            rows.append({"f1": signal, "f2": rng.normal()})
            groups.append(pid)
            labels.append(y)
    X = pd.DataFrame(rows)
    y = pd.Series(labels)
    g = pd.Series(groups)
    params = {
        "n_estimators": 15,
        "max_depth": 2,
        "enable_categorical": True,
        "tree_method": "hist",
        "random_state": 42,
    }
    oof = oof_probabilities(X, y, g, params=params, n_splits=3)
    assert oof.shape == (len(X),)
    assert not np.isnan(oof).any()  # every row predicted exactly once
    assert ((oof >= 0) & (oof <= 1)).all()
    # Signal should be learnable: mean OOF prob higher for positives than negatives.
    assert oof[y.values == 1].mean() > oof[y.values == 0].mean()
