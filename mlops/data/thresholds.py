"""
thresholds — operating-threshold calibration & decision analytics (pure, KFP-free).

The model is trained and evaluated **probability-first**; this module owns the
*decision layer*: turning predicted probabilities into a single operating
threshold and the threshold-dependent diagnostics that follow from it.

Design choices:
  * The threshold is selected on **out-of-fold**, patient-grouped predictions
    (``StratifiedGroupKFold`` on ``subject_id``) so it is chosen on data the
    fold's model never saw — the same leakage guard used in HPO/feature
    selection. Selecting on the training fit would bias the threshold optimistic.
  * The objective is **F-beta** (``beta`` weights recall over precision;
    ``beta=2`` => recall twice as important). The objective is a transient
    selection rule — only the resulting threshold is a persisted, operating
    value.
  * Decision Curve Analysis (net benefit) quantifies clinical usefulness across
    the plausible range of risk thresholds, versus treat-all / treat-none.

Everything here is hermetic (no kfp / GCP imports) so it is unit-testable and is
reused by the ``calibrate_threshold`` and ``evaluate_test`` components.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

# Default operating-threshold search grid (probabilities).
DEFAULT_GRID = np.round(np.arange(0.01, 1.00, 0.01), 4)


# --------------------------------------------------------------------------- #
# Out-of-fold probabilities                                                    #
# --------------------------------------------------------------------------- #
def oof_probabilities(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    params: dict,
    n_splits: int = 5,
) -> np.ndarray:
    """Patient-grouped out-of-fold predicted probabilities, aligned to ``X`` rows.

    Every row is predicted exactly once, by a model trained on the other folds
    with no patient (``groups``) straddling the split.
    """
    y = pd.Series(np.asarray(y).astype(int)).reset_index(drop=True)
    X = X.reset_index(drop=True)
    groups = pd.Series(np.asarray(groups)).reset_index(drop=True)

    oof = np.full(len(X), np.nan, dtype=float)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in sgkf.split(X, y, groups):
        model = XGBClassifier(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx], verbose=False)
        oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    return oof


# --------------------------------------------------------------------------- #
# Confusion counts & point metrics at a threshold                              #
# --------------------------------------------------------------------------- #
def confusion_counts(y_true, y_proba, threshold: float) -> dict:
    """TP/FP/TN/FN for ``y_proba >= threshold``."""
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(y_proba, dtype=float) >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def fbeta_from_counts(tp: int, fp: int, fn: int, beta: float) -> float:
    """F-beta from confusion counts (0.0 when undefined)."""
    b2 = float(beta) ** 2
    denom = (1 + b2) * tp + b2 * fn + fp
    return float((1 + b2) * tp / denom) if denom > 0 else 0.0


def point_metrics(y_true, y_proba, threshold: float, *, beta: float = 2.0) -> dict:
    """Threshold-dependent metrics: precision(=PPV)/recall/specificity/NPV/F-beta."""
    c = confusion_counts(y_true, y_proba, threshold)
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": float(precision),
        "ppv": float(precision),
        "recall": float(recall),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "npv": float(npv),
        "fbeta": fbeta_from_counts(tp, fp, fn, beta),
        "beta": float(beta),
    }


# --------------------------------------------------------------------------- #
# F-beta threshold selection                                                   #
# --------------------------------------------------------------------------- #
def threshold_curve(y_true, y_proba, *, beta: float = 2.0, grid=None) -> list[dict]:
    """precision/recall/F-beta at each candidate threshold in ``grid``."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    grid = DEFAULT_GRID if grid is None else np.asarray(grid, dtype=float)
    curve = []
    for t in grid:
        c = confusion_counts(y_true, y_proba, float(t))
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        curve.append(
            {
                "threshold": float(t),
                "precision": float(precision),
                "recall": float(recall),
                "fbeta": fbeta_from_counts(tp, fp, fn, beta),
            }
        )
    return curve


def select_threshold_fbeta(
    y_true, y_proba, *, beta: float = 2.0, grid=None
) -> tuple[float, float, list[dict]]:
    """Return ``(best_threshold, best_fbeta, curve)`` maximizing F-beta.

    Ties break toward the **lower** threshold (higher recall) — consistent with
    a recall-weighted objective and deterministic.
    """
    curve = threshold_curve(y_true, y_proba, beta=beta, grid=grid)
    best = max(curve, key=lambda r: (r["fbeta"], -r["threshold"]))
    return best["threshold"], best["fbeta"], curve


# --------------------------------------------------------------------------- #
# Decision Curve Analysis (net benefit)                                        #
# --------------------------------------------------------------------------- #
def net_benefit(y_true, y_proba, pt: float) -> float:
    """Net benefit of the model at risk-threshold probability ``pt``.

    NB = TP/n - FP/n * (pt / (1 - pt)), classifying positive when p >= pt
    (Vickers & Elkin, 2006). ``pt`` in (0, 1).
    """
    y_true = np.asarray(y_true).astype(int)
    n = len(y_true)
    if n == 0 or pt <= 0 or pt >= 1:
        return float("nan")
    c = confusion_counts(y_true, y_proba, pt)
    w = pt / (1 - pt)
    return float(c["tp"] / n - c["fp"] / n * w)


def net_benefit_curve(y_true, y_proba, *, pts=None) -> list[dict]:
    """Net-benefit curve: model vs treat-all vs treat-none over ``pts``.

    treat-none NB = 0 everywhere; treat-all NB = prevalence - (1-prevalence)*w.
    Default ``pts`` span 0.01..0.50 (the clinically plausible range for a
    readmission-intervention decision).
    """
    y_true = np.asarray(y_true).astype(int)
    prevalence = float(y_true.mean()) if len(y_true) else float("nan")
    pts = np.round(np.arange(0.01, 0.51, 0.01), 4) if pts is None else np.asarray(pts, dtype=float)
    curve = []
    for pt in pts:
        w = pt / (1 - pt)
        curve.append(
            {
                "pt": float(pt),
                "model": net_benefit(y_true, y_proba, float(pt)),
                "treat_all": float(prevalence - (1 - prevalence) * w),
                "treat_none": 0.0,
            }
        )
    return curve
