"""
feature_selection — leakage-controlled feature selection utilities.

Mirrors the production modeling regime so features are validated under the SAME
conditions the final model uses (otherwise a feature set can look validated here
yet mislead downstream):

  * patient-grouped, class-stratified CV (StratifiedGroupKFold on subject_id) —
    no patient's admissions straddle a fold, matching the outer split's leakage
    guard;
  * NATIVE XGBoost categorical handling (no ordinal ``.cat.codes`` — that
    imposed a false ordering and differed from the production encoding);
  * class-imbalance weighting via ``scale_pos_weight``;
  * AUCPR (average_precision) scoring throughout.

The core is ``grouped_cv_rfe`` — a recursive feature elimination that preserves
native category dtype (sklearn's RFECV cannot, which forced the previous
ordinal-code workaround) and scores each candidate subset with grouped CV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier


def scale_pos_weight(y: pd.Series) -> float:
    """Empirical class-imbalance ratio (neg/pos); 1.0 if there are no positives."""
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return float(neg / pos) if pos > 0 else 1.0


def grouped_folds(X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int = 5):
    """Patient-grouped, class-stratified CV folds (list of (train_idx, val_idx))."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return list(sgkf.split(X, y, groups))


def default_params(spw: float) -> dict:
    """Selection estimator params — aligned with the production XGBoost regime."""
    return {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "aucpr",
        "enable_categorical": True,
        "tree_method": "hist",
        "scale_pos_weight": spw,
    }


def _cv_fit(X, y, groups, *, params, n_splits=5):
    """Sequential grouped CV; each fit uses all cores (params' n_jobs).

    Returns (mean AUCPR, std, fold-averaged gain). Folds run sequentially so
    each XGBoost fit gets the full core count (faster per fit than splitting
    cores across concurrent single-threaded fits), and the fold models' gain is
    reused for elimination — no separate full-data fit per step.
    """
    aps = []
    importances = []
    for train_idx, val_idx in grouped_folds(X, y, groups, n_splits):
        model = XGBClassifier(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx], verbose=False)
        proba = model.predict_proba(X.iloc[val_idx])[:, 1]
        aps.append(average_precision_score(y.iloc[val_idx], proba))
        importances.append(model.feature_importances_)
    mean_importance = np.mean(np.vstack(importances), axis=0)
    return float(np.mean(aps)), float(np.std(aps)), mean_importance


def cv_aucpr(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    params: dict,
    n_splits: int = 5,
) -> tuple[float, float]:
    """Mean/std grouped-CV AUCPR for a feature subset (native categoricals)."""
    mean, std, _ = _cv_fit(X, y, groups, params=params, n_splits=n_splits)
    return mean, std


def gain_ranking(X: pd.DataFrame, y: pd.Series, *, params: dict) -> pd.DataFrame:
    """XGBoost gain importance ranking (native categoricals, imbalance-weighted)."""
    model = XGBClassifier(**params)
    model.fit(X, y, verbose=False)
    df = pd.DataFrame({"feature": list(X.columns), "gain": model.feature_importances_})
    df = df.sort_values("gain", ascending=False).reset_index(drop=True)
    total = float(df["gain"].sum())
    df["gain_pct"] = (df["gain"] / total * 100) if total > 0 else 0.0
    df["gain_rank"] = range(1, len(df) + 1)
    return df


def select_one_se(curve: list[dict], n_splits: int = 5) -> dict:
    """Apply the one-standard-error rule to an elimination curve.

    Returns the curve entry for the **smallest** feature set whose mean CV
    AUCPR is within one standard error of the best (peak) mean. The standard
    error of the peak is ``std_across_folds / sqrt(n_splits)`` (Hastie,
    Tibshirani & Friedman, ESL §7.10). This prefers the simplest model that is
    statistically indistinguishable from the best — the principled way to turn a
    flat performance plateau into a parsimonious selection instead of chasing a
    noise-level argmax.
    """
    best = max(curve, key=lambda r: r["mean_aucpr"])
    se = best["std_aucpr"] / (n_splits ** 0.5)
    threshold = best["mean_aucpr"] - se
    within = [r for r in curve if r["mean_aucpr"] >= threshold]
    return min(within, key=lambda r: r["n_features"])


def grouped_cv_rfe(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    params: dict,
    n_splits: int = 5,
    min_features: int = 5,
    verbose: bool = False,
    one_se: bool = False,
) -> tuple[list[str], list[dict]]:
    """Recursive feature elimination with grouped CV, preserving categoricals.

    At each step: score the current feature subset with grouped CV, then drop
    the single weakest feature by gain **averaged over the fold models just
    fit** — so no separate full-data fit is needed, and the drop decision is
    redundancy-aware. Continues down to ``min_features``. Set ``verbose=True``
    to print progress per step.

    Selection rule for the returned subset:
      * ``one_se=False`` (default): the peak mean CV AUCPR (argmax).
      * ``one_se=True``: the one-standard-error rule — the smallest feature set
        within one SE of the peak (see ``select_one_se``). Prefer this when the
        curve is flat, so dimensionality actually drops.

    Returns ``(selected_features, curve)`` where ``curve`` is a list of dicts
    (``n_features``, ``mean_aucpr``, ``std_aucpr``, ``features``) ordered from
    all features down to ``min_features``.
    """
    features = list(X.columns)
    curve: list[dict] = []

    while True:
        mean, std, importances = _cv_fit(
            X[features], y, groups, params=params, n_splits=n_splits,
        )
        curve.append(
            {
                "n_features": len(features),
                "mean_aucpr": mean,
                "std_aucpr": std,
                "features": list(features),
            }
        )
        if verbose:
            print(f"  {len(features):3d} features → CV AUCPR {mean:.4f} ± {std:.4f}", flush=True)
        if len(features) <= min_features:
            break
        # Drop the weakest feature by fold-averaged gain (reuses the fits above).
        weakest = features[int(np.argmin(importances))]
        features.remove(weakest)

    if one_se:
        chosen = select_one_se(curve, n_splits)
    else:
        chosen = max(curve, key=lambda row: row["mean_aucpr"])
    return chosen["features"], curve



