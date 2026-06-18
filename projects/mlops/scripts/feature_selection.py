#!/usr/bin/env python3
"""
feature_selection.py — 5-method feature selection for readmission prediction.

Executes the 5 selection methods from workflow.md §2.5, aggregates results
via voting, and writes a feature shortlist. All results logged to the shared
Vertex AI Experiment 'readmission-mlops'.

Usage:
    python projects/mlops/scripts/feature_selection.py
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    chi2,
    f_classif,
    mutual_info_classif,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Ensure projects/mlops is on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.bigquery import run_query
from src.config import (
    CATEGORICAL_FEATURES,
    FULL_TABLE_REF,
    LABEL_COLUMN,
    NON_FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    PROJECT_ID,
    SPLIT_COLUMN,
    SPLITS,
)
from src.imputer import MissingnessImputer

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Experiment identity
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "readmission-mlops"

# ---------------------------------------------------------------------------
# Paths & thresholds
# ---------------------------------------------------------------------------
RID = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S")
FEATURES_VERSION = "v3"
ARTIFACTS_DIR = _PROJECT_ROOT / "artifacts" / "feature_selection" / RID
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

TIER1_TOP_N = 20   # features passed from Tier 1 to Tier 2
VOTE_THRESHOLD = 3  # methods needed to KEEP a feature

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load train and validation splits, impute, return X_train, X_val, y_train, y_val."""
    split_list = ", ".join(f"'{s}'" for s in SPLITS.values())
    sql = f"""
        SELECT *
        FROM `{FULL_TABLE_REF}`
        WHERE {SPLIT_COLUMN} IN ({split_list})
    """
    print(f"Querying {FULL_TABLE_REF} …")
    df = run_query(sql)

    train_df = df[df[SPLIT_COLUMN] == SPLITS["train"]].reset_index(drop=True)
    val_df = df[df[SPLIT_COLUMN] == SPLITS["validation"]].reset_index(drop=True)
    # Drop ID/meta columns
    drop_cols = [c for c in NON_FEATURE_COLUMNS if c in train_df.columns and c != LABEL_COLUMN]
    drop_cols.append(SPLIT_COLUMN)
    train_df = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    val_df = val_df.drop(columns=[c for c in drop_cols if c in val_df.columns])

    # Impute
    imputer = MissingnessImputer()
    train_imputed = imputer.fit_transform(train_df)
    val_imputed = imputer.transform(val_df)

    y_train = train_imputed[LABEL_COLUMN].astype(int)
    y_val = val_imputed[LABEL_COLUMN].astype(int)
    X_train = train_imputed.drop(columns=[LABEL_COLUMN])
    X_val = val_imputed.drop(columns=[LABEL_COLUMN])

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"  Features after imputation: {X_train.shape[1]}")
    print(f"  Label prevalence (train): {y_train.mean():.4f}")
    return X_train, X_val, y_train, y_val


def _encode_categoricals(X_train, X_val):
    """Label-encode categorical columns consistently across train/val."""
    Xt = X_train.copy()
    Xv = X_val.copy()
    encoders = {}
    for col in Xt.columns:
        if Xt[col].dtype == "object" or str(Xt[col].dtype) == "category":
            le = LabelEncoder()
            Xt[col] = Xt[col].astype(str).fillna("missing")
            Xv[col] = Xv[col].astype(str).fillna("missing")
            le.fit(Xt[col])
            Xt[col] = le.transform(Xt[col])
            Xv[col] = le.transform(Xv[col])
            encoders[col] = le
    return Xt, Xv


def _log_to_experiment(run_name: str, params: dict, metrics: dict, artifacts: dict | None = None):
    """Log a run to the shared experiment."""
    try:
        from google.cloud import aiplatform
        aiplatform.init(project=PROJECT_ID, location="us-east1", experiment=EXPERIMENT_NAME)
        with aiplatform.start_run(run_name) as run:
            run.log_params(params)
            run.log_metrics(metrics)
        print(f"  ✓ Logged to experiment '{EXPERIMENT_NAME}': {run_name}")
    except Exception as e:
        print(f"  ⚠ Experiment logging failed: {e}")


# ---------------------------------------------------------------------------
# Tier 1 — Cheap Methods
# ---------------------------------------------------------------------------

def run_filter(X_train, y_train) -> pd.DataFrame:
    """Spearman + mutual info. Returns ranked DataFrame."""
    print("\n── Filter (Spearman + MI) ──")
    run_name = f"filter-{RID}"

    Xt, _ = _encode_categoricals(X_train, X_train)
    # Sample for speed — rank-ordering is stable with 50k rows
    n_sample = min(50000, len(Xt))
    idx = np.random.RandomState(42).choice(len(Xt), n_sample, replace=False)
    Xs = Xt.iloc[idx]
    ys = y_train.iloc[idx]

    scores = []
    for col in Xs.columns:
        r = Xs[col].corr(ys, method="spearman")
        scores.append({"feature": col, "spearman_r": abs(r) if not np.isnan(r) else 0})

    mi = mutual_info_classif(Xs.fillna(0), ys, random_state=42)
    for i, col in enumerate(Xs.columns):
        scores[i]["mutual_info"] = mi[i]

    df_scores = pd.DataFrame(scores).sort_values("mutual_info", ascending=False).reset_index(drop=True)
    df_scores["rank_mi"] = df_scores["mutual_info"].rank(ascending=False)
    df_scores["rank_spearman"] = df_scores["spearman_r"].rank(ascending=False)
    df_scores["rank_avg"] = (df_scores["rank_mi"] + df_scores["rank_spearman"]) / 2

    out = ARTIFACTS_DIR / "filter_importance.csv"
    df_scores.to_csv(out, index=False)
    print(f"  ✓ {out.name}")

    _log_to_experiment(run_name, {"method": "filter"}, {
        "filter_n": len(df_scores),
        f"filter_top5": ", ".join(df_scores.head(5)["feature"].tolist()),
    })
    return df_scores


def run_lasso(X_train, X_val, y_train, y_val) -> pd.DataFrame:
    """LASSO logistic regression. Returns coefficients."""
    print("\n── Embedded LASSO ──")
    run_name = f"lasso-{RID}"

    Xt, Xv = _encode_categoricals(X_train, X_val)
    scaler = StandardScaler()
    # Sample for speed — LASSO rank-ordering is stable with 50k rows
    n_sample = min(50000, len(Xt))
    idx = np.random.RandomState(42).choice(len(Xt), n_sample, replace=False)
    Xt_s = scaler.fit_transform(Xt.fillna(0).iloc[idx])
    yt_s = y_train.iloc[idx]
    Xv_s = scaler.transform(Xv.fillna(0))

    model = LogisticRegression(penalty="l1", solver="saga", C=0.1, max_iter=500, random_state=42)
    model.fit(Xt_s, yt_s)
    val_aucpr = average_precision_score(y_val, model.predict_proba(Xv_s)[:, 1])

    df = pd.DataFrame({
        "feature": X_train.columns,
        "coefficient": model.coef_[0],
        "abs_coef": np.abs(model.coef_[0]),
        "selected": model.coef_[0] != 0,
    }).sort_values("abs_coef", ascending=False).reset_index(drop=True)
    df["rank_lasso"] = df["abs_coef"].rank(ascending=False)

    out = ARTIFACTS_DIR / "lasso_coefficients.csv"
    df.to_csv(out, index=False)
    print(f"  ✓ {out.name} — {df.selected.sum()} selected, val AUCPR={val_aucpr:.4f}")

    _log_to_experiment(run_name, {"method": "lasso", "C": 0.1}, {
        "lasso_n_selected": int(df.selected.sum()),
        "lasso_val_aucpr": val_aucpr,
    })
    return df


def run_lgbm(X_train, X_val, y_train, y_val) -> pd.DataFrame:
    """LightGBM gain importance."""
    print("\n── Embedded LGBM ──")
    run_name = f"lgbm-{RID}"

    from lightgbm import LGBMClassifier

    Xt, Xv = _encode_categoricals(X_train, X_val)
    model = LGBMClassifier(n_estimators=200, max_depth=5, random_state=42, verbose=-1)
    model.fit(Xt, y_train, eval_set=[(Xv, y_val)])
    val_aucpr = average_precision_score(y_val, model.predict_proba(Xv)[:, 1])

    df = pd.DataFrame({
        "feature": X_train.columns,
        "gain": model.feature_importances_,
        "gain_pct": model.feature_importances_ / model.feature_importances_.sum() * 100,
    }).sort_values("gain", ascending=False).reset_index(drop=True)
    df["rank_lgbm"] = df["gain"].rank(ascending=False)

    out = ARTIFACTS_DIR / "lgbm_gain.csv"
    df.to_csv(out, index=False)
    print(f"  ✓ {out.name} — top 5: {', '.join(df.head(5)['feature'].tolist())}")
    print(f"    val AUCPR={val_aucpr:.4f}")

    _log_to_experiment(run_name, {"method": "lgbm"}, {
        "lgbm_val_aucpr": val_aucpr,
        f"lgbm_top5": ", ".join(df.head(5)["feature"].tolist()),
    })
    return df


# ---------------------------------------------------------------------------
# Tier 2 — Expensive Methods (on reduced set)
# ---------------------------------------------------------------------------

def run_rfe(X_train, X_val, y_train, y_val, features_subset) -> pd.DataFrame:
    """Recursive feature elimination with XGBoost."""
    print("\n── Wrapper RFE ──")
    run_name = f"rfe-{RID}"

    from xgboost import XGBClassifier

    subset = [f for f in features_subset if f in X_train.columns]
    Xt_all = X_train[subset]
    Xv_all = X_val[subset]
    Xt, Xv = _encode_categoricals(Xt_all, Xv_all)
    Xt = Xt.fillna(0)
    Xv = Xv.fillna(0)

    model = XGBClassifier(n_estimators=100, max_depth=4, random_state=42, verbosity=0)
    rfe = RFE(model, n_features_to_select=max(5, len(subset) // 2), step=1)
    rfe.fit(Xt, y_train)

    df = pd.DataFrame({
        "feature": subset,
        "rfe_rank": rfe.ranking_,
        "rfe_support": rfe.support_,
    }).sort_values("rfe_rank").reset_index(drop=True)

    out = ARTIFACTS_DIR / "rfe_ranking.csv"
    df.to_csv(out, index=False)
    retained = df["rfe_support"].sum()
    print(f"  ✓ {out.name} — retained {retained}/{len(subset)}")

    _log_to_experiment(run_name, {"method": "rfe", "n_input_features": len(subset)}, {
        "rfe_n_retained": int(retained),
    })
    return df


def run_boruta(X_train, y_train, features_subset) -> pd.DataFrame:
    """Boruta feature selection with shadow features."""
    print("\n── Wrapper Boruta ──")
    run_name = f"boruta-{RID}"

    try:
        from boruta import BorutaPy
    except ImportError:
        print("  ⚠ boruta not installed — skipping. pip install boruta")
        _log_to_experiment(run_name, {"method": "boruta", "status": "skipped"}, {})
        return pd.DataFrame(columns=["feature", "decision"])

    subset = [f for f in features_subset if f in X_train.columns]
    Xt, _ = _encode_categoricals(X_train[subset], X_train[subset])
    Xt = Xt.fillna(0)

    rf = RandomForestClassifier(n_jobs=-1, max_depth=5, random_state=42)
    boruta = BorutaPy(rf, n_estimators="auto", perc=100, random_state=42)
    boruta.fit(Xt.values, y_train.values)

    decisions = ["Confirmed" if s else "Rejected" for s in boruta.support_]
    df = pd.DataFrame({
        "feature": subset,
        "decision": decisions,
        "boruta_rank": boruta.ranking_,
    }).sort_values("boruta_rank").reset_index(drop=True)

    out = ARTIFACTS_DIR / "boruta_decisions.csv"
    df.to_csv(out, index=False)
    n_confirmed = (df["decision"] == "Confirmed").sum()
    print(f"  ✓ {out.name} — confirmed {n_confirmed}/{len(subset)}")

    _log_to_experiment(run_name, {"method": "boruta", "n_input_features": len(subset)}, {
        "boruta_n_confirmed": int(n_confirmed),
    })
    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_tier1(df_filter, df_lasso, df_lgbm, n_top=TIER1_TOP_N) -> list[str]:
    """Borda-count rank fusion across Tier 1 methods."""
    print(f"\n── Aggregate Tier 1 (top {n_top}) ──")

    scores = {}
    for _, r in df_filter.iterrows():
        scores[r["feature"]] = scores.get(r["feature"], 0) + r.get("rank_avg", 999)
    for _, r in df_lasso.iterrows():
        scores[r["feature"]] = scores.get(r["feature"], 0) + r.get("rank_lasso", 999)
    for _, r in df_lgbm.iterrows():
        scores[r["feature"]] = scores.get(r["feature"], 0) + r.get("rank_lgbm", 999)

    ranked = sorted(scores.items(), key=lambda x: x[1])
    top = [f for f, _ in ranked[:n_top]]
    print(f"  Top {len(top)}: {', '.join(top[:8])}...")
    return top


def aggregate_final(
    df_filter, df_lasso, df_lgbm, df_rfe, df_boruta, all_features
) -> pd.DataFrame:
    """Vote across all 5 methods: KEEP (≥3), REVIEW (2), DROP (≤1)."""
    print(f"\n── Aggregate Final (threshold={VOTE_THRESHOLD}) ──")

    votes: dict[str, int] = {f: 0 for f in all_features}

    # Filter: top TIER1_TOP_N by avg rank
    top_filter = set(df_filter.sort_values("rank_avg").head(TIER1_TOP_N)["feature"])
    for f in top_filter:
        votes[f] += 1

    # LASSO: coefficient != 0
    lasso_selected = set(df_lasso[df_lasso["selected"]]["feature"])
    for f in lasso_selected:
        votes[f] += 1

    # LGBM: gain > 1% of total
    lgbm_selected = set(df_lgbm[df_lgbm["gain_pct"] > 1.0]["feature"])
    for f in lgbm_selected:
        votes[f] += 1

    # RFE: support
    if not df_rfe.empty:
        rfe_selected = set(df_rfe[df_rfe["rfe_support"]]["feature"])
        for f in rfe_selected:
            votes[f] += 1

    # Boruta: confirmed
    if not df_boruta.empty:
        boruta_selected = set(df_boruta[df_boruta["decision"] == "Confirmed"]["feature"])
        for f in boruta_selected:
            votes[f] += 1

    result = []
    for f, v in votes.items():
        if v >= VOTE_THRESHOLD:
            decision = "KEEP"
        elif v == 2:
            decision = "REVIEW"
        else:
            decision = "DROP"
        result.append({"feature": f, "votes": v, "decision": decision})

    df = pd.DataFrame(result).sort_values(["decision", "votes"], ascending=[True, False]).reset_index(drop=True)
    print(f"  KEEP: {(df.decision == 'KEEP').sum()}, REVIEW: {(df.decision == 'REVIEW').sum()}, DROP: {(df.decision == 'DROP').sum()}")

    out = ARTIFACTS_DIR / "feature_shortlist.csv"
    df.to_csv(out, index=False)
    print(f"  ✓ {out.name}")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print(f"Feature Selection — {RID}")
    print(f"Experiment:  {EXPERIMENT_NAME}")
    print(f"Artifacts:   {ARTIFACTS_DIR.relative_to(_PROJECT_ROOT)}")
    print("=" * 60)

    # 0. Load & impute data
    X_train, X_val, y_train, y_val = load_data()
    all_features = list(X_train.columns)

    # 1. Tier 1 — cheap methods (sequential for clarity; can parallelize later)
    df_filter = run_filter(X_train, y_train)
    df_lasso = run_lasso(X_train, X_val, y_train, y_val)
    df_lgbm = run_lgbm(X_train, X_val, y_train, y_val)

    # 2. Aggregate Tier 1
    tier1_features = aggregate_tier1(df_filter, df_lasso, df_lgbm)

    # 3. Tier 2 — expensive methods on reduced set
    df_rfe = run_rfe(X_train, X_val, y_train, y_val, tier1_features)
    df_boruta = run_boruta(X_train, y_train, tier1_features)

    # 4. Final aggregation
    df_final = aggregate_final(df_filter, df_lasso, df_lgbm, df_rfe, df_boruta, all_features)

    # 5. Log summary to experiment
    keep_n = int((df_final.decision == "KEEP").sum())
    review_n = int((df_final.decision == "REVIEW").sum())
    print(f"\n{'=' * 60}")
    print(f"Feature Selection Complete")
    print(f"  KEEP:   {keep_n}")
    print(f"  REVIEW: {review_n}")
    print(f"  DROP:   {int((df_final.decision == 'DROP').sum())}")
    print(f"{'=' * 60}")

    _log_to_experiment(f"feature-selection-{RID}", {
        "tier1_top_n": TIER1_TOP_N,
        "vote_threshold": VOTE_THRESHOLD,
        "features_version": FEATURES_VERSION,
    }, {
        "fs_n_keep": keep_n,
        "fs_n_review": review_n,
    })

    return 0


if __name__ == "__main__":
    sys.exit(main())
