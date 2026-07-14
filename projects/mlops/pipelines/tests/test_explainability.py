"""Tests for shap_explain and fairness_audit (pipeline module 8).

These run the real components on model-ready data that mirrors production:
category-dtype features (train-only categories) and float labs with NaN. Prior
to this module both components had zero test coverage.
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

from pipelines.components.data import fit_imputer, prepare_splits
from pipelines.components.fairness_audit import (
    _omb_race,
    _wilson_ci,
    run_fairness_audit,
)
from pipelines.components.shap_explain import run_shap_explain

LABEL = "readmission_30d"
SELECTED = ["age", "gender", "race", "insurance", "discharge_location", "glucose_last"]
CAT = ["gender", "race", "insurance", "discharge_location"]


def _raw(rng, n):
    glucose = rng.normal(120, 30, n)
    glucose[rng.random(n) < 0.1] = np.nan  # some missing labs
    y = ((np.nan_to_num(glucose, nan=120) > 120) ^ (rng.random(n) < 0.3)).astype(int)
    return pd.DataFrame(
        {
            "age": rng.integers(40, 90, n).astype(float),
            "gender": rng.choice(["M", "F"], n),
            "race": rng.choice(["WHITE", "BLACK", "OTHER"], n),
            "insurance": rng.choice(["Medicare", "Medicaid"], n),
            "discharge_location": rng.choice(["HOME", "SNF"], n),
            "glucose_last": glucose,
            LABEL: y,
        }
    )


@pytest.fixture
def model_and_test(tmp_path):
    rng = np.random.default_rng(0)
    train_df, val_df, test_df = _raw(rng, 300), _raw(rng, 100), _raw(rng, 200)
    imputer = fit_imputer(train_df)
    out = prepare_splits(
        train_df, val_df, test_df,
        imputer=imputer, selected_features=SELECTED, cat_features=CAT, label_col=LABEL,
    )
    model = XGBClassifier(n_estimators=15, max_depth=3, enable_categorical=True,
                          tree_method="hist", random_state=42)
    model.fit(out["X_train"], out["y_train"])

    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)
    x_test = tmp_path / "x_test.parquet"
    y_test = tmp_path / "y_test.parquet"
    out["X_test"].to_parquet(x_test, index=False)
    out["y_test"].to_frame(LABEL).to_parquet(y_test, index=False)
    return str(model_path), str(x_test), str(y_test)


# ---------------------------------------------------------------------------
# shap_explain
# ---------------------------------------------------------------------------

def test_shap_explain_outputs_and_importance(tmp_path, model_and_test):
    model_path, x_test, _ = model_and_test
    summary = tmp_path / "summary.png"
    beeswarm = tmp_path / "beeswarm.png"
    waterfall = tmp_path / "waterfall.png"
    local_dir = tmp_path / "local"
    values = tmp_path / "shap.parquet"

    importance = run_shap_explain(
        x_test_path=x_test,
        model_artifact_path=model_path,
        shap_summary_png=str(summary),
        shap_beeswarm_png=str(beeswarm),
        shap_waterfall_png=str(waterfall),
        shap_local_plots_dir=str(local_dir),
        shap_values_parquet=str(values),
        top_n=6,
    )
    assert summary.exists() and beeswarm.exists() and waterfall.exists()
    assert values.exists()
    assert set(importance).issubset(set(SELECTED))
    assert all(np.isfinite(v) and v >= 0 for v in importance.values())


# ---------------------------------------------------------------------------
# fairness_audit
# ---------------------------------------------------------------------------

def test_omb_race_rolls_up_to_standard_buckets():
    assert _omb_race("WHITE - RUSSIAN") == "White"
    assert _omb_race("BLACK/CAPE VERDEAN") == "Black or African American"
    assert _omb_race("ASIAN - CHINESE") == "Asian"
    assert _omb_race("HISPANIC/LATINO - CUBAN") == "Hispanic or Latino"
    assert _omb_race("AMERICAN INDIAN/ALASKA NATIVE") == "American Indian or Alaska Native"
    assert _omb_race(None) == "Other/Unknown"
    assert _omb_race("UNABLE TO OBTAIN") == "Other/Unknown"


def test_wilson_ci_bounds_and_edges():
    lo, hi = _wilson_ci(5, 10)
    assert 0.0 <= lo < 0.5 < hi <= 1.0
    import math as _m
    a, b = _wilson_ci(0, 0)
    assert _m.isnan(a) and _m.isnan(b)


def test_fairness_audit_error_rate_parity_report(tmp_path, model_and_test):
    model_path, x_test, y_test = model_and_test
    report_json = tmp_path / "fairness.json"
    report_html = tmp_path / "fairness.html"

    report = run_fairness_audit(
        x_test_path=x_test,
        y_test_path=y_test,
        model_artifact_path=model_path,
        fairness_report_json=str(report_json),
        fairness_html_path=str(report_html),
        tuned_threshold=0.3,
    )
    assert report_json.exists() and report_html.exists()
    assert isinstance(report["equal_opportunity_pass"], bool)
    assert isinstance(report["predictive_equality_pass"], bool)
    assert report["primary_signal"] == "equal_opportunity_tpr_parity"

    # gender + insurance audited (n>50 per level); each level carries TPR/FPR + CIs.
    assert "gender" in report["subgroups"] and "insurance" in report["subgroups"]
    for level in report["subgroups"]["gender"].values():
        assert "tpr" in level and "fpr" in level
        assert len(level["tpr_ci"]) == 2 and len(level["fpr_ci"]) == 2
    # gaps are computed for each audited subgroup.
    assert "gender" in report["gaps"] and "insurance" in report["gaps"]
