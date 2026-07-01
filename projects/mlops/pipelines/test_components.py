#!/usr/bin/env python3
"""
test_components.py — Local dry-run of every pipeline component.

Runs each run_*() function in sequence with a small data sample and
reduced parameters.  Catches import errors, shape mismatches, and
dependency issues before the pipeline hits Vertex AI.

Usage:
    python pipelines/test_components.py

Prerequisites:
    - BigQuery credentials available (gcloud auth application-default login)
    - Fitted imputer uploaded to GCS (IMPUTER_GCS_PATH env var or default)
    - All pip dependencies installed in the current venv
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    PROJECT_ID, FULL_TABLE_REF, LABEL_COLUMN,
    SPLIT_COLUMN, SPLITS,
)

# ---------------------------------------------------------------------------
# Test configuration — small data, fast params.
# ---------------------------------------------------------------------------
IMPUTER_GCS_PATH = os.environ.get(
    "IMPUTER_GCS_PATH",
    f"gs://{PROJECT_ID}-pipeline-artifacts/imputer/imputer.joblib",
)
N_TEST_ROWS = 5000
SELECTED_FEATURES = [
    "prior_inpatient_days", "oncology_flag", "prior_admission_count",
    "rdw_max", "recent_ed_visits", "has_procedure", "admission_type",
    "discharge_location", "sodium_min", "hemoglobin_min", "procedure_count",
    "sodium_last", "medication_order_count", "age", "index_los_days",
    "race", "gender", "monocytes_min", "medication_count", "rbc_min",
    "rbc_last", "hemoglobin_last", "on_anticoagulant", "monocytes_last",
    "insurance", "glucose_delta", "monocytes_delta", "rdw_last",
    "glucose_max", "diagnosis_count", "sodium_max", "rbc_max", "sodium_delta",
]
CAT_FEATURES = ["gender", "race", "admission_type", "insurance", "discharge_location"]

DEFAULT_XGB_PARAMS = dict(
    n_estimators=10, max_depth=3, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, eval_metric="logloss", enable_categorical=True,
)

passed = 0
failed = 0
tmpdir = tempfile.mkdtemp(prefix="pipeline_test_")
artifacts: dict[str, str] = {}  # component_name -> output_path


def _ok(name: str):
    global passed
    passed += 1
    print(f"  [{name}] PASS\n")


def _fail(name: str, error: Exception):
    global failed
    failed += 1
    print(f"  [{name}] FAIL: {type(error).__name__}: {error}\n")


# ===================================================================
# 1. Load Data
# ===================================================================
print("=" * 60)
print("1. load_data")
print("=" * 60)

try:
    from pipelines.components.load_data import run_load_data

    x_train = f"{tmpdir}/x_train.parquet"
    y_train = f"{tmpdir}/y_train.parquet"
    x_val = f"{tmpdir}/x_val.parquet"
    y_val = f"{tmpdir}/y_val.parquet"
    x_test = f"{tmpdir}/x_test.parquet"
    y_test = f"{tmpdir}/y_test.parquet"

    run_load_data(
        project_id=PROJECT_ID,
        full_table_ref=FULL_TABLE_REF,
        label_col=LABEL_COLUMN,
        split_col=SPLIT_COLUMN,
        train_split=SPLITS["train"],
        val_split=SPLITS["validation"],
        test_split=SPLITS["test"],
        selected_features=SELECTED_FEATURES,
        cat_features=CAT_FEATURES,
        imputer_gcs_path=IMPUTER_GCS_PATH,
        x_train_path=x_train, y_train_path=y_train,
        x_val_path=x_val, y_val_path=y_val,
        x_test_path=x_test, y_test_path=y_test,
    )
    artifacts["x_train"] = x_train
    artifacts["y_train"] = y_train
    artifacts["x_val"] = x_val
    artifacts["y_val"] = y_val
    artifacts["x_test"] = x_test
    artifacts["y_test"] = y_test
    _ok("load_data")
except Exception as e:
    _fail("load_data", e)

# ===================================================================
# 2. Validate Data
# ===================================================================
print("=" * 60)
print("2. validate_data")
print("=" * 60)

try:
    from pipelines.components.validate_data import run_validate_data

    drift_html = f"{tmpdir}/drift.html"
    quality_html = f"{tmpdir}/quality.html"

    result = run_validate_data(
        x_train_path=artifacts["x_train"],
        x_val_path=artifacts["x_val"],
        drift_report_html=drift_html,
        quality_report_html=quality_html,
        max_drifted_share=1.0,  # always pass during test
    )
    artifacts["drift_html"] = drift_html
    artifacts["quality_html"] = quality_html
    _ok("validate_data")
except Exception as e:
    _fail("validate_data", e)

# ===================================================================
# 3. Benchmark XGBoost
# ===================================================================
print("=" * 60)
print("3. benchmark_xgboost")
print("=" * 60)

try:
    from pipelines.components.benchmark_xgboost import run_benchmark_xgboost

    model_path = f"{tmpdir}/benchmark_model.joblib"
    aucpr = run_benchmark_xgboost(
        x_train_path=artifacts["x_train"],
        y_train_path=artifacts["y_train"],
        x_val_path=artifacts["x_val"],
        y_val_path=artifacts["y_val"],
        xgb_params=DEFAULT_XGB_PARAMS,
        cat_features=CAT_FEATURES,
        model_artifact_path=model_path,
    )
    artifacts["benchmark_model"] = model_path
    artifacts["benchmark_aucpr"] = aucpr
    print(f"  Benchmark AUCPR: {aucpr:.4f}")
    _ok("benchmark_xgboost")
except Exception as e:
    _fail("benchmark_xgboost", e)

# ===================================================================
# 4. Benchmark Gate
# ===================================================================
print("=" * 60)
print("4. benchmark_gate")
print("=" * 60)

try:
    from pipelines.components.benchmark_gate import run_benchmark_gate

    if "benchmark_aucpr" in artifacts:
        aucpr_val = artifacts["benchmark_aucpr"]
        # For testing, we accept any result — just validate the function runs.
        try:
            run_benchmark_gate(benchmark_aucpr=aucpr_val)
        except ValueError:
            pass  # expected if AUCPR < HOSPITAL — test data is too small
        _ok("benchmark_gate")
    else:
        _fail("benchmark_gate", RuntimeError("benchmark not run"))
except Exception as e:
    _fail("benchmark_gate", e)

# ===================================================================
# 5. Optuna HPO (3 trials only)
# ===================================================================
print("=" * 60)
print("5. optuna_hpo (3 trials)")
print("=" * 60)

try:
    from pipelines.components.optuna_hpo import run_optuna_hpo

    best_params_path = f"{tmpdir}/best_params.json"
    best_aucpr = run_optuna_hpo(
        x_train_path=artifacts["x_train"],
        y_train_path=artifacts["y_train"],
        cat_features=CAT_FEATURES,
        n_trials=3,
        best_params_path=best_params_path,
    )
    artifacts["best_params"] = best_params_path
    artifacts["hpo_aucpr"] = best_aucpr
    print(f"  Best AUCPR: {best_aucpr:.4f}")
    _ok("optuna_hpo")
except Exception as e:
    _fail("optuna_hpo", e)

# ===================================================================
# 6. Train Final
# ===================================================================
print("=" * 60)
print("6. train_final")
print("=" * 60)

try:
    from pipelines.components.train_final import run_train_final

    final_model_path = f"{tmpdir}/final_model.joblib"
    final_aucpr = run_train_final(
        x_train_path=artifacts["x_train"],
        y_train_path=artifacts["y_train"],
        x_val_path=artifacts["x_val"],
        y_val_path=artifacts["y_val"],
        best_params_path=artifacts.get("best_params", best_params_path),
        cat_features=CAT_FEATURES,
        model_artifact_path=final_model_path,
    )
    artifacts["final_model"] = final_model_path
    artifacts["final_aucpr"] = final_aucpr
    print(f"  Final AUCPR: {final_aucpr:.4f}")
    _ok("train_final")
except Exception as e:
    _fail("train_final", e)

# ===================================================================
# 7. Evaluate Test
# ===================================================================
print("=" * 60)
print("7. evaluate_test")
print("=" * 60)

try:
    from pipelines.components.evaluate_test import run_evaluate_test

    model_to_eval = artifacts.get("final_model", artifacts.get("benchmark_model"))
    test_aucpr, beat_hospital, stable = run_evaluate_test(
        x_test_path=artifacts["x_test"],
        y_test_path=artifacts["y_test"],
        model_artifact_path=model_to_eval,
        final_val_aucpr=artifacts.get("final_aucpr", 0.0),
        benchmark_aucpr=artifacts.get("benchmark_aucpr", 0.0),
    )
    artifacts["test_aucpr"] = test_aucpr
    print(f"  Test AUCPR: {test_aucpr:.4f}")
    _ok("evaluate_test")
except Exception as e:
    _fail("evaluate_test", e)

# ===================================================================
# 8. SHAP Explain
# ===================================================================
print("=" * 60)
print("8. shap_explain")
print("=" * 60)

try:
    from pipelines.components.shap_explain import run_shap_explain

    shap_png = f"{tmpdir}/shap_summary.png"
    shap_pqt = f"{tmpdir}/shap_values.parquet"

    run_shap_explain(
        x_test_path=artifacts["x_test"],
        model_artifact_path=model_to_eval,
        shap_summary_png=shap_png,
        shap_values_parquet=shap_pqt,
        top_n=10,
    )
    artifacts["shap_png"] = shap_png
    artifacts["shap_pqt"] = shap_pqt
    _ok("shap_explain")
except Exception as e:
    _fail("shap_explain", e)

# ===================================================================
# 9. Fairness Audit
# ===================================================================
print("=" * 60)
print("9. fairness_audit")
print("=" * 60)

try:
    from pipelines.components.fairness_audit import run_fairness_audit

    fairness_json = f"{tmpdir}/fairness_report.json"
    ppv_ok, npv_ok = run_fairness_audit(
        x_test_path=artifacts["x_test"],
        y_test_path=artifacts["y_test"],
        model_artifact_path=model_to_eval,
        fairness_report_json=fairness_json,
    )
    artifacts["fairness_json"] = fairness_json
    _ok("fairness_audit")
except Exception as e:
    _fail("fairness_audit", e)

# ===================================================================
# Summary
# ===================================================================
total = passed + failed
print("=" * 60)
print(f"RESULTS: {passed}/{total} passed, {failed}/{total} failed")
print(f"Artifacts: {tmpdir}")
print("=" * 60)

sys.exit(0 if failed == 0 else 1)
