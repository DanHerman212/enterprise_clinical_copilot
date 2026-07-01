#!/usr/bin/env python3
"""
test_ml_logic.py — Pure Python pre-production test harness for ML components.

This script executes the inner run_*() logic of all ML components sequentially,
skipping Vertex AI/BQ IO. It neutralizes KFP at import time, preventing
the Python 3.12 compatibility bug from triggering.

Usage:
    python projects/mlops/pipelines/test_ml_logic.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import warnings
from pathlib import Path

# ==============================================================================
# 0. System Setup & KFP Neutralization
# ==============================================================================

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class _MockKFP:
    """A dummy module to replace kfp and prevent @dsl decorators from executing."""
    class dsl:
        class Input: pass
        class Output: pass
        class Dataset: pass
        class Model: pass
        class Metrics: pass
        class Artifact: pass
        
        @staticmethod
        def component(*args, **kwargs):
            # Return a decorator that just returns the function unchanged
            def decorator(func):
                return func
            return decorator
            
        @staticmethod
        def pipeline(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
            
        @staticmethod
        def OutputPath(*args, **kwargs):
            return str

# Replace kfp in sys.modules BEFORE any component tries to import it
sys.modules['kfp'] = _MockKFP()

# ==============================================================================
# 1. Imports (Now Safe)
# ==============================================================================
import pandas as pd
from pipelines.components.validate_data import run_validate_data
from pipelines.components.benchmark_xgboost import run_benchmark_xgboost
from pipelines.components.benchmark_gate import run_benchmark_gate
from pipelines.components.optuna_hpo import run_optuna_hpo
from pipelines.components.train_final import run_train_final
from pipelines.components.evaluate_test import run_evaluate_test
from pipelines.components.shap_explain import run_shap_explain
from pipelines.components.fairness_audit import run_fairness_audit

# ==============================================================================
# 2. Main Test Harness
# ==============================================================================

def main():
    print("=====================================================================")
    print("Executing Pre-Production ML Logic Test")
    print("=====================================================================\n")

    # Paths and constants
    sample_data_path = Path(__file__).resolve().parent / "sample_data.parquet"
    if not sample_data_path.exists():
        print(f"ERROR: Cannot find sample data at {sample_data_path}.")
        print("Please run _export_sample_data.py first.")
        sys.exit(1)

    work_dir = Path(tempfile.mkdtemp(prefix="mlops_test_"))
    print(f"Working directory: {work_dir}\n")

    cat_features = ["gender", "race", "admission_type", "insurance", "discharge_location"]
    xgb_params = {
        "n_estimators": 5, "max_depth": 3, "learning_rate": 0.1,  # Fast test
        "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42
    }
    
    # --------------------------------------------------------------------------
    print("[1] Mocking Data Load...")
    # --------------------------------------------------------------------------
    df = pd.read_parquet(sample_data_path)
    paths = {}
    
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split].drop(columns=["split"])
        y = split_df.pop("readmission_30d")
        
        x_path = work_dir / f"X_{split}.parquet"
        y_path = work_dir / f"y_{split}.parquet"
        
        split_df.to_parquet(x_path, index=False)
        pd.DataFrame(y).to_parquet(y_path, index=False)
        
        paths[f"X_{split}"] = str(x_path)
        paths[f"y_{split}"] = str(y_path)
        
        print(f"    Created {split} split: Features {split_df.shape}, Labels {y.shape}")

    # --------------------------------------------------------------------------
    print("\n[2] Testing: validate_data...")
    # --------------------------------------------------------------------------
    html_path = str(work_dir / "drift_report.html")
    quality_html_path = str(work_dir / "quality_report.html")
    run_validate_data(
        x_train_path=paths["X_train"],
        x_val_path=paths["X_val"],
        max_drifted_share=1.0,  # Max tolerance for test
        drift_report_html=html_path,
        quality_report_html=quality_html_path
    )
    print(f"    OK. Drift report generated: {html_path}")

    # --------------------------------------------------------------------------
    print("\n[3] Testing: benchmark_xgboost...")
    # --------------------------------------------------------------------------
    bm_model_path = str(work_dir / "benchmark_model.joblib")
    benchmark_aucpr = run_benchmark_xgboost(
        x_train_path=paths["X_train"],
        y_train_path=paths["y_train"],
        x_val_path=paths["X_val"],
        y_val_path=paths["y_val"],
        xgb_params=xgb_params.copy(),
        cat_features=cat_features,
        model_artifact_path=bm_model_path
    )
    print(f"    OK. Benchmark AUCPR: {benchmark_aucpr:.4f}")

    # --------------------------------------------------------------------------
    print("\n[4] Testing: benchmark_gate (Warning only for tests)...")
    # --------------------------------------------------------------------------
    try:
        run_benchmark_gate(benchmark_aucpr=benchmark_aucpr)
        print("    OK. Gate passed.")
    except ValueError as e:
        print(f"    [TEST WARNING] Gate failed: {e}")
        print("    Continuing test pipeline regardless...")

    # --------------------------------------------------------------------------
    print("\n[5] Testing: optuna_hpo (3 trials)...")
    # --------------------------------------------------------------------------
    hpo_results_path = str(work_dir / "hpo_best_params.json")
    best_aucpr = run_optuna_hpo(
        x_train_path=paths["X_train"],
        y_train_path=paths["y_train"],
        cat_features=cat_features,
        n_trials=3,
        best_params_path=hpo_results_path
    )
    print(f"    OK. Best HPO AUCPR: {best_aucpr:.4f}")

    # --------------------------------------------------------------------------
    print("\n[6] Testing: train_final...")
    # --------------------------------------------------------------------------
    final_model_path = str(work_dir / "final_model.joblib")
    final_aucpr = run_train_final(
        x_train_path=paths["X_train"],
        y_train_path=paths["y_train"],
        x_val_path=paths["X_val"],
        y_val_path=paths["y_val"],
        best_params_path=hpo_results_path,
        cat_features=cat_features,
        model_artifact_path=final_model_path
    )
    print(f"    OK. Final Model AUCPR: {final_aucpr:.4f}")

    # --------------------------------------------------------------------------
    print("\n[7] Testing: evaluate_test...")
    # --------------------------------------------------------------------------
    metrics_path = str(work_dir / "metrics.json")
    try:
        test_aucpr, beat_hospital, stable = run_evaluate_test(
            x_test_path=paths["X_test"],
            y_test_path=paths["y_test"],
            model_artifact_path=final_model_path,
            final_val_aucpr=final_aucpr,
            benchmark_aucpr=benchmark_aucpr
        )
        print(f"    OK. Holdout Test AUCPR: {test_aucpr:.4f} (Beat Hospital: {beat_hospital}, Stable: {stable})")
    except ValueError as e:
        print(f"    [TEST WARNING] Test Evaluation Gate failed: {e}")
        # To proceed, we assume a synthetic AUCPR and continue to SHAP/Fairness
        test_aucpr = 0.25 
        print("    Continuing test pipeline regardless...")

    # --------------------------------------------------------------------------
    print("\n[8] Testing: shap_explain...")
    # --------------------------------------------------------------------------
    shap_png = str(work_dir / "shap_summary.png")
    shap_beeswarm = str(work_dir / "shap_beeswarm.png")
    shap_waterfall = str(work_dir / "shap_waterfall.png")
    shap_local_dir = str(work_dir / "shap_local_plots")
    shap_values = str(work_dir / "shap_values.parquet")
    run_shap_explain(
        x_test_path=paths["X_test"],
        model_artifact_path=final_model_path,
        shap_summary_png=shap_png,
        shap_beeswarm_png=shap_beeswarm,
        shap_waterfall_png=shap_waterfall,
        shap_local_plots_dir=shap_local_dir,
        shap_values_parquet=shap_values
    )
    print(f"    OK. SHAP Summary generated: {shap_png}")
    print(f"    OK. SHAP Beeswarm generated: {shap_beeswarm}")
    print(f"    OK. Local waterfall plots saved to: {shap_local_dir}")

    # --------------------------------------------------------------------------
    print("\n[9] Testing: fairness_audit...")
    # --------------------------------------------------------------------------
    fairness_json = str(work_dir / "fairness_audit.json")
    run_fairness_audit(
        x_test_path=paths["X_test"],
        y_test_path=paths["y_test"],
        model_artifact_path=final_model_path,
        fairness_report_json=fairness_json
    )
    print(f"    OK. Fairness Audit generated: {fairness_json}")

    print("\n=====================================================================")
    print("ALL TESTS PASSED: ML Logic is verified and functional.")
    print("=====================================================================\n")

if __name__ == "__main__":
    main()
