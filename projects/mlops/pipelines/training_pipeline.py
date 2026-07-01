#!/usr/bin/env python3
"""
training_pipeline.py — Phase 3 training pipeline with Optuna HPO.

DAG: load-data → validate → benchmark → gate → optuna → train-final → register
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import kfp
from kfp import dsl

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    PROJECT_ID, FULL_TABLE_REF, LABEL_COLUMN,
    SPLIT_COLUMN, SPLITS, EVIDENTLY_CONFIG,
)

from pipelines.components.load_data import load_data
from pipelines.components.validate_data import validate_data
from pipelines.components.benchmark_xgboost import benchmark_xgboost
from pipelines.components.benchmark_gate import benchmark_gate
from pipelines.components.optuna_hpo import optuna_hpo
from pipelines.components.train_final import train_final
from pipelines.components.evaluate_test import evaluate_test
from pipelines.components.shap_explain import shap_explain
from pipelines.components.fairness_audit import fairness_audit
from pipelines.components.register_model import register_model

# ---------------------------------------------------------------------------
PIPELINE_NAME = "readmission-training"
PIPELINE_ROOT = f"gs://{PROJECT_ID}-pipeline-artifacts/{PIPELINE_NAME}"
REGION = "us-east1"
SERVICE_ACCOUNT = os.environ.get(
    "VERTEX_SA", f"{PROJECT_ID}@appspot.gserviceaccount.com",
)

DEFAULT_XGB_PARAMS = dict(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, eval_metric="logloss", enable_categorical=True,
)

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

CATEGORICAL_FEATURES = [
    "gender", "race", "admission_type", "insurance", "discharge_location",
]

HPO_TRIALS = 50


# ===================================================================
# Pipeline
# ===================================================================

@dsl.pipeline(
    name=PIPELINE_NAME,
    description="Readmission ML Training — benchmark + Optuna HPO",
    pipeline_root=PIPELINE_ROOT,
)
def training_pipeline(
    project_id: str = PROJECT_ID,
    full_table_ref: str = FULL_TABLE_REF,
    label_col: str = LABEL_COLUMN,
    split_col: str = SPLIT_COLUMN,
    train_split: str = SPLITS["train"],
    val_split: str = SPLITS["validation"],
    test_split: str = SPLITS["test"],
    selected_features: list = SELECTED_FEATURES,
    cat_features: list = CATEGORICAL_FEATURES,
    imputer_gcs_path: str = "",
    xgb_params: dict = DEFAULT_XGB_PARAMS,
    n_trials: int = HPO_TRIALS,
    max_drifted_share: float = EVIDENTLY_CONFIG.get("max_drifted_share", 0.2),
):
    """Phase 3 training pipeline — benchmark → Optuna → final → test → register."""

    # 1. Load data.
    data = load_data(
        project_id=project_id, full_table_ref=full_table_ref,
        label_col=label_col, split_col=split_col,
        train_split=train_split, val_split=val_split,
        test_split=test_split,
        selected_features=selected_features, cat_features=cat_features,
        imputer_gcs_path=imputer_gcs_path,
    )

    # 2. Validate data.
    validation = validate_data(
        x_train_path=data.outputs["x_train_path"],
        x_val_path=data.outputs["x_val_path"],
        max_drifted_share=max_drifted_share,
    )
    validation.outputs["passed"]

    # 3. Benchmark XGBoost.
    benchmark = benchmark_xgboost(
        x_train_path=data.outputs["x_train_path"],
        y_train_path=data.outputs["y_train_path"],
        x_val_path=data.outputs["x_val_path"],
        y_val_path=data.outputs["y_val_path"],
        xgb_params=xgb_params,
        cat_features=cat_features,
    )

    # 4. Gate: benchmark must beat HOSPITAL (0.3325).
    gate = benchmark_gate(
        benchmark_aucpr=benchmark.outputs["benchmark_aucpr"],
    )
    gate.outputs["passed"]

    # 5. Optuna HPO.
    hpo = optuna_hpo(
        x_train_path=data.outputs["x_train_path"],
        y_train_path=data.outputs["y_train_path"],
        cat_features=cat_features,
        n_trials=n_trials,
    )

    # 6. Train final model with best params.
    final = train_final(
        x_train_path=data.outputs["x_train_path"],
        y_train_path=data.outputs["y_train_path"],
        x_val_path=data.outputs["x_val_path"],
        y_val_path=data.outputs["y_val_path"],
        best_params_path=hpo.outputs["best_params_path"],
        cat_features=cat_features,
    )

    # 7. Evaluate on held-out test set (gates registration).
    #    SHAP and fairness run in parallel — read-only, do not gate.
    test_eval = evaluate_test(
        x_test_path=data.outputs["x_test_path"],
        y_test_path=data.outputs["y_test_path"],
        model_artifact_path=final.outputs["model_artifact_path"],
        final_val_aucpr=final.outputs["final_aucpr"],
        benchmark_aucpr=benchmark.outputs["benchmark_aucpr"],
    )
    test_eval.outputs["beat_hospital"]

    shap = shap_explain(
        x_test_path=data.outputs["x_test_path"],
        model_artifact_path=final.outputs["model_artifact_path"],
    )

    fairness = fairness_audit(
        x_test_path=data.outputs["x_test_path"],
        y_test_path=data.outputs["y_test_path"],
        model_artifact_path=final.outputs["model_artifact_path"],
    )

    # 8. Register model.
    registry = register_model(
        project_id=project_id,
        model_artifact_path=final.outputs["model_artifact_path"],
        test_aucpr=test_eval.outputs["test_aucpr"],
        final_val_aucpr=final.outputs["final_aucpr"],
        benchmark_aucpr=benchmark.outputs["benchmark_aucpr"],
    )


# ===================================================================
# CLI
# ===================================================================

def _compile():
    import kfp.compiler
    output = Path(__file__).with_suffix(".json")
    kfp.compiler.Compiler().compile(training_pipeline, str(output))
    print(f"Compiled -> {output}")


def _submit():
    import kfp.compiler
    from google.cloud import aiplatform
    from google.cloud.aiplatform import pipeline_jobs

    aiplatform.init(project=PROJECT_ID, location=REGION)
    json_path = Path(__file__).with_suffix(".json")
    kfp.compiler.Compiler().compile(training_pipeline, str(json_path))

    job = pipeline_jobs.PipelineJob(
        display_name=f"{PIPELINE_NAME}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        template_path=str(json_path),
        pipeline_root=PIPELINE_ROOT,
        parameter_values={
            "project_id": PROJECT_ID,
            "imputer_gcs_path": os.environ.get(
                "IMPUTER_GCS_PATH",
                f"gs://{PROJECT_ID}-pipeline-artifacts/imputer/imputer.joblib",
            ),
        },
        enable_caching=True,
    )
    job.submit(service_account=SERVICE_ACCOUNT)
    print(f"Submitted: {job.resource_name}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["compile", "submit"], default="compile", nargs="?")
    args = parser.parse_args()
    {"compile": _compile, "submit": _submit}[args.action]()
