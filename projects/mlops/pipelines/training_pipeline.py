"""
training_pipeline — Vertex AI (KFP v2) readmission training pipeline DAG.

Wiring::

    fit_imputer ─▶ load_data ─▶ validate_data ─▶ benchmark_xgboost ─▶ benchmark_gate
                                                                           │
                                                                           ▼
                                                                       optuna_hpo
                                                                           │
                                                                           ▼
                                                                      train_final
                                                                           │
                       ┌───────────────────┬───────────────────┬──────────┘
                       ▼                   ▼                   ▼
                 evaluate_test        shap_explain       fairness_audit
                       │
                       ▼
                 register_model

The feature contract is **pinned explicitly** below (sourced from
``artifacts/feature_selection/20260701t225649/run_summary.json``) and passed as
a default pipeline parameter, so a rerun is fully reproducible and does not
silently change when a new feature-selection run lands. To adopt a new feature
set, update ``SELECTED_FEATURES`` deliberately from the chosen run summary.
"""

import json
import sys
from pathlib import Path

from kfp import compiler, dsl

from pipelines.components.benchmark_gate import benchmark_gate
from pipelines.components.benchmark_xgboost import benchmark_xgboost
from pipelines.components.calibrate_threshold import calibrate_threshold
from pipelines.components.evaluate_test import evaluate_test
from pipelines.components.fairness_audit import fairness_audit
from pipelines.components.fit_imputer import fit_imputer_op
from pipelines.components.load_data import load_data
from pipelines.components.optuna_hpo import optuna_hpo
from pipelines.components.register_model import register_model
from pipelines.components.shap_explain import shap_explain
from pipelines.components.train_final import train_final
from pipelines.components.validate_data import validate_data

PIPELINE_NAME = "readmission-training"

# Vertex AI Experiment shared with the HOSPITAL baseline and feature-selection
# runs, so training runs land alongside them for side-by-side comparison.
EXPERIMENT_NAME = "readmission-mlops"

# --- PINNED feature contract (source: run_summary.json 20260702t163137) -------
# Leakage-controlled selection: grouped CV (StratifiedGroupKFold on subject_id),
# native categoricals, scale_pos_weight, 1-standard-error parsimony rule.
# 22 features (down from 50), grouped-CV AUCPR 0.4078 — statistically tied with
# the full set (0.4084) but 56% smaller. ``insurance`` is added on top of the
# selection run as a model feature (it also serves as the fairness-audit SES
# slice, read straight off the encoded test frame).
SELECTED_FEATURES = [
    "age", "gender", "race", "admission_type", "discharge_location", "insurance",
    "prior_admission_count", "prior_inpatient_days", "recent_ed_visits",
    "index_los_days", "procedure_count", "has_procedure",
    "medication_count", "medication_order_count",
    "rbc_last", "rbc_min", "rdw_max", "monocytes_min", "hemoglobin_min",
    "sodium_last", "sodium_max", "sodium_min", "oncology_flag",
]

# Categorical subset of SELECTED_FEATURES (config.CATEGORICAL_FEATURES ∩ pinned).
CAT_FEATURES = [
    "gender", "race", "admission_type", "discharge_location", "insurance",
    "has_procedure", "oncology_flag",
]

# Benchmark XGBoost defaults (mirror the feature-selection reference model).
DEFAULT_XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}


def _load_hospital_baseline() -> float:
    """Single source of truth for the HOSPITAL baseline AUCPR (build-time)."""
    path = Path(__file__).resolve().parents[1] / "artifacts" / "hospital_baseline.json"
    return float(json.loads(path.read_text())["aucpr"])


# Resolved once at build/submit time and baked into the compiled pipeline.
HOSPITAL_AUCPR = _load_hospital_baseline()


@dsl.pipeline(
    name=PIPELINE_NAME,
    description="30-day readmission risk: impute → validate → benchmark → HPO → "
    "train → evaluate/explain/audit → register.",
)
def training_pipeline(
    project_id: str,
    full_table_ref: str,
    label_col: str = "readmission_30d",
    split_col: str = "split_name",
    train_split: str = "train",
    val_split: str = "validation",
    test_split: str = "test",
    selected_features: list = SELECTED_FEATURES,
    cat_features: list = CAT_FEATURES,
    xgb_params: dict = DEFAULT_XGB_PARAMS,
    n_trials: int = 50,
    hpo_timeout_seconds: int = 2700,
    fbeta_beta: float = 2.0,
    max_drifted_share: float = 0.2,
    hospital_aucpr: float = HOSPITAL_AUCPR,
    serving_container_image_uri: str = "",
):
    """Assemble the readmission training DAG."""
    fit = fit_imputer_op(
        project_id=project_id,
        full_table_ref=full_table_ref,
        split_col=split_col,
        train_split=train_split,
    )

    data = load_data(
        project_id=project_id,
        full_table_ref=full_table_ref,
        label_col=label_col,
        split_col=split_col,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        selected_features=selected_features,
        cat_features=cat_features,
        imputer=fit.outputs["imputer"],
    )

    validate = validate_data(
        x_train=data.outputs["x_train"],
        x_val=data.outputs["x_val"],
        max_drifted_share=max_drifted_share,
    )

    bench = benchmark_xgboost(
        x_train=data.outputs["x_train"],
        y_train=data.outputs["y_train"],
        x_val=data.outputs["x_val"],
        y_val=data.outputs["y_val"],
        xgb_params=xgb_params,
        cat_features=cat_features,
    ).after(validate)

    gate = benchmark_gate(
        benchmark_aucpr=bench.outputs["Output"],
        hospital_aucpr=hospital_aucpr,
    )

    hpo = optuna_hpo(
        x_train=data.outputs["x_train"],
        y_train=data.outputs["y_train"],
        groups=data.outputs["groups_train"],
        cat_features=cat_features,
        n_trials=n_trials,
        timeout_seconds=hpo_timeout_seconds,
        project_id=project_id,
        experiment_name=EXPERIMENT_NAME,
        pipeline_job_name=dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
    ).after(gate)

    final = train_final(
        x_train=data.outputs["x_train"],
        y_train=data.outputs["y_train"],
        x_val=data.outputs["x_val"],
        y_val=data.outputs["y_val"],
        best_params=hpo.outputs["best_params"],
        cat_features=cat_features,
    )

    # Operating threshold: F-beta-optimal on out-of-fold (patient-grouped) TRAIN
    # predictions with the tuned HPO params. Probability-first model; this
    # threshold is metadata for the decision layer only.
    calib = calibrate_threshold(
        x_train=data.outputs["x_train"],
        y_train=data.outputs["y_train"],
        groups=data.outputs["groups_train"],
        best_params=hpo.outputs["best_params"],
        cat_features=cat_features,
        beta=fbeta_beta,
        project_id=project_id,
        experiment_name=EXPERIMENT_NAME,
        pipeline_job_name=dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
    )

    # The honest pre-test generalization estimate is the HPO validation AUCPR,
    # NOT train_final's combined-fit metric (the final model trained on val).
    evalt = evaluate_test(
        x_test=data.outputs["x_test"],
        y_test=data.outputs["y_test"],
        model_artifact=final.outputs["model_artifact"],
        tuned_threshold=calib.outputs["Output"],
        hpo_val_aucpr=hpo.outputs["Output"],
        benchmark_aucpr=bench.outputs["Output"],
        hospital_aucpr=hospital_aucpr,
        beta=fbeta_beta,
        project_id=project_id,
        experiment_name=EXPERIMENT_NAME,
        pipeline_job_name=dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
    )

    shap_explain(
        x_test_path=data.outputs["x_test"],
        model_artifact_path=final.outputs["model_artifact"],
    )

    fairness_audit(
        x_test=data.outputs["x_test"],
        y_test=data.outputs["y_test"],
        model_artifact=final.outputs["model_artifact"],
        tuned_threshold=calib.outputs["Output"],
        project_id=project_id,
        experiment_name=EXPERIMENT_NAME,
        pipeline_job_name=dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
    )

    register_model(
        project_id=project_id,
        model_artifact=final.outputs["model_artifact"],
        imputer=fit.outputs["imputer"],
        schema=data.outputs["schema"],
        serving_container_image_uri=serving_container_image_uri,
        test_aucpr=evalt.outputs["test_aucpr"],
        hpo_val_aucpr=hpo.outputs["Output"],
        benchmark_aucpr=bench.outputs["Output"],
        tuned_threshold=calib.outputs["Output"],
        beta=fbeta_beta,
    ).after(evalt)


def compile_pipeline(package_path: str = "readmission_training_pipeline.yaml") -> str:
    """Compile the pipeline to a KFP IR YAML and return the path."""
    compiler.Compiler().compile(
        pipeline_func=training_pipeline, package_path=package_path
    )
    return package_path


def submit() -> None:
    """Compile and submit the pipeline to Vertex AI Pipelines."""
    import os
    from datetime import datetime, timezone

    from google.cloud import aiplatform

    from src.config import FULL_TABLE_REF, PROJECT_ID

    package_path = compile_pipeline()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    aiplatform.init(project=PROJECT_ID, location="us-east1", experiment=EXPERIMENT_NAME)
    job = aiplatform.PipelineJob(
        display_name=f"{PIPELINE_NAME}-{ts}",
        template_path=package_path,
        # GCS staging root for pipeline artifacts (required by Vertex).
        pipeline_root=os.environ.get("PIPELINE_ROOT"),
        parameter_values={
            "project_id": PROJECT_ID,
            "full_table_ref": FULL_TABLE_REF,
            # HPO trials — set N_TRIALS low (e.g. 5) for a quick wiring check.
            "n_trials": int(os.environ.get("N_TRIALS", "50")),
            # Wall-clock backstop for HPO (seconds); stops launching trials past
            # this budget and returns best-so-far. Default 45 min.
            "hpo_timeout_seconds": int(os.environ.get("HPO_TIMEOUT", "2700")),
            # Custom real-time predictor image (see pipelines/serving/).
            "serving_container_image_uri": os.environ.get("SERVING_IMAGE_URI", ""),
        },
        # Every run logs fresh (no cached step reuse) so the experiment record
        # reflects the actual execution.
        enable_caching=False,
    )
    # Associate the run with the shared Vertex Experiment: pipeline parameters
    # and system.Metrics artifacts are auto-logged for comparison against the
    # baseline and feature-selection runs. Runs as PIPELINE_SA if set.
    job.submit(
        experiment=EXPERIMENT_NAME,
        service_account=os.environ.get("PIPELINE_SA") or None,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "submit":
        submit()
    else:
        path = compile_pipeline()
        print(f"Compiled pipeline -> {path}")
