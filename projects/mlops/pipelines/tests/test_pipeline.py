"""Tests for the training pipeline DAG assembly.

These pin two things:
  * the whole pipeline compiles to KFP IR with every expected task present and
    the correct dependency edges, and
  * the explicitly pinned feature list stays in sync with its documented source
    (the feature-selection run summary), so a silent drift is caught.
"""

import json
from pathlib import Path

import yaml

from pipelines.training_pipeline import (
    CAT_FEATURES,
    SELECTED_FEATURES,
    compile_pipeline,
    training_pipeline,
)

_MLOPS_ROOT = Path(__file__).resolve().parents[2]
_RUN_SUMMARY = (
    _MLOPS_ROOT
    / "artifacts" / "feature_selection" / "20260702t163137" / "run_summary.json"
)

EXPECTED_TASKS = {
    "fit-imputer-op", "load-data", "validate-data", "benchmark-xgboost",
    "benchmark-gate", "optuna-hpo", "train-final", "evaluate-test",
    "shap-explain", "fairness-audit", "register-model",
}


def test_pipeline_compiles_with_all_tasks(tmp_path):
    out = tmp_path / "pipeline.yaml"
    compile_pipeline(str(out))
    assert out.exists() and out.stat().st_size > 0

    spec = yaml.safe_load(out.read_text())
    tasks = set(spec["root"]["dag"]["tasks"].keys())
    assert EXPECTED_TASKS.issubset(tasks), f"missing tasks: {EXPECTED_TASKS - tasks}"


def test_pipeline_dependency_edges(tmp_path):
    out = tmp_path / "pipeline.yaml"
    compile_pipeline(str(out))
    spec = yaml.safe_load(out.read_text())
    tasks = spec["root"]["dag"]["tasks"]

    def depends_on(task: str, upstream: str) -> bool:
        # An edge exists via explicit dependsOn or via an input wired from the
        # upstream task's outputs.
        t = tasks[task]
        if upstream in t.get("dependentTasks", []):
            return True
        blob = json.dumps(t.get("inputs", {}))
        return upstream in blob

    # load_data must consume the fitted imputer; training must follow the gate.
    assert depends_on("load-data", "fit-imputer-op")
    assert depends_on("benchmark-gate", "benchmark-xgboost")
    assert depends_on("optuna-hpo", "benchmark-gate")
    assert depends_on("train-final", "optuna-hpo")
    assert depends_on("register-model", "evaluate-test")


def test_pinned_features_match_run_summary():
    summary = json.loads(_RUN_SUMMARY.read_text())
    # ``insurance`` is added on top of the feature-selection run as a model
    # feature (and the fairness-audit SES slice); everything else must still
    # match the documented run summary exactly.
    intentional_additions = {"insurance"}
    assert set(SELECTED_FEATURES) == set(summary["selected_features"]) | intentional_additions, (
        "Pinned SELECTED_FEATURES drifted from the documented run summary "
        "(beyond the intentional additions). Update it deliberately from the "
        "chosen feature-selection run."
    )


def test_cat_features_are_subset_of_selected():
    assert set(CAT_FEATURES).issubset(set(SELECTED_FEATURES))


def test_pipeline_callable_is_a_kfp_pipeline():
    # The @dsl.pipeline decorator attaches a pipeline spec.
    assert hasattr(training_pipeline, "pipeline_spec")


def test_hospital_baseline_matches_artifact():
    from pipelines.training_pipeline import HOSPITAL_AUCPR

    baseline = json.loads(
        (_MLOPS_ROOT / "artifacts" / "hospital_baseline.json").read_text()
    )
    assert HOSPITAL_AUCPR == float(baseline["aucpr"])
