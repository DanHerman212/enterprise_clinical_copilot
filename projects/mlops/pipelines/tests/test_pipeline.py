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

from src.encoding import feature_order
from pipelines.training_pipeline import (
    CAT_FEATURES,
    SELECTED_FEATURES,
    compile_pipeline,
    training_pipeline,
)

_MLOPS_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_TASKS = {
    "load-data", "validate-data", "benchmark-xgboost",
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


def test_no_module_constant_defaults_in_component_signatures(tmp_path):
    """Guard the KFP re-exec gotcha.

    KFP serializes each @component wrapper's *source* and re-execs it inside the
    training container, where module-level names from the authoring module are
    NOT defined. So a parameter default like ``x: int = _SOME_CONSTANT`` raises
    ``NameError`` at function definition -> the task exits 1 immediately (with no
    useful stdout). Component defaults must be literals; keep module constants in
    the ``run_*`` helpers (which are imported from the image, not serialized).
    """
    import re

    out = tmp_path / "pipeline.yaml"
    compile_pipeline(str(out))
    offenders = re.findall(
        r": (?:int|float|str|list|dict|bool) = _[A-Za-z]\w*", out.read_text()
    )
    assert not offenders, (
        f"component parameter default references a module constant: {offenders}"
    )


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

    # Encoding is static in BigQuery now (no imputer node); training must
    # follow the benchmark gate.
    assert depends_on("benchmark-gate", "benchmark-xgboost")
    assert depends_on("optuna-hpo", "benchmark-gate")
    assert depends_on("train-final", "optuna-hpo")
    assert depends_on("register-model", "evaluate-test")


def test_selected_features_match_encoding_order():
    # Feature encoding is the single source of truth (``src.encoding``); the
    # pipeline's SELECTED_FEATURES must equal the encoded view's column order
    # exactly, in order, so the serving feature vector lines up with training.
    assert SELECTED_FEATURES == feature_order()


def test_cat_features_is_empty_after_onehot():
    # One-hot encoding is now static in BigQuery, so the pipeline carries no
    # in-model categorical columns.
    assert CAT_FEATURES == []


def test_pipeline_callable_is_a_kfp_pipeline():
    # The @dsl.pipeline decorator attaches a pipeline spec.
    assert hasattr(training_pipeline, "pipeline_spec")


def test_hospital_baseline_matches_artifact():
    from pipelines.training_pipeline import HOSPITAL_AUCPR

    baseline = json.loads(
        (_MLOPS_ROOT / "artifacts" / "hospital_baseline.json").read_text()
    )
    assert HOSPITAL_AUCPR == float(baseline["aucpr"])
