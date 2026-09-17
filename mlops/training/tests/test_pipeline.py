"""Tests for the training pipeline DAG assembly.

These pin three things:
  * the whole pipeline compiles to KFP IR with every expected task present and
    the correct dependency edges,
  * the committed IR matches what the source compiles to, so the artifact a run
    is submitted from cannot go stale (the rename left it importing
    ``pipelines.components.*`` until it was repaired), and
  * the explicitly pinned feature list stays in sync with its documented source
    (the feature-selection run summary), so a silent drift is caught.
"""

import json
from pathlib import Path

import yaml

from mlops.data.encoding import feature_order
from mlops.training.training_pipeline import (
    CAT_FEATURES,
    IR_FILENAME,
    SELECTED_FEATURES,
    compile_pipeline,
    training_pipeline,
)

_MLOPS_ROOT = Path(__file__).resolve().parents[2]
_COMMITTED_IR = Path(__file__).resolve().parents[1] / IR_FILENAME

EXPECTED_TASKS = {
    "importer", "load-data", "validate-data", "benchmark-xgboost",
    "benchmark-gate", "optuna-hpo", "train-final", "calibrate-threshold",
    "evaluate-test", "shap-explain", "fairness-audit", "register-model",
}


def test_pipeline_compiles_with_all_tasks(tmp_path):
    out = tmp_path / "pipeline.yaml"
    compile_pipeline(str(out))
    assert out.exists() and out.stat().st_size > 0

    spec = yaml.safe_load(out.read_text())
    tasks = set(spec["root"]["dag"]["tasks"].keys())
    # Equality, not containment: a step that vanishes from the DAG is as much a
    # defect as one that appears, and ``issubset`` let that pass silently.
    assert tasks == EXPECTED_TASKS, f"task set differs: {tasks ^ EXPECTED_TASKS}"


def test_the_training_table_is_a_lineage_artifact_not_just_a_parameter(tmp_path):
    """The dataset the model learned from has to be a node in the graph.

    A table passed as a parameter and forgotten leaves no edge to traverse, so
    "which data produced this model" has no answer in the lineage store. The
    importer gives the table a `system.Dataset` artifact, and load-data consumes
    it — which is what makes the artifact an input of an execution rather than a
    value in a run's parameters.
    """
    out = tmp_path / "pipeline.yaml"
    compile_pipeline(str(out))
    spec = yaml.safe_load(out.read_text())

    importer = spec["components"]["comp-importer"]
    artifact = importer["outputDefinitions"]["artifacts"]["artifact"]
    assert artifact["artifactType"]["schemaTitle"] == "system.Dataset"

    bound = spec["root"]["dag"]["tasks"]["load-data"]["inputs"]["artifacts"]
    assert bound["training_table"]["taskOutputArtifact"]["producerTask"] == "importer"
    # The URI is built from the parameter, so no table is baked into the IR.
    assert "bq://" in json.dumps(
        spec["root"]["dag"]["tasks"]["importer"]["inputs"]
    )


def _images(spec: dict) -> set[str]:
    """Every container image named anywhere in the IR, deduplicated.

    The images live under ``components.<name>.executor.container.image``, not on
    the DAG tasks — a task only carries an ``executorLabel``.
    """
    found: set[str] = set()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "image" and isinstance(value, str):
                    found.add(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(spec)
    return found


def _without_images(node):
    """The IR with every image value blanked, for comparing graphs.

    The image legitimately depends on ``TRAINING_IMAGE_URI`` at compile time, so
    it is compared separately from the rest of the spec.
    """
    if isinstance(node, dict):
        return {
            key: ("<image>" if key == "image" else _without_images(value))
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [_without_images(item) for item in node]
    return node


def test_committed_ir_matches_what_the_source_compiles_to(tmp_path):
    """The committed IR is the artifact E5 points at; keep it true.

    ``compile_pipeline()`` writes it next to the module, so a submit refreshes
    it — but a source change without a submit leaves it behind, and the copy in
    the repository is what a reviewer reads and what a console submission runs.
    """
    fresh = tmp_path / "fresh.yaml"
    compile_pipeline(str(fresh))

    assert _COMMITTED_IR.exists(), (
        f"{_COMMITTED_IR.name} is missing — run compile_pipeline() to write it"
    )
    fresh_spec = yaml.safe_load(fresh.read_text())
    committed_spec = yaml.safe_load(_COMMITTED_IR.read_text())

    assert _without_images(fresh_spec) == _without_images(committed_spec), (
        "the committed IR no longer matches the source — recompile and commit it"
    )
    # One image for the whole run: a half-updated IR would mix versions of the
    # code inside a single execution.
    assert len(_images(committed_spec)) == 1, (
        f"the committed IR names more than one image: {_images(committed_spec)}"
    )


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
    # Feature encoding is the single source of truth (``mlops.data.encoding``); the
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


def test_hospital_baseline_is_read_from_the_artifact_the_gates_read():
    """The number the gates gate against is the one the versioned artifact holds."""
    from mlops.training.components._baselines import BASELINE_ARTIFACT, hospital_aucpr

    baseline = json.loads(
        (_MLOPS_ROOT / "artifacts" / "hospital_baseline.json").read_text()
    )
    assert BASELINE_ARTIFACT == _MLOPS_ROOT / "artifacts" / "hospital_baseline.json"
    assert hospital_aucpr() == float(baseline["aucpr"])


def test_no_gate_threshold_can_be_set_at_submit_time():
    """A gate threshold that arrives as a parameter is the submitter's choice.

    `hospital_aucpr=0.01` used to pass validation and neutralize the benchmark
    gate and the test gate together. Neither threshold is in the compiled
    pipeline now, so there is nothing to pass — which is a stronger guarantee
    than a value check, because it cannot be argued with.
    """
    spec = yaml.safe_load(_COMMITTED_IR.read_text())
    params = set((spec["root"]["inputDefinitions"] or {}).get("parameters", {}))

    assert "hospital_aucpr" not in params
    assert "max_drifted_share" not in params

    for task in ("validate-data", "benchmark-gate", "evaluate-test"):
        inputs = spec["root"]["dag"]["tasks"][task].get("inputs") or {}
        declared = set((inputs.get("parameters") or {}))
        assert not declared & {"hospital_aucpr", "max_drifted_share"}, task


def test_the_baseline_ships_in_the_image_the_gates_run_in():
    """The gate reads it inside its own container, so it has to be built in."""
    ignore = (_MLOPS_ROOT / ".gcloudignore").read_text()

    assert "!artifacts/hospital_baseline.json" in ignore
    # ...and the artifact the code points at is the one on disk.
    from mlops.training.components._baselines import BASELINE_ARTIFACT

    assert BASELINE_ARTIFACT.exists()
