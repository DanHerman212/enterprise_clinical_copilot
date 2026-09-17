"""Gap 5 — a registry entry says which run produced it and which code.

The entry used to carry a name, a score, and nothing traceable: the pipeline job
name was threaded into five components and not into the one that publishes an
artifact a person looks up later, no revision was recorded anywhere, and the
`parent_model` parameter that was supposed to give version lineage was declared,
plumbed through the DAG, passed to `Model.upload` — and never set by anything.

These tests pin what replaced it: the run and the revision travel into the
registration and land on the entry, the encoding is registry-legal, and an
unknown value is left off rather than written as a placeholder.
"""

import pathlib
import re

import pytest
import yaml

from mlops.training.components.register_model import LABEL_MAX, provenance_labels

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
TRAINING_DIR = REPO_ROOT / "mlops/training"
COMMITTED_IR = TRAINING_DIR / "readmission_training_pipeline.yaml"

LABEL_VALUE = re.compile(r"^[a-z0-9_-]{0,63}$")


def _labels(**overrides):
    kwargs = {
        "test_aucpr": 0.3294,
        "tuned_threshold": 0.11,
        "pipeline_job_name": "readmission-training-20260917130517",
        "git_revision": "2b62c6c0",
    }
    kwargs.update(overrides)
    return provenance_labels(**kwargs)


def test_the_entry_names_the_run_and_the_revision():
    labels = _labels()

    assert labels["run"] == "readmission-training-20260917130517"
    assert labels["commit"] == "2b62c6c0"


def test_an_unknown_run_or_revision_is_left_off_rather_than_guessed():
    """Absent means not recorded; a placeholder would read as a real value."""
    labels = _labels(pipeline_job_name="", git_revision="")

    assert "run" not in labels
    assert "commit" not in labels
    # The rest of the entry is unaffected — a model is not less registered for
    # having been submitted outside a script that resolved a revision.
    assert labels["stage"] == "final"


@pytest.mark.parametrize(
    "value",
    [
        "readmission-training-20260917130517",
        "2b62c6c0-dirty",
        "FEATURE/Branch_Name#1",
        "x" * 200,
    ],
)
def test_every_label_value_is_registry_legal(value):
    labels = _labels(pipeline_job_name=value, git_revision=value)

    assert LABEL_VALUE.match(labels["run"]), labels["run"]
    assert LABEL_VALUE.match(labels["commit"]), labels["commit"]
    assert len(labels["run"]) <= 63


def test_the_metrics_are_still_encoded_for_a_label():
    """Registry labels allow [a-z0-9_-], so the decimal point is a dash."""
    labels = _labels()

    assert labels["test_aucpr"] == "0-3294"
    assert labels["tuned_threshold"] == "0-1100"


def test_the_dead_version_parameter_is_gone():
    """A lineage parameter that nothing sets reads as a capability that exists."""
    for path in (
        TRAINING_DIR / "components/register_model.py",
        TRAINING_DIR / "training_pipeline.py",
    ):
        assert "parent_model" not in path.read_text(), f"{path.name} still carries parent_model"


def test_registration_receives_the_run_it_belongs_to():
    source = (TRAINING_DIR / "training_pipeline.py").read_text()
    call = source.split("register_model(", 1)[1]

    assert "pipeline_job_name=dsl.PIPELINE_JOB_NAME_PLACEHOLDER" in call
    assert "git_revision=git_revision" in call


def test_the_revision_is_not_baked_into_the_committed_ir():
    """A compiled-in revision would describe the machine that compiled it."""
    spec = yaml.safe_load(COMMITTED_IR.read_text())
    params = spec["root"]["inputDefinitions"]["parameters"]

    assert params["git_revision"]["defaultValue"] == ""
