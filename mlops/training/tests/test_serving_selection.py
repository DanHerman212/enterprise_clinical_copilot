"""Gap 4 — which record serves, and who is allowed to take that position.

Every selection in this system is one query: newest by `create_time` on the
`readmission-final-*` name. That meant "the most recent successful run" only
while the pipeline was the sole writer of the name — and a second writer
(`register_serving_model.py`) published the same name from a bundle it did not
train, with no gate metrics and no digests of its own.

These tests pin the rule that replaced it: a record serves only if the training
pipeline registered it, meaning the name prefix *and* the stage label
`register_model.py` writes with it. The manual path now publishes its own name
and its own label, so it cannot be selected at all.
"""

import pathlib

import pytest

from mlops.serving.deploy_cpr import (
    PIPELINE_STAGE,
    score_of,
    select_serving_record,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

# Every place that resolves which record serves, and must apply the same rule.
RESOLVERS = [
    "mlops/serving/deploy_cpr.py",
    "services/mcp/dependencies/features/manifest.py",
    "mlops/training/smoke_test.py",
]


class _Model:
    """The three fields the rule reads off a registry record."""

    def __init__(self, display_name, labels=None, uri="gs://bucket/bundle"):
        self.display_name = display_name
        self.labels = labels or {}
        self.gca_resource = type("_Gca", (), {"artifact_uri": uri})()


def _pipeline_record(ts, aucpr="0-3282", threshold="0-1100"):
    return _Model(
        f"readmission-final-{ts}",
        {"pipeline": "readmission-training", "stage": PIPELINE_STAGE,
         "test_aucpr": aucpr, "tuned_threshold": threshold},
    )


def test_newest_pipeline_registration_wins():
    newest = _pipeline_record("20260902014308")
    models = [newest, _pipeline_record("20260901222119")]

    assert select_serving_record(models) is newest


def test_a_hand_registration_cannot_take_the_newest_position():
    """The defect this gap is about: hand-written record, newest, must not serve."""
    hand = _Model(
        "readmission-manual-20260918000000",
        {"pipeline": "readmission-training", "stage": "manual"},
    )
    trained = _pipeline_record("20260902014308")

    assert select_serving_record([hand, trained]) is trained
    assert select_serving_record([hand]) is None


def test_a_record_with_the_name_but_not_the_label_is_not_selected():
    """The name is not the rule. Only the pipeline writes the label with it."""
    impostor = _Model("readmission-final-20260918000000", {"stage": "cpr"})

    assert select_serving_record([impostor]) is None


def test_a_record_with_no_labels_at_all_is_not_selected():
    assert select_serving_record([_Model("readmission-final-20260918000000")]) is None


def test_the_manual_namespace_and_the_pipeline_namespace_do_not_overlap():
    """The manual script must publish somewhere no resolver looks."""
    source = (REPO_ROOT / "mlops/serving/register_serving_model.py").read_text()

    assert 'display_name = f"readmission-manual-{ts}"' in source
    assert 'display_name = f"readmission-final-' not in source
    assert '"stage": "manual"' in source


def test_score_of_decodes_the_labels_the_registry_allows():
    """Registry labels allow [a-z0-9_-], so the decimal point is a dash."""
    assert score_of({"test_aucpr": "0-3294", "tuned_threshold": "0-1100"}) == (
        "test AUCPR 0.3294, threshold 0.11"
    )


def test_score_of_admits_when_a_record_carries_no_metrics():
    assert score_of({}) == "test AUCPR ?, threshold ?"


@pytest.mark.parametrize("path", RESOLVERS)
def test_every_resolver_requires_the_stage_label(path):
    """A new resolver that selects on the name alone re-opens this gap."""
    source = (REPO_ROOT / path).read_text()

    assert 'get("stage")' in source, f"{path} selects without checking the stage label"
    assert "PIPELINE_STAGE" in source, f"{path} does not name the stage it requires"


@pytest.mark.parametrize("path", RESOLVERS)
def test_the_stage_every_resolver_requires_is_the_one_the_pipeline_writes(path):
    """One writer, three readers: the label is the contract between them."""
    registrar = REPO_ROOT / "mlops/training/components/register_model.py"

    assert f'"stage": "{PIPELINE_STAGE}"' in registrar.read_text()
    assert PIPELINE_STAGE == "final"
