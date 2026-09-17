"""Gap 9 — one feature contract, and no way to mistake a missing column for it.

The 49 names the model consumes were written out by hand in four places: the
training contract plus three demo-side generators whose job is to build a
synthetic table a real booster can score. They agreed, and nothing kept them
agreeing — and the read path made any disagreement invisible, because it filled a
column the table did not have with None, which the booster reads as a measurement
that was not taken.

These tests pin both halves: the generators derive the list from the contract, and
the feature source distinguishes an absent column from a null value.
"""

import ast
import pathlib

import pytest

from services.mcp.dependencies.features import bigquery_source
from mlops.data.encoding import feature_order

REPO = pathlib.Path(__file__).resolve().parents[2]

GENERATORS = (
    "scripts/agent/load_synthetic_features.py",
    "scripts/agent/load_hybrid_notes.py",
    "scripts/agent/generate_hybrid_features_v2.py",
)

LIST_NAMES = ("_FEATURE_NAMES", "FEATURE_NAMES")


@pytest.mark.parametrize("path", GENERATORS)
def test_the_generator_derives_the_contract_instead_of_retyping_it(path):
    """A retyped list is a second contract, and only one of them is enforced."""
    tree = ast.parse((REPO / path).read_text())

    assignments = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", "") in LIST_NAMES for t in node.targets)
    ]
    assert assignments, f"{path} no longer defines a feature-name list"

    value = assignments[0].value
    assert isinstance(value, ast.Call), (
        f"{path} assigns a literal list — it must call the contract instead"
    )
    assert getattr(value.func, "id", "") == "feature_order"


@pytest.mark.parametrize("path", GENERATORS)
def test_the_contract_is_the_one_the_model_was_trained_on(path):
    """The generator and the model must agree, provably, not by inspection."""
    source = (REPO / path).read_text()

    assert "from mlops.data.encoding import feature_order" in source
    assert len(feature_order()) == 49


def _fetch_with(monkeypatch, table_columns: dict):
    """Run the real source.fetch against a fake BigQuery client.

    `feature_order` is patched to the training contract. The real one resolves the
    serving bundle over the network and caches it, which is right for the live
    service and wrong for a unit test — this test is about what the source does
    with the contract, not about how the contract is located.
    """
    class _Row(dict):
        def items(self):
            return super().items()

    class _Job:
        def result(self):
            return [_Row(table_columns)]

    class _Client:
        def query(self, *args, **kwargs):
            return _Job()

    class _Config:
        def __init__(self, **kwargs):
            pass

    source = bigquery_source.BigQueryFeatureSource.__new__(
        bigquery_source.BigQueryFeatureSource
    )
    source._client = _Client()
    source._table = "readmission.hybrid_features"
    monkeypatch.setattr(bigquery_source, "feature_order", feature_order)
    monkeypatch.setattr(bigquery_source.bigquery, "QueryJobConfig", _Config)
    monkeypatch.setattr(
        bigquery_source.bigquery, "ScalarQueryParameter", lambda *a, **k: None
    )
    return source.fetch(1)


def test_a_null_value_keeps_its_column_so_it_still_scores(monkeypatch):
    """A measurement that was not taken is normal; the model is trained for it."""
    row = {name: 1.0 for name in feature_order()}
    row["sodium_last"] = None

    fetched = _fetch_with(monkeypatch, row)

    assert "sodium_last" in fetched
    assert fetched["sodium_last"] is None
    assert len(fetched) == 49


def test_an_absent_column_is_left_out_so_the_row_is_refused(monkeypatch):
    """The table and the model disagreeing is not the same as a missing value."""
    row = {name: 1.0 for name in feature_order() if name != "sodium_last"}

    fetched = _fetch_with(monkeypatch, row)

    assert "sodium_last" not in fetched, (
        "an absent column was invented as None, which hides the disagreement"
    )
    missing = [col for col in feature_order() if col not in fetched]
    assert missing == ["sodium_last"]


def test_the_label_never_reaches_the_model_input(monkeypatch):
    """SELECT * carries the label; letting it through would be a silent bug."""
    row = {name: 1.0 for name in feature_order()}
    row["readmission_30d"] = 1.0
    row["subject_id"] = 90000001

    fetched = _fetch_with(monkeypatch, row)

    assert "readmission_30d" not in fetched
    assert "subject_id" not in fetched
