"""Gap 6 — the model says what it was trained on.

The bundle recorded everything about the model — booster, manifest, threshold,
gate numbers, digests — and nothing about the data. The table name was passed as
a parameter and forgotten, so two models registered a month apart were
indistinguishable in what they learned from, and a surprising score could not be
traced to "the data or the model?".

These tests pin what replaced it: the table's own state as observed at read time,
carried in the manifest (and therefore in the bundle, under the checksums), and
surfaced on the registry entry.
"""

import datetime
import json
import pathlib
import types

from mlops.training.components.load_data import _data_reference
from mlops.training.components.register_model import (
    provenance_description,
    provenance_labels,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
TABLE = "trim-icon-498815-a0.readmission.analytics_dataset_encoded"


class _FakeClient:
    def __init__(self, *, num_rows, modified):
        self._table = types.SimpleNamespace(num_rows=num_rows, modified=modified)

    def get_table(self, ref):
        self.requested = ref
        return self._table


def _reference(**overrides):
    kwargs = {
        "num_rows": 219744,
        "modified": datetime.datetime(2026, 9, 2, 1, 8, 47,
                                      tzinfo=datetime.timezone.utc),
    }
    kwargs.update(overrides)
    return _data_reference(
        _FakeClient(**kwargs),
        TABLE,
        {"train": 153820, "validation": 32961, "test": 32963},
    )


def test_the_reference_names_the_table_and_its_state_when_read():
    reference = _reference()

    assert reference["table"] == TABLE
    assert reference["row_count"] == 219744
    assert reference["last_modified_utc"].startswith("2026-09-02T01:08:47")
    assert reference["observed_rows_by_split"]["train"] == 153820


def test_an_unknown_modification_time_is_recorded_as_absent():
    """A table with no recorded write time is not the same as an unmoved one."""
    assert _reference(modified=None)["last_modified_utc"] is None


def test_the_reference_is_observed_rather_than_recomputed():
    """It comes from the table's metadata, so no query result can disagree."""
    client = _FakeClient(num_rows=7, modified=None)

    _data_reference(client, TABLE, {})

    assert client.requested == TABLE


def test_the_entry_description_names_run_code_and_data():
    description = provenance_description(
        test_aucpr=0.3294,
        tuned_threshold=0.11,
        pipeline_job_name="readmission-training-20260917130517",
        git_revision="ee584b6",
        data_reference=_reference(),
    )

    assert "readmission-training-20260917130517" in description
    assert "ee584b6" in description
    assert TABLE in description
    assert "219744 rows" in description
    assert "0.3294" in description


def test_the_description_admits_when_the_data_is_unrecorded():
    description = provenance_description(
        test_aucpr=0.3294,
        tuned_threshold=0.11,
        data_reference=None,
    )

    assert "data reference not recorded" in description
    assert "unknown" in description


def test_the_row_count_is_a_label_so_the_entry_is_queryable():
    labels = provenance_labels(
        test_aucpr=0.3294,
        tuned_threshold=0.11,
        data_row_count=219744,
    )

    assert labels["data_rows"] == "219744"


def test_a_missing_row_count_adds_no_label():
    labels = provenance_labels(test_aucpr=0.3294, tuned_threshold=0.11)

    assert "data_rows" not in labels


def test_the_manifest_load_data_writes_is_the_one_that_carries_the_reference():
    """The reference rides in the manifest so it ships inside the bundle."""
    source = (
        REPO_ROOT / "mlops/training/components/load_data.py"
    ).read_text()

    assert '{**encoding.manifest(), "data": data_reference}' in source


def test_registration_reads_the_reference_back_out_of_the_bundle():
    """One source for the reference, not a second parameter to keep in step."""
    source = (
        REPO_ROOT / "mlops/training/components/register_model.py"
    ).read_text()

    assert 'json.load(f).get("data")' in source
    assert "manifest.json" in source
