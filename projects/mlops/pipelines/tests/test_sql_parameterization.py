"""Tests for Cluster Q SQL-identifier fixes (mlops side, ECC-63).

run_load_data must bind split-name values as query parameters and validate
the table ref / column identifiers against a strict shape — a crafted
`train_split` can no longer reach the SQL text.
"""

import pytest

from pipelines.components.load_data import (
    _validated_ident,
    _validated_table_ref,
    run_load_data,
)


def test_table_ref_accepts_project_dataset_table():
    ref = "trim-icon-498815-a0.readmission.analytics_dataset_encoded"
    assert _validated_table_ref(ref) == ref


@pytest.mark.parametrize("ref", [
    "readmission.analytics_dataset_encoded",          # missing project
    "proj.dataset.table` WHERE 1=1; --",
    "proj.data set.table",
    "proj.dataset.table.extra",
    "",
])
def test_table_ref_rejects_malformed(ref):
    with pytest.raises(ValueError, match="full_table_ref"):
        _validated_table_ref(ref)


def test_ident_accepts_plain_columns():
    assert _validated_ident("split_name", "split_col") == "split_name"
    assert _validated_ident("subject_id", "id_col") == "subject_id"


@pytest.mark.parametrize("ident", [
    "split_name'; DROP TABLE x; --",
    "1starts_with_digit",
    "col name",
    "col,other",
    "",
])
def test_ident_rejects_malformed(ident):
    with pytest.raises(ValueError, match="not a valid column identifier"):
        _validated_ident(ident, "split_col")


def test_run_load_data_validates_before_any_query():
    """A malformed table ref fails loudly before a BigQuery client exists."""
    with pytest.raises(ValueError, match="full_table_ref"):
        run_load_data(
            project_id="p",
            full_table_ref="proj.dataset.table; DROP TABLE x",
            label_col="readmission_30d",
            split_col="split_name",
            id_col="subject_id",
            train_split="train",
            val_split="val",
            test_split="test",
            x_train_path="/dev/null", y_train_path="/dev/null",
            x_val_path="/dev/null", y_val_path="/dev/null",
            x_test_path="/dev/null", y_test_path="/dev/null",
            groups_train_path="/dev/null", groups_val_path="/dev/null",
            manifest_path="/dev/null",
        )


def test_run_load_data_rejects_injected_split_col():
    with pytest.raises(ValueError, match="split_col"):
        run_load_data(
            project_id="p",
            full_table_ref="proj.dataset.table",
            label_col="readmission_30d",
            split_col="split_name = 'train' OR 1=1 --",
            id_col="subject_id",
            train_split="train",
            val_split="val",
            test_split="test",
            x_train_path="/dev/null", y_train_path="/dev/null",
            x_val_path="/dev/null", y_val_path="/dev/null",
            x_test_path="/dev/null", y_test_path="/dev/null",
            groups_train_path="/dev/null", groups_val_path="/dev/null",
            manifest_path="/dev/null",
        )
