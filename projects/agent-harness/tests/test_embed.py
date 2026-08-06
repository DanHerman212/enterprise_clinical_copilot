"""Tests for embedding helpers (rag/embed.py). Pure, offline, harness venv."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.embed import (  # noqa: E402
    DOCUMENT_TASK_TYPE,
    EMBEDDING_MODEL,
    OUTPUT_DIMENSIONALITY,
    QUERY_TASK_TYPE,
    RESTRICT_NAMESPACE,
    batch_input_row,
    datapoint_id,
    vector_search_record,
)


def test_datapoint_id_folds_colons():
    assert datapoint_id("12345:brief_hospital_course:2") == "12345_brief_hospital_course_2"


def test_datapoint_id_is_vector_search_safe():
    result = datapoint_id("12345:brief_hospital_course:2")
    assert all(c.isalnum() or c in "_-" for c in result)


def test_batch_input_row_schema():
    row = batch_input_row("7:pertinent_results:1", "07:15AM WBC-8.0")
    assert row["key"] == "7:pertinent_results:1"
    assert row["request"]["content"]["parts"][0]["text"] == "07:15AM WBC-8.0"
    assert row["embed_content_config"]["output_dimensionality"] == 768
    assert row["embed_content_config"]["task_type"] == DOCUMENT_TASK_TYPE


def test_batch_input_row_custom_task_type():
    row = batch_input_row("7:brief_hospital_course:1", "text",
                          task_type=QUERY_TASK_TYPE)
    assert row["embed_content_config"]["task_type"] == QUERY_TASK_TYPE


def test_vector_search_record():
    rec = vector_search_record("7:brief_hospital_course:1", 20924467,
                               [0.1, 0.2, 0.3])
    assert rec["id"] == "7_brief_hospital_course_1"
    assert rec["embedding"] == [0.1, 0.2, 0.3]
    assert rec["restricts"] == [
        {"namespace": RESTRICT_NAMESPACE, "allow": ["20924467"]}
    ]


def test_constants_sane():
    assert EMBEDDING_MODEL == "gemini-embedding-001"
    assert OUTPUT_DIMENSIONALITY == 768
    assert RESTRICT_NAMESPACE == "hadm_id"
