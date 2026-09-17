"""Tests for embedding helpers (rag/embed.py). Pure, offline, harness venv."""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.mcp.retrieval.embed import (  # noqa: E402
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


def test_the_embedding_space_is_defined_once(monkeypatch):
    """Gap 5: every consumer resolves THIS object, not an equal value.

    Identity is the assertion, not equality: a copy that happens to agree today is
    exactly the drift surface this gap is about — the index is built with one
    definition and the query is embedded with the other, and the failure is
    plausible neighbours rather than an error.

    The environment is set for the duration, so a path that still reads a setting
    fails here rather than in production.
    """
    import importlib

    import services.mcp.retrieval.config as pipeline_config
    import services.mcp.tools.retrieval as serving
    from services.mcp.pipelines import rag_ingest_pipeline as pipeline

    monkeypatch.setenv("EMBEDDING_MODEL", "gemini-imaginary-1")
    monkeypatch.setenv("EMBEDDING_DIM", "999")

    importlib.reload(serving)

    # The serving path: same objects, whatever the environment says.
    assert serving.EMBEDDING_MODEL is EMBEDDING_MODEL
    assert serving.OUTPUT_DIMENSIONALITY is OUTPUT_DIMENSIONALITY
    assert serving.RESTRICT_NAMESPACE is RESTRICT_NAMESPACE

    # The pipeline: same objects again, resolved rather than read from the YAML.
    cfg = pipeline_config.load()
    assert cfg.embedding_model is EMBEDDING_MODEL
    assert cfg.dimensions is OUTPUT_DIMENSIONALITY
    assert cfg.query_task_type is QUERY_TASK_TYPE

    # The pipeline's own default, which is how a DAG is compiled. The decorated object
    # is a KFP GraphComponent, so the function is reached through `pipeline_func`.
    from inspect import signature

    compiled = pipeline.rag_ingest_pipeline.pipeline_func
    default = signature(compiled).parameters["dimensions"].default
    assert default is OUTPUT_DIMENSIONALITY

    # And the copy that was removed has not crept back.
    assert "embedding" not in yaml.safe_load(
        pipeline_config.CONFIG_PATH.read_text()
    )


def test_a_returning_embedding_block_is_an_error(tmp_path):
    """Deleting the copy is the fix; refusing its return is what keeps it deleted."""
    import services.mcp.retrieval.config as pipeline_config

    doc = yaml.safe_load(pipeline_config.CONFIG_PATH.read_text())
    doc["embedding"] = {"model": "gemini-embedding-001", "dimensions": 768,
                        "query_task_type": "RETRIEVAL_QUERY"}
    path = tmp_path / "rag_config.yaml"
    path.write_text(yaml.safe_dump(doc))

    with pytest.raises(ValueError, match="embedding block"):
        pipeline_config.load(path)
