"""Gap 3 — an ingest records the data version it read, and never reuses a vector
from an unidentified vector space.

Two defects are pinned here. The first is that the artifacts an ingest produces
described counts and parameters but never the state of the source tables, so an
index could not be told apart from one built against a different corpus state.
The second is that embedding reuse was keyed on the datapoint identifier alone.
Those identifiers are derived from the note, the section and an ordinal, so they
are stable across runs, corpora and embedding models: an identifier match says
nothing about the vector carrying it, and reusing across a model change would
put two incompatible geometries in one index while every count and dimension
check still passed.

The tests are offline. The GCS client is a stub, because what is under test is
the decision the component makes from what a manifest says, not the transport.
"""

import ast
import json
import pathlib

import pytest

from services.mcp.pipelines.components import embed_chunks
from services.mcp.pipelines.rag_ingest_pipeline import reuse_uri
from services.mcp.retrieval.embed import (
    DOCUMENT_TASK_TYPE,
    EMBEDDING_MODEL,
    OUTPUT_DIMENSIONALITY,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
PIPELINES = REPO / "services/mcp/pipelines"
COMPONENTS = PIPELINES / "components"

REUSE_URI = "gs://bucket/rag/embeddings/demo/abc123/embed_ingest.jsonl.gz"


class _Blob:
    def __init__(self, text: str | None):
        self._text = text

    def exists(self) -> bool:
        return self._text is not None

    def download_as_text(self) -> str:
        assert self._text is not None
        return self._text


class _Bucket:
    def __init__(self, objects: dict[str, str]):
        self._objects = objects

    def blob(self, name: str) -> _Blob:
        return _Blob(self._objects.get(name))


class _Storage:
    """A GCS client that answers from a dictionary keyed by object name."""

    def __init__(self, objects: dict[str, str]):
        self._objects = objects

    def bucket(self, _name: str) -> _Bucket:
        return _Bucket(self._objects)


def _manifest(**overrides) -> str:
    described = {
        "model": EMBEDDING_MODEL,
        "output_dimensionality": OUTPUT_DIMENSIONALITY,
        "task_type": DOCUMENT_TASK_TYPE,
    }
    described.update(overrides)
    return json.dumps(described)


def _reuse_with(objects: dict[str, str]):
    client = _Storage(objects)
    # The helper looks for the manifest under the conventional sibling names.
    return embed_chunks._verified_reuse_source(client, REUSE_URI)


def test_the_reuse_source_is_accepted_when_the_vector_space_agrees():
    described = _reuse_with({
        "rag/embeddings/demo/abc123/embed_ingest.manifest.json": _manifest(),
    })

    assert described["model"] == EMBEDDING_MODEL


def test_a_source_built_with_another_model_is_refused():
    """An identifier match says nothing about the vector carrying it."""
    with pytest.raises(SystemExit) as exc:
        _reuse_with({
            "rag/embeddings/demo/abc123/embed_ingest.manifest.json":
                _manifest(model="text-embedding-005"),
        })

    assert "refusing to reuse" in str(exc.value)
    assert "text-embedding-005" in str(exc.value)


def test_a_source_built_with_another_dimension_is_refused():
    with pytest.raises(SystemExit):
        _reuse_with({
            "rag/embeddings/demo/abc123/embed_ingest.manifest.json":
                _manifest(output_dimensionality=256),
        })


def test_a_source_built_with_another_task_type_is_refused():
    with pytest.raises(SystemExit):
        _reuse_with({
            "rag/embeddings/demo/abc123/embed_ingest.manifest.json":
                _manifest(task_type="RETRIEVAL_QUERY"),
        })


def test_a_source_with_no_manifest_is_refused_rather_than_trusted():
    """Absent provenance is not the same as good provenance."""
    with pytest.raises(SystemExit) as exc:
        _reuse_with({})

    assert "no manifest found" in str(exc.value)


def test_the_pipeline_manifest_also_counts_as_a_description():
    """A pipeline run writes its ingest beside a sibling named `manifest`."""
    described = _reuse_with({
        "rag/embeddings/demo/abc123/manifest":
            json.dumps({"dimensions": OUTPUT_DIMENSIONALITY,
                        "model": EMBEDDING_MODEL,
                        "task_type": DOCUMENT_TASK_TYPE}),
    })

    assert described["dimensions"] == OUTPUT_DIMENSIONALITY


def test_the_reuse_path_is_keyed_by_corpus_and_data_version():
    uri = reuse_uri("proj", "demo", "fingerprint1")

    assert uri == "gs://proj-mlops/rag/embeddings/demo/fingerprint1/embed_ingest.jsonl.gz"
    assert reuse_uri("proj", "mimic", "fingerprint1") != uri
    assert reuse_uri("proj", "demo", "fingerprint2") != uri


@pytest.mark.parametrize(
    "component, key",
    [
        ("chunk_notes.py", "data_fingerprint"),
        ("embed_chunks.py", "data_fingerprint"),
        ("build_index.py", "data_fingerprint"),
    ],
)
def test_every_artifact_an_ingest_writes_records_the_data_version(component, key):
    """The fingerprint is the only thing tying an artifact to a data state."""
    tree = ast.parse((COMPONENTS / component).read_text())
    literals = {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert key in literals, f"{component} does not record {key}"


def test_the_embed_manifest_also_records_the_vector_space():
    """A description of the artifact is what makes the next reuse checkable."""
    literals = {
        node.value for node in ast.walk(ast.parse((COMPONENTS / "embed_chunks.py").read_text()))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    for key in ("vector_space", "reuse_source", "model", "task_type"):
        assert key in literals, f"the embed manifest does not record {key}"


def test_the_vector_space_the_manifest_records_is_the_serving_one():
    """The manifest must name the serving module's constants, not literals.

    A written-out model name or dimensionality would satisfy a reader and drift
    from the serving path the moment either changed. Both write paths in the
    component record the vector space, so both are checked.
    """
    tree = ast.parse((COMPONENTS / "embed_chunks.py").read_text())
    blocks = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and key.value == "vector_space":
                assert isinstance(value, ast.Dict), "vector_space is not a mapping"
                blocks.append(value)

    assert len(blocks) == 2, f"expected both write paths to record it, found {len(blocks)}"
    for block in blocks:
        recorded = {
            key.value: value
            for key, value in zip(block.keys, block.values)
            if isinstance(key, ast.Constant)
        }
        assert set(recorded) == {"model", "dimensions", "task_type"}
        for field, expected in (
            ("model", "EMBEDDING_MODEL"),
            ("dimensions", "OUTPUT_DIMENSIONALITY"),
            ("task_type", "DOCUMENT_TASK_TYPE"),
        ):
            value = recorded[field]
            assert isinstance(value, ast.Name), f"{field} is written out as a literal"
            assert value.id == expected, f"{field} is recorded as {value.id}"

    # And those names have to arrive from the module the serving path imports.
    sources = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and isinstance(node.module, str)
        and node.module.endswith("retrieval.embed")
    ]
    assert sources, "the component does not import its vector space from the serving module"


def test_the_standalone_driver_describes_its_artifact_the_way_reuse_requires():
    """The driver uploads the artifact a later ingest may reuse, so the two
    files have to agree on what a description of that artifact looks like."""
    tree = ast.parse((REPO / "scripts/agent/embed_chunks.py").read_text())
    keys = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "result"
            for target in node.targets
        ):
            keys = [key.value for key in node.value.keys]  # type: ignore[union-attr]

    assert keys is not None
    for key in ("model", "output_dimensionality", "task_type", "data_fingerprint"):
        assert key in keys, f"the driver's manifest does not record {key}"


def test_the_driver_uploads_where_the_pipeline_looks_for_a_reuse_source():
    """The upload key and the reuse key are the same convention in two files."""
    source = (REPO / "scripts/agent/embed_chunks.py").read_text()

    assert 'f"rag/embeddings/{CORPUS}/{fingerprint}/"' in source, source
    assert reuse_uri("proj", "demo", "fp").endswith(
        "/rag/embeddings/demo/fp/embed_ingest.jsonl.gz"
    )


def test_the_committed_ir_carries_the_fingerprint_to_every_step():
    """The compiled artifact is what runs, so the wiring has to be in it."""
    ir = (PIPELINES / "rag_ingest_pipeline.yaml").read_text()

    # Three consumers: chunk, embed and index each receive the fingerprint.
    assert ir.count("data_fingerprint") >= 3
