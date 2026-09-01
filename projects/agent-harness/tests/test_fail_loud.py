"""Tests for Cluster E fail-loud index-build fixes.

Pins:
  * ECC-45: iter_chunks verifies the chunk cache against its manifest (a
    truncated corpus can no longer read as short-but-valid);
  * ECC-40: deploy_index.py requires --expected for tree-ah instead of the
    old hardcoded real-corpus count.
"""

import gzip
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load_notes():
    import rag.notes as notes
    return notes


def test_iter_chunks_matches_manifest(tmp_path, monkeypatch):
    notes = _load_notes()
    chunks = tmp_path / "chunks.jsonl.gz"
    manifest = tmp_path / "chunks.manifest.json"
    with gzip.open(chunks, "wt", encoding="utf-8") as handle:
        for i in range(3):
            handle.write(json.dumps({"chunk_id": f"c{i}"}) + "\n")
    manifest.write_text(json.dumps({"chunk_count": 3}) + "\n")

    monkeypatch.setattr(notes, "CHUNKS_PATH", chunks)
    monkeypatch.setattr(notes, "CHUNKS_MANIFEST", manifest)
    assert [c["chunk_id"] for c in notes.iter_chunks()] == ["c0", "c1", "c2"]


def test_iter_chunks_fails_on_truncated_corpus(tmp_path, monkeypatch):
    notes = _load_notes()
    chunks = tmp_path / "chunks.jsonl.gz"
    manifest = tmp_path / "chunks.manifest.json"
    with gzip.open(chunks, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"chunk_id": "c0"}) + "\n")
    manifest.write_text(json.dumps({"chunk_count": 5}) + "\n")

    monkeypatch.setattr(notes, "CHUNKS_PATH", chunks)
    monkeypatch.setattr(notes, "CHUNKS_MANIFEST", manifest)
    with pytest.raises(RuntimeError, match="Chunk cache is corrupt"):
        list(notes.iter_chunks())


def test_iter_chunks_requires_manifest(tmp_path, monkeypatch):
    notes = _load_notes()
    chunks = tmp_path / "chunks.jsonl.gz"
    chunks.write_bytes(b"")  # exists, but no manifest
    monkeypatch.setattr(notes, "CHUNKS_PATH", chunks)
    monkeypatch.setattr(notes, "CHUNKS_MANIFEST", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError, match="chunk manifest"):
        list(notes.iter_chunks())


def test_tree_ah_requires_expected(monkeypatch):
    import deploy_index  # noqa: E402 (scripts dir already on sys.path)

    monkeypatch.setattr(sys, "argv", ["deploy_index.py", "--mode", "tree-ah"])
    with pytest.raises(SystemExit, match="requires --expected"):
        deploy_index.main()
