"""Unit tests for the offline eval_ingest integrity gate (pure, no cloud)."""

import json

import pytest

from pipelines.components.eval_ingest import run_eval_ingest


def _write_ingest(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _record(id_, dims=3):
    return {"id": id_, "embedding": [0.1] * dims, "restricts": []}


def test_passes_and_writes_report(tmp_path):
    ingest = tmp_path / "ingest.json"
    _write_ingest(ingest, [_record(f"a_{i}") for i in range(5)])

    ok = run_eval_ingest(
        ingest_path=str(ingest), expected=5, dimensions=3,
        report_dir=str(tmp_path / "report"),
        corpus="mimic", index_name="ingest",
    )
    assert ok is True
    assert (tmp_path / "report" / "eval_report.html").exists()
    assert (tmp_path / "report" / "eval_results.json").exists()


def test_count_mismatch_raises(tmp_path):
    ingest = tmp_path / "ingest.json"
    _write_ingest(ingest, [_record(f"a_{i}") for i in range(5)])

    with pytest.raises(SystemExit, match="vector count"):
        run_eval_ingest(
            ingest_path=str(ingest), expected=3, dimensions=3,
            report_dir=str(tmp_path / "report"),
            corpus="mimic", index_name="ingest",
        )


def test_duplicate_id_raises(tmp_path):
    ingest = tmp_path / "ingest.json"
    _write_ingest(ingest, [_record("dup"), _record("dup")])

    with pytest.raises(SystemExit, match="duplicate"):
        run_eval_ingest(
            ingest_path=str(ingest), expected=2, dimensions=3,
            report_dir=str(tmp_path / "report"),
            corpus="mimic", index_name="ingest",
        )


def test_dimension_mismatch_raises(tmp_path):
    ingest = tmp_path / "ingest.json"
    _write_ingest(ingest, [_record("a", dims=2)])

    with pytest.raises(SystemExit, match="dims"):
        run_eval_ingest(
            ingest_path=str(ingest), expected=1, dimensions=3,
            report_dir=str(tmp_path / "report"),
            corpus="mimic", index_name="ingest",
        )
