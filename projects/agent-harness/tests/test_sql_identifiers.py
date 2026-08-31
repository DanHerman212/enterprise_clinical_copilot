"""Tests for Cluster Q SQL-identifier fixes (agent-harness side).

Pins:
  * ECC-37: chunk_notes binds `split_name` as a query parameter and validates
    both table refs against a strict identifier shape;
  * ECC-29: mcp_server.config validates FEATURE_TABLE / DISCHARGE_TABLE env
    values at import — an env var carrying SQL fails the boot instead of
    reaching a query string.
"""

import gzip
import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

import pytest

from mcp_server.config import _validated_table_ref as config_validate

# The mlops repo owns the `pipelines` package name in the merged test run, so
# the agent-harness component is loaded by file path under a synthetic package
# (the relative `._image` import needs a parent package with a __path__).
_COMP_DIR = Path(__file__).resolve().parents[1] / "pipelines" / "components"
if "ah_components" not in sys.modules:
    _pkg = types.ModuleType("ah_components")
    _pkg.__path__ = [str(_COMP_DIR)]
    sys.modules["ah_components"] = _pkg
_spec = importlib.util.spec_from_file_location(
    "ah_components.chunk_notes", _COMP_DIR / "chunk_notes.py"
)
cn = importlib.util.module_from_spec(_spec)
sys.modules["ah_components.chunk_notes"] = cn
_spec.loader.exec_module(cn)


# ---------------------------------------------------------------------------
# ECC-29 — config-level table ref validation
# ---------------------------------------------------------------------------

def test_config_accepts_plain_table_refs():
    assert config_validate("readmission.hybrid_features", "FEATURE_TABLE",
                           parts=(2,)) == "readmission.hybrid_features"
    assert config_validate("my-proj.readmission.hybrid_notes",
                           "DISCHARGE_TABLE", parts=(3,)) \
        == "my-proj.readmission.hybrid_notes"


@pytest.mark.parametrize("ref", [
    "readmission.notes`; DROP TABLE x; --",
    "readmission.notes WHERE 1=1",
    "proj.dataset.table.extra",
    "readmission",
    "proj.data set.table",
    "proj.dataset.ta'ble",
])
def test_config_rejects_malformed_refs(ref):
    with pytest.raises(RuntimeError, match="not a valid BigQuery table"):
        config_validate(ref, "DISCHARGE_TABLE", parts=(3,))


def test_config_wrong_part_count_rejected():
    # FEATURE_TABLE must be dataset.table — a full FQN would double-prefix.
    with pytest.raises(RuntimeError):
        config_validate("proj.dataset.table", "FEATURE_TABLE", parts=(2,))


# ---------------------------------------------------------------------------
# ECC-37 — chunk_notes parameterization + table ref validation
# ---------------------------------------------------------------------------

class _FakeBQClient:
    """Records the SQL and job_config; returns no rows."""

    def __init__(self, project=None):
        _FakeBQClient.last = self

    def query(self, sql, job_config=None):
        self.sql = sql
        self.job_config = job_config

        class _Job:
            @staticmethod
            def result():
                return []
        return _Job()


def test_chunk_notes_rejects_malformed_table_ref():
    with pytest.raises(ValueError, match="notes_table_ref"):
        cn.run_chunk_notes(
            project_id="p",
            notes_table_ref="proj.dataset.notes`; DROP TABLE x",
            split_table_ref="proj.dataset.split",
            split_name="test",
            pack_to=700,
            sections_csv="hospital course",
            chunks_path="/dev/null",
            manifest_path="/dev/null",
        )


def test_chunk_notes_binds_split_name_as_parameter(monkeypatch):
    monkeypatch.setattr(cn.bigquery, "Client", _FakeBQClient)
    with tempfile.TemporaryDirectory() as tmp:
        chunks_path = str(Path(tmp) / "chunks.jsonl.gz")
        manifest_path = str(Path(tmp) / "manifest.json")
        cn.run_chunk_notes(
            project_id="p",
            notes_table_ref="proj.dataset.notes",
            split_table_ref="proj.dataset.split",
            split_name="test'; DROP TABLE x; --",
            pack_to=700,
            sections_csv="hospital course",
            chunks_path=chunks_path,
            manifest_path=manifest_path,
        )
        client = _FakeBQClient.last
        # The value never appears in the SQL text; it travels as a parameter.
        assert "@split_name" in client.sql
        assert "DROP TABLE" not in client.sql
        params = client.job_config.query_parameters
        assert [p.name for p in params] == ["split_name"]
        assert params[0].value == "test'; DROP TABLE x; --"
        # Empty result still writes a coherent (zero-chunk) manifest.
        with gzip.open(chunks_path, "rt") as fh:
            assert fh.read() == ""
        assert json.loads(Path(manifest_path).read_text())["chunks"] == 0
