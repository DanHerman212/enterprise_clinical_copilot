"""Tests for Cluster M deploy-safety fixes (agent-harness side).

Pins:
  * ECC-36/53: the shared synthetic-scale guard refuses to deploy a
    real-corpus (>100k vector) index to the public Vector Search endpoint;
  * ECC-58: the agent/MCP Cloud Build configs pin a dedicated service account
    (and keep the private-service posture).
"""

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import _deploy_guard  # noqa: E402


def test_guard_allows_synthetic_scale():
    # The synthetic demo cohort is a few hundred vectors.
    _deploy_guard.assert_synthetic_scale(136, "rag-tree-ah-20260831")
    _deploy_guard.assert_synthetic_scale(100_000, "rag-tree-ah-20260831")


def test_guard_refuses_real_corpus_scale():
    with pytest.raises(SystemExit, match="MIMIC-derived corpus"):
        _deploy_guard.assert_synthetic_scale(555_770, "rag-tree-ah-real")


def test_guard_refuses_anything_over_limit():
    with pytest.raises(SystemExit):
        _deploy_guard.assert_synthetic_scale(100_001, "rag-tree-ah-x")


def _cloudbuild_text(name: str) -> str:
    return (Path(__file__).resolve().parents[1] / name).read_text()


def test_cloudbuild_configs_pin_build_service_account():
    for name in ("cloudbuild.agent.yaml", "cloudbuild.mcp.yaml"):
        text = _cloudbuild_text(name)
        assert "serviceAccount:" in text, name
        assert "cicd-deployer@" in text, name


def test_cloudbuild_configs_keep_private_service_posture():
    for name in ("cloudbuild.agent.yaml", "cloudbuild.mcp.yaml"):
        assert "--no-allow-unauthenticated" in _cloudbuild_text(name), name
