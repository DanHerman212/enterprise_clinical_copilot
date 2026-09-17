"""Gap 10 — every file in the serving bundle is covered by the digest that guards it.

`assemble_serving_bundle` records a SHA-256 per bundle file in `checksums.json`,
and the endpoint refuses to start unless every file that file names is present and
matches. It used to walk a hardcoded list and skip any name that did not exist
yet, while `gate_metrics.json` was written *after* the digests were taken. On the
bundle that serves, four files were in the directory and three were in
`checksums.json`; the gate metrics — the numbers that say why the model was
allowed to register — were the file nothing verified.

The fix is structural rather than a reordering: the digest list *is* the list of
files the function wrote, so a written file cannot be missed and a name that
cannot resolve cannot be listed. These tests hold that property, and hold that the
digests are real digests of the bytes on disk.
"""

import ast
import hashlib
import json
import pathlib

import pytest

from mlops.training.components.register_model import assemble_serving_bundle

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
COMPONENT = REPO_ROOT / "mlops/training/components/register_model.py"

GATE_METRICS = {
    "test_aucpr": 0.3294,
    "hpo_val_aucpr": 0.3177,
    "benchmark_aucpr": 0.1799,
    "tuned_threshold": 0.11,
    "beta": 2.0,
    "registered_at_utc": "2026-09-17T17:05:00+00:00",
}


def _sources(tmp_path: pathlib.Path) -> tuple[str, str]:
    """A booster and a manifest to copy in, so no GCS round-trip is needed."""
    booster = tmp_path / "source_model.bst"
    booster.write_bytes(b"not-a-real-booster")
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(json.dumps({"feature_order": ["a", "b"], "data": {}}))
    return str(booster), str(manifest)


def _assemble(tmp_path: pathlib.Path, **overrides) -> tuple[pathlib.Path, dict]:
    booster, manifest = _sources(tmp_path)
    bundle = tmp_path / "serving_model"
    kwargs = {
        "booster_path": booster,
        "manifest_path": manifest,
        "bundle_dir": str(bundle),
        "tuned_threshold": 0.11,
        "beta": 2.0,
        "gate_metrics": GATE_METRICS,
    }
    kwargs.update(overrides)
    assemble_serving_bundle(**kwargs)
    return bundle, json.loads((bundle / "checksums.json").read_text())


def _files(bundle: pathlib.Path) -> set[str]:
    """Every file in the bundle except the digest file itself."""
    return {p.name for p in bundle.iterdir() if p.name != "checksums.json"}


def test_the_gate_metrics_are_covered_by_the_digests(tmp_path):
    """The defect, stated directly: the file that justifies registration."""

    _, checksums = _assemble(tmp_path)

    assert "gate_metrics.json" in checksums


def test_nothing_in_the_bundle_escapes_the_digests(tmp_path):
    """The property, stated generally — the equality is what makes it a guard."""

    bundle, checksums = _assemble(tmp_path)

    assert set(checksums) == _files(bundle)
    assert set(checksums) != {"model.bst", "manifest.json", "threshold.json"}, (
        "gate_metrics.json is missing from the digests again"
    )


def test_every_digest_is_the_sha256_of_the_file_it_names(tmp_path):
    """The endpoint's own check, run here so a bundle can be trusted before it ships."""

    bundle, checksums = _assemble(tmp_path)

    for name, expected in checksums.items():
        path = bundle / name
        assert path.exists(), f"{name} is digested but was never written"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_the_digest_list_is_not_a_fixed_list_that_can_go_stale(tmp_path):
    """A file the function is told to write is covered; one it is not, is not."""
    bundle, checksums = _assemble(tmp_path, gate_metrics=None)

    assert "gate_metrics.json" not in checksums
    assert not (bundle / "gate_metrics.json").exists()
    assert set(checksums) == _files(bundle)


def test_a_bundle_without_a_threshold_still_names_only_what_it_wrote(tmp_path):
    """The threshold is optional, and its absence must not become an unresolved name."""
    bundle, checksums = _assemble(tmp_path, tuned_threshold=None)

    assert "threshold.json" not in checksums
    assert not (bundle / "threshold.json").exists()
    assert set(checksums) == _files(bundle)


def test_the_digest_pass_has_no_existence_check_to_skip_a_file():
    """A skip is how a missing file passed for a covered one; the shape is the bug."""
    tree = ast.parse(COMPONENT.read_text())
    function = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "assemble_serving_bundle"
    )

    called = [
        getattr(node.func, "attr", "")
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
    ]

    assert "exists" not in called, (
        "assemble_serving_bundle skips what is absent, which is how "
        "gate_metrics.json went undigested in the first place"
    )
