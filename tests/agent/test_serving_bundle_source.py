"""Gap 8 — the working copy is one model, not a mixture.

The repository root used to hold `model.bst`, `manifest.json` and `threshold.json`,
read directly by the cohort scripts. It held July's booster (`4d429896…`) beside a
threshold from September (0.11) — a pair that never existed as a registered model —
while one of the scripts that read the pair applied a third number, hardcoded, with
a comment claiming it came from the file.

These tests pin the replacement: the local scoring path resolves ONE registered
bundle, all three files land in one directory keyed to that bundle, and nothing
reads a loose copy from the repository root again.
"""

import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
SCORER = REPO / "scripts/agent/fill_features.py"

# Files that must never be reconstructed at the repository root: each one is a copy
# of something the serving bundle already carries, and a copy is what drifted.
ROOT_FILES = ("model.bst", "threshold.json", "manifest.json")

ROOTS_TO_SCAN = (
    REPO / "scripts",
    REPO / "services",
    REPO / "tests",
    REPO / "mlops",
)


def _source_files():
    for root in ROOTS_TO_SCAN:
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            yield path


@pytest.mark.parametrize("name", ROOT_FILES)
def test_no_loose_copy_at_the_repository_root(name):
    assert not (REPO / name).exists(), (
        f"{name} is back at the repository root — the scoring path should read it "
        "from a registered bundle, not from a separate copy that can drift"
    )


@pytest.mark.parametrize("name", ROOT_FILES)
def test_no_code_reads_a_root_level_bundle_file(name):
    """The mixture was possible because a loose file was readable in the first place."""
    pattern = re.compile(rf'REPO\s*/\s*"{re.escape(name)}"')

    offenders = [str(p.relative_to(REPO)) for p in _source_files()
                 if pattern.search(p.read_text())]

    assert not offenders, f"root-level {name} is read by: {offenders}"


def test_the_scoring_path_takes_all_three_files_from_one_bundle():
    """Booster, manifest and threshold have to come from the same run.

    Read from three separate places they can silently disagree — which is exactly
    what happened, and the reason the fixture and the endpoint disagreed about ten
    patients.
    """
    source = SCORER.read_text()

    assert "_serving_bundle()" in source
    assert "select_serving_record" in source, (
        "the record must be resolved by the same rule the deploy script uses"
    )
    for name in ROOT_FILES:
        assert f'"{name}"' in source, f"{name} is not downloaded from the bundle"
    assert "BOOSTER_PATH = _BUNDLE" in source
    assert '(_BUNDLE / "manifest.json")' in source
    assert '(_BUNDLE / "threshold.json")' in source


def test_the_bundle_cache_is_keyed_by_the_bundle():
    """A shared cache directory would rebuild the mixture one run later.

    Downloading into one fixed directory means the second bundle's manifest can sit
    beside the first bundle's booster — the same defect with more steps.
    """
    source = SCORER.read_text()

    assert "hashlib.sha256(uri.encode()).hexdigest()" in source, (
        "the cache directory is not keyed to the bundle it holds"
    )
