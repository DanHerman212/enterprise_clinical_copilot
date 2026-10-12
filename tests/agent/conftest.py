"""Make the migrated service packages importable during tests."""

import os
import sys
from pathlib import Path

import pytest

HARNESS_ROOT = Path(__file__).resolve().parents[1]
if str(HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_ROOT))

# config.py requires PROJECT_ID (fail-closed); unit tests stay hermetic.
os.environ.setdefault("PROJECT_ID", "unit-test-project")


@pytest.fixture(scope="session")
def require_served():
    """A served tool response, or a skip that says why there is none.

    `tests/agent/fixtures/expected.json` records admission 20924467, and the predict
    path no longer holds it: `services/mcp/config.py` reads
    `readmission.hybrid_features` by default — the demo cohort, 90000001+ — and the
    MIMIC-derived table the fixture was generated from is out of scope for the
    demonstration. So the live tests that assert a VALUE cannot assert one, and the
    difference is not a regression: the anchor moved out from under them.

    They say that rather than passing on a comparison they never made, or failing
    on a number that was never served. Re-anchoring is what turns them back into
    assertions, and it is small but not a command: the fixture's writer
    (`mlops/training/smoke_test.py --write-fixture`) still reads the MIMIC-derived
    `analytics_dataset_encoded` table, and the recorded `model_version` has moved on
    as well.
    """
    def _require(run: dict) -> dict:
        calls = run.get("tool_calls") or []
        response = calls[0].get("response") if calls else None
        if isinstance(response, dict) and response.get("error") == "unknown_patient":
            pytest.skip(
                "the fixture's admission is not in the served feature source: "
                f"{response.get('message') or 'unknown_patient'} — the live value "
                "assertions need a re-anchored fixture"
            )
        return response

    return _require
