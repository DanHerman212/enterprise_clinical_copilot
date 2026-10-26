"""Make the migrated service packages importable during tests."""

import os
import sys
from pathlib import Path

import pytest

# Tests must never write to a real Langfuse project.
#
# On 2026-10-20 the evaluation work exported LANGFUSE_* into a persistent shell to seed
# datasets; every later `pytest tests` run in that shell then emitted real traces from
# the agent tests (110 of them, no scores) into production observability. Removing the
# variables at conftest import makes that impossible regardless of the caller's shell,
# and costs nothing: tracing is designed to be absent-and-silent when unconfigured.
for _tracing_var in ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
    os.environ.pop(_tracing_var, None)

REPO_ROOT = Path(__file__).resolve().parents[2]

HARNESS_ROOT = Path(__file__).resolve().parents[1]
if str(HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_ROOT))


def _project_id_from_repo_env() -> str | None:
    """PROJECT_ID out of the repo-root `.env`, where the services read it too.

    `config.py` fails closed without a project, so a default is needed to keep the
    unit tests importable. It must not be a default that SHADOWS the real one: the
    live tests spawn the MCP server, the subprocess inherits this environment, and a
    placeholder project makes every BigQuery call fail — which the stdio tests then
    reported as "MCP stdio session failed", blaming the transport for a setting
    (2026-10-12).

    Parsed here rather than imported from `services.mcp.runtime`, because this runs
    before anything downstream is importable. Same shape of read.
    """
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return None
    for line in env_file.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("PROJECT_ID="):
            value = stripped.split("=", 1)[1].strip()
            if value:
                return value
    return None


os.environ.setdefault("PROJECT_ID", _project_id_from_repo_env() or "unit-test-project")


@pytest.fixture(scope="session")
def require_served():
    """A served tool response, or a skip that says why there is none.

    A guard against anchor drift rather than a workaround. The fixture
    (`tests/agent/fixtures/expected.json`) declares its admission in
    `anchor_hadm_id` and the live tests assert values against it, which is only
    possible while the served feature source holds that admission.

    It did not, until 2026-10-12: the anchor was 20924467, a MIMIC-derived
    admission, and the predict path serves `readmission.hybrid_features` — the demo
    cohort. The chain was two stale paths after the test tree moved into
    `tests/agent/`: the cohort builder stopped pinning fixture patients (it read
    `tests/fixtures/`), so the cohort was rebuilt without them, and the tests kept
    measuring a patient the service could not find.

    So when the anchor is missing they say so, rather than passing on a comparison
    they cannot make or failing on a number nobody served.
    """
    def _require(run: dict) -> dict:
        calls = run.get("tool_calls") or []
        # The prediction, by name, rather than whichever call ran first: `/ehr` carries
        # the estimate as a supplementary call beside its section retrieval, and it is
        # the only command that reaches the predictor at all since `/risk` was retired
        # (2026-10-26). Reading `calls[0]` measured the retrieval instead.
        chosen = next((call for call in calls
                       if call.get("name") == "predict_readmission"), None)
        if chosen is None and calls:
            chosen = calls[0]
        response = chosen.get("response") if chosen else None
        if isinstance(response, dict) and response.get("error") == "unknown_patient":
            pytest.skip(
                "the fixture's anchor admission is not in the served feature source: "
                f"{response.get('message') or 'unknown_patient'} — re-anchor the "
                "fixture (mlops/training/smoke_test.py <hadm_id> --write-fixture)"
            )
        return response

    return _require
