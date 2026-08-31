"""Make the two feature sources importable as `mcp_server.*` without packaging."""

import os
import sys
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parents[1]
if str(HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_ROOT))

# config.py requires PROJECT_ID (fail-closed); unit tests stay hermetic.
os.environ.setdefault("PROJECT_ID", "unit-test-project")
