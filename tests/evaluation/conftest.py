"""Make the harness modules importable, and keep tests off a real Langfuse project."""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The same guard the agent tests carry: a shell that exported the credentials for a
# live run must not turn the harness tests into producers of real traces.
for _tracing_var in ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
    os.environ.pop(_tracing_var, None)
