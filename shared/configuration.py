"""Configuration values shared by deployable services."""

import os


def project_id() -> str:
    """Return the configured project or fail closed."""
    value = os.environ.get("PROJECT_ID", "").strip()
    if not value:
        raise RuntimeError("PROJECT_ID must be configured; no production default exists.")
    return value


def location(default: str = "us-east1") -> str:
    """Return the configured Google Cloud location."""
    return os.environ.get("LOCATION", default)
