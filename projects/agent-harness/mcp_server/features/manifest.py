"""Serving-bundle manifest: feature order and parent groups.

Cached at module level. `smoke_test.py` refetches per run, which is fine for a
CLI; in a long-lived MCP server that would be a GCS round-trip on every tool
call.
"""

import json
import sys
from functools import lru_cache

from google.cloud import aiplatform, storage

from ..config import BUNDLE_URI_OVERRIDE, FINAL_MODEL_PREFIX, LOCATION, PROJECT


@lru_cache(maxsize=1)
def bundle_uri() -> str:
    """GCS dir of the serving bundle, from the newest provenance record."""
    if BUNDLE_URI_OVERRIDE:
        return BUNDLE_URI_OVERRIDE.rstrip("/")

    aiplatform.init(project=PROJECT, location=LOCATION)
    models = [
        m for m in aiplatform.Model.list(order_by="create_time desc")
        if m.display_name.startswith(FINAL_MODEL_PREFIX)
    ]
    if not models:
        sys.exit(f"No '{FINAL_MODEL_PREFIX}*' model found; set BUNDLE_URI to override.")
    return models[0].gca_resource.artifact_uri.rstrip("/")


@lru_cache(maxsize=1)
def manifest() -> dict:
    """Read manifest.json from GCS.

    Uses the storage client rather than shelling out to `gsutil`: once
    aiplatform has opened its gRPC channels, subprocess's fork() can deadlock
    in gRPC's pthread_atfork handler. It is racy, so it presents as an
    intermittent hang with no child process ever appearing.
    """
    uri = bundle_uri()
    bucket_name, _, prefix = uri[len("gs://"):].partition("/")
    blob = storage.Client(project=PROJECT).bucket(bucket_name).blob(f"{prefix}/manifest.json")
    return json.loads(blob.download_as_text())


def feature_order() -> list[str]:
    return manifest()["feature_order"]


def groups() -> dict[str, list[str]]:
    return manifest().get("groups", {})
