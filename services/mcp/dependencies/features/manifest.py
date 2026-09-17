"""Serving-bundle location: which bundle to read the feature order from.

Cached at module level. `smoke_test.py` refetches per run, which is fine for a
CLI; in a long-lived MCP server that would be a GCS round-trip on every tool
call.

This resolves *where the feature order is read from*. It deliberately does not
resolve which model is serving, which is a question only the endpoint can
answer — see `services/mcp/tools/prediction.py::_model_identity`.
"""

import json
from functools import lru_cache

from google.cloud import aiplatform, storage

from ...config import (
    BUNDLE_URI_OVERRIDE,
    FINAL_MODEL_PREFIX,
    LOCATION,
    PIPELINE_STAGE,
    PROJECT,
)


@lru_cache(maxsize=1)
def _discover() -> str:
    """The GCS dir of the newest pipeline-registered version's serving bundle.

    The rule: the model that serves is the bundle from the most recent successful
    pipeline run, so a record qualifies only if the training pipeline registered
    it — the final name prefix *and* the stage label `register_model.py` writes
    with it. The same predicate is applied in `mlops/serving/deploy_cpr.py` and
    `mlops/training/smoke_test.py`, which is what makes "which bundle do we read
    the feature order from" and "which bundle do we deploy" the same answer.
    Selecting on the name alone is what let a hand re-registration take the newest
    position; that path now publishes `readmission-manual-*` with `stage=manual`.

    Raises rather than calling sys.exit: this is imported by a long-lived
    server, where killing the process on a lookup failure would take every
    other in-flight request with it.
    """
    if BUNDLE_URI_OVERRIDE:
        return BUNDLE_URI_OVERRIDE.rstrip("/")

    aiplatform.init(project=PROJECT, location=LOCATION)
    models = [
        m for m in aiplatform.Model.list(order_by="create_time desc")
        if m.display_name.startswith(FINAL_MODEL_PREFIX)
        and (m.labels or {}).get("stage") == PIPELINE_STAGE
    ]
    if not models:
        raise RuntimeError(
            f"No pipeline-registered '{FINAL_MODEL_PREFIX}*' model found in "
            f"{PROJECT}/{LOCATION} (stage={PIPELINE_STAGE}); run the training "
            "pipeline or set BUNDLE_URI to override."
        )
    return models[0].gca_resource.artifact_uri.rstrip("/")


def bundle_uri() -> str:
    """GCS dir of the serving bundle."""
    return _discover()


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
