"""Thin wrapper around Vertex AI Experiments.

Goals:
  * One canonical place to call `aiplatform.init` so project, region,
    staging bucket, and experiment name are consistent across every
    notebook and pipeline step.
  * A `log_run` context manager so run lifecycle (start / log / end)
    is uniform and exception-safe.
  * Helpers to upload run artifacts to GCS under a per-run prefix and
    log the resulting URIs as run params (kept as params rather than
    Vertex Artifacts so they survive even when the SDK Artifact
    surface changes).

Intentionally small. Anything model-specific (feature importances,
calibration curves, etc.) belongs in the caller — `log_run` accepts
arbitrary params/metrics dicts, and `upload_artifact` returns a URI
the caller can pass back into `log_params`.
"""
from __future__ import annotations

import io
import json
from contextlib import contextmanager
from typing import Any, Iterator, Mapping

import pandas as pd
from google.cloud import aiplatform, storage

from . import config

_initialized: bool = False


def init(experiment: str = config.EXPERIMENT_NAME) -> None:
    """Idempotently configure the Vertex AI SDK for this process.

    Safe to call multiple times; only the first call hits the SDK.
    """
    global _initialized
    if _initialized:
        return
    aiplatform.init(
        project=config.PROJECT_ID,
        location=config.VERTEX_REGION,
        staging_bucket=config.GCS_STAGING_URI,
        experiment=experiment,
    )
    _initialized = True


def _flatten_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Vertex Experiments params must be scalar (str/int/float/bool).

    Coerce tuples/lists to pipe-joined strings, dicts to compact JSON,
    and everything else to str() as a last resort.
    """
    out: dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = "|".join(str(x) for x in v)
        elif isinstance(v, dict):
            out[k] = json.dumps(v, sort_keys=True, separators=(",", ":"))
        else:
            out[k] = str(v)
    return out


@contextmanager
def log_run(
    run_name: str,
    params: Mapping[str, Any] | None = None,
    *,
    experiment: str = config.EXPERIMENT_NAME,
    resume: bool = False,
) -> Iterator[str]:
    """Open a Vertex AI Experiments run.

    Parameters
    ----------
    run_name :
        Run name. Convention: ``<family>-v<n>`` (``baseline-v1``,
        ``logreg-v1``, ``gbt-v2`` …).
    params :
        Optional initial param dict logged at run start. Augment
        further inside the ``with`` block via :func:`log_params`.
    experiment :
        Vertex experiment name. Defaults to :data:`config.EXPERIMENT_NAME`.
    resume :
        Forwarded to ``aiplatform.start_run``.

    Yields
    ------
    str
        The run name (so callers can build artifact prefixes from it).
    """
    init(experiment=experiment)
    enriched = {"git_sha": config.git_sha(), "mimic_version": config.MIMIC_VERSION}
    if params:
        enriched.update(params)

    run_ctx = aiplatform.start_run(run_name, resume=resume)
    run_ctx.__enter__()
    try:
        log_params(enriched)
        yield run_name
    finally:
        run_ctx.__exit__(None, None, None)


def log_params(params: Mapping[str, Any]) -> None:
    aiplatform.log_params(_flatten_params(params))


def log_metrics(metrics: Mapping[str, float]) -> None:
    # Metrics must be numeric.
    aiplatform.log_metrics({k: float(v) for k, v in metrics.items()})


# --- Artifact helpers (GCS) -------------------------------------------

def _bucket() -> storage.Bucket:
    return storage.Client(project=config.PROJECT_ID).bucket(config.GCS_BUCKET)


def artifact_prefix(run_name: str) -> str:
    """Canonical GCS prefix for per-run artifacts."""
    return f"{config.GCS_ARTIFACTS_URI}/{run_name}"


def upload_dataframe(run_name: str, name: str, df: pd.DataFrame) -> str:
    """Upload a DataFrame as CSV under the run's artifact prefix.

    Returns the full ``gs://...`` URI.
    """
    blob_path = f"artifacts/{run_name}/{name}"
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    blob = _bucket().blob(blob_path)
    blob.upload_from_string(buf.getvalue(), content_type="text/csv")
    return f"{config.GCS_BUCKET_URI}/{blob_path}"


def upload_text(run_name: str, name: str, text: str, content_type: str = "text/plain") -> str:
    blob_path = f"artifacts/{run_name}/{name}"
    blob = _bucket().blob(blob_path)
    blob.upload_from_string(text, content_type=content_type)
    return f"{config.GCS_BUCKET_URI}/{blob_path}"


def run_console_url(run_name: str, experiment: str = config.EXPERIMENT_NAME) -> str:
    return (
        f"https://console.cloud.google.com/vertex-ai/experiments/locations/"
        f"{config.VERTEX_REGION}/experiments/{experiment}/runs"
        f"?project={config.PROJECT_ID}"
    )
