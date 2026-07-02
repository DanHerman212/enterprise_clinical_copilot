"""
app — custom serving container HTTP app for the readmission predictor.

Implements the Vertex AI custom-container contract:
  * a health route (GET) returning 200 when the model is loaded;
  * a predict route (POST) accepting ``{"instances": [ {feature: value, ...} ]}``
    and returning ``{"predictions": [ prob, ... ]}``.

The serving bundle (model.joblib + imputer.joblib + schema.json) is loaded at
startup from ``AIP_STORAGE_URI`` (a GCS directory, downloaded locally) or from
``MODEL_DIR`` (a local directory, used in tests). Predictions are produced by
the shared ``Predictor``, which reproduces the exact training encoding.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from pipelines.serving.predictor import Predictor

HEALTH_ROUTE = os.environ.get("AIP_HEALTH_ROUTE", "/health")
PREDICT_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/predict")

_predictor = Predictor()


def _localize(uri: str) -> str:
    """Return a local directory for the serving bundle, downloading from GCS."""
    if uri.startswith("gs://"):
        from google.cloud import storage

        bucket_name, _, prefix = uri[len("gs://"):].partition("/")
        local_dir = tempfile.mkdtemp(prefix="serving_bundle_")
        client = storage.Client()
        for blob in client.list_blobs(bucket_name, prefix=prefix):
            name = os.path.basename(blob.name)
            if name:  # skip "directory" placeholder blobs
                blob.download_to_filename(os.path.join(local_dir, name))
        return local_dir
    return uri


def load_artifacts() -> None:
    """Resolve the bundle location and load it into the predictor."""
    uri = os.environ.get("AIP_STORAGE_URI") or os.environ.get("MODEL_DIR")
    if not uri:
        raise RuntimeError(
            "Set AIP_STORAGE_URI (Vertex) or MODEL_DIR to the serving bundle."
        )
    _predictor.load(_localize(uri))


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(title="readmission-predictor", lifespan=lifespan)


@app.get(HEALTH_ROUTE)
def health() -> dict:
    return {"status": "healthy"}


@app.post(PREDICT_ROUTE)
async def predict(request: Request) -> dict:
    body = await request.json()
    instances = body.get("instances", [])
    predictions = _predictor.predict(instances)
    return {"predictions": predictions}
