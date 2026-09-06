# Readmission CPR serving image.
#
# This is a hand-maintained equivalent of the Dockerfile that
# LocalModel.build_cpr_model(...) generated locally, so the image can be built
# on Cloud Build (native linux/amd64) instead of the developer's machine.
# Keep the ENV wiring and layer order in sync with the SDK contract.
#
# The image is model-agnostic: the trained model is downloaded at runtime from
# AIP_STORAGE_URI (the bundle artifact_uri). Only rebuild when predictor.py or
# requirements.txt change (deploy_cpr.py enforces this via a content hash tag).
# Pinned by digest (ECC-55): a mutable base tag would make the content-hash
# image tag lie — the same hash could build materially different images.
# Multi-arch manifest-list digest — resolves linux/amd64 on Cloud Build and
# arm64 locally. Bumped 3.10 → 3.12-slim (2026-09-01): the 3.10 base carried
# 3 critical / 25 high CVEs and nears EOL; 3.12-slim (trixie) is 0 critical /
# 0 high per docker scout.
FROM python:3.12-slim@sha256:e5c9fa26ffb76e11e0f054f30dc2523a2f9693f0c36c0cf1e39b27e152d899fc

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8080

ENTRYPOINT ["python", "-m", "google.cloud.aiplatform.prediction.model_server"]

# Writable workdir/home for any UID (Vertex runs the container as non-root).
RUN mkdir -m 777 -p /usr/app /home
WORKDIR /usr/app
ENV HOME=/home

# libgomp1: xgboost's manylinux wheels link OpenMP at runtime; the slim base
# omits it (the full python image bundles it via buildpack-deps). Installed
# against the pinned base digest so the apt state stays deterministic.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPR prediction-server SDK (separate layer so it caches independently of user
# code). Pinned to an exact version and NOT --force-reinstall'd (ECC-55): the
# previous >=1.27.0 floor + force-reinstall made the image drift under a
# stable content hash. Any version bump is hashed via this Dockerfile.
# Residual (2026-09-01): this pin caps fastapi<=0.124.4 → starlette<0.51, which
# leaves 2 HIGH starlette CVEs (SSRF in form parsing + DoS) fixable only in
# starlette 1.1/1.3.1. The endpoint accepts JSON instances only and sits behind
# Vertex's managed endpoint, so exploit surface is minimal; a future aiplatform
# release that lifts the fastapi cap clears them.
RUN pip install --no-cache-dir 'google-cloud-aiplatform[prediction]==1.161.0'

# CPR handler/predictor wiring (read by the model server at startup).
ENV HANDLER_MODULE=google.cloud.aiplatform.prediction.handler
ENV HANDLER_CLASS=PredictionHandler
ENV PREDICTOR_MODULE=predictor
ENV PREDICTOR_CLASS=ReadmissionPredictor

# Copy the source directory into the container.
COPY . .

# User serving dependencies (xgboost, numpy). Not --force-reinstall'd: the
# image must be reproducible from its content hash (ECC-55).
RUN pip install --no-cache-dir -r requirements.txt
