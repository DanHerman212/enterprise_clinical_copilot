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
# Digest resolved for linux/amd64 python:3.10 (2026-08-31).
FROM python:3.10@sha256:d47f2f44e7976e19c4097d16a515ad0420e1ed44c02ac0ad6bd1131049afc668

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8080

ENTRYPOINT ["python", "-m", "google.cloud.aiplatform.prediction.model_server"]

# Writable workdir/home for any UID (Vertex runs the container as non-root).
RUN mkdir -m 777 -p /usr/app /home
WORKDIR /usr/app
ENV HOME=/home

# CPR prediction-server SDK (separate layer so it caches independently of user
# code). Pinned to an exact version and NOT --force-reinstall'd (ECC-55): the
# previous >=1.27.0 floor + force-reinstall made the image drift under a
# stable content hash. Any version bump is hashed via this Dockerfile.
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
