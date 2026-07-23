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
FROM python:3.10

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8080

ENTRYPOINT ["python", "-m", "google.cloud.aiplatform.prediction.model_server"]

# Writable workdir/home for any UID (Vertex runs the container as non-root).
RUN mkdir -m 777 -p /usr/app /home
WORKDIR /usr/app
ENV HOME=/home

# CPR prediction-server SDK (separate layer so it caches independently of user code).
RUN pip install --no-cache-dir --force-reinstall 'google-cloud-aiplatform[prediction]>=1.27.0'

# CPR handler/predictor wiring (read by the model server at startup).
ENV HANDLER_MODULE=google.cloud.aiplatform.prediction.handler
ENV HANDLER_CLASS=PredictionHandler
ENV PREDICTOR_MODULE=predictor
ENV PREDICTOR_CLASS=ReadmissionPredictor

# Copy the source directory into the container.
COPY . .

# User serving dependencies (xgboost, numpy).
RUN pip install --no-cache-dir --force-reinstall -r requirements.txt
