# Custom real-time serving container for the readmission predictor (Option 2).
#
# Build context is projects/mlops (so `src/` and `pipelines/` are copied):
#   gcloud builds submit --config pipelines/serving/cloudbuild.yaml projects/mlops
# or locally:
#   docker build -f pipelines/serving/Dockerfile -t readmission-serving projects/mlops
FROM python:3.10-slim

WORKDIR /app

COPY pipelines/serving/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Code needed by the predictor: the shared encoding (components/data.py), the
# imputer/config (src/), and the serving app.
COPY src /app/src
COPY pipelines/components /app/pipelines/components
COPY pipelines/serving /app/pipelines/serving

ENV PYTHONPATH=/app
# src.config resolves a GCP project id at import; serving does no BigQuery, so a
# placeholder keeps the import from failing.
ENV PROJECT_ID=serving
ENV AIP_HTTP_PORT=8080
EXPOSE 8080

CMD exec uvicorn pipelines.serving.app:app --host 0.0.0.0 --port ${AIP_HTTP_PORT:-8080}
