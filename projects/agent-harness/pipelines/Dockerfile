# RAG ingest pipeline image. Lean: no training stack — the ingest components
# only move data between BigQuery, GCS, and the Vertex embeddings API.
# Build context is projects/agent-harness.
FROM python:3.12-slim

RUN pip install --no-cache-dir \
    "google-cloud-bigquery>=3.20" \
    "google-cloud-storage>=2.14" \
    "google-cloud-aiplatform>=1.60" \
    "google-genai>=1.0" \
    "kfp==2.16.1"

# Bake the RAG source in so @dsl.component wrappers can import their run_*
# helpers (KFP lightweight components only serialize the decorated function
# body; shared code must be on PYTHONPATH inside the container).
WORKDIR /app
COPY rag /app/rag
COPY pipelines /app/pipelines
ENV PYTHONPATH=/app
