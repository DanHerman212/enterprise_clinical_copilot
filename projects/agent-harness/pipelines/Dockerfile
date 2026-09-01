# RAG ingest pipeline image. Lean: no training stack — the ingest components
# only move data between BigQuery, GCS, and the Vertex embeddings API.
# Build context is projects/agent-harness.
# Pinned by digest: reproducible rebuilds (resolves linux/amd64 on Cloud Build,
# arm64 locally).
FROM python:3.12-slim@sha256:e5c9fa26ffb76e11e0f054f30dc2523a2f9693f0c36c0cf1e39b27e152d899fc

RUN pip install --no-cache-dir \
    "google-cloud-bigquery>=3.20" \
    "google-cloud-storage>=2.14" \
    "google-cloud-aiplatform>=1.60" \
    "google-genai>=1.0" \
    "kfp==2.16.1" \
    "numpy>=1.24"

# Bake the RAG source in so @dsl.component wrappers can import their run_*
# helpers (KFP lightweight components only serialize the decorated function
# body; shared code must be on PYTHONPATH inside the container).
WORKDIR /app
COPY rag /app/rag
COPY pipelines /app/pipelines
# recall@k job (exact vs tree-AH) lives in pipelines/ so it is COPY'd into the
# image (scripts/ is excluded from the build context by .dockerignore).
ENV PYTHONPATH=/app
