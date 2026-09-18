# RAG ingest pipeline image. Lean: no training stack — the ingest components
# only move data between BigQuery, GCS, and the Vertex embeddings API.
#
# Build context is services/mcp (the .dockerignore beside this file trims it).
# The layout inside the image mirrors that context, because the code imports its
# own package as `pipelines.*` and the retrieval layer as `services.mcp.retrieval.*`.
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

# Bake the source in so @dsl.component wrappers can import their run_*
# helpers (KFP lightweight components only serialize the decorated function
# body; shared code must be on PYTHONPATH inside the container).
#
# This block copied `rag/` until 2026-09-18, a directory that no longer exists:
# the ingest subtree moved from services/{rag,pipelines} to
# services/mcp/{retrieval,pipelines} and the image went unbuilt from 2026-08-21
# because of it. A stale image here is not cosmetic — the recall job runs from
# it, so the deploy gate measured an index with a component that could not
# report `empty_result_rate` and refused a promotion over a metric that was
# never measured, while recall on the same run was 0.992 against a 0.90 floor.
# `recall_k.py` (the exact-vs-tree-AH job) lives in pipelines/, and scripts/ is
# excluded from the context by .dockerignore.
WORKDIR /app
COPY pipelines /app/pipelines
COPY retrieval /app/services/mcp/retrieval
# config.py and the corpus configuration resolve relative to the package
# (retrieval/config.py reads ../rag_config.yaml), so they ship together.
COPY __init__.py config.py rag_config.yaml /app/services/mcp/
ENV PYTHONPATH=/app
