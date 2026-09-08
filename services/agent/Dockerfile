# Build from the harness root, not from services/agent/: the agent imports
# services.mcp.config for the Gemini and project settings, so both packages must
# be in the context.
#
#   gcloud builds submit --config services/agent/cloudbuild.yaml \
#     --substitutions _IMAGE=...  .

# Pinned by digest (ECC-16): a mutable base tag would make rebuilds
# non-reproducible. Multi-arch manifest-list digest — resolves linux/amd64 on
# Cloud Build and arm64 locally (a single-arch child digest would pin one
# platform and flag a platform mismatch on Apple Silicon).
FROM python:3.12-slim@sha256:e5c9fa26ffb76e11e0f054f30dc2523a2f9693f0c36c0cf1e39b27e152d899fc

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY services/agent/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY services/agent/ ./services/agent/

# Only the modules the agent actually imports. The rest of services/mcp
# (features, endpoint, manifest) imports google-cloud-aiplatform and BigQuery,
# which are deliberately not in this image — the agent reaches all of that
# through the MCP server. retrieval/sections.py is the canonical discharge-note
# heading vocabulary used by the citation layer (pure stdlib).
COPY services/mcp/__init__.py services/mcp/config.py services/mcp/runtime.py ./services/mcp/
COPY services/mcp/retrieval/__init__.py services/mcp/retrieval/sections.py ./services/mcp/retrieval/

USER 1000

EXPOSE 8080

# No --port: Cloud Run injects PORT and services/agent/http.py reads it.
CMD ["python", "-m", "services.agent.http"]
