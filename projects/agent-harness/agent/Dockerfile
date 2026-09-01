# Build from the harness root, not from agent/: the agent imports
# mcp_server.config for the Gemini and project settings, so both packages must
# be in the context.
#
#   gcloud builds submit --config projects/agent-harness/cloudbuild.agent.yaml \
#     --substitutions _IMAGE=...  projects/agent-harness

# Pinned by digest (ECC-16): a mutable base tag would make rebuilds
# non-reproducible. Digest resolved for linux/amd64 python:3.12-slim (2026-09-01).
FROM python:3.12-slim@sha256:2fe5997d249a808b8eeea52c58a1dbffbba28754dc11699ef5c029f2d818ce79

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY agent/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY agent/ ./agent/

# Only the config module. The rest of mcp_server (features, endpoint, manifest)
# imports google-cloud-aiplatform and BigQuery, which are deliberately not in
# this image — the agent reaches all of that through the MCP server.
COPY mcp_server/__init__.py mcp_server/config.py ./mcp_server/

USER 1000

EXPOSE 8080

# No --port: Cloud Run injects PORT and agent/server.py reads it.
CMD ["python", "-m", "agent.server"]
