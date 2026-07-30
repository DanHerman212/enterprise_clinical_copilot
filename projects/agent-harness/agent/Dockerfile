# Build from the harness root, not from agent/: the agent imports
# mcp_server.config for the Gemini and project settings, so both packages must
# be in the context.
#
#   gcloud builds submit --config projects/agent-harness/cloudbuild.agent.yaml \
#     --substitutions _IMAGE=...  projects/agent-harness

FROM python:3.12-slim

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
