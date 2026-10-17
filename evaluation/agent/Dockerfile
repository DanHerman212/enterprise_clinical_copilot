# The evaluation harness, as a container.
#
# Build from the REPOSITORY ROOT, not from this directory: the harness imports
# `services.agent.contracts`, `services.agent.questions` and `services.mcp.config`
# so that a case's wire names and a question's wording come from the same source the
# deployment uses. Copying only `evaluation/` would mean re-typing them here.
#
#   gcloud builds submit --config evaluation/agent/cloudbuild.yaml \
#       --substitutions=_TAG=$(git rev-parse --short HEAD)
#
# What is deliberately NOT installed: the agent's own dependencies (LangChain, the
# MCP SDK, vertexai). The harness talks to the deployment over HTTP and to BigQuery
# for the census. A harness that shares the agent's process is one that can pass by
# agreeing with itself.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# Dependencies first, so a change to the harness does not re-resolve them.
COPY evaluation/agent/requirements.txt /app/evaluation/agent/requirements.txt
RUN pip install --no-cache-dir -r /app/evaluation/agent/requirements.txt

COPY services/ /app/services/
COPY evaluation/agent/ /app/evaluation/agent/

WORKDIR /app/evaluation/agent

# PHASE selects the job; there is no default, because a job that silently picks a
# phase is a job that can report on the wrong one.
ENTRYPOINT ["python", "entrypoint.py"]
