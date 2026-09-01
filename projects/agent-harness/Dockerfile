# Build context is projects/agent-harness/ (not mcp_server/) so requirements.txt
# sits beside the package and the layer cache works: dependencies only reinstall
# when requirements.txt changes, not on every code edit.
# Pinned by digest: reproducible rebuilds (resolves linux/amd64 on Cloud Build,
# arm64 locally).
FROM python:3.12-slim@sha256:e5c9fa26ffb76e11e0f054f30dc2523a2f9693f0c36c0cf1e39b27e152d899fc

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# No gcc / build-essential: every dependency here ships wheels.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mcp_server/ ./mcp_server/

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8080

# JSON form, so the process runs as PID 1 and receives SIGTERM directly rather
# than being wrapped in a shell that swallows it. Port comes from Cloud Run's
# PORT env var, which server.py reads.
CMD ["python", "-m", "mcp_server.server", "--transport", "http"]
