# Build context is the harness root, so both the package and the pure-stdlib
# rag/ section parser (imported by rag_search.py) are available:
#   gcloud builds submit --tag ... projects/agent-harness
# Pinned by digest: reproducible rebuilds (resolves linux/amd64 on Cloud Build,
# arm64 locally).
FROM python:3.12-slim@sha256:e5c9fa26ffb76e11e0f054f30dc2523a2f9693f0c36c0cf1e39b27e152d899fc

# No gcc / libpq-dev: nothing here compiles. The website image carried them for
# months before anyone checked, at ~150 MB of nothing.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY mcp_server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# The package must sit at /app/mcp_server so `python -m mcp_server.server`
# resolves — the source directory name (agent-harness) contains a hyphen and is
# not importable, which is why the context is the harness root and the package
# is copied to a clean /app/mcp_server path.
COPY mcp_server/ ./mcp_server/
# rag/ is pure stdlib (re, dataclasses only) and rag_search.py anchors section
# queries to the section parser it defines, so it ships alongside the server.
COPY rag/ ./rag/

# Cloud Run sets PORT; the server reads it. Non-root by default.
USER 1000
EXPOSE 8080

# JSON form, so the process is PID 1 and receives SIGTERM directly. With the
# shell form, sh swallows the signal and Cloud Run waits out the full grace
# period on every revision swap.
CMD ["python", "-m", "mcp_server.server", "--transport", "http"]
