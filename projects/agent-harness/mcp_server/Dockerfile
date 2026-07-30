# Build context is this directory:
#   gcloud builds submit --tag ... projects/agent-harness/mcp_server
FROM python:3.12-slim

# No gcc / libpq-dev: nothing here compiles. The website image carried them for
# months before anyone checked, at ~150 MB of nothing.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# The package must sit at /app/mcp_server so `python -m mcp_server.server`
# resolves — the source directory name (agent-harness) contains a hyphen and is
# not importable, which is why the build context is the package itself.
COPY . ./mcp_server/

# Cloud Run sets PORT; the server reads it. Non-root by default.
USER 1000
EXPOSE 8080

# JSON form, so the process is PID 1 and receives SIGTERM directly. With the
# shell form, sh swallows the signal and Cloud Run waits out the full grace
# period on every revision swap.
CMD ["python", "-m", "mcp_server.server", "--transport", "http"]
