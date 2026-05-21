# Phase D — TFDV runner image.
#
# TFDV publishes wheels for linux x86_64 + Python 3.9–3.11 only. The
# project's primary venv is macOS arm64 + Python 3.12, so we run TFDV inside
# this container as a batch worker. The notebook (`data_validation.ipynb`)
# on the host invokes this image via `scripts/run_validation.sh`, mounting
# the project's `artifacts/validation/` directory for outputs and ADC
# credentials for BigQuery access. The notebook then reads back the
# generated Facets HTML + anomalies report — no `import tfdv` needed on
# the host.

FROM --platform=linux/amd64 python:3.11-slim

# System deps for pyarrow + grpc + protobuf compiled wheels.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
 && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip wheel \
 && pip install -r /app/requirements.txt

COPY run_validation.py /app/run_validation.py

# Artifacts live on a bind mount; default matches scripts/run_validation.sh.
ENV OUT_DIR=/artifacts

ENTRYPOINT ["python", "/app/run_validation.py"]
