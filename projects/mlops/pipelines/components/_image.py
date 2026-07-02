"""
_image.py — Docker image URI for the pipeline components.

Defaults to the Vertex prebuilt XGBoost training image; components then install
their extra dependencies at runtime via ``packages_to_install``. Set
``TRAINING_IMAGE_URI`` (at compile/submit time) to a custom pre-built image
(see ``pipelines/Dockerfile``) to get pinned, reproducible dependencies baked
in and skip the per-step installs, e.g.::

    export TRAINING_IMAGE_URI=us-east1-docker.pkg.dev/$PROJECT_ID/readmission/training:latest
"""

import os

_DEFAULT_BASE_IMAGE = "us-docker.pkg.dev/vertex-ai/training/xgboost-cpu.2-1:latest"

TRAINING_IMAGE = os.environ.get("TRAINING_IMAGE_URI", _DEFAULT_BASE_IMAGE)