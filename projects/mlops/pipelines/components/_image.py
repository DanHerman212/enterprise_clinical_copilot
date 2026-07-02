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

from kfp import dsl

_DEFAULT_BASE_IMAGE = "us-docker.pkg.dev/vertex-ai/training/xgboost-cpu.2-1:latest"

TRAINING_IMAGE = os.environ.get("TRAINING_IMAGE_URI", _DEFAULT_BASE_IMAGE)


def component(*args, **kwargs):
    """``dsl.component`` at authoring time, a no-op decorator at execution time.

    The thin ``@component`` wrappers import their ``run_*`` helpers from their
    own module at runtime (so shared code is reachable inside the Vertex
    executor). Importing that module re-evaluates the decorator, but KFP omits
    ``kfp.dsl.component`` when ``_KFP_RUNTIME=true`` (inside the executor), so a
    real ``@dsl.component`` there raises ``AttributeError``. Returning an
    identity decorator at runtime lets the import succeed; the compiled
    component object is only ever needed at authoring/compile time.
    """
    if os.environ.get("_KFP_RUNTIME", "false") == "true":
        def _identity(func):
            return func

        return _identity
    return dsl.component(*args, **kwargs)