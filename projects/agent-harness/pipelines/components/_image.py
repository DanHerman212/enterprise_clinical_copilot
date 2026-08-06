"""_image.py — Docker image URI for the RAG ingest pipeline components.

Mirror of projects/mlops/pipelines/components/_image.py. The `component()`
helper is an identity decorator at runtime (inside the KFP executor, where
kfp.dsl.component is unavailable and the run_* helper is imported directly).
"""

import os

from kfp import dsl

_DEFAULT_BASE_IMAGE = "python:3.12-slim"

RAG_IMAGE = os.environ.get("RAG_IMAGE_URI", _DEFAULT_BASE_IMAGE)


def component(*args, **kwargs):
    if os.environ.get("_KFP_RUNTIME", "false") == "true":
        def _identity(func):
            return func

        return _identity
    return dsl.component(*args, **kwargs)
