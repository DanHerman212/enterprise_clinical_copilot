"""Pytest configuration and shared fixtures for the MLOps pipeline tests.

Puts the mlops project root (``projects/mlops``) on ``sys.path`` so tests can
import the ``pipelines`` and ``src`` packages, and provides small synthetic
fixtures that mirror the *raw* BigQuery output (object-dtype categoricals,
float labs with NaN) — deliberately including the edge cases that caused the
prior pipeline's leakage bugs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Import path + hermetic config
# ---------------------------------------------------------------------------

_MLOPS_ROOT = Path(__file__).resolve().parents[2]  # projects/mlops
if str(_MLOPS_ROOT) not in sys.path:
    sys.path.insert(0, str(_MLOPS_ROOT))

# src.config resolves a GCP project id at import time; give it a value so the
# unit tests never depend on a local .env or real credentials.
os.environ.setdefault("PROJECT_ID", "unit-test-project")

SAMPLE_DATA_PATH = _MLOPS_ROOT / "pipelines" / "sample_data.parquet"

LABEL_COL = "readmission_30d"


# ---------------------------------------------------------------------------
# Synthetic raw splits (mirror BigQuery: object categoricals, float labs+NaN)
# ---------------------------------------------------------------------------

@pytest.fixture
def selected_features() -> list[str]:
    return ["age", "discharge_location", "insurance", "gender", "glucose_last"]


@pytest.fixture
def cat_features() -> list[str]:
    return ["discharge_location", "insurance", "gender"]


@pytest.fixture
def train_df() -> pd.DataFrame:
    # insurance train mode = "Medicare" (4 of 5 non-null).
    # discharge_location has a NaN -> constant_unknown -> "Unknown".
    # glucose_last has a NaN -> no_missing -> stays NaN.
    return pd.DataFrame(
        {
            "age": [50, 60, 70, 55, 65, 45],
            "gender": ["M", "F", "M", "F", "M", "F"],
            "insurance": ["Medicare", "Medicare", "Medicaid", "Medicare", None, "Medicare"],
            "discharge_location": ["HOME", "SNF", None, "HOME", "HOME", "SNF"],
            "glucose_last": [100.0, 110.0, np.nan, 95.0, 105.0, 99.0],
            LABEL_COL: [0, 1, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def val_df() -> pd.DataFrame:
    # insurance mode within val would be "Medicaid"; the NaN must be filled
    # with the TRAIN mode ("Medicare") to prove there is no val leakage.
    return pd.DataFrame(
        {
            "age": [52, 58, 63, 47],
            "gender": ["M", "F", "M", "F"],
            "insurance": ["Medicaid", "Medicaid", None, "Medicaid"],
            "discharge_location": ["HOME", None, "SNF", "HOME"],
            "glucose_last": [100.0, np.nan, 90.0, 88.0],
            LABEL_COL: [0, 1, 0, 1],
        }
    )


@pytest.fixture
def test_df() -> pd.DataFrame:
    # gender "U" is unseen in train -> must encode to NaN (no leakage), not a
    # brand-new category. glucose_last NaN must pass through.
    return pd.DataFrame(
        {
            "age": [51, 59],
            "gender": ["U", "F"],
            "insurance": ["Medicare", "Medicaid"],
            "discharge_location": ["HOME", "SNF"],
            "glucose_last": [np.nan, 92.0],
            LABEL_COL: [0, 1],
        }
    )
