"""
Shared configuration for the Enterprise Clinical Copilot MLOps project.

Resolves project identity from env var or repo-root .env — no
personal project IDs are committed to git.
"""

from __future__ import annotations

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Project identity
# ---------------------------------------------------------------------------

def _resolve_project_id() -> str:
    """Resolve GCP project ID from env var or repo-root .env file."""
    if os.environ.get("PROJECT_ID"):
        return os.environ["PROJECT_ID"]

    for directory in [Path.cwd(), *Path.cwd().resolve().parents]:
        env_file = directory / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                stripped = line.strip()
                if stripped.startswith("PROJECT_ID="):
                    return stripped.split("=", 1)[1].strip()
            break

    raise RuntimeError(
        "PROJECT_ID is not set. Export it, or copy .env.example -> .env "
        "at the repo root and set PROJECT_ID to your GCP project."
    )


def get_project_id() -> str:
    """Resolve the GCP project id lazily — only when actually needed.

    Prefer this over importing ``PROJECT_ID`` so that merely importing this
    module (e.g. transitively from the serving predictor, which never touches
    BigQuery) does not require a project id to be configured.
    """
    return _resolve_project_id()


# ---------------------------------------------------------------------------
# BigQuery — analytics dataset (built by Dataform)
# ---------------------------------------------------------------------------

BQ_DATASET = "readmission"
BQ_TABLE = "analytics_dataset"


def _full_table_ref() -> str:
    return f"{get_project_id()}.{BQ_DATASET}.{BQ_TABLE}"


def __getattr__(name: str):
    """Resolve PROJECT_ID / FULL_TABLE_REF lazily (PEP 562).

    They remain importable (``from src.config import PROJECT_ID``) but are only
    resolved on first access, so importing this module never fails when no
    project id is configured (e.g. inside the serving container).
    """
    if name == "PROJECT_ID":
        return _resolve_project_id()
    if name == "FULL_TABLE_REF":
        return _full_table_ref()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# ---------------------------------------------------------------------------
# Splits (mirrors definitions/staging/cohort_split.sqlx)
# ---------------------------------------------------------------------------

SPLIT_COLUMN = "split_name"

SPLITS = {
    "train":      "train",
    "validation": "validation",
    "test":       "test",
}

ALL_SPLITS = ("train", "validation", "test", "prod_test", "demo")

# ---------------------------------------------------------------------------
# Label
# ---------------------------------------------------------------------------

LABEL_COLUMN = "readmission_30d"

# ---------------------------------------------------------------------------
# Columns to exclude from feature analysis
# ---------------------------------------------------------------------------

ID_COLUMNS = [
    "subject_id",
    "hadm_id",
    "admittime",
    "dischtime",
]

META_COLUMNS = [
    "split_bucket",
    "split_name",
]

# All columns that are not features
NON_FEATURE_COLUMNS = ID_COLUMNS + META_COLUMNS + [LABEL_COLUMN]

# ---------------------------------------------------------------------------
# Feature families (mirrors the groupings in missingness_analysis.ipynb)
# ---------------------------------------------------------------------------

FEATURE_FAMILIES: dict[str, list[str]] = {
    "demographics": [
        "age", "gender", "marital_status", "language", "race",
        "admission_type", "insurance", "discharge_location",
    ],
    "utilization": [
        "prior_admission_count", "prior_inpatient_days",
        "recent_ed_visits", "index_los_days",
    ],
    "codes": [
        "diagnosis_count", "procedure_count", "has_procedure",
    ],
    "medications": [
        "medication_count", "medication_order_count",
        "on_anticoagulant", "on_insulin",
    ],
    "labs": [
        "rbc_last", "rbc_max", "rbc_min", "rbc_delta", "rbc_measured",
        "rdw_last", "rdw_max", "rdw_min", "rdw_delta", "rdw_measured",
        "glucose_last", "glucose_max", "glucose_min", "glucose_delta",
        "glucose_measured",
        "monocytes_last", "monocytes_max", "monocytes_min",
        "monocytes_delta", "monocytes_measured",
        "hemoglobin_last", "hemoglobin_max", "hemoglobin_min",
        "hemoglobin_delta", "hemoglobin_measured",
        "sodium_last", "sodium_max", "sodium_min",
        "sodium_delta", "sodium_measured",
    ],
    "oncology": [
        "oncology_flag",
    ],
}

# ---------------------------------------------------------------------------
# Numeric vs categorical feature classification
# ---------------------------------------------------------------------------

NUMERIC_FEATURES: list[str] = [
    "age",
    "prior_admission_count", "prior_inpatient_days",
    "recent_ed_visits", "index_los_days",
    "diagnosis_count", "procedure_count",
    "medication_count", "medication_order_count",
    "rbc_last", "rbc_max", "rbc_min", "rbc_delta",
    "rdw_last", "rdw_max", "rdw_min", "rdw_delta",
    "glucose_last", "glucose_max", "glucose_min", "glucose_delta",
    "monocytes_last", "monocytes_max", "monocytes_min", "monocytes_delta",
    "hemoglobin_last", "hemoglobin_max", "hemoglobin_min", "hemoglobin_delta",
    "sodium_last", "sodium_max", "sodium_min", "sodium_delta",
]

CATEGORICAL_FEATURES: list[str] = [
    "gender", "marital_status", "language", "race",
    "admission_type", "insurance", "discharge_location",
    "has_procedure", "on_anticoagulant", "on_insulin",
    "rbc_measured", "rdw_measured", "glucose_measured", "monocytes_measured",
    "hemoglobin_measured", "sodium_measured",
    "oncology_flag",
]

# ---------------------------------------------------------------------------
# Evidently AI thresholds
# ---------------------------------------------------------------------------

EVIDENTLY_CONFIG: dict = {
    # Data Drift: per-column drift score threshold (Wasserstein / JS distance)
    "drift_threshold": 0.1,
    # Data Drift: maximum share of drifted columns before the gate fails
    "max_drifted_share": 0.2,
    # Data Quality: tolerance for missingness increase (absolute percentage)
    "missingness_tolerance_pct": 5.0,
}

# ---------------------------------------------------------------------------
# Artifact paths (relative to repo root)
# ---------------------------------------------------------------------------

ARTIFACTS_DIR = Path("projects/mlops/artifacts")
VALIDATION_DIR = ARTIFACTS_DIR / "validation"
BASELINE_DIR = ARTIFACTS_DIR / "baseline"

# ---------------------------------------------------------------------------
# Missingness policy (output of missingness_analysis.ipynb)
# ---------------------------------------------------------------------------

MISSINGNESS_POLICY_CSV = Path(__file__).resolve().parent.parent / "artifacts" / "missingness_policy.csv"
