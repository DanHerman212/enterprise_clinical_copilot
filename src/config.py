"""Project-wide configuration constants.

Single source of truth for project ID, regions, BigQuery table FQNs,
GCS paths, and dataset versioning. Imported by `src.tracking` and by
every notebook / pipeline step that talks to GCP. Keep this module
free of side-effects and free of GCP client construction.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass

# --- GCP project / locations ------------------------------------------
PROJECT_ID: str = "enterprise-clinical-copilot"

# BigQuery dataset is multi-region US (set at dataset creation time).
BQ_LOCATION: str = "US"

# Vertex AI / GCS region. Must match the bucket location and the
# region the bootstrap script provisions resources in.
VERTEX_REGION: str = "us-east1"

# --- GCS layout (bucket created by scripts/bootstrap_project.sh) -------
GCS_BUCKET: str = f"{PROJECT_ID}-mlops"
GCS_BUCKET_URI: str = f"gs://{GCS_BUCKET}"
GCS_STAGING_URI: str = f"{GCS_BUCKET_URI}/staging"
GCS_ARTIFACTS_URI: str = f"{GCS_BUCKET_URI}/artifacts"
GCS_TFDV_URI: str = f"{GCS_BUCKET_URI}/tfdv"
GCS_PIPELINES_URI: str = f"{GCS_BUCKET_URI}/pipelines"

# --- BigQuery dataset + table FQNs ------------------------------------
BQ_DATASET: str = "readmission"
BQ_DATASET_FQN: str = f"{PROJECT_ID}.{BQ_DATASET}"

COHORT_TABLE: str         = f"{BQ_DATASET_FQN}.cohort_admissions"
LABELED_TABLE: str        = f"{BQ_DATASET_FQN}.cohort_labeled"
LACE_TABLE: str           = f"{BQ_DATASET_FQN}.cohort_lace"
HOSPITAL_TABLE: str       = f"{BQ_DATASET_FQN}.cohort_hospital"
BASELINES_TABLE: str      = f"{BQ_DATASET_FQN}.cohort_baselines"
SPLITS_TABLE: str         = f"{BQ_DATASET_FQN}.cohort_splits"
# Created in Phase C:
FEATURES_TABLE: str       = f"{BQ_DATASET_FQN}.cohort_features_v2"

# Feature-set version. Logged as a run param by every later model run
# so each experiment row is traceable to a specific feature contract.
# Bump whenever the schema of `cohort_features` changes.
# v1: 74 raw features (Phase C output).
# v2: 68 features after Phase E selection + per-feature missingness policy
#     (see docs/feature_shortlist_v2.md). Materialized as a BQ view that
#     projects from cohort_features.
FEATURES_VERSION: str = "v2"

# --- MIMIC-IV source datasets (PhysioNet public project) --------------
MIMIC_PROJECT: str  = "physionet-data"
MIMIC_HOSP: str     = f"{MIMIC_PROJECT}.mimiciv_3_1_hosp"
MIMIC_ICU: str      = f"{MIMIC_PROJECT}.mimiciv_3_1_icu"
MIMIC_DERIVED: str  = f"{MIMIC_PROJECT}.mimiciv_3_1_derived"
MIMIC_ED: str       = f"{MIMIC_PROJECT}.mimiciv_ed"
MIMIC_NOTE: str     = f"{MIMIC_PROJECT}.mimiciv_note"

# --- Data versioning --------------------------------------------------
MIMIC_VERSION: str = "v3_1"

# Eight `admission_type` values that mark an admission as "unplanned".
# Used by both the 30-day label and LACE-A. Single canonical tuple.
ACUTE_ADMISSION_TYPES: tuple[str, ...] = (
    "URGENT",
    "EMERGENCY",
    "EW EMER.",
    "DIRECT EMER.",
    "DIRECT OBSERVATION",
    "EU OBSERVATION",
    "OBSERVATION ADMIT",
    "AMBULATORY OBSERVATION",
)

# --- Experiment tracking ----------------------------------------------
EXPERIMENT_NAME: str = "readmission-30d"

# --- Service account (for Vertex jobs / pipelines) --------------------
SERVICE_ACCOUNT: str = (
    f"clinical-copilot-mlops@{PROJECT_ID}.iam.gserviceaccount.com"
)


@dataclass(frozen=True)
class SplitContract:
    """Hash-bucket boundaries for the four-way patient-level split.

    bucket = ABS(MOD(FARM_FINGERPRINT(CAST(subject_id AS STRING)), 1000))
    """
    demo_max:  int = 5     # bucket  <   5  -> demo  (~0.5%)
    train_max: int = 705   # bucket  < 705  -> train ( 70%)
    val_max:   int = 855   # bucket  < 855  -> val   ( 15%)
    # bucket < 1000 -> test (~14.5%)
    modulus:   int = 1000

    @property
    def strategy(self) -> str:
        return f"FARM_FINGERPRINT(subject_id) % {self.modulus}"


SPLIT_CONTRACT: SplitContract = SplitContract()


def git_sha(short: bool = True) -> str:
    """Best-effort current git SHA for run provenance.

    Returns "unknown" if git is unavailable or the working dir is not
    a repo. Never raises — provenance metadata must not break a run.
    """
    args = ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"]
    if short:
        args = ["git", "rev-parse", "--short", "HEAD"]
    else:
        args = ["git", "rev-parse", "HEAD"]
    try:
        out = subprocess.run(
            args, capture_output=True, text=True, check=True, timeout=2
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"
