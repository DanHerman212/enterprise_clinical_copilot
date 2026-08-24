"""Shared constants for the MCP server and its feature sources."""

import os

# Overridable so the container can be pointed at another project/region without
# a rebuild. The defaults are the real ones, so local runs need no environment.
PROJECT = os.environ.get("PROJECT_ID", "trim-icon-498815-a0")
LOCATION = os.environ.get("LOCATION", "us-east1")

# Vertex serving
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "readmission-endpoint")
FINAL_MODEL_PREFIX = "readmission-final-"
BUNDLE_URI_OVERRIDE = os.environ.get("BUNDLE_URI")

# BigQuery feature source
DATASET = "readmission"
# Defaults to the HYBRID features table — the eval/demo cohort is the hybrid
# admissions (90000001+), whose feature rows live in readmission.hybrid_features.
# The real MIMIC-derived analytics_dataset_encoded table is out of scope for
# the demo and never carries the synthetic/hybrid admissions.
TABLE = os.environ.get("FEATURE_TABLE", f"{DATASET}.hybrid_features")
TABLE_FQN = f"{PROJECT}.{TABLE}"
ENTITY_ID_COLUMN = "hadm_id"

# The hand-picked demo cohort. Built by scripts/build_demo_cohort.py.
COHORT_TABLE = f"{DATASET}.demo_cohort"
COHORT_TABLE_FQN = f"{PROJECT}.{COHORT_TABLE}"

# RAG / Vector Search serving.
# The index endpoint is resolved by display name (like ENDPOINT_NAME) so the
# deployed id never needs to be hardcoded, and so teardown/stand-up just works.
INDEX_ENDPOINT_NAME = os.environ.get("INDEX_ENDPOINT_NAME", "readmission-rag-index")
DEPLOYED_INDEX_ID = os.environ.get("DEPLOYED_INDEX_ID", "rag_tree_ah")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "768"))
RESTRICT_NAMESPACE = "hadm_id"
# The discharge notes table (note_id -> hadm_id mapping, text by note_id).
# Defaults to the HYBRID notes table — the deployed RAG index is built from
# these MT-* notes, so serving must read passage text from the same place.
# The real MIMIC-derived table is out of scope for the demo and must never
# resolve passage text from real patient notes.
DISCHARGE_TABLE = os.environ.get(
    "DISCHARGE_TABLE", f"{PROJECT}.readmission.hybrid_notes"
)
DEFAULT_TOP_K = int(os.environ.get("RAG_TOP_K", "5"))

# Gemini. Verified reachable from us-east1 on 2026-07-30, so the agent stays
# co-located with the prediction endpoint; "global" is the fallback, not the default.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# 2.5 models are thinking models and this budget covers thinking AND the answer.
# Too small and the whole budget is spent on thoughts: the call returns 200 with
# empty text and finish_reason=MAX_TOKENS, no exception. 16 tokens was enough to
# reproduce that. Keep this generous.
GEMINI_MAX_OUTPUT_TOKENS = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "2048"))

API_ENDPOINT = f"{LOCATION}-aiplatform.googleapis.com"
