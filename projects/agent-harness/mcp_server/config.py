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
TABLE = f"{DATASET}.analytics_dataset_encoded"
TABLE_FQN = f"{PROJECT}.{TABLE}"
ENTITY_ID_COLUMN = "hadm_id"

# Feature Store serves the demo cohort only, not the full 352,699-row table.
# Syncing everything into Bigtable to answer queries about a few dozen patients
# is an hour of export and ongoing storage for ~0.01% of what it holds.
# Built by scripts/build_demo_cohort.py; ids outside it fall back to BigQuery.
COHORT_TABLE = f"{DATASET}.demo_cohort"
COHORT_TABLE_FQN = f"{PROJECT}.{COHORT_TABLE}"

# Feature Store online serving. The teardown script matches online stores by the
# "readmission" prefix, so these names must keep it.
ONLINE_STORE_ID = os.environ.get("ONLINE_STORE_ID", "readmission_online")
FEATURE_VIEW_ID = os.environ.get("FEATURE_VIEW_ID", "readmission_cohort")

# "bigquery" (default: dev, CI, tests) | "feature_store" (live demos)
FEATURE_SOURCE = os.environ.get("FEATURE_SOURCE", "bigquery")

# Gemini. Verified reachable from us-east1 on 2026-07-30, so the agent stays
# co-located with the prediction endpoint; "global" is the fallback, not the default.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# 2.5 models are thinking models and this budget covers thinking AND the answer.
# Too small and the whole budget is spent on thoughts: the call returns 200 with
# empty text and finish_reason=MAX_TOKENS, no exception. 16 tokens was enough to
# reproduce that. Keep this generous.
GEMINI_MAX_OUTPUT_TOKENS = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "2048"))

API_ENDPOINT = f"{LOCATION}-aiplatform.googleapis.com"
