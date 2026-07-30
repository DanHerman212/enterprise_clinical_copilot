"""Shared constants for the MCP server and its feature sources."""

import os

PROJECT = "trim-icon-498815-a0"
LOCATION = "us-east1"

# Vertex serving
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "readmission-endpoint")
FINAL_MODEL_PREFIX = "readmission-final-"
BUNDLE_URI_OVERRIDE = os.environ.get("BUNDLE_URI")

# BigQuery feature source
DATASET = "readmission"
TABLE = f"{DATASET}.analytics_dataset_encoded"
TABLE_FQN = f"{PROJECT}.{TABLE}"
ENTITY_ID_COLUMN = "hadm_id"

# Feature Store online serving. The teardown script matches online stores by the
# "readmission" prefix, so these names must keep it.
ONLINE_STORE_ID = os.environ.get("ONLINE_STORE_ID", "readmission_online")
FEATURE_VIEW_ID = os.environ.get("FEATURE_VIEW_ID", "readmission_features")

# "bigquery" (default: dev, CI, tests) | "feature_store" (live demos)
FEATURE_SOURCE = os.environ.get("FEATURE_SOURCE", "bigquery")

API_ENDPOINT = f"{LOCATION}-aiplatform.googleapis.com"
