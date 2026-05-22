"""One-shot: re-run patched Family 5 (vitals) + assembly cells.

Pulls the source of those two cells from notebooks/feature_engineering.ipynb
and execs them in a namespace that mirrors the notebook's setup cell.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

bq = bigquery.Client(project=config.PROJECT_ID, location=config.BQ_LOCATION)

V_DEMO  = f"{config.BQ_DATASET_FQN}.v_features_demographics"
V_ADMIN = f"{config.BQ_DATASET_FQN}.v_features_index_admin"
V_COMO  = f"{config.BQ_DATASET_FQN}.v_features_comorbidities"
V_PRIOR = f"{config.BQ_DATASET_FQN}.v_features_prior_utilization"
V_VIT   = f"{config.BQ_DATASET_FQN}.v_features_vitals_24h"
V_LAB   = f"{config.BQ_DATASET_FQN}.v_features_labs_last"
V_SEV   = f"{config.BQ_DATASET_FQN}.v_features_severity"


def run(sql: str) -> bigquery.QueryJob:
    job = bq.query(sql)
    job.result()
    return job


def show(sql: str) -> pd.DataFrame:
    df = bq.query(sql).result().to_dataframe()
    print(df.to_string(index=False))
    return df


def _cell_src(predicate):
    nb = json.loads(Path("notebooks/feature_engineering.ipynb").read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if predicate(src):
            return src
    raise RuntimeError("cell not found")


ns = {
    "config": config, "bq": bq, "pd": pd, "run": run, "show": show,
    "V_DEMO": V_DEMO, "V_ADMIN": V_ADMIN, "V_COMO": V_COMO,
    "V_PRIOR": V_PRIOR, "V_VIT": V_VIT, "V_LAB": V_LAB, "V_SEV": V_SEV,
}

print(">>> re-running vitals view (Family 5) ...")
exec(_cell_src(lambda s: "vitals_sql" in s and "CREATE OR REPLACE VIEW" in s and "{V_VIT}" in s), ns)

print("\n>>> re-running assembly (cohort_features) ...")
exec(_cell_src(lambda s: "assemble_sql" in s and "CREATE OR REPLACE TABLE" in s), ns)

print("\n>>> verifying temp_* dtypes ...")
schema = bq.get_table(config.FEATURES_TABLE).schema
for f in schema:
    if f.name.startswith("temp_"):
        print(f"  {f.name}: {f.field_type}")
