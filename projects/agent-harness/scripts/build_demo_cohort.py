"""SUPERSEDED by seed_demo_cohort.py — do not run both.

Kept for the reasoning below, which still explains why a cohort table exists at
all. Selection here balances on the *label*, which produced a cohort with no
patient just below the 0.12 threshold; §14 replaced it with risk-band selection.

Build the demo cohort table that backs Feature Store online serving.

Why this exists: syncing all 352,699 admissions into Bigtable to serve a demo
over a few dozen patients is a full-table export, an hour of wall clock, and
ongoing storage — to serve ~0.01% of what it holds. The cohort table is the
same data, scoped to what the demo actually queries.

Selection rules:
  - test split only. Demo patients must never be rows the model trained on;
    a memorised prediction is not a demonstration of anything.
  - balanced on readmission_30d, so the demo can show both a high-risk and a
    low-risk case without hunting.
  - deterministic. Ordering by FARM_FINGERPRINT(hadm_id) is stable across runs
    without biasing toward low ids the way ORDER BY hadm_id would.
  - fixture patients are pinned in regardless, so Tier 1 keeps working.

Provisional: §14 owns final cohort selection. This is a defensible default,
not a decision.
"""

import argparse
import json
import sys
from pathlib import Path

from google.cloud import bigquery

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = REPO_ROOT / "projects" / "agent-harness" / "tests" / "fixtures" / "expected.json"

PROJECT = "trim-icon-498815-a0"
SOURCE_TABLE = f"{PROJECT}.readmission.analytics_dataset_encoded"
COHORT_TABLE = f"{PROJECT}.readmission.demo_cohort"


def _pinned_ids() -> list[int]:
    """hadm_ids the Tier 1 fixture depends on."""
    if not FIXTURE_PATH.exists():
        return []
    fixture = json.loads(FIXTURE_PATH.read_text())
    return [int(k) for k in fixture.get("patients", {})]


def build(per_class: int) -> None:
    bq = bigquery.Client(project=PROJECT)
    pinned = _pinned_ids()
    print(f"Pinned from fixture: {pinned or 'none'}")

    sql = f"""
    CREATE OR REPLACE TABLE `{COHORT_TABLE}` AS
    WITH ranked AS (
      SELECT *, ROW_NUMBER() OVER (
                  PARTITION BY readmission_30d
                  ORDER BY FARM_FINGERPRINT(CAST(hadm_id AS STRING))
                ) AS rn
      FROM `{SOURCE_TABLE}`
      WHERE split_name = 'test'
    )
    SELECT * EXCEPT (rn) FROM ranked WHERE rn <= @per_class
    UNION DISTINCT
    SELECT * FROM `{SOURCE_TABLE}` WHERE hadm_id IN UNNEST(@pinned)
    """
    job = bq.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("per_class", "INT64", per_class),
        bigquery.ArrayQueryParameter("pinned", "INT64", pinned),
    ]))
    job.result()

    table = bq.get_table(COHORT_TABLE)
    print(f"Built {COHORT_TABLE}: {table.num_rows} rows, "
          f"{table.num_bytes / 1024:.0f} KB")

    rows = list(bq.query(f"""
        SELECT readmission_30d, COUNT(*) AS n
        FROM `{COHORT_TABLE}` GROUP BY 1 ORDER BY 1
    """).result())
    for r in rows:
        print(f"  readmission_30d={r.readmission_30d}: {r.n}")

    # Fail loudly rather than syncing a cohort the fixture cannot use.
    if pinned:
        found = {r.hadm_id for r in bq.query(
            f"SELECT hadm_id FROM `{COHORT_TABLE}` WHERE hadm_id IN UNNEST(@p)",
            job_config=bigquery.QueryJobConfig(query_parameters=[
                bigquery.ArrayQueryParameter("p", "INT64", pinned)]),
        ).result()}
        missing = set(pinned) - found
        if missing:
            sys.exit(f"Fixture patients missing from cohort: {sorted(missing)}")
        print(f"  all {len(pinned)} fixture patient(s) present")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-class", type=int, default=20,
                        help="admissions per readmission_30d class (default 20)")
    build(parser.parse_args().per_class)


if __name__ == "__main__":
    main()
