"""authorise_demo_cohort — derive the authorisation boundary from the served data.

`services/mcp/config.py` calls `readmission.demo_cohort` the authorisation
boundary: the site rejects an admission that is not in it before the agent is
called. A boundary is only a boundary if it is the same set as the data the
tools read. It was not. The table held 24 admissions, 20 of which were served,
while the corpus the tools and the index serve holds 89, so 69 admissions were
reachable by anything that reached the tools without passing the site's
allowlist — and the tables cannot leak real MIMIC data, which is why the
mismatch went unnoticed: the PHI argument held while the authorisation argument
did not.

The served corpus is authoritative (decision of 2026-09-18), so the boundary is
derived from it rather than selected beside it. Two selections of the same thing
drift; one derivation cannot.

    .venv/bin/python scripts/agent/authorise_demo_cohort.py [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from google.cloud import bigquery

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.mcp.config import (  # noqa: E402
    COHORT_TABLE_FQN,
    PROJECT,
    TABLE_FQN,
)

DEMO_MARKER = "hybrid"


def _ids(client: bigquery.Client, table: str) -> set[str]:
    rows = client.query(f"SELECT DISTINCT hadm_id FROM `{table}`")
    return {str(row["hadm_id"]) for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # The boundary authorises the demonstration corpus. Deriving it from
    # anything else — the real corpus above all — would authorise data this
    # environment is not permitted to serve.
    if DEMO_MARKER not in TABLE_FQN:
        raise SystemExit(
            f"refusing to derive the demo boundary from {TABLE_FQN}: it is not "
            f"the demonstration corpus (expected a table named with "
            f"{DEMO_MARKER!r}). Set FEATURE_TABLE to the demo table."
        )

    client = bigquery.Client(project=PROJECT)
    served = _ids(client, TABLE_FQN)
    if not served:
        raise SystemExit(f"{TABLE_FQN} holds no admissions; nothing to authorise")
    authorised = _ids(client, COHORT_TABLE_FQN)

    missing = served - authorised
    extra = authorised - served
    print(f"served     {TABLE_FQN}: {len(served)} admissions")
    print(f"authorised {COHORT_TABLE_FQN}: {len(authorised)} admissions")
    print(f"  served but not authorised: {len(missing)}")
    print(f"  authorised but not served: {len(extra)}")

    if not missing and not extra:
        print("the boundary already matches the served corpus; nothing to do")
        return 0
    if args.dry_run:
        print("dry-run: would rewrite the boundary to the served corpus")
        return 0

    # The same shape as the served table, taken from it, so the boundary cannot
    # describe admissions the tools would then answer differently about.
    client.query(
        f"CREATE OR REPLACE TABLE `{COHORT_TABLE_FQN}` AS "
        f"SELECT * FROM `{TABLE_FQN}`"
    ).result()

    after = _ids(client, COHORT_TABLE_FQN)
    if after != served:
        raise SystemExit(
            f"the boundary does not match the served corpus after the rewrite: "
            f"{len(after)} authorised, {len(served)} served, "
            f"{len(served ^ after)} differing"
        )
    print(f"rewrote {COHORT_TABLE_FQN}: {len(authorised)} -> {len(after)} "
          f"admissions, matching the served corpus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
