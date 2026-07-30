"""
setup_feature_store.py — Provision the Feature Store online store and feature view.

Creates a Bigtable-backed online store over the same encoded dataset BigQuery
reads, keyed by hadm_id, then runs an initial sync. Idempotent: existing
resources are reused rather than recreated.

    .venv/bin/python projects/agent-harness/scripts/setup_feature_store.py
    .venv/bin/python projects/agent-harness/scripts/setup_feature_store.py --sync-only

Verify afterwards:
    FEATURE_SOURCE=feature_store .venv/bin/python -m pytest \\
        projects/agent-harness/tests/test_feature_parity.py -v

COST: a provisioned online store bills continuously, like the endpoint. Remove
it with `scripts/teardown.py` when you are done demoing.
"""

import argparse
import sys
import time
from pathlib import Path

from google.api_core.exceptions import AlreadyExists, NotFound
from google.cloud import aiplatform_v1

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp_server.config import (  # noqa: E402
    API_ENDPOINT,
    ENTITY_ID_COLUMN,
    FEATURE_VIEW_ID,
    LOCATION,
    ONLINE_STORE_ID,
    PROJECT,
    TABLE_FQN,
)

PARENT = f"projects/{PROJECT}/locations/{LOCATION}"
STORE_PATH = f"{PARENT}/featureOnlineStores/{ONLINE_STORE_ID}"
VIEW_PATH = f"{STORE_PATH}/featureViews/{FEATURE_VIEW_ID}"

SYNC_POLL_SECONDS = 20
SYNC_TIMEOUT_SECONDS = 45 * 60


def _admin() -> aiplatform_v1.FeatureOnlineStoreAdminServiceClient:
    return aiplatform_v1.FeatureOnlineStoreAdminServiceClient(
        client_options={"api_endpoint": API_ENDPOINT}
    )


def ensure_online_store(admin) -> None:
    try:
        admin.get_feature_online_store(name=STORE_PATH)
        print(f"  Online store exists: {ONLINE_STORE_ID}")
        return
    except NotFound:
        pass

    print(f"  Creating online store {ONLINE_STORE_ID} (a few minutes)…")
    store = aiplatform_v1.FeatureOnlineStore(
        bigtable=aiplatform_v1.FeatureOnlineStore.Bigtable(
            auto_scaling=aiplatform_v1.FeatureOnlineStore.Bigtable.AutoScaling(
                # Smallest footprint that serves. This is the billable part.
                min_node_count=1,
                max_node_count=1,
                cpu_utilization_target=50,
            )
        )
    )
    try:
        admin.create_feature_online_store(
            parent=PARENT,
            feature_online_store_id=ONLINE_STORE_ID,
            feature_online_store=store,
        ).result()
        print("  Created")
    except AlreadyExists:
        print("  Already exists (race) — reusing")


def ensure_feature_view(admin) -> None:
    try:
        admin.get_feature_view(name=VIEW_PATH)
        print(f"  Feature view exists: {FEATURE_VIEW_ID}")
        return
    except NotFound:
        pass

    print(f"  Creating feature view {FEATURE_VIEW_ID} over bq://{TABLE_FQN}…")
    view = aiplatform_v1.FeatureView(
        big_query_source=aiplatform_v1.FeatureView.BigQuerySource(
            uri=f"bq://{TABLE_FQN}",
            entity_id_columns=[ENTITY_ID_COLUMN],
        ),
        # No cron: the dataset is static, so syncs are run on demand from here
        # rather than on a schedule that would bill for nothing.
        sync_config=aiplatform_v1.FeatureView.SyncConfig(cron=""),
    )
    try:
        admin.create_feature_view(
            parent=STORE_PATH, feature_view_id=FEATURE_VIEW_ID, feature_view=view
        ).result()
        print("  Created")
    except AlreadyExists:
        print("  Already exists (race) — reusing")


def run_sync(admin) -> None:
    print("  Starting sync…")
    response = admin.sync_feature_view(feature_view=VIEW_PATH)
    sync_name = response.feature_view_sync
    print(f"  Sync: {sync_name.rsplit('/', 1)[-1]}")

    deadline = time.time() + SYNC_TIMEOUT_SECONDS
    while time.time() < deadline:
        sync = admin.get_feature_view_sync(name=sync_name)
        # end_time is populated only once the sync finishes.
        if sync.run_time.end_time:
            rows = sync.sync_summary.row_synced
            print(f"  Sync complete — {rows} rows synced")
            return
        print("    still syncing…")
        time.sleep(SYNC_POLL_SECONDS)

    sys.exit("  Sync did not finish before the timeout; check the Vertex console.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--sync-only", action="store_true",
                        help="skip creation, just re-sync an existing view")
    args = parser.parse_args()

    print(f"Feature Store setup — {PROJECT} / {LOCATION}")
    admin = _admin()

    if not args.sync_only:
        ensure_online_store(admin)
        ensure_feature_view(admin)
    run_sync(admin)

    print("\nDone. Verify parity with:")
    print("  FEATURE_SOURCE=feature_store .venv/bin/python -m pytest "
          "projects/agent-harness/tests/test_feature_parity.py -v")
    print("\nThis store bills continuously. Remove it with scripts/teardown.py")


if __name__ == "__main__":
    main()
