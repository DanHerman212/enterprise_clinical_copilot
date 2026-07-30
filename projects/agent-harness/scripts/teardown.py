"""
teardown.py — Remove the billable serving resources for the readmission demo.

The Vertex endpoint (`n1-standard-2`) bills continuously — roughly $50-70/month —
whether or not anything calls it. Cloud Run services scale to zero and cost
nothing, so they are deliberately NOT touched here. This script is the single
biggest cost lever in the project.

What it removes:
    1. Deployed models on the readmission endpoint, then the endpoint itself
    2. The Feature Store online store and its feature views (BUILD_GUIDE section 5),
       which bill for provisioned serving nodes

What it deliberately keeps:
    - Model registry entries (readmission-final-*). They are free, and they carry
      the provenance that smoke_test.py uses to discover the serving bundle.
    - GCS bundles, BigQuery tables, Artifact Registry images. Storage is pennies
      and rebuilding them is expensive.

Rebuild with:
    .venv/bin/python projects/mlops/scripts/deploy_cpr.py

Usage (from repo root):
    .venv/bin/python projects/agent-harness/scripts/teardown.py --dry-run
    .venv/bin/python projects/agent-harness/scripts/teardown.py
    .venv/bin/python projects/agent-harness/scripts/teardown.py --yes
"""

import argparse
import os
import sys

from google.cloud import aiplatform

PROJECT = "trim-icon-498815-a0"
LOCATION = "us-east1"
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "readmission-endpoint")

# Feature Store resources are matched by prefix so a teardown written now still
# catches whatever section 5 ends up naming them.
FEATURE_STORE_PREFIX = os.environ.get("FEATURE_STORE_PREFIX", "readmission")


def _find_endpoints() -> list:
    return [
        ep for ep in aiplatform.Endpoint.list(order_by="create_time desc")
        if ep.display_name == ENDPOINT_NAME
    ]


def _teardown_endpoints(dry_run: bool) -> int:
    """Undeploy models and delete matching endpoints. Returns count removed."""
    endpoints = _find_endpoints()
    if not endpoints:
        print(f"  Endpoints: none named '{ENDPOINT_NAME}' — nothing to do")
        return 0

    removed = 0
    for ep in endpoints:
        deployed = list(ep.gca_resource.deployed_models)
        print(f"  Endpoint: {ep.display_name}  ({ep.resource_name})")
        for dm in deployed:
            print(f"      deployed model: {dm.display_name}  id={dm.id}")
        if not deployed:
            print("      (no deployed models)")

        if dry_run:
            print("      would undeploy all models, then delete the endpoint")
            removed += 1
            continue

        if deployed:
            print("      undeploying…")
            ep.undeploy_all(sync=True)
        print("      deleting endpoint…")
        ep.delete()
        removed += 1
        print("      removed")
    return removed


def _teardown_feature_store(dry_run: bool) -> int:
    """Delete matching Feature Store online stores and their views."""
    try:
        from google.cloud import aiplatform_v1
    except ImportError:
        print("  Feature Store: aiplatform_v1 unavailable — skipped")
        return 0

    api_endpoint = f"{LOCATION}-aiplatform.googleapis.com"
    parent = f"projects/{PROJECT}/locations/{LOCATION}"
    try:
        admin = aiplatform_v1.FeatureOnlineStoreAdminServiceClient(
            client_options={"api_endpoint": api_endpoint}
        )
        stores = [
            s for s in admin.list_feature_online_stores(parent=parent)
            if s.name.rsplit("/", 1)[-1].startswith(FEATURE_STORE_PREFIX)
        ]
    except Exception as e:  # API disabled, no permission, or not yet created
        print(f"  Feature Store: not queryable ({type(e).__name__}) — skipped")
        return 0

    if not stores:
        print(f"  Feature Store: no online store starting with "
              f"'{FEATURE_STORE_PREFIX}' — nothing to do")
        return 0

    removed = 0
    for store in stores:
        print(f"  Feature online store: {store.name}")
        try:
            views = list(admin.list_feature_views(parent=store.name))
        except Exception:
            views = []
        for v in views:
            print(f"      feature view: {v.name.rsplit('/', 1)[-1]}")

        if dry_run:
            print("      would delete feature views, then the online store")
            removed += 1
            continue

        for v in views:
            print(f"      deleting view {v.name.rsplit('/', 1)[-1]}…")
            admin.delete_feature_view(name=v.name).result()
        print("      deleting online store…")
        # force=True also removes any view this listing missed.
        admin.delete_feature_online_store(name=store.name, force=True).result()
        removed += 1
        print("      removed")
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be removed, change nothing")
    parser.add_argument("--yes", action="store_true",
                        help="skip the confirmation prompt")
    args = parser.parse_args()

    aiplatform.init(project=PROJECT, location=LOCATION)
    mode = "DRY RUN — nothing will be deleted" if args.dry_run else "LIVE"
    print(f"Teardown ({mode})")
    print(f"Project: {PROJECT}   Location: {LOCATION}\n")

    if not args.dry_run and not args.yes:
        print("This deletes the serving endpoint and any Feature Store online store.")
        print("Model registry entries, GCS bundles and BigQuery tables are kept.")
        print(f"Rebuild with: .venv/bin/python projects/mlops/scripts/deploy_cpr.py\n")
        if input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
            sys.exit("Aborted — nothing changed.")
        print()

    n_endpoints = _teardown_endpoints(args.dry_run)
    n_stores = _teardown_feature_store(args.dry_run)

    verb = "would remove" if args.dry_run else "removed"
    print(f"\nSummary: {verb} {n_endpoints} endpoint(s), {n_stores} feature store(s)")
    if not args.dry_run and n_endpoints == 0 and n_stores == 0:
        print("Nothing was billing — no action needed.")


if __name__ == "__main__":
    main()
