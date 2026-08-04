"""
teardown.py — Remove the billable serving resources for the readmission demo.

Two resources in this project bill by the hour for as long as they exist, with
no relationship to how much traffic they see:

    1. The Vertex prediction endpoint (n1-standard-2, ~$0.11/hour, ~$80/month)
    2. The Vector Search deployed index endpoint (e2-standard-2, ~$0.09/hour,
       ~$68/month)

Cloud Run services scale to zero and cost nothing idle, so they are deliberately
NOT touched here. This script is the single biggest cost lever in the project.

What it deliberately keeps:
    - Model registry entries (readmission-final-*). Free, and they carry the
      provenance that smoke_test.py uses to discover the serving bundle.
    - The Vector Search *index* itself. Only the deployed endpoint bills by the
      hour; the index is storage. Keeping it avoids paying the $3.00/GiB rebuild
      charge every time the demo is stood back up.
    - GCS bundles, BigQuery tables, Artifact Registry images. Storage is pennies
      and rebuilding them is expensive.

Rebuild with:
    .venv/bin/python projects/mlops/scripts/deploy_cpr.py             # endpoint
    .venv/bin/python projects/agent-harness/scripts/deploy_index.py   # vector

The two resources are independent and are wanted at different times — the
endpoint is needed for any prediction at all, the index endpoint only for
retrieval. Hence --only.

Usage (from repo root):
    .venv/bin/python projects/agent-harness/scripts/teardown.py --dry-run
    .venv/bin/python projects/agent-harness/scripts/teardown.py
    .venv/bin/python projects/agent-harness/scripts/teardown.py --yes
    .venv/bin/python projects/agent-harness/scripts/teardown.py --only endpoint
    .venv/bin/python projects/agent-harness/scripts/teardown.py --only vector-index
"""

import argparse
import os
import sys

from google.cloud import aiplatform

PROJECT = "trim-icon-498815-a0"
LOCATION = "us-east1"
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "readmission-endpoint")

# Matched by prefix so a teardown written now still catches the index endpoint
# whatever it ends up being named.
VECTOR_ENDPOINT_PREFIX = os.environ.get("VECTOR_ENDPOINT_PREFIX", "readmission")


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


def _teardown_vector_index(dry_run: bool) -> int:
    """Undeploy indexes and delete matching Vector Search index endpoints.

    The index itself is left alone on purpose — see the module docstring.
    """
    try:
        endpoints = [
            ep for ep in aiplatform.MatchingEngineIndexEndpoint.list()
            if ep.display_name.startswith(VECTOR_ENDPOINT_PREFIX)
        ]
    except Exception as e:  # API disabled, no permission, or none created yet
        print(f"  Vector Search: not queryable ({type(e).__name__}) — skipped")
        return 0

    if not endpoints:
        print(f"  Vector Search: no index endpoint starting with "
              f"'{VECTOR_ENDPOINT_PREFIX}' — nothing to do")
        return 0

    removed = 0
    for ep in endpoints:
        deployed = list(ep.gca_resource.deployed_indexes)
        print(f"  Index endpoint: {ep.display_name}  ({ep.resource_name})")
        for di in deployed:
            print(f"      deployed index: {di.id}  ({di.index})")
        if not deployed:
            print("      (no deployed indexes)")

        if dry_run:
            print("      would undeploy all indexes, then delete the endpoint")
            print("      (the index resource itself would be kept)")
            removed += 1
            continue

        for di in deployed:
            print(f"      undeploying {di.id}…")
            ep.undeploy_index(deployed_index_id=di.id)
        print("      deleting index endpoint…")
        ep.delete()
        removed += 1
        print("      removed")
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be removed, change nothing")
    parser.add_argument("--yes", action="store_true",
                        help="skip the confirmation prompt")
    parser.add_argument("--only", choices=("endpoint", "vector-index"),
                        help="restrict teardown to one resource (default: both)")
    args = parser.parse_args()

    do_endpoint = args.only in (None, "endpoint")
    do_vector = args.only in (None, "vector-index")

    aiplatform.init(project=PROJECT, location=LOCATION)
    mode = "DRY RUN — nothing will be deleted" if args.dry_run else "LIVE"
    print(f"Teardown ({mode})")
    print(f"Project: {PROJECT}   Location: {LOCATION}\n")

    if not args.dry_run and not args.yes:
        targets = " and ".join(
            t for t, wanted in (("the serving endpoint", do_endpoint),
                                ("any Vector Search index endpoint", do_vector))
            if wanted
        )
        print(f"This deletes {targets}.")
        print("Model registry entries, the vector index, GCS bundles and "
              "BigQuery tables are kept.")
        if do_endpoint:
            print("Rebuild endpoint: .venv/bin/python projects/mlops/scripts/deploy_cpr.py")
        if do_vector:
            print("Rebuild vector:   .venv/bin/python "
                  "projects/agent-harness/scripts/deploy_index.py")
        print()
        if input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
            sys.exit("Aborted — nothing changed.")
        print()

    if do_endpoint:
        n_endpoints = _teardown_endpoints(args.dry_run)
    else:
        n_endpoints = 0
        print("  Endpoints: skipped (--only vector-index)")

    if do_vector:
        n_vector = _teardown_vector_index(args.dry_run)
    else:
        n_vector = 0
        print("  Vector Search: skipped (--only endpoint)")

    verb = "would remove" if args.dry_run else "removed"
    print(f"\nSummary: {verb} {n_endpoints} endpoint(s), "
          f"{n_vector} index endpoint(s)")
    if not args.dry_run and n_endpoints == 0 and n_vector == 0:
        print("Nothing in scope was billing — no action needed.")


if __name__ == "__main__":
    main()
