"""Deploy the rag tree-AH index to the existing (or a new) index endpoint.

Idempotent: if the endpoint already has `rag_tree_ah` deployed, it reports that
and exits 0. Otherwise it creates the endpoint if needed and deploys the index,
polling the LRO so a failure surfaces as a clear error instead of a silent hang.

Usage:
    .venv/bin/python projects/agent-harness/scripts/deploy_rag_endpoint.py
"""

import os
import sys
import time

from google.api_core.exceptions import AlreadyExists
from google.cloud.aiplatform_v1 import IndexEndpointServiceClient
from google.cloud.aiplatform_v1.types import (
    DeployedIndex,
    DedicatedResources,
    MachineSpec,
)

PROJECT = os.environ.get("PROJECT_ID", "trim-icon-498815-a0")
LOCATION = os.environ.get("LOCATION", "us-east1")
INDEX_ID = os.environ.get("INDEX_ID", "2371299135438454784")
ENDPOINT_NAME = os.environ.get("INDEX_ENDPOINT_NAME", "readmission-rag-index")
DEPLOYED_ID = os.environ.get("DEPLOYED_INDEX_ID", "rag_tree_ah")
MACHINE = os.environ.get("INDEX_MACHINE_TYPE", "e2-standard-16")
PARENT = f"projects/{PROJECT}/locations/{LOCATION}"


def _client() -> IndexEndpointServiceClient:
    return IndexEndpointServiceClient(
        client_options={"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"})


def _find_endpoint(c: IndexEndpointServiceClient) -> str | None:
    for ep in c.list_index_endpoints(parent=PARENT):
        if ep.display_name == ENDPOINT_NAME:
            return ep.name
    return None


def main() -> int:
    c = _client()
    index_name = f"{PARENT}/indexes/{INDEX_ID}"

    ep_name = _find_endpoint(c)
    if ep_name is None:
        print(f"creating endpoint {ENDPOINT_NAME!r}…", flush=True)
        op = c.create_index_endpoint(
            parent=PARENT,
            index_endpoint={
                "display_name": ENDPOINT_NAME,
                "public_endpoint_enabled": True,
            },
        )
        ep = op.result()
        ep_name = ep.name
        print(f"endpoint created: {ep_name}", flush=True)
    else:
        print(f"using existing endpoint: {ep_name}", flush=True)

    # Already deployed?
    current = c.get_index_endpoint(name=ep_name)
    for d in current.deployed_indexes:
        if d.id == DEPLOYED_ID:
            print(f"already deployed {DEPLOYED_ID} → {d.index}", flush=True)
            return 0

    di = DeployedIndex(
        id=DEPLOYED_ID,
        index=index_name,
        dedicated_resources=DedicatedResources(
            machine_spec=MachineSpec(machine_type=MACHINE),
            min_replica_count=1,
            max_replica_count=1,
        ),
    )
    print(f"deploying {DEPLOYED_ID} → {index_name} ({MACHINE})…", flush=True)
    try:
        op = c.deploy_index(index_endpoint=ep_name, deployed_index=di)
    except AlreadyExists:
        # A deploy with this id is already in flight (left by an earlier run
        # whose client died mid-poll). Just wait for it to finish below.
        print("  deploy already in progress; waiting for it to finish…", flush=True)
        op = None

    if op is not None:
        started = time.monotonic()
        while not op.done():
            if int(time.monotonic() - started) % 30 < 1:
                print(f"  …waiting {int(time.monotonic()-started)}s", flush=True)
            time.sleep(5)
        err = op.exception()
        if err is not None:
            print(f"DEPLOY FAILED: {err}", flush=True)
            return 1

    # Confirm the endpoint actually serves the index. This also covers the
    # AlreadyExists path: the earlier in-flight deploy may still be finishing,
    # and the deployed index only appears in the endpoint once deployment is
    # complete.
    started = time.monotonic()
    while True:
        ep = c.get_index_endpoint(name=ep_name)
        if any(d.id == DEPLOYED_ID for d in ep.deployed_indexes):
            print("DEPLOY DONE", flush=True)
            print("deployed:", [(d.id, d.index.split("/")[-1]) for d in ep.deployed_indexes])
            return 0
        if time.monotonic() - started > 1800:
            print("TIMEOUT waiting for deployed index to appear", flush=True)
            return 1
        if int(time.monotonic() - started) % 30 < 1:
            print(f"  …confirming {int(time.monotonic()-started)}s", flush=True)
        time.sleep(10)


if __name__ == "__main__":
    raise SystemExit(main())
