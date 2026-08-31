"""Wait for the in-flight rag_tree_ah deploy to complete on the endpoint.

Does NOT create or deploy anything — only polls the endpoint until the
deployed index appears (or a timeout). Safe to run repeatedly.

Usage:
    .venv/bin/python projects/agent-harness/scripts/wait_rag_deploy.py
"""

import os
import sys
import time
from pathlib import Path

from google.cloud.aiplatform_v1 import IndexEndpointServiceClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp_server.config import (  # noqa: E402
    DEPLOYED_INDEX_ID as DEPLOYED_ID,
    LOCATION,
    PROJECT,
)


def main() -> int:
    c = IndexEndpointServiceClient(
        client_options={"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"})
    parent = f"projects/{PROJECT}/locations/{LOCATION}"
    ep_name = None
    for ep in c.list_index_endpoints(parent=parent):
        if ep.display_name == "readmission-rag-index":
            ep_name = ep.name
            break
    if ep_name is None:
        print("no readmission-rag-index endpoint found", flush=True)
        return 1

    started = time.monotonic()
    while time.monotonic() - started < 900:  # up to 15 min
        ep = c.get_index_endpoint(name=ep_name)
        for d in ep.deployed_indexes:
            if d.id == DEPLOYED_ID:
                print(f"DEPLOYED: {DEPLOYED_ID} → {d.index.split('/')[-1]} "
                      f"sync_time={d.index_sync_time}", flush=True)
                return 0
        print(f"  …waiting ({int(time.monotonic()-started)}s): "
              f"{len(ep.deployed_indexes)} deployed", flush=True)
        time.sleep(10)
    print("TIMEOUT: deploy did not complete in 15 min", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
