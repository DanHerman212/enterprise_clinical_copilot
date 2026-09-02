"""deploy_rag — blue-green deploy of the demo RAG index to the public endpoint.

Replaces the manual deploy_synthetic_rag / wait_rag_deploy / chain_rag_deploy
sequence with one scripted, gated step:

  1. Pick the newest ``rag-tree-ah-*`` index.
  2. Refuse anything over the synthetic-scale limit (ECC-36/53) — this is a
     PUBLIC endpoint, so only the demo (MTSamples) corpus may ever land on it.
  3. Deploy the index under a STAGING id, leaving the live id untouched.
  4. Gate: if ``--recall-report`` is given, fail the deploy unless the report
     passes the configured thresholds (recall@k from ``pipelines/recall_k.py``).
  5. Pass → undeploy the old live id, promote staging → live id.
     Fail → undeploy staging, keep the live id serving.

The full recall@k job (``pipelines/recall_k.py``) requires an endpoint to query;
for the demo corpus it is run against the staging id BEFORE promotion. For the
MIMIC corpus this same flow would need a private (PSC/VPC) endpoint — out of
scope for the public demo and left to Step 6/7.

Usage:
    .venv/bin/python projects/agent-harness/scripts/deploy_rag.py [--dry-run] \
        [--recall-report gs://.../recall_report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google.cloud.aiplatform_v1 import (
    IndexEndpointServiceClient,
    IndexServiceClient,
)
from google.cloud.aiplatform_v1.types import (
    DeployedIndex,
    DedicatedResources,
    MachineSpec,
)

from mcp_server.config import (  # noqa: E402
    DEPLOYED_INDEX_ID as LIVE_ID,
    INDEX_ENDPOINT_NAME as ENDPOINT_NAME,
    LOCATION,
    PROJECT,
)
from rag.config import load as load_config  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _deploy_guard import assert_synthetic_scale  # noqa: E402

PARENT = f"projects/{PROJECT}/locations/{LOCATION}"
STAGING_ID = "rag_tree_ah_staging"


def _client() -> IndexEndpointServiceClient:
    return IndexEndpointServiceClient(
        client_options={"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"})


def _index_client() -> IndexServiceClient:
    return IndexServiceClient(
        client_options={"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"})


def newest_tree_index() -> tuple[str, str, int]:
    """(resource_name, display_name, vectors) of the newest rag-tree-ah-* index."""
    client = _index_client()
    candidates = []
    for idx in client.list_indexes(parent=PARENT):
        if idx.display_name.startswith("rag-tree-ah-"):
            vectors = idx.index_stats.vectors_count if idx.index_stats else 0
            candidates.append((idx.create_time, idx.name, idx.display_name, vectors))
    if not candidates:
        raise SystemExit("no rag-tree-ah-* index found — run the ingest pipeline first")
    candidates.sort(reverse=True)
    for created, name, display, vectors in candidates:
        print(f"  {display:32s} vectors={vectors:<8}")
    return candidates[0][1], candidates[0][2], candidates[0][3]


def _find_endpoint(c: IndexEndpointServiceClient) -> str | None:
    for ep in c.list_index_endpoints(parent=PARENT):
        if ep.display_name == ENDPOINT_NAME:
            return ep.name
    return None


def _deploy(c: IndexEndpointServiceClient, ep_name: str, index_name: str,
            deployed_id: str, machine: str) -> None:
    di = DeployedIndex(
        id=deployed_id,
        index=index_name,
        dedicated_resources=DedicatedResources(
            machine_spec=MachineSpec(machine_type=machine),
            min_replica_count=1,
            max_replica_count=1,
        ),
    )
    print(f"deploying {deployed_id} -> {index_name.split('/')[-1]} ({machine})…")
    op = c.deploy_index(index_endpoint=ep_name, deployed_index=di)
    while not op.done():
        time.sleep(5)
    if op.exception() is not None:
        raise SystemExit(f"deploy failed: {op.exception()}")


def _undeploy(c: IndexEndpointServiceClient, ep_name: str, deployed_id: str) -> None:
    print(f"undeploying {deployed_id}…")
    c.undeploy_index(index_endpoint=ep_name, deployed_index_id=deployed_id)


def _deployed_ids(c: IndexEndpointServiceClient, ep_name: str) -> list[str]:
    return [d.id for d in c.get_index_endpoint(name=ep_name).deployed_indexes]


def _gate_passes(report_uri: str | None) -> bool:
    """True if the recall report passes the configured thresholds.

    When no report is supplied, the gate is vacuous (deploy proceeds) — the
    caller is responsible for running recall_k.py against the staging id first.
    """
    if not report_uri:
        return True

    from google.cloud import storage

    from rag.eval_report import write_artifacts
    from rag.gate import recall_result

    cfg = load_config()
    bucket, obj = report_uri.replace("gs://", "", 1).split("/", 1)
    report = json.loads(
        storage.Client().bucket(bucket).blob(obj).download_as_text())

    result = recall_result(
        report=report,
        corpus=cfg.corpus.name,
        index_name="demo-index",
        recall_min={"recall_at_10": cfg.recall_at_10_min},
        empty_max={"empty_result_rate": cfg.empty_result_rate_max},
    )
    write_artifacts(result, Path("/tmp/rag_eval"))
    passed, failing = result.verdict()
    if not passed:
        print(f"recall gate FAILED: {failing}")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--recall-report", default=None,
                        help="gs:// path to a recall_k.py report to gate on")
    args = parser.parse_args()

    cfg = load_config()
    machine = cfg.machine_type
    index_name, display, vectors = newest_tree_index()
    assert_synthetic_scale(vectors, display)

    c = _client()
    ep_name = _find_endpoint(c)
    if ep_name is None:
        if args.dry_run:
            print(f"dry-run: would create endpoint {ENDPOINT_NAME!r}")
        else:
            op = c.create_index_endpoint(
                parent=PARENT,
                index_endpoint={
                    "display_name": ENDPOINT_NAME,
                    "public_endpoint_enabled": True,
                },
            )
            ep_name = op.result().name
            print(f"endpoint created: {ep_name}")
    else:
        print(f"using endpoint: {ep_name}")

    live_present = LIVE_ID in _deployed_ids(c, ep_name) if not args.dry_run else False

    if args.dry_run:
        print(f"dry-run: deploy {display} as {STAGING_ID} → gate → "
              f"{'promote to ' + LIVE_ID if _gate_passes(args.recall_report) else 'rollback'}")
        return 0

    # Blue: deploy the new index under a staging id, live id untouched.
    _deploy(c, ep_name, index_name, STAGING_ID, machine)

    # Green: gate, then promote or roll back.
    if _gate_passes(args.recall_report):
        if live_present:
            _undeploy(c, ep_name, LIVE_ID)
        _deploy(c, ep_name, index_name, LIVE_ID, machine)
        _undeploy(c, ep_name, STAGING_ID)
        print(f"PROMOTED {display} to {LIVE_ID}")
    else:
        _undeploy(c, ep_name, STAGING_ID)
        print("ROLLED BACK — live id untouched")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
